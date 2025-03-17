import time
import gurobipy as gp
from gurobipy import GRB
import gc
import os

class OR_model:

    def __init__(self, df=None, P=None, F=None, l_v=None, u_v=None, utils=None, model_path=None):
        """
        Initializes OR_model. Either loads a model from a file or sets up a new model.

        Parameters:
        - df: DataFrame (Optional, required if creating a new model)
        - P: List of allowed moves (Optional, required if creating a new model)
        - F: List of floors (Optional, required if creating a new model)
        - l_v: List of lower bound of service v (Optional, required if creating a new model)
        - u_v: List of upper bound of service v (Optional, required if creating a new model)
        - utils: Utilities object (Optional, required if creating a new model)
        - model_path: File path to a saved Gurobi model (MPS, LP, JSON)
        """
        self.model = None  # Initialize model attribute
        self.is_loaded_model = False

        if model_path and df:
            # Load model from file
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found.")
            
            print(f"Loading model from '{model_path}'...")
            self.model = gp.read(model_path)  # Load Gurobi model
            self.is_loaded_model = True
            print("Model loaded successfully!")
        else:
            # Initialize a new model
            if df is None or P is None or F is None or l_v is None or u_v is None or utils is None:
                raise ValueError("Missing required parameters for creating a new model.")
            
            self.df = df
            self.P = P
            self.F = F
            self.l_v = l_v
            self.u_v = u_v
            self.utils = utils

            # Create a new Gurobi model
            self.model = gp.Model("OR opt")
            print("New model created successfully.")



    def init(self):
        if self.is_loaded_model:
            raise RuntimeError("Cannot initialize a loaded model.")
        # Define sets
        print("\n\n", "-"*10, "Reading sets from df and precompute", "-"*10)
        start = time.time()

        W = self.df['Weekday'].unique()            #weekday
        N = self.df['Week Number'].unique()        #OR
        O = self.df['ORs'].unique()                #OR
        B = self.df['Block'].unique()              #Block
        S = self.df['Surgeon ID'].unique()         #Surgeon
        V = self.df['Service ID'].unique()         #Service

        peak = 338


        # Variables defined here is some precompute values, this is used to reduce complexity of constraints
        Di_dict = {(b, w, s): not self.df[(self.df['Block'] == b) & (self.df['Week Number'] == w) & (self.df['Surgeon ID'] == s)].empty
                for b in B for w in N for s in S}



        print(f"Finish, uses {time.time() - start:.5f} seconds")



        # Define decision variables
        print("\n\n", "-"*10, "Setting decision variables", "-"*10)
        start = time.time()

        x_p = self.model.addVars(self.P, vtype=GRB.BINARY, name="x_p")  # Allowed Moves
        z = self.model.addVars(self.F, vtype=GRB.INTEGER, name="z")

        summed_values = self.model.addVars(W, N, vtype=GRB.INTEGER, name="sum_var") # Total bed counts for weekday d in week i

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # Define constraint
        print("\n\n", "-"*10, "Setting constraints", "-"*10)


        # Additional (not from original paper)

        print("Setting additional constraints")
        start = time.time()

        for b1 in B:  # This constraint makes sure that each block can only move to one place
            self.model.addConstr(gp.quicksum(x_p[p] for p in self.P if b1 == self.utils.getBs(p)[0]) == 1)

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # (2)

        print("Setting constraints (2)")
        start = time.time()

        for b2 in B:
            self.model.addConstr(gp.quicksum(x_p[p] for p in self.P if b2 == self.utils.getBs(p)[1]) == 1)

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # (3)

        print("Setting constraints (3)")
        start = time.time()

        for f in self.F:
            for d in W: # Weekday
                for i in N:
                    self.model.addConstr(gp.quicksum(self.utils.cases(self.utils.getBs(p)[0])[i-1] * x_p[p] for p in self.P if self.utils.getDay(self.utils.getBs(p)[1]) == d) == summed_values[d, i])
                    self.model.addConstr(summed_values[d, i] <= z[f])

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # (4)

        print("Setting constraints (4)")
        start = time.time()

        for s in S:
            for i in N:
                for d in W:

                    # model.addConstr(gp.quicksum(x_p[p] for p in P if getDay(getBs(p)[1]) == d and Di(getBs(p)[0], i, s)) <= 1)
                    self.model.addConstr(gp.quicksum(x_p[p] for p in self.P if self.utils.getDay(self.utils.getBs(p)[1]) == d and Di_dict.get((self.utils.getBs(p)[0], i, s), False)) <= 1)

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # (5)

        print("Setting constraints (5)")
        start = time.time()

        for v in V:
            for d in W:
                self.model.addConstr(gp.quicksum(self.utils.alloc(self.utils.getBs(p)[0], v) * x_p[p] for p in self.P if self.utils.getDay(self.utils.getBs(p)[1]) == d) <= self.u_v[v])
                self.model.addConstr(gp.quicksum(self.utils.alloc(self.utils.getBs(p)[0], v) * x_p[p] for p in self.P if self.utils.getDay(self.utils.getBs(p)[1]) == d) >= self.l_v[v])

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # Objective
        print("-"*10, "Setting objection", "-"*10)
        start = time.time()

        self.model.setObjective(sum(peak - z[f] for f in self.F), GRB.MAXIMIZE)

        print(f"Finish, uses {time.time() - start:.5f} seconds")




    def optimize(self):
        if self.is_loaded_model:
            raise RuntimeError("Cannot optimize a loaded model.")
        # Optimize the model
        print("\n\n", "-"*10, "Optimizing", "-"*10)
        start = time.time()

        self.model.optimize()

        print(f"Finish, uses {time.time() - start:.5f} seconds")



    def get_model(self):
        return self.model



    def printResults(self):
        """Prints all decision variables and their values."""
        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal Solution Found!")

            # Iterate through all decision variables in the model
            print("\nDecision Variables:")
            for var in self.model.getVars():
                print(f"{var.varName} = {var.x}")

        elif self.model.status == GRB.INFEASIBLE:
            print("\ No feasible solution found. The model is infeasible.")

        elif self.model.status == GRB.UNBOUNDED:
            print("\nThe model is unbounded.")

        else:
            print("\nNo optimal solution found. Model status:", self.model.status)




    def destroy(self):
        """Explicitly releases memory used by the model."""
        print("\n\n", "-" * 10, "Destroying model", "-" * 10)
        
        if hasattr(self, 'model') and self.model is not None:
            self.model.dispose()  # Free Gurobi model memory
            self.model = None
        
        # Delete variables explicitly to release references
        del self.x_p, self.z, self.summed_values
        
        # Force garbage collection
        gc.collect()

        print("Model successfully destroyed.")



    def saveModel(self, filename="or_model", format="mps"):
        """
        Saves the Gurobi model to a file.

        Parameters:
        - filename (str): Name of the file (without extension).
        - format (str): Format of the file ('mps', 'lp', 'json', 'sol').
        """
        if not hasattr(self, "model") or self.model is None:
            print("No model found to save.")
            return

        # Define valid formats
        valid_formats = {"mps", "lp", "json", "sol"}

        if format not in valid_formats:
            print(f"Invalid format '{format}'. Using default 'mps'.")
            format = "mps"

        filepath = f"{filename}.{format}"

        try:
            self.model.write(filepath)
            print(f"Model saved successfully as '{filepath}'")
        except Exception as e:
            print(f"Failed to save model: {e}")
