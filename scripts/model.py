import time
import gurobipy as gp
from gurobipy import GRB
import gc
import os

class OR_model:

    def __init__(self, df=None, P=None, F=None, l_v=None, u_v=None, alpha=0, utils=None, model_path=None):
        """
        Initializes OR_model. Either loads a model (and solution) from files or sets up a new model.

        Parameters:
        - df: DataFrame (Optional, required if creating a new model)
        - P: List of allowed moves (Optional, required if creating a new model)
        - F: List of floors (Optional, required if creating a new model)
        - l_v: List of lower bounds of service v (Optional, required if creating a new model)
        - u_v: List of upper bounds of service v (Optional, required if creating a new model)
        - alpha: Penalty term of the amount of moving blocks
        - utils: Utilities object (Optional, required if creating a new model)
        - model_path: File path to a saved Gurobi model (MPS, LP, SAV)
        - sol_path: Optional path to a Gurobi solution file (.sol) to load solution values
        """
        self.model = None  # Initialize model attribute
        self.is_loaded_model = False

        if model_path and df is not None:

            self.df = df

            sol_path = f"{model_path}.sol"
            model_path = f"{model_path}.mps"
            # Load model from file
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found.")

            print("\n\n", "-" * 10, f"Loading model from '{model_path}'...", "-" * 10)
            self.model = gp.read(model_path)  # Load Gurobi model
            self.is_loaded_model = True
            print("Model loaded successfully!")

            # Try to load the solution
            if not os.path.exists(sol_path):
                print(f"Solution file '{sol_path}' not found. Skipping solution load.")
            else:
                try:
                    self.model.read(sol_path)  # Apply solution values to the model
                    print(f"Solution loaded successfully from '{sol_path}'")
                except Exception as e:
                    print(f"Failed to load solution file: {e}")

        else:
            # Initialize a new model
            if df is None or P is None or F is None or l_v is None or u_v is None or utils is None:
                raise ValueError("Missing required parameters for creating a new model.")

            self.df = df
            self.P = P
            self.F = F
            self.l_v = l_v
            self.u_v = u_v
            self.alpha = alpha
            self.utils = utils

            print("\n\n", "-" * 10, "Creating model", "-" * 10)
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
                    self.model.addConstr(gp.quicksum(x_p[p] for p in self.P if self.utils.getDay(self.utils.getBs(p)[1]) == d and self.utils.Di_dict.get((self.utils.getBs(p)[0], i, s), False)) <= 1)

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

        self.model.setObjective(sum(peak - z[f] for f in self.F) + self.alpha * sum(x_p[p] for p in self.utils.repeat_blocks), GRB.MAXIMIZE)

        print(f"Finish, uses {time.time() - start:.5f} seconds")




    def optimize(self):
        # if self.is_loaded_model:
        #     raise RuntimeError("Cannot optimize a loaded model.")
        # Optimize the model
        print("\n\n", "-"*10, "Optimizing", "-"*10)
        start = time.time()

        self.model.optimize()

        print(f"Finish, uses {time.time() - start:.5f} seconds")



    def get_model(self):
        return self.model



    def printResults(self, ignore_0=True):
        """Prints all decision variables and their values."""
        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal Solution Found!")

            # Iterate through all decision variables in the model
            print("\nDecision Variables:")
            for var in self.model.getVars():
                if ignore_0 and var.x > 0.5:
                    print(f"{var.varName} = {var.x}")

        elif self.model.status == GRB.INFEASIBLE:
            print("\nNo feasible solution found. The model is infeasible.")

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
        
        # Force garbage collection
        gc.collect()

        print("Model successfully destroyed.")



    def saveModel(self, filename="or_model"):
        """
        Saves the Gurobi model and its solution to .mps and .sol files.

        - Model is saved as 'filename.mps'
        - Solution (if available) is saved as 'filename.sol'
        """
        print("\n\n", "-" * 10, "Saving model and solution", "-" * 10)

        if not hasattr(self, "model") or self.model is None:
            print("No model found to save.")
            return

        try:
            # Save the model in .mps format
            model_path = f"{filename}.mps"
            self.model.write(model_path)
            print(f"Model saved successfully as '{model_path}'")

            # Check if the model has been solved and has a solution
            if self.model.SolCount > 0:
                sol_path = f"{filename}.sol"
                self.model.write(sol_path)
                print(f"Solution saved successfully as '{sol_path}'")
            else:
                print("No solution found. Skipping saving solution file.")

        except Exception as e:
            print(f"Failed to save model or solution: {e}")
