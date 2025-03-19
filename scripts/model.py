from scripts.resultData import ResultData

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

import gc
import os
import re
import time

class OR_model:

    def __init__(self, df=None, P=None, F=None, l_v=None, u_v=None, alpha=0, utils=None):
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
        """
        self.model = None  # Initialize model attribute
        self.result = None  # Initialize result attribute

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

        # for d in W: # Weekday
        #     if not d == 4:
        #         self.model.addConstr()

        print(f"Finish, uses {time.time() - start:.5f} seconds")


        # Thursday

        t_d = self.model.addVars(W, vtype=GRB.BINARY, name="t_d")

        self.model.addConstr(gp.quicksum(t_d[d] for d in W) == 1)

        for p in self.P:
            if self.utils.getDay(self.utils.getBs(p)[0]) == 4:
                p_d = self.utils.getDay(self.utils.getBs(p)[1]) # destination day
                self.model.addConstr(x_p[p] <= t_d[p_d])


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
        # Optimize the model
        print("\n\n", "-"*10, "Optimizing", "-"*10)
        start = time.time()

        self.model.optimize()

        print(f"Finish, uses {time.time() - start:.5f} seconds")

        self.result = ResultData(model=self.model, df=self.df, utils=self.utils)


    def getModel(self):
        return self.model
    
    def getResult(self):
        return self.result


    def destroy(self):
        """Explicitly releases memory used by the model."""
        print("\n\n", "-" * 10, "Destroying model", "-" * 10)
        
        if hasattr(self, 'model') and self.model is not None:
            self.model.dispose()  # Free Gurobi model memory
            self.model = None
        
        # Force garbage collection
        gc.collect()

        print("Model successfully destroyed.")


if __name__ == "__main__":
    
    import pandas as pd
    from utils import utils
    import itertools

    pass