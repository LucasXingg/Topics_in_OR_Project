import pandas as pd
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
m = df.shape[0]


# Define days and operating rooms
days = [f'd{i}' for i in range(1, 8)]
rooms = [f'o{i}' for i in range(1, 53)]

# Generate all valid block pairs P
P = []
for (d1, o1), (d2, o2) in itertools.product(itertools.product(days, rooms), repeat=2):
    room_num1, room_num2 = int(o1[1:]), int(o2[1:])  # Extract room numbers
    if abs(room_num1 - room_num2) <= 10:
        new_item = (f"({d1},{o1})", f"({d2},{o2})")
        if not new_item in P:
            P.append(new_item)

# Print a sample of P to check

def getDay(b):
    day = b.split('d')[1][0]
    return int(day)

def cases(b):
    # print(b)
    line = df[(df['Block'] == b)]
    # print(list(line['Beds']))
    return list(line['Beds'])

def alloc(b, v):
    line = df[(df['Block'] == b) & (df['Service ID'] == v)]
    return len(line)

def Di(b, w, s):
    line = df[(df['Block'] == b) & (df['Week Number'] == w) & (df['Surgeon ID'] == s)]
    return not line.empty  # Returns True if at least one row exists

def getBs(p):
    b1, b2 = p
    return [b1, b2]

import gurobipy as gp
from gurobipy import GRB

# Define sets
W = df['Weekday'].unique()            #weekday
N = df['Week Number'].unique()        #OR
O = df['ORs'].unique()                #OR
B = df['Block'].unique()              #Block
S = df['Surgeon ID'].unique()         #Surgeon
V = df['Service ID'].unique()         #Service
totalBedCount = sum(df['Beds'].to_numpy())

l_v = [0,0,0,0,0]                     # lower bound for each service
u_v = [150,150,150,150,150]           # upper bound for each service



F = ['1']
peak = 338

# Create a new model
model = gp.Model("OR opt")

# Define decision variables
x_p = model.addVars(P, vtype=GRB.BINARY, name="x_p")  # Allowed Moves
z = model.addVars(F, vtype=GRB.INTEGER, name="z")

summed_values = model.addVars(W, N, vtype=GRB.INTEGER, name="sum_var") # Total bed counts for weekday d in week i


# Define constraint

# Additional (not from original paper)

for b1 in B:  # This constraint makes sure that each block can only move to one place
    model.addConstr(gp.quicksum(x_p[p] for p in P if b1 == getBs(p)[0]) == 1)

model.addConstr(gp.quicksum(summed_values[d, i] for d in W for i in N) == totalBedCount)

# (2)

for b2 in B:
    model.addConstr(gp.quicksum(x_p[p] for p in P if b2 == getBs(p)[1]) == 1)

# (3)

for f in F:
    for d in W: # Weekday
        for i in N:
            model.addConstr(gp.quicksum(cases(getBs(p)[0])[i-1] * x_p[p] for p in P if getDay(getBs(p)[1]) == d) == summed_values[d, i])
        for i in N:
            model.addConstr(summed_values[d, i] <= z[f])

# (4)

for s in S:
    for i in N:
        for d in W:

            model.addConstr(gp.quicksum(x_p[p] for p in P if getDay(getBs(p)[1]) == d and Di(getBs(p)[0], i, s)) <= 1)

# (5)

for v in V:
    for d in W:
        model.addConstr(gp.quicksum(alloc(getBs(p)[0], v) * x_p[p] for p in P if getDay(getBs(p)[1]) == d) <= u_v[v])
        model.addConstr(gp.quicksum(alloc(getBs(p)[0], v) * x_p[p] for p in P if getDay(getBs(p)[1]) == d) >= l_v[v])


model.setObjective(sum(peak - z[f] for f in F), GRB.MAXIMIZE)

# Optimize the model
model.optimize()