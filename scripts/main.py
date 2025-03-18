from utils import utils
from model import OR_model

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB



def readData():
    print("\n\n", "-"*10, "Reading data", "-"*10)
    df = pd.read_csv('dataset.csv')

    print("Data size:", df.shape)

    print("\n\n", "-"*10, "Do precalculations", "-"*10)
    start = time.time()

    utils_ = utils(df)

    print(f"Finish, uses {time.time() - start:.5f} seconds")

    return df, utils_


def getAllowedMoves(df, utils):
    B = df['Block'].unique()
    P_set = set()

    for b1, b2 in itertools.product(B, repeat=2):
        o1 = utils.getRoom(b1)
        o2 = utils.getRoom(b2)
        if abs(o1 - o2) <= 10:
            P_set.add((b1,b2))

    return list(P_set)



if __name__ == "__main__":
    df, utils = readData()
    P = getAllowedMoves(df, utils)

    # Hyperparameters
    F = ['1']
    l_v = [0,0,0,0,0]
    u_v = [150,150,150,150,150]


    model = OR_model(df=df, P=P, F=F, l_v=l_v, u_v=u_v, utils=utils)
    model.init()
    model.optimize()
    model.saveModel(filename="basic_model")





