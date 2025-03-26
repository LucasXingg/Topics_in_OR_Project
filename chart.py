from scripts.utils import utils
from scripts.model import OR_model
from scripts.resultData import ResultData

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('dataset.csv')

def readData(pre=True):
    print("\n\n", "-"*10, "Reading data", "-"*10)
    df = pd.read_csv('dataset.csv')

    print("Data size:", df.shape)

    print("\n\n", "-"*10, "Do precalculations", "-"*10)
    start = time.time()

    utils_ = utils(df, pre=pre)

    print(f"Finish, uses {time.time() - start:.5f} seconds")

    return df, utils_

df, utils_ = readData(pre=False)

load_result = ResultData(filepath='./saves/result_P7_a0.csv',df=df, utils=utils_)
load_result.plot(save="Chief_20.png")