"""
Test a learner.  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---
"""

import sys
import numpy as np
import pandas as pd
import math
from DTLearner import DTLearner
from BagLearner import BagLearner
from RTLearner import RTLearner
import matplotlib.pyplot as plt
import time

def calc_rmse(actY, predY):
    rmse = math.sqrt(((actY - predY) ** 2).sum()/actY.shape[0])
    return rmse

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1], index_col='date')
    data = df.to_numpy()
    rows = np.random.permutation(data.shape[0])
    cutoff = int(data.shape[0] * .6)
    train_data = data[rows[:cutoff], :]
    test_data = data[rows[cutoff:], :]
    trainX = train_data[:, :-1]
    trainY = train_data[:, -1]
    testX = test_data[:, :-1]
    testY = test_data[:, -1]

    train_rmse = []
    test_rmse = []
    max_leaf = 50
    for i in range(1, max_leaf + 1):
        learner = DTLearner(i)
        learner.addEvidence(trainX, trainY)
        train_rmse.append(calc_rmse(trainY, learner.query(trainX)))
        test_rmse.append(calc_rmse(testY, learner.query(testX)))

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(train_rmse, label='Train RMSE', marker='o')
    plt.plot(test_rmse, label='Test RMSE', marker='o')
    plt.xlim(1, max_leaf)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("RMSE of DTLearner with leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(0, max_leaf, 5))
    plt.yticks(np.arange(0, 1, 0.1) * .01)
    plt.savefig("dt_learner_leaf.png", format="PNG")

    train_rmse = []
    test_rmse = []
    max_leaf = 50
    bags = 30

    for i in range(1, max_leaf + 1):
        learner = BagLearner(DTLearner, {"leaf_size": i}, bags)
        learner.addEvidence(trainX, trainY)
        train_rmse.append(calc_rmse(trainY, learner.query(trainX)))
        test_rmse.append(calc_rmse(testY, learner.query(testX)))

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(train_rmse, label='Train RMSE', marker='o')
    plt.plot(test_rmse, label='Test RMSE', marker='o')
    plt.xlim(1, max_leaf)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("RMSE of BagLearner with leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(0, max_leaf, 5))
    plt.yticks(np.arange(0, 1, 0.1) * .01)
    plt.savefig("bag_learner_leaf.png", format="PNG")

    rt_train_rmse = []
    rt_test_rmse = []
    dt_train_rmse = []
    dt_test_rmse = []
    max_leaf = 50
    for i in range(1, max_leaf + 1):
        rt = RTLearner(i)
        rt.addEvidence(trainX, trainY)
        rt_train_rmse.append(calc_rmse(trainY, rt.query(trainX)))
        rt_test_rmse.append(calc_rmse(testY, rt.query(testX)))
        dt = DTLearner(i)
        dt.addEvidence(trainX, trainY)
        dt_train_rmse.append(calc_rmse(trainY, dt.query(trainX)))
        dt_test_rmse.append(calc_rmse(testY, dt.query(testX)))

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(rt_test_rmse, label='RTLearner RMSE', marker='o')
    plt.plot(dt_test_rmse, label='DTLearner RMSE', marker='o')
    plt.xlim(1, max_leaf)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("RMSE comparison of DT Learner with RT Learner wrt. leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(0, max_leaf, 5))
    plt.yticks(np.arange(6, 10, .5) * .001)
    plt.savefig("rt_vs_dt_error.png", format="PNG")

    rt_time = []
    dt_time = []
    rt_size = []
    training_samples = []
    for i in range(1, data.shape[0], 10):
        rows = np.random.permutation(data.shape[0])
        tmp_data = data[rows[:i]]
        training_samples.append(tmp_data.shape[0])
        trainX = tmp_data[:, :-1]
        trainY = tmp_data[:, -1]
        rt = RTLearner(20)
        start = time.time()
        rt.addEvidence(trainX, trainY)
        end = time.time()
        rt_time.append(end - start)
        dt = DTLearner(20)
        start = time.time()
        dt.addEvidence(trainX, trainY)
        end = time.time()
        dt_time.append(end - start)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(training_samples, rt_time, label='RTLearner', marker='o')
    plt.plot(training_samples, dt_time, label='DTLearner', marker='o')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.title("Training time of RT and DT Learner")
    plt.xlabel("Number of training samples")
    plt.ylabel("Time in seconds")
    plt.savefig("rt_vs_dt_time.png", format="PNG")

    rt_space = []
    dt_space = []
    rt_size = []
    training_samples = []
    for i in range(1, data.shape[0], 10):
        rows = np.random.permutation(data.shape[0])
        tmp_data = data[rows[:i]]
        training_samples.append(tmp_data.shape[0])
        trainX = tmp_data[:, :-1]
        trainY = tmp_data[:, -1]
        rt = RTLearner(20)
        rt.addEvidence(trainX, trainY)
        rt_space.append(rt.tree.shape[0])
        dt = DTLearner(20)
        dt.addEvidence(trainX, trainY)
        dt_space.append(dt.tree.shape[0])

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(training_samples, rt_space, label='RTLearner', marker='o')
    plt.plot(training_samples, dt_space, label='DTLearner', marker='o')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.title("RT vs DT Learner in tree size")
    plt.xlabel("Number of training samples")
    plt.ylabel("Size of tree (#nodes)")
    plt.savefig("rt_vs_dt_space.png", format="PNG")
