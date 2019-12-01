# Student Name: Sanjana Garg (replace with your name)
# GT User ID: sgarg96 (replace with your User ID)
# GT ID: 903475801 (replace with your GT ID)

import numpy as np
import pandas as pd
import math
from scipy import stats


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size

    def author(self):
        return 'sgarg96'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        data = np.concatenate((dataX, dataY.reshape(-1, 1)), axis=1)
        self.tree = self.buildTree(data)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to
        a specific query.
        @returns the estimated values according to the saved model.
        """
        predictions = []
        for i in range(points.shape[0]):
            predictions.append(self.getPred(points[i]))
        return np.array(predictions)

    def getPred(self, point):
        idx = 0
        while not math.isnan(self.tree[idx, 0]):
            if point[self.tree[idx, 0]] <= self.tree[idx, 1]:
                idx = idx + self.tree[idx, 2]
            else:
                idx = idx + self.tree[idx, 3]
        return self.tree[idx, 1]

    def buildTree(self, data):
        if data.shape[0] <= self.leaf_size:
            return np.array([[np.nan, stats.mode(data[:, -1]).mode[0], np.nan,
                              np.nan]],
                            dtype=object)
        if self.isYSame(data):
            return np.array([[np.nan, data[0, -1], np.nan, np.nan]],
                            dtype=object)

        feature = self.getBestFeatureToSplit(data)
        sv = self.getSplitVal(data, feature)
        left_split = data[data[:, feature] <= sv]
        right_split = data[data[:, feature] > sv]
        if np.array_equal(left_split, data):
            return np.array([[np.nan, stats.mode(data[:, -1]).mode[0], np.nan,
                              np.nan]],
                            dtype=object)
        left = self.buildTree(left_split)
        right = self.buildTree(right_split)
        root = np.array([[feature, sv, 1, left.shape[0] + 1]], dtype=object)
        tree = np.concatenate((root.reshape(1, -1), left, right))
        return tree

    def isYSame(self, data):
        if np.unique(data[:, -1]).shape[0] == 1:
            return True
        return False

    def getBestFeatureToSplit(self, data):
        num_features = data.shape[1] - 1
        return np.random.randint(0, num_features)

    def getSplitVal(self, data, feature):
        return np.median(data[:, feature])
