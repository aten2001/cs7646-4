import numpy as np
import pandas as pd
import math


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
                idx = idx + int(self.tree[idx, 2])
            else:
                idx = idx + int(self.tree[idx, 3])
        return self.tree[idx, 1]

    def buildTree(self, data):
        if data.shape[0] <= self.leaf_size:
            return np.array(
                [np.nan, np.mean(data[:, -1]), np.nan.np.nan]).reshape(
                (1, -1))
        if self.isYSame(data):
            return np.array([np.nan, data[0, -1], np.nan.np.nan]).reshape(
                (1, -1))
        if self.isXSame(data):
            return np.array(
                [np.nan, np.mean(data[:, -1]), np.nan, np.nan]).reshape(
                (1, -1))
        feature = self.getBestFeatureToSplit(data)
        sv = self.getSplitVal(data, feature)
        left = self.buildTree(data[data[:, feature] <= sv])
        right = self.buildTree(data[data[:, feature] > sv])
        root = np.array([feature, sv, 1, left.shape[0]])
        tree = np.concatenate((root.reshape(1, -1), left, right))
        return tree

    def isYSame(self, data):
        y = data[:, -1]
        if y[y[0] == y].shape[0] == y.shape[0]:
            return True
        return False

    def isXSame(self, data):
        x = data[:, :-1]
        if (x[0] == x).size == x.size:
            return True
        return False

    def getBestFeatureToSplit(self, data):
        df = pd.DataFrame(data)
        return df.corr().iloc[:, -1][:-1].idxmax()

    def getSplitVal(self, data, feature):
        return np.median(data[:, feature])
