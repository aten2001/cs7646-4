import numpy as np
import pandas as pd
import math


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):
        kwargs["verbose"] = verbose
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))
        pass

    def author(self):
        return 'sgarg96'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        num_samples = dataX.shape[0]
        for learner in self.learners:
            indices = np.random.choice(np.arange(num_samples), num_samples)
            learner.addEvidence(dataX[indices], dataY[indices])


    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to
        a specific query.
        @returns the estimated values according to the saved model.
        """
        results = []
        for learner in self.learners:
            results.append(learner.query(points))
        results = np.array(results)
        return np.mean(results, axis=0)