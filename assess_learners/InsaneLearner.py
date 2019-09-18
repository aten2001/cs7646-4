from BagLearner import BagLearner
from LinRegLearner import LinRegLearner


class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner = BagLearner(BagLearner, {"learner": LinRegLearner,
                                               "kwargs": {},
                                               "bags": 20, "verbose": verbose},
                                  20, False,
                                  verbose)

    def author(self):
        return 'sgarg96'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.learner.addEvidence(dataX, dataY)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to
        a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.learner.query(points)
