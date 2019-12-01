"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

Student Name: Sanjana Garg (replace with your name)
GT User ID: sgarg96 (replace with your User ID)
GT ID: 903475801 (replace with your GT ID)
"""

import datetime as dt

import pandas as pd

import util as ut
from BagLearner import BagLearner
from RTLearner import RTLearner
from indicators import calc_bb_ratio, calc_sma_ratio, \
    calc_momentum


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False, impact=0):
        self.verbose = verbose
        self.impact = impact
        self.learner = BagLearner(learner=RTLearner, kwargs={"leaf_size": 5},
                                  bags=20, boost=False, verbose=False)
        self.lookback = 14
        self.lookforward = 14
        self.impact = impact

    def getFeatures(self, prices, symbol):
        sma_ratio = calc_sma_ratio(prices, self.lookback)
        bbratio = calc_bb_ratio(prices, self.lookback)
        momentum = calc_momentum(prices, self.lookback)
        sma_ratio.rename(columns={symbol: "smaratio"}, inplace=True)
        bbratio.rename(columns={symbol: "bbratio"}, inplace=True)
        momentum.rename(columns={symbol: "momentum"}, inplace=True)

        X = sma_ratio.join([bbratio, momentum])
        X.dropna(inplace=True)

        if self.verbose:
            print(X.shape, X.columns)
        return X

    def get_trades(self, predY):
        trades = []
        net_holdings = 0
        min_holdings = -1000
        max_holdings = 1000

        for i in range(predY.shape[0]):
            if predY[i] == -1 and net_holdings > min_holdings:  # sell
                num_shares = min_holdings - net_holdings
                trades.append(num_shares)
                net_holdings = min_holdings
            elif predY[i] == 1 and net_holdings < max_holdings:  # buy
                num_shares = max_holdings - net_holdings
                trades.append(num_shares)
                net_holdings = max_holdings
            else:
                trades.append(0)
        return trades

    def get_pos(self, x):
        if x > 0.05 + self.impact:
            return 1
        elif x < -0.02 - self.impact:
            return -1
        else:
            return 0

    def getTargetVariable(self, prices, symbol):
        returns = (prices.shift(-1 * self.lookforward) / prices - 1)
        returns = returns.dropna()
        signals = (returns[symbol].apply(lambda x: self.get_pos(x))).to_frame()
        return signals, returns

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="JPM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31),
                    sv=100000):

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        if self.verbose:
            print("Prices loaded")

        trainX = self.getFeatures(prices, symbol)
        trainY, returns = self.getTargetVariable(prices, symbol)

        self.data = trainX.join(trainY, how='outer').dropna()
        trainX = self.data[trainX.columns]
        trainY = self.data[trainY.columns]
        assert trainY.shape[0] == trainX.shape[0]

        if self.verbose:
            print("Data shapes", trainX.shape, trainY.shape)
            print("Starting learning")

        self.learner.addEvidence(trainX.to_numpy(), trainY.to_numpy())
        return prices, trainX, trainY, returns

    # this method should use the existing policy and test it against new data
    def testPolicy(self,
                   symbol="JPM",
                   sd=dt.datetime(2010, 1, 1),
                   ed=dt.datetime(2011, 12, 31),
                   sv=100000):
        if self.verbose:
            print("Testing policy")
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        testX = self.getFeatures(prices, symbol)
        predY = self.learner.query(testX.to_numpy())
        assert predY.shape[0] == testX.shape[0]

        if self.verbose:
            print(testX.shape, predY.shape)

        trades = self.get_trades(predY)
        df_trades = pd.DataFrame(trades, index=testX.index)
        return df_trades

    def author(self):
        return 'sgarg96'


if __name__ == "__main__":
    print("Do nothing.")
