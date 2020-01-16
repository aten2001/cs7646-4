"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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

import random as rand

import numpy as np


class QLearner(object):

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.lr = alpha
        self.dr = gamma
        self.rar = rar
        self.radr = radr

        self.Q = np.zeros((self.num_states, self.num_actions))
        self.dyna = dyna
        if self.dyna > 0:
            self.count = np.ones((self.num_states, self.num_actions,
                                  self.num_states)) * .00001
            self.model = self.count / self.count.sum(axis=2, keepdims=True)
            self.R = np.zeros((self.num_states, self.num_actions))
        self.history = []
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])

        self.s = s
        self.a = action
        self.history.append((s, action))
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def update_Q(self, s, a, s_prime, r):
        max_a = np.argmax(self.Q[s_prime])
        prev = (1 - self.lr) * self.Q[s, a]
        new = self.lr * (r + self.dr * self.Q[s_prime, max_a])
        self.Q[s, a] = prev + new

    def update_model(self, s, a, s_prime, r):
        self.count[s, a, s_prime] = self.count[s, a, s_prime] + 1
        self.model = self.count / self.count.sum(axis=2, keepdims=True)
        self.R[s, a] = (1 - 0.9) * self.R[s, a] + 0.9 * r

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        self.update_Q(self.s, self.a, s_prime, r)

        if self.dyna > 0:
            self.update_model(self.s, self.a, s_prime, r)
            indices = np.random.choice(len(self.history), self.dyna, replace=True)
            random_prob = np.random.random(self.dyna)
            cumsum = np.cumsum(self.model, axis=2)
            for i in range(self.dyna):
                s_tilde, a_tilde = self.history[indices[i]]
                s_prime_tilde = np.where(cumsum[s_tilde, a_tilde] >=
                                         random_prob[i])[0][0]
                r_tilde = self.R[s_tilde, a_tilde]
                self.update_Q(s_tilde, a_tilde, s_prime_tilde, r_tilde)

        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])
        self.rar = self.rar * self.radr
        self.history.append((s_prime, action))
        self.s = s_prime
        self.a = action
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return 'sgarg96'


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
