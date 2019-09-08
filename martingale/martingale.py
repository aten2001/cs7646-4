"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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

import matplotlib.pyplot as plt
import numpy as np


def author():
    return 'sgarg96'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 903475801  # replace with your GT ID number


def get_spin_result(win_prob):
    # result = False
    # if np.random.random() <= win_prob:
    #     result = True
    # return result
    winnings = []
    epsiode_winnings = 0
    winnings.append(epsiode_winnings)
    num_bets = 1000
    bet = 1
    for i in range(num_bets):
        if epsiode_winnings < 80:
            if np.random.random() <= win_prob:
                epsiode_winnings = epsiode_winnings + bet
                bet = 1
            else:
                epsiode_winnings = epsiode_winnings - bet
                bet = bet * 2
        winnings.append(epsiode_winnings)
    return np.array(winnings)

def get_realistic_spin_result(win_prob, cash):
    winnings = []
    epsiode_winnings = 0
    winnings.append(epsiode_winnings)
    num_bets = 1000
    bet = 1
    for i in range(num_bets):
        if epsiode_winnings < 80 and cash > 0:
            if bet > cash:
                bet = cash
            if np.random.random() <= win_prob:
                epsiode_winnings = epsiode_winnings + bet
                cash = cash + bet
                bet = 1
            else:
                epsiode_winnings = epsiode_winnings - bet
                cash = cash - bet
                bet = bet * 2
        winnings.append(epsiode_winnings)
    return np.array(winnings)

def simple_sim(num_iter):
    win_prob = 18/38
    results = []
    for i in range(num_iter):
        winnings = get_spin_result(win_prob)
        results.append(winnings)
    return np.array(results)

def realistic_sim(num_iter):
    win_prob = 18/38
    results = []
    for i in range(num_iter):
        winnings = get_realistic_spin_result(win_prob, 256)
        results.append(winnings)
    return np.array(results)

def save_plot(statistic, std, labels, title, filename):
    plt.figure()
    plt.plot(statistic, label=labels[0])
    plt.plot(statistic + std, label=labels[1])
    plt.plot(statistic - std, label=labels[2])
    plt.axis(axis_limits)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Winnings")
    plt.savefig(filename, format="PNG")


# add your code here to implement the experiments

if __name__ == "__main__":
    np.random.seed(gtid())
    axis_limits = [0, 300, -256, 100]
    results1 = simple_sim(10)
    plt.figure()
    plt.plot(np.arange(results1.shape[1]), results1.T)
    plt.axis(axis_limits)
    plt.title("10 simulations of simple simulator")
    plt.xlabel("Iterations")
    plt.ylabel("Winnings")
    plt.savefig("fig1.png", format="PNG")

    results2 = simple_sim(1000)
    mean1 = np.mean(results2, axis=0)
    std1 = np.std(results2, axis=0)
    median1 = np.median(results2, axis=0)

    labels = ["mean", "mean + std", "mean - std"]
    save_plot(mean1, std1, labels, "Mean & deviation of 1000 runs (Simple Simulator)", "fig2.png")

    labels = ["median", "median + std", "median - std"]
    save_plot(mean1, std1, labels, "median & deviation of 1000 runs (Simple Simulator)", "fig3.png")

    results3 = realistic_sim(1000)
    mean2 = np.mean(results3, axis=0)
    std2 = np.std(results3, axis=0)
    median2 = np.median(results3, axis=0)

    labels = ["mean", "mean + std", "mean - std"]
    save_plot(mean2, std2, labels, "Mean & deviation of 1000 runs (Realistic Simulator)", "fig4.png")

    labels = ["median", "median + std", "median - std"]
    save_plot(mean2, std2, labels, "median & deviation of 1000 runs (Realistic Simulator)", "fig5.png")