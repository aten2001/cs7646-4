"""MC2-P1: Market simulator.

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

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
    return 'sgarg96'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True,
                         na_values=['nan'])
    orders = orders.sort_index()
    symbols = orders['Symbol'].unique().tolist()
    start_date = orders.index[0]
    end_date = orders.index[-1]

    data = get_data(symbols, pd.date_range(start_date, end_date))
    data = data.ffill()
    data = data.bfill()
    trade_days = data.index

    prices = data[symbols]
    num_stocks = pd.DataFrame(0, index=trade_days, columns=symbols)

    cols = ['value', 'cash']
    port_value = pd.DataFrame(index=trade_days, columns=cols)

    cash = start_val
    for date, row in orders.iterrows():
        if date not in trade_days:
            continue
        stock = row['Symbol']
        order = row['Order']
        shares = row['Shares']

        order_type = 1 if order == 'BUY' else -1

        num_stocks.at[date, stock] += order_type * shares
        transac_cost = impact * shares * prices.loc[date, stock] + commission
        cash += (prices.loc[date, stock] * shares * (
            -1) * order_type - transac_cost)
        port_value.at[date, 'cash'] = cash

    num_stocks = num_stocks.cumsum()
    port_value['cash'] = port_value['cash'].ffill()
    port_value['stock_val'] = (num_stocks * prices).sum(axis=1)
    port_value['value'] = port_value['cash'] + port_value['stock_val']
    return port_value[['value']]

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

if __name__ == "__main__":
    test_code()
