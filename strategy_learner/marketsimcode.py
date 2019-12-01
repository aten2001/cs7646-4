# Student Name: Sanjana Garg (replace with your name)
# GT User ID: sgarg96 (replace with your User ID)
# GT ID: 903475801 (replace with your GT ID)

import datetime as dt

import pandas as pd

from util import get_data


def author():
    return 'sgarg96'


def compute_portvals(orders, start_val=1000000, commission=9.95,
                     impact=0.005, sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2010, 12, 31)):
    orders = orders.sort_index()
    symbols = orders['Symbol'].unique().tolist()
    start_date = sd
    end_date = ed

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
    port_value['cash'] = port_value['cash'].fillna(start_val)
    port_value['value'] = port_value['cash'] + port_value['stock_val']
    return port_value[['value']]
