# Student Name: Sanjana Garg (replace with your name)
# GT User ID: sgarg96 (replace with your User ID)
# GT ID: 903475801 (replace with your GT ID)

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from util import get_data


def author():
    return 'sgarg96'


def calc_sma(price, n):
    return price.rolling(window=n).mean()


def calc_avg_daily_returns(port_val):
    daily_returns = calc_daily_returns(port_val)
    return round(daily_returns['value'][1:].mean(), 4)


def calc_std_daily_returns(port_val):
    daily_returns = calc_daily_returns(port_val)
    return round(daily_returns['value'][1:].std(), 4)


def calc_std(price, n):
    return price.rolling(n).std()


def calc_obv(vol, price):
    obv = [0]
    for i in range(1, price.shape[0]):
        if price.iloc[i] > price.iloc[i - 1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif price.iloc[i] < price.iloc[i - 1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.DataFrame(obv, index=price.index)


def calc_obv_slope(vol, price, n):
    obv = calc_obv(vol, price)
    return obv / obv.shift(n - 1) - 1


def calc_cum_return(port_val):
    cum_ret = (port_val.iloc[-1, 0] / port_val.iloc[0, 0]) - 1
    return round(cum_ret, 4)


def bollinger_bands(price, n):
    sma = calc_sma(price, n)
    std = calc_std(price, n)
    bu = sma + 2 * std
    bl = sma - 2 * std
    return bu, sma, bl


def calc_bb_ratio(price, n):
    sma = calc_sma(price, n)
    std = calc_std(price, n)
    return (price - sma) / (2 * std)


def calc_sma_ratio(price, n):
    sma = calc_sma(price, n)
    return price / sma


def calc_momentum(price, n):
    return price / price.shift(n - 1) - 1


def calc_cci(price, n):
    pass


def calc_sharpe_ratio(adr, sddr):
    k = np.sqrt(252.0)
    sr = k * adr / sddr
    return round(sr,4)


def plot_sma(price, sma, sma_ratio):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=True,
                                   figsize=(10, 10))
    x = sma.index
    ax1.plot(x, price, label="Price")
    ax1.plot(x, sma, label="SMA")
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2.plot(x, np.ones(price.shape[0]), label="y = 1")
    ax2.plot(x, sma_ratio, label="Price/SMA")
    ax2.grid(True)
    ax2.legend(loc='lower right')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.savefig("sma_pot.png")


def plot_bb(price, bu, sma, bl, bb_ratio):
    x = price.index
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    ax.plot(x, bl, x, bu, color='black')
    ax.plot([], [], color='black', label='Bollinger bands')
    ax.fill_between(x, bl, bu, where=bu >= bl, alpha=0.1)
    ax.plot(x, price, label='Price')
    ax.plot(x, sma, label='SMA')
    ax.grid(True)
    ax.legend(loc='lower right')

    ax2.plot(x, bb_ratio, label='BB Ratio')
    ax2.plot(x, np.ones(price.shape[0]), color='red', label="y = 1.0")
    ax2.plot(x, -1 * np.ones(price.shape[0]), color='red', label="y = -1.0")
    ax2.grid(True)
    ax2.legend(loc='lower right')
    plt.xlabel('Date')
    plt.savefig("bb_plot.png")


def plot_momentum(price, momentum):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    x = price.index
    ax1.plot(x, price, label="Price")
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2.plot(x, momentum, label="momentum")
    ax2.plot(x, np.zeros(price.shape[0]), label="y = 0")
    ax2.grid(True)
    ax2.legend(loc='lower right')
    plt.xlabel('Date')
    plt.savefig("momentum.png")


def calc_daily_returns(price):
    return (price / price.shift(1)) - 1


def plot_obv_slope(price, obv_slope):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    x = price.index
    ax1.plot(x, price, label="Price")
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax1.set_ylabel('Normalized Price')

    ax2.plot(x, obv_slope, label="obv_slope")
    ax2.plot(x, np.zeros(price.shape[0]), label="y = 0")
    ax2.grid(True)
    ax2.legend(loc='lower right')
    plt.xlabel('Date')
    ax2.set_ylabel('OBV Slope')
    plt.savefig("obv.png")
    # plt.ylabel('Normalized Price')


if __name__ == '__main__':
    register_matplotlib_converters()
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    data = get_data([symbol], pd.date_range(start_date, end_date))
    data = data / data.iloc[0]
    price = data['JPM']
    vol = \
        get_data([symbol], pd.date_range(start_date, end_date),
                 colname='Volume')[
            'JPM']
    n = 20

    sma = calc_sma(data['JPM'], n)
    sma_ratio = calc_sma_ratio(data['JPM'], n)
    plot_sma(data['JPM'], sma, sma_ratio)

    bu, sma, bl = bollinger_bands(data['JPM'], n)
    bb_ratio = calc_bb_ratio(data['JPM'], n)
    plot_bb(data['JPM'], bu, sma, bl, bb_ratio)

    momentum = calc_momentum(data['JPM'], n)
    plot_momentum(data['JPM'], momentum)

    obv_slope = calc_obv_slope(vol, price, 20)
    plot_obv_slope(price, obv_slope)
