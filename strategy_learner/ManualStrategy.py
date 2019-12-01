import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from indicators import calc_sma_ratio, calc_bb_ratio, \
    calc_avg_daily_returns, calc_std_daily_returns, calc_cum_return, \
    calc_momentum
from marketsimcode import compute_portvals
from util import get_data


class ManualStrategy:
    def __init__(self):
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2010, 12, 31), sv=100000):
        data = get_data([symbol], pd.date_range(sd, ed))
        data = data / data.iloc[0]
        price = data[symbol]

        n = 14
        sma_ratio = calc_sma_ratio(price, n)
        bb_ratio = calc_bb_ratio(price, n)
        momentum = calc_momentum(price, n)
        net_holdings = 0
        min_holdings = -1000
        max_holdings = 1000
        trades = [0] * (n - 1)

        for date in price.index[n - 1:]:
            trades.append(0)
            if (sma_ratio[date] > 1 and bb_ratio[date] > 1) or 0.2 < momentum[
                date] < 1:
                if net_holdings > min_holdings:
                    num_shares = min_holdings - net_holdings
                    trades[-1] = num_shares
                    net_holdings = min_holdings
            elif (sma_ratio[date] < 1 and bb_ratio[date] < -1) and -0.2 < \
                    momentum[date] < 0:
                if net_holdings < max_holdings:
                    num_shares = max_holdings - net_holdings
                    trades[-1] = num_shares
                    net_holdings = max_holdings
        df_trades = pd.DataFrame(index=price.index)
        df_trades[symbol] = trades
        return df_trades


def author():
    return 'sgarg96'


def get_orders_df(df_trades):
    dates = []
    order_type = []
    shares = []
    for date in df_trades.index:
        if df_trades.loc[date]['JPM'] > 0:
            dates.append(date)
            order_type.append('BUY')
            shares.append(df_trades.loc[date]['JPM'])
        elif df_trades.loc[date]['JPM'] < 0:
            dates.append(date)
            order_type.append('SELL')
            shares.append(abs(df_trades.loc[date]['JPM']))

    orders = pd.DataFrame(index=dates)
    orders['Symbol'] = 'JPM'
    orders['Order'] = order_type
    orders['Shares'] = shares
    return orders


def plot_cmp(x, manual, benchmark, orders=None, plot_trades=False,
             file="manual_vs_bench.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, manual, label='ManualStrategy', color='red')
    ax.plot(x, benchmark, label='Benchmark', color='green')
    ax.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    if plot_trades:
        for date in orders.index:
            if orders.loc[date]['Order'] == 'BUY':
                plt.axvline(x=date, color='blue')
            else:
                plt.axvline(x=date, color='black')
    ax.grid()
    plt.savefig(file)


def get_benchmark_trades(sd, ed, symbol):
    dates = pd.date_range(sd, ed)
    syms = [symbol]
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    orders_df = pd.DataFrame(index=prices.index[:1])
    orders_df['Symbol'] = 'JPM'
    orders_df['Order'] = 'BUY'
    orders_df['Shares'] = 1000
    return orders_df


if __name__ == '__main__':
    register_matplotlib_converters()
    ms = ManualStrategy()
    symbol = "JPM"
    impact = 0.005
    commission = 0

    # in sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    bench_orders = get_benchmark_trades(sd, ed, symbol)


    df_trades = ms.testPolicy(symbol, sd, ed, 100000)
    orders = get_orders_df(df_trades)

    benchmark_df = pd.DataFrame(index=[dt.datetime(2008, 1, 2)])
    benchmark_df['Symbol'] = 'JPM'
    benchmark_df['Order'] = 'BUY'
    benchmark_df['Shares'] = 1000

    manual_port = compute_portvals(orders, 100000, 9.95, .005, sd, ed)
    benchmark_port = compute_portvals(benchmark_df, 100000, 0, 0, sd, ed)

    norm_manual = manual_port / manual_port.iloc[0]
    norm_bench = benchmark_port / benchmark_port.iloc[0]
    x = manual_port.index
    plot_cmp(x, norm_manual['value'], norm_bench['value'], orders, True,
             "manual_vs_bench.png")
    print(calc_cum_return(manual_port), calc_avg_daily_returns(manual_port),
          calc_std_daily_returns(manual_port))
    print(calc_cum_return(benchmark_port),
          calc_avg_daily_returns(benchmark_port),
          calc_std_daily_returns(benchmark_port))

    # out sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    df_trades = ms.testPolicy(symbol, sd, ed, 100000)
    orders = get_orders_df(df_trades)

    benchmark_df = pd.DataFrame(index=[dt.datetime(2010, 1, 4)])
    benchmark_df['Symbol'] = 'JPM'
    benchmark_df['Order'] = 'BUY'
    benchmark_df['Shares'] = 1000

    manual_port = compute_portvals(orders, 100000, 9.95, .005, sd, ed)
    benchmark_port = compute_portvals(benchmark_df, 100000, 0, 0, sd, ed)

    norm_manual = manual_port / manual_port.iloc[0]
    norm_bench = benchmark_port / benchmark_port.iloc[0]
    x = manual_port.index
    plot_cmp(x, norm_manual['value'], norm_bench['value'], None, False,
             "manual_vs_bench_out.png")
    print(calc_cum_return(manual_port), calc_avg_daily_returns(manual_port),
          calc_std_daily_returns(manual_port))
    print(calc_cum_return(benchmark_port),
          calc_avg_daily_returns(benchmark_port),
          calc_std_daily_returns(benchmark_port))
