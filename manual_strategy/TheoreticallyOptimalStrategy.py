import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from indicators import calc_daily_returns, calc_std_daily_returns, \
    calc_cum_return, calc_avg_daily_returns
from marketsimcode import compute_portvals
from util import get_data


class TheoreticallyOptimalStrategy:
    def __init__(self):
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2010, 12, 31), sv=100000):
        data = get_data([symbol], pd.date_range(sd, ed))
        data = data / data.iloc[0]

        price = data[symbol]

        daily_returns = calc_daily_returns(price)

        net_holdings = 0
        min_holdings = -1000
        max_holdings = 1000
        trades = []
        for date in price.index[1:]:
            trades.append(0)
            if daily_returns[date] < 0:
                if net_holdings > min_holdings:
                    num_shares = min_holdings - net_holdings
                    trades[-1] = num_shares
                    net_holdings = min_holdings
            elif daily_returns[date] > 0:
                if net_holdings < max_holdings:
                    num_shares = max_holdings - net_holdings
                    trades[-1] = num_shares
                    net_holdings = max_holdings
        trades.append(0)
        df_trades = pd.DataFrame(index=price.index)
        df_trades[symbol] = trades
        return df_trades


def author():
    return 'sgarg96'


if __name__ == '__main__':
    register_matplotlib_converters()
    th = TheoreticallyOptimalStrategy()
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    df_trades = th.testPolicy('JPM', sd, ed, 100000)

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

    benchmark_df = pd.DataFrame(index=[dt.datetime(2008, 1, 2)])
    benchmark_df['Symbol'] = 'JPM'
    benchmark_df['Order'] = 'BUY'
    benchmark_df['Shares'] = 1000

    theo_port = compute_portvals(orders, 100000, 0, 0, sd, ed)
    benchmark_port = compute_portvals(benchmark_df, 100000, 0, 0, sd, ed)

    norm_theo = theo_port / theo_port.iloc[0]
    norm_bench = benchmark_port / benchmark_port.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = theo_port.index
    ax.plot(x, norm_theo['value'], label='TheoreticallyOptimalStrategy',
            color='red')
    ax.plot(x, norm_bench['value'], label='Benchmark', color='green')
    ax.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    ax.grid()
    plt.savefig("theo_vs_bench.png")
    print(calc_cum_return(theo_port), calc_avg_daily_returns(theo_port),
          calc_std_daily_returns(theo_port))
    print(calc_cum_return(benchmark_port),
          calc_avg_daily_returns(benchmark_port),
          calc_std_daily_returns(benchmark_port))
