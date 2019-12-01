from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from indicators import *
from marketsimcode import compute_portvals


def plot_cmp(x, manual, benchmark, strategy, orders=None, plot_trades=False,
             file="manual_vs_bench.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, manual, label='ManualStrategy', color='red')
    ax.plot(x, benchmark, label='Benchmark', color='green')
    ax.plot(x, strategy, label="Strategy Learner", color='blue')
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


def get_orders_df(df_trades, symbol):
    dates = []
    order_type = []
    shares = []
    columns = df_trades.columns
    pos = columns[0]
    for date in df_trades.index:
        if df_trades.loc[date][pos] > 0:
            dates.append(date)
            order_type.append('BUY')
            shares.append(df_trades.loc[date][pos])
        elif df_trades.loc[date][pos] < 0:
            dates.append(date)
            order_type.append('SELL')
            shares.append(abs(df_trades.loc[date][pos]))

    orders = pd.DataFrame(index=dates)
    orders['Symbol'] = symbol
    orders['Order'] = order_type
    orders['Shares'] = shares
    return orders


def print_stats(portfolios):
    for portfolio in portfolios:
        cr = calc_cum_return(portfolio)
        avg_dr = calc_avg_daily_returns(portfolio)
        std_dr = calc_std_daily_returns(portfolio)
        sr = calc_sharpe_ratio(avg_dr, std_dr)
        print(f"CumRet: {cr} AvgDailyRet: {avg_dr} StdDailyRet: {std_dr} "
              f"SharpeRatio: {sr}")


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


def test(symbol, sd, ed, sv, plot_file, sl, ms, commission, impact):
    # get trades
    trades_sl = sl.testPolicy(symbol, sd, ed, sv)
    trades_ms = ms.testPolicy(symbol, sd, ed, sv)

    # build order book
    orders_sl = get_orders_df(trades_sl, symbol)
    orders_ms = get_orders_df(trades_ms, symbol)
    orders_bench = get_benchmark_trades(sd, ed, symbol)

    # execute orders
    portfolio_ms = compute_portvals(orders_ms, sv, commission, impact, sd, ed)
    portfolio_sl = compute_portvals(orders_sl, sv, commission, impact, sd, ed)
    portfolio_bench = compute_portvals(orders_bench, 100000, commission, 0, sd,
                                       ed)

    # normalize portfolio values
    portfolio_ms = portfolio_ms / portfolio_ms.iloc[0]
    portfolio_bench = portfolio_bench / portfolio_bench.iloc[0]
    portfolio_sl = portfolio_sl / portfolio_sl.iloc[0]

    x = portfolio_bench.index
    plot_cmp(x,
             portfolio_ms['value'],
             portfolio_bench['value'],
             portfolio_sl['value'],
             None,
             False,
             plot_file)
    print_stats([portfolio_sl, portfolio_ms, portfolio_bench])


if __name__ == '__main__':
    register_matplotlib_converters()
    np.random.seed(42)
    impact = 0.005
    symbol = "JPM"
    commission = 0.0
    sv = 100000

    sl = StrategyLearner(impact=impact)
    ms = ManualStrategy()

    # in sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # learning step
    sl.addEvidence(symbol, sd, ed, sv)

    # test on in-sample data
    test(symbol, sd, ed, sv, "sl_insample.png", sl, ms, commission, impact)

    # out sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    # test on out-sample data
    test(symbol, sd, ed, sv, "sl_outsample.png", sl, ms, commission, impact)
