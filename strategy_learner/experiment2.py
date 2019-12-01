# Student Name: Sanjana Garg (replace with your name)
# GT User ID: sgarg96 (replace with your User ID)
# GT ID: 903475801 (replace with your GT ID)

from StrategyLearner import StrategyLearner
from indicators import *
from marketsimcode import compute_portvals


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
    orders_df['Symbol'] = symbol
    orders_df['Order'] = 'BUY'
    orders_df['Shares'] = 1000
    return orders_df


def plot_cmp(x, benchmark, portfolio_list, impact_values, orders=None,
             plot_trades=False,
             file="impact_cmp.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, benchmark, label='Benchmark')
    for portfolio, impact in zip(portfolio_list, impact_values):
        ax.plot(x, portfolio, label=f"StrategyLearner impact {impact}")
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


def test(symbol, sd, ed, sv, sl, commission, impact):
    # get trades
    trades_sl = sl.testPolicy(symbol, sd, ed, sv)

    # build order book
    orders_sl = get_orders_df(trades_sl, symbol)
    orders_bench = get_benchmark_trades(sd, ed, symbol)

    # execute orders
    portfolio_sl = compute_portvals(orders_sl, sv, commission, impact, sd, ed)
    portfolio_bench = compute_portvals(orders_bench, 100000, commission, 0, sd,
                                       ed)
    print(portfolio_sl.shape, portfolio_bench.shape)
    # normalize portfolio values
    portfolio_bench = portfolio_bench / portfolio_bench.iloc[0]
    portfolio_sl = portfolio_sl / portfolio_sl.iloc[0]

    return portfolio_bench, portfolio_sl


def author():
    return 'sgarg96'


if __name__ == '__main__':
    register_matplotlib_converters()
    np.random.seed(42)
    impact_values = [0.0005, 0.005, 0.05]
    symbol = "JPM"
    commission = 0.0
    sv = 100000
    # in sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    portfolio_list = []
    orders_list = []
    for impact in impact_values:
        sl = StrategyLearner(impact=impact)
        np.random.seed(42)
        # learning step
        sl.addEvidence(symbol, sd, ed, sv)

        # get trades
        trades_sl = sl.testPolicy(symbol, sd, ed, sv)

        # build orders
        orders_sl = get_orders_df(trades_sl, symbol)
        orders_list.append(orders_sl)

        # execute orders
        portfolio_sl = compute_portvals(orders_sl, sv, commission, impact, sd,
                                        ed)
        portfolio_sl = portfolio_sl / portfolio_sl.iloc[0]

        portfolio_list.append(portfolio_sl)
    orders_bench = get_benchmark_trades(sd, ed, symbol)
    portfolio_bench = compute_portvals(orders_bench, 100000, commission, 0, sd,
                                       ed)
    portfolio_bench = portfolio_bench / portfolio_bench.iloc[0]

    # portfolio_list.append(portfolio_bench)

    plot_cmp(portfolio_bench.index,
             portfolio_bench,
             portfolio_list,
             impact_values)

    plot_cmp(portfolio_bench.index,
             portfolio_bench,
             [portfolio_list[0]],
             [impact_values[0]],
             orders_list[0],
             True,
             "impact0005.png")

    plot_cmp(portfolio_bench.index,
             portfolio_bench,
             [portfolio_list[-1]],
             [impact_values[-1]],
             orders_list[-1],
             True,
             "impact05.png")

    portfolio_list.append(portfolio_bench)
    print_stats(portfolio_list)
