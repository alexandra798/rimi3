"""回测模块 - validation/backtest.py"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import logging


from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def backtest_formulas(formulas, X_test, y_test):
    """
    回测已发现的公式

    Parameters:
    - formulas: 要测试的公式列表
    - X_test: 测试特征数据
    - y_test: 测试目标数据

    Returns:
    - results: 公式及其IC值的字典
    """
    evaluator = FormulaEvaluator()
    results = {}

    for formula in formulas:
        # 使用统一的评估函数
        feature = evaluator.evaluate(formula, X_test)

        # 对齐数据
        valid_indices = ~(feature.isna() | y_test.isna())
        feature_clean = feature[valid_indices]
        y_test_clean = y_test[valid_indices]

        # 计算IC
        if len(feature_clean) > 1:
            ic, _ = spearmanr(feature_clean, y_test_clean)
            results[formula] = ic if not np.isnan(ic) else 0
        else:
            results[formula] = 0
            logger.warning(f"Insufficient data for formula: {formula}")

    return results


# validation/backtest.py 新增
def backtest_with_trading_simulation(formulas, X_test, y_test, price_data,
                                     top_k=40, rebalance_freq=5,
                                     initial_capital=1000000):
    """
    论文5.3节的完整交易模拟

    Parameters:
    - formulas: alpha公式列表
    - X_test: 测试特征数据
    - y_test: 实际收益率
    - price_data: 包含价格信息的DataFrame
    - top_k: 每次选择的股票数量
    - rebalance_freq: 重新平衡频率（天）
    - initial_capital: 初始资金
    """
    evaluator = FormulaEvaluator()

    # 获取时间索引（稳健支持 MultiIndex 或列字段）
    if isinstance(X_test.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(set(X_test.index.names)):
        dates = X_test.index.get_level_values('date').unique().sort_values()

        def get_daily_data(df, d):
            return df.xs(d, level='date', drop_level=False)

        def tickers_of(df_day):
            return df_day.index.get_level_values('ticker').unique()

        def slice_ticker(df_day, t):
            return df_day.xs(t, level='ticker', drop_level=False)
    else:
        if 'date' not in X_test.columns or 'ticker' not in X_test.columns:
            raise ValueError("X_test 必须是 MultiIndex(date,ticker) 或含有 'date' 与 'ticker' 列")
        dates = pd.Index(sorted(X_test['date'].unique()))

        def get_daily_data(df, d):
            return df[df['date'] == d]

        def tickers_of(df_day):
            return pd.Index(df_day['ticker'].unique())

        def slice_ticker(df_day, t):
            return df_day[df_day['ticker'] == t]

    portfolio_values = [initial_capital]
    holdings = {}  # 当前持仓

    def get_close(price_df, d, t):
        # 稳健取 close，兼容 MultiIndex 或列字段
        if isinstance(price_df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(set(price_df.index.names)):
            return float(price_df.loc[(d, t), 'close'])
        elif {'date', 'ticker', 'close'}.issubset(set(price_df.columns)):
            row = price_df[(price_df['date'] == d) & (price_df['ticker'] == t)]
            if len(row):
                return float(row['close'].iloc[0])
            raise KeyError(f"Missing price for {d} {t}")
        else:
            raise ValueError("price_data 需为 MultiIndex(date,ticker) 或包含 date/ticker/close 列")

    for i, date in enumerate(dates):
        # 每rebalance_freq天重新平衡
        if i % rebalance_freq == 0:
            # 计算所有股票的alpha信号
            daily_data = get_daily_data(X_test, date)

            alpha_scores = {}
            for ticker in tickers_of(daily_data):
                ticker_data = slice_ticker(daily_data, ticker)

                # 使用所有公式的平均信号
                scores = []
                for formula in formulas:
                    score = evaluator.evaluate(formula, ticker_data)
                    if not pd.isna(score).all():
                        score = (score.dropna().iloc[-1] if hasattr(score, 'dropna') else score)
                        scores.append(float(score))

                if scores:
                    alpha_scores[ticker] = np.mean(scores)

            # 选择top-k股票
            if not alpha_scores:
                portfolio_values.append(portfolio_values[-1])  # 无信号，持仓不变
                continue

            sorted_tickers = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tickers = [t[0] for t in sorted_tickers[:top_k]]

            if len(selected_tickers) == 0:
                portfolio_values.append(portfolio_values[-1])
                continue

            # 计算每只股票的投资金额（等权重）
            current_value = portfolio_values[-1]
            position_size = current_value / len(selected_tickers)

            # 更新持仓
            new_holdings = {}
            for ticker in selected_tickers:
                try:
                    # 获取当前价格
                    current_price = get_close(price_data, date, ticker)
                    shares = position_size / current_price
                    new_holdings[ticker] = shares
                except Exception as e:
                    logger.warning(f"Price missing for {ticker} @ {date}: {e}")
                    # 跳过无法获取价格的股票

            holdings = new_holdings

        # 计算当日组合价值
        daily_value = 0
        for ticker, shares in holdings.items():
            try:
                current_price = get_close(price_data, date, ticker)#
                daily_value += shares * current_price
            except Exception as e:
                logger.warning(f"Price missing @ {date} {ticker}: {e}")
                daily_value = portfolio_values[-1]  # 保持前一日价值

        if daily_value == 0:
            daily_value = portfolio_values[-1]  # 保持前一日价值

        portfolio_values.append(daily_value)

    # 计算性能指标
    portfolio_returns = np.diff(portfolio_values) / np.asarray(portfolio_values[:-1], dtype=float)

    # 累积收益率
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # 使用统一的指标函数
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.0, periods=252)
    max_drawdown = calculate_max_drawdown(np.asarray(portfolio_values, dtype=float))

    results = {
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'portfolio_values': portfolio_values,
        'daily_returns': portfolio_returns
    }

    return results