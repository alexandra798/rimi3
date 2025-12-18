"""Backtesting module"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import logging


from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def backtest_formulas(formulas, X_test, y_test):
    """
    Backtest discovered formulas.

    Parameters:
    - formulas: List of formulas to evaluate.
    - X_test: Test feature data.
    - y_test: Test target data.

    Returns:
    - results: Dict mapping formula -> IC value.
    """
    evaluator = FormulaEvaluator()
    results = {}

    for formula in formulas:
        # Evaluate using the unified evaluator.
        feature = evaluator.evaluate(formula, X_test)

        # Align and clean data.
        valid_indices = ~(feature.isna() | y_test.isna())
        feature_clean = feature[valid_indices]
        y_test_clean = y_test[valid_indices]

        # Compute IC.
        if len(feature_clean) > 1:
            ic, _ = spearmanr(feature_clean, y_test_clean)
            results[formula] = ic if not np.isnan(ic) else 0
        else:
            results[formula] = 0
            logger.warning(f"Insufficient data for formula: {formula}")

    return results


# Added in validation/backtest.py
def backtest_with_trading_simulation(formulas, X_test, y_test, price_data,
                                     top_k=40, rebalance_freq=5,
                                     initial_capital=1000000):
    """
    Full trading simulation from Section 5.3 of the paper.

    Parameters:
    - formulas: List of alpha formulas.
    - X_test: Test feature data.
    - y_test: Realized returns (ground truth).
    - price_data: DataFrame containing price information.
    - top_k: Number of stocks selected at each rebalance.
    - rebalance_freq: Rebalance frequency in days.
    - initial_capital: Starting capital.
    """
    evaluator = FormulaEvaluator()

    # Get the date index (robustly supports MultiIndex or explicit columns).
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
    holdings = {}  # Current holdings (ticker -> shares).

    def get_close(price_df, d, t):
        # Robust close-price lookup (supports MultiIndex or explicit columns).
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
        # Rebalance every `rebalance_freq` days.
        if i % rebalance_freq == 0:
            # Compute alpha signals for all tickers on this day.
            daily_data = get_daily_data(X_test, date)

            alpha_scores = {}
            for ticker in tickers_of(daily_data):
                ticker_data = slice_ticker(daily_data, ticker)

                # Use the average signal across all formulas.
                scores = []
                for formula in formulas:
                    score = evaluator.evaluate(formula, ticker_data)
                    if not pd.isna(score).all():
                        score = (score.dropna().iloc[-1] if hasattr(score, 'dropna') else score)
                        scores.append(float(score))

                if scores:
                    alpha_scores[ticker] = np.mean(scores)

            # Select top-k tickers.
            if not alpha_scores:
                portfolio_values.append(portfolio_values[-1])  # No signal: keep holdings/value unchanged.
                continue

            sorted_tickers = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tickers = [t[0] for t in sorted_tickers[:top_k]]

            if len(selected_tickers) == 0:
                portfolio_values.append(portfolio_values[-1])
                continue

            # Equal-weight position sizing.
            current_value = portfolio_values[-1]
            position_size = current_value / len(selected_tickers)

            # Update holdings.
            new_holdings = {}
            for ticker in selected_tickers:
                try:
                    # Get current price.
                    current_price = get_close(price_data, date, ticker)
                    shares = position_size / current_price
                    new_holdings[ticker] = shares
                except Exception as e:
                    logger.warning(f"Price missing for {ticker} @ {date}: {e}")
                    # Skip tickers with missing prices.

            holdings = new_holdings

        # Compute daily portfolio value.
        daily_value = 0
        for ticker, shares in holdings.items():
            try:
                current_price = get_close(price_data, date, ticker)#
                daily_value += shares * current_price
            except Exception as e:
                logger.warning(f"Price missing @ {date} {ticker}: {e}")
                daily_value = portfolio_values[-1]  # Fall back to previous day's value.

        if daily_value == 0:
            daily_value = portfolio_values[-1]  # Fall back to previous day's value.

        portfolio_values.append(daily_value)

    # Performance metrics.
    portfolio_returns = np.diff(portfolio_values) / np.asarray(portfolio_values[:-1], dtype=float)

    # Cumulative return.
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # Use unified metric helpers.
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
