"""utils/metrics.py"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def calculate_ic(predictions, targets, method='pearman'):
    """Compute the Information Coefficient (IC) with safe index alignment."""
    import warnings
    from scipy.stats import ConstantInputWarning, spearmanr, pearsonr
    # Safe index alignment when both inputs are pandas Series.
    if isinstance(predictions, pd.Series) and isinstance(targets, pd.Series):
        # Inner-join alignment on index.
        df = pd.concat([predictions.rename('pred'), targets.rename('target')],
                       axis=1, join='inner')
        # Drop NaNs.
        df = df.dropna()

        if len(df) < 2:
            return 0.0

        x = df['pred'].values
        y = df['target'].values
    else:
        # Legacy behavior for non-Series inputs (backward compatible).
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        if hasattr(targets, 'values'):
            targets = targets.values

        x = np.array(predictions).flatten()
        y = np.array(targets).flatten()

        # Length alignment.
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # Drop NaNs.
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 2:
            return 0.0
        x, y = x[valid], y[valid]
    # Constant-vector guard.
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        if method == 'pearson':
            corr, _ = pearsonr(x, y)
        else:
            corr, _ = spearmanr(x, y)

    return float(corr) if not np.isnan(corr) else 0.0


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    arr = np.asarray(getattr(returns, 'values', returns), dtype=float).ravel()

    rf = risk_free_rate / periods
    arr = arr - rf
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0

    mu = np.nanmean(arr)
    sigma = np.nanstd(arr, ddof=1)

    if sigma < 1e-10 or not np.isfinite(sigma):
        return 0.0
    sharpe = float(np.sqrt(periods) * mu / sigma)
    if abs(sharpe) > 100:  # Abnormally high Sharpe ratio.
        return np.sign(sharpe) * 100  # Clip to a reasonable range.
    return float(np.sqrt(periods) * mu / sigma)



def calculate_max_drawdown(cumulative_returns):
    """Maximum drawdown with a zero-denominator safeguard."""
    cr = np.asarray(cumulative_returns, dtype=float)
    if len(cr) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cr)
    running_max = np.where(running_max == 0.0, 1e-12, running_max)
    drawdown = (cr - running_max) / running_max
    return float(abs(np.min(drawdown)))


def calculate_icir(ic_series):
    """
    Compute the IC Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC)

    Measures the stability/consistency of IC over time.
    """
    ic_array = np.array(ic_series)
    ic_array = ic_array[~np.isnan(ic_array)]  # Remove NaNs.

    if len(ic_array) < 2:
        return 0.0

    mean_ic = np.mean(ic_array)
    std_ic = np.std(ic_array)

    if std_ic == 0:
        return 0.0 if mean_ic == 0 else np.inf

    return mean_ic / std_ic


def calculate_rank_ic(predictions, targets):
    """
    Compute Rank IC (rank-based correlation).

    More robust and less sensitive to outliers.
    """
    from scipy.stats import rankdata

    # Convert inputs to flat arrays.
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(targets, 'values'):
        targets = targets.values

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Drop NaNs.
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if valid_mask.sum() < 2:
        return 0.0

    # Rank-transform both series.
    pred_ranks = rankdata(predictions[valid_mask])
    target_ranks = rankdata(targets[valid_mask])

    # Compute correlation on ranks.
    corr, _ = pearsonr(pred_ranks, target_ranks)

    return corr if not np.isnan(corr) else 0.0
