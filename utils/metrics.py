"""utils/metrics.py"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def calculate_ic(predictions, targets, method='pearman'):
    """计算信息系数(IC)；对齐索引"""
    import warnings
    from scipy.stats import ConstantInputWarning, spearmanr, pearsonr
    # 索引安全对齐（两者都是 Series 时）
    if isinstance(predictions, pd.Series) and isinstance(targets, pd.Series):
        # 使用内连接对齐索引
        df = pd.concat([predictions.rename('pred'), targets.rename('target')],
                       axis=1, join='inner')
        # 去除NaN
        df = df.dropna()

        if len(df) < 2:
            return 0.0

        x = df['pred'].values
        y = df['target'].values
    else:
        # 非Series时的原逻辑（保持向后兼容）
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        if hasattr(targets, 'values'):
            targets = targets.values

        x = np.array(predictions).flatten()
        y = np.array(targets).flatten()

        # 长度对齐
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # 去NaN
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 2:
            return 0.0
        x, y = x[valid], y[valid]
    # 常数检测
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
    if abs(sharpe) > 100:  # 异常高的夏普比率
        return np.sign(sharpe) * 100  # 截断到合理范围
    return float(np.sqrt(periods) * mu / sigma)



def calculate_max_drawdown(cumulative_returns):
    """最大回撤，分母做零保护。"""
    cr = np.asarray(cumulative_returns, dtype=float)
    if len(cr) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cr)
    running_max = np.where(running_max == 0.0, 1e-12, running_max)
    drawdown = (cr - running_max) / running_max
    return float(abs(np.min(drawdown)))


def calculate_icir(ic_series):
    """
    计算IC Information Ratio
    ICIR = mean(IC) / std(IC)
    衡量IC的稳定性
    """
    ic_array = np.array(ic_series)
    ic_array = ic_array[~np.isnan(ic_array)]  # 移除NaN

    if len(ic_array) < 2:
        return 0.0

    mean_ic = np.mean(ic_array)
    std_ic = np.std(ic_array)

    if std_ic == 0:
        return 0.0 if mean_ic == 0 else np.inf

    return mean_ic / std_ic


def calculate_rank_ic(predictions, targets):
    """
    计算Rank IC（基于排序的相关系数）
    更robust，对异常值不敏感
    """
    from scipy.stats import rankdata

    # 转换为数组
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(targets, 'values'):
        targets = targets.values

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # 移除NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if valid_mask.sum() < 2:
        return 0.0

    # 排序
    pred_ranks = rankdata(predictions[valid_mask])
    target_ranks = rankdata(targets[valid_mask])

    # 计算相关系数
    corr, _ = pearsonr(pred_ranks, target_ranks)

    return corr if not np.isnan(corr) else 0.0