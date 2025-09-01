"""core/operators.py"""
import numpy as np
import pandas as pd
import logging

from scipy import stats


def _get_cache(series):
    """
    获取/初始化 Series 级别的滚动结果缓存字典。
    对于 pandas 的派生 Series，attrs 可能不会自动继承，这里只用于“原始列反复滚动”的高频场景。
    """
    try:
        return series.attrs.setdefault('_op_cache', {})
    except Exception:
        # 某些 pandas 版本或对象不支持 attrs；则直接禁用缓存
        return None

MAX_VALUE = 1e8  # 数值上限
MIN_VALUE = -1e8  # 数值下限
EPSILON = 1e-10  # 防止除零的最小值

logger = logging.getLogger(__name__)


class Operators:
    """所有操作符的静态方法集合"""



    @staticmethod
    def ensure_series_or_array(operand, data_length=None, data_index=None):
        """确保操作数是Series或Array"""
        if isinstance(operand, (int, float)) and data_length:
            if data_index is not None:
                return pd.Series(operand, index=data_index)
            else:
                return pd.Series([operand] * data_length)
        return operand
    # =================================
    @staticmethod
    def safe_divide(x, y, default_value=0):
        """安全除法函数，避免除零错误"""
        if isinstance(x, pd.Series):
            # 修复：将无穷大也替换为 default_value
            return x.div(y).replace([np.inf, -np.inf], default_value).fillna(default_value)
        else:
            return np.divide(x, y, out=np.full_like(x, default_value, dtype=float), where=y != 0)

    # 一元操作符====================

    @staticmethod
    def csrank(operand, data_length=None, data_index=None):
        """横截面排名"""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            if isinstance(operand.index, pd.MultiIndex):
                return operand.groupby(level=1).rank(pct=True)
            else:
                return operand.rank(pct=True)
        else:
            # NumPy数组
            return stats.rankdata(operand, method='average') / len(operand)

    @staticmethod
    def sign(operand, data_length=None, data_index=None):
        """符号函数：正数返回1，非正返回0"""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            return (operand > 0).astype(float)
        else:
            return np.where(operand > 0, 1.0, 0.0)

    @staticmethod
    def abs(operand, data_length=None, data_index=None):
        """绝对值操作符"""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        return np.abs(operand)

    @staticmethod
    def log(operand, data_length=None, data_index=None):
        """安全的log操作: log(max(|x|+1e-10, 1e-10))"""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            return np.log(np.maximum(operand.abs() + 1e-10, 1e-10))
        else:
            return np.log(np.maximum(np.abs(operand) + 1e-10, 1e-10))


    # 二元操作符========================================
    @staticmethod
    def _align_operands(operand1, operand2):
        """对齐两个操作数的形状"""
        if isinstance(operand1, (int, float)) and isinstance(operand2, (pd.Series, np.ndarray)):
            if isinstance(operand2, pd.Series):
                operand1 = pd.Series(operand1, index=operand2.index)
            else:
                operand1 = np.full(len(operand2), operand1)
        elif isinstance(operand2, (int, float)) and isinstance(operand1, (pd.Series, np.ndarray)):
            if isinstance(operand1, pd.Series):
                operand2 = pd.Series(operand2, index=operand1.index)
            else:
                operand2 = np.full(len(operand1), operand2)
        return operand1, operand2

    @staticmethod
    def add(operand1, operand2, data_length=None, data_index=None):
        """加法操作符"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return operand1 + operand2

    @staticmethod
    def sub(operand1, operand2, data_length=None, data_index=None):
        """减法操作符"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return operand1 - operand2

    @staticmethod
    def mul(operand1, operand2, data_length=None, data_index=None):
        """乘法操作符（添加数值裁剪）"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        with np.errstate(over='ignore', invalid='ignore'):
            result = operand1 * operand2
            # 裁剪到合理范围
            if isinstance(result, pd.Series):
                result = result.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            else:
                result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result

    @staticmethod
    def div(operand1, operand2, data_length=None, data_index=None):
        """安全除法操作符（改进版）"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)

        # 防止极小除数
        if isinstance(operand2, pd.Series):
            operand2 = operand2.where(operand2.abs() > EPSILON, EPSILON)
        else:
            operand2 = np.where(np.abs(operand2) > EPSILON, operand2, EPSILON)

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            result = Operators.safe_divide(operand1, operand2)
            # 裁剪结果
            if isinstance(result, pd.Series):
                result = result.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            else:
                result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result

    @staticmethod
    def greater(operand1, operand2, data_length=None, data_index=None):
        """大于比较：x > y 返回1，否则0"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return (operand1 > operand2).astype(float)

    @staticmethod
    def less(operand1, operand2, data_length=None, data_index=None):
        """小于比较：x < y 返回1，否则0"""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return (operand1 < operand2).astype(float)

    # 时序操作符=====================================

    @staticmethod
    def _ensure_window_int(window):
        """确保window是整数"""
        if isinstance(window, (pd.Series, np.ndarray)):
            window = int(window[0]) if len(window) > 0 else 5
        else:
            window = int(window)
        return max(1, min(window, 100))  # 限制窗口大小

    @staticmethod
    def ts_ref(data, window):
        """引用t天前的值"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            return data.shift(window)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            result[:window] = np.nan
            if window < len(data):
                result[window:] = data[:-window]
            return result

    @staticmethod
    def ts_rank(data, window):
        """窗口内排名百分位"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            def rank_in_window(x):
                if len(x) < 2:
                    return 0.5
                return (x.iloc[-1] > x).sum() / len(x)

            result = data.rolling(window=window, min_periods=1).apply(rank_in_window, raw=False)
            return result.fillna(0.5)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]

                if len(window_data) < 2:
                    result[i] = 0.5
                else:
                    current_val = data[i]
                    rank = (current_val > window_data).sum() / len(window_data)
                    result[i] = rank
            return result

    @staticmethod
    def ts_mean(data, window):
        """移动平均"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            cache = _get_cache(data)
            key = ('ts_mean', int(window))
            if cache is not None and key in cache:
                return cache[key]

            result = data.rolling(window=window, min_periods=1).mean().bfill().fillna(0)

            if cache is not None:
                cache[key] = result
            return result
        else:
            # 原 Numpy 实现保持不变
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.mean(window_data)
            return result

    @staticmethod
    def ts_med(data, window):
        """移动中位数"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).median()
            return result.bfill().fillna(0)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.median(window_data)
            return result

    @staticmethod
    def ts_sum(data, window):
        """移动求和"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).sum()
            result = result.clip(lower=-1e10, upper=1e10)
            return result.fillna(0)
        else:
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.clip(np.sum(window_data), -1e10, 1e10)
            return result

    @staticmethod
    def ts_std(data, window):
        """标准差（防溢出 + 窗口缓存）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            cache = _get_cache(data)
            key = ('ts_std', int(window))
            if cache is not None and key in cache:
                return cache[key]

            # 先裁剪，保持你原来的稳健做法
            data_clipped = data.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            with np.errstate(all='ignore'):
                result = data_clipped.rolling(window=window, min_periods=min(3, window)).std()
            result = result.replace([np.inf, -np.inf], 0).bfill().fillna(0)

            if cache is not None:
                cache[key] = result
            return result
        else:
            # 原 Numpy 分支保持不变
            data = np.clip(np.asarray(data), MIN_VALUE, MAX_VALUE)
            result = np.zeros_like(data, dtype=np.float64)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                if len(window_data) >= 2:
                    with np.errstate(all='ignore'):
                        std_val = np.std(window_data, ddof=1)
                        result[i] = 0 if not np.isfinite(std_val) else std_val
                else:
                    result[i] = 0
            return result

    @staticmethod
    def ts_var(data, window):
        """方差（智能处理小窗口）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            with np.errstate(over='ignore', invalid='ignore'):
                result = data.rolling(window=window, min_periods=min(3, window)).var()
            return result.replace([np.inf, -np.inf], 0).bfill().fillna(0)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            if window < 3:
                # 小窗口：使用差分方差
                diff = np.diff(data, prepend=data[0])

                for i in range(len(data)):
                    start_idx = max(0, i - max(window, 2) + 1)
                    window_diff = diff[start_idx:i + 1]
                    if len(window_diff) > 0:
                        result[i] = np.var(window_diff)
                    else:
                        result[i] = 0
            else:
                # 正常窗口
                for i in range(len(data)):
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 2:
                        result[i] = np.var(window_data, ddof=1)
                    else:
                        if i > 0:
                            result[i] = ((data[i] - data[i - 1]) / np.sqrt(2)) ** 2
                        else:
                            result[i] = 0
            return result

    @staticmethod
    def ts_max(data, window):
        """移动最大值"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).max()
            return result.bfill().fillna(data.fillna(0))
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.max(window_data)
            return result

    @staticmethod
    def ts_min(data, window):
        """移动最小值"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).min()
            return result.bfill().fillna(data.fillna(0))
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.min(window_data)
            return result

    @staticmethod
    def ts_skew(data, window):
        """偏度（智能处理小窗口）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            if window < 5:
                return pd.Series(0, index=data.index)
            else:
                min_periods = min(5, window)
                result = data.rolling(window=window, min_periods=min_periods).skew()
                return result.fillna(0)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                if window < 5:
                    # 小窗口：简化偏度估计
                    start_idx = max(0, i - max(window, 3) + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        mean = np.mean(window_data)
                        std = np.std(window_data)

                        if std > 1e-8:
                            deviation = window_data - mean
                            pos_dev = np.sum(deviation[deviation > 0])
                            neg_dev = np.sum(np.abs(deviation[deviation < 0]))

                            if pos_dev + neg_dev > 0:
                                skew_proxy = (pos_dev - neg_dev) / (pos_dev + neg_dev)
                                result[i] = skew_proxy * 3
                            else:
                                result[i] = 0
                        else:
                            result[i] = 0
                    else:
                        result[i] = 0
                else:
                    # 正常窗口
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        try:
                            if np.std(window_data) < 1e-10:
                                result[i] = 0
                            else:
                                val = stats.skew(window_data)
                                result[i] = 0 if not np.isfinite(val) else val
                        except Exception:
                            result[i] = 0
                    else:
                        result[i] = 0

            return result

    @staticmethod
    def ts_kurt(data, window):
        """峰度（智能处理小窗口）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            if window < 5:
                return pd.Series(0, index=data.index)
            else:
                min_periods = min(5, window)
                result = data.rolling(window=window, min_periods=min_periods).kurt()
                return result.fillna(0)
        else:
            # NumPy实现
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                if window < 5:
                    # 小窗口：简化峰度估计
                    start_idx = max(0, i - max(window, 3) + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        mean = np.mean(window_data)
                        std = np.std(window_data)

                        if std > 1e-8:
                            normalized = (window_data - mean) / std
                            extreme_ratio = np.sum(np.abs(normalized) > 2) / len(normalized)
                            result[i] = extreme_ratio * 10
                        else:
                            result[i] = 0
                    else:
                        result[i] = 0
                else:
                    # 正常窗口
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 4:
                        try:
                            if np.std(window_data) < 1e-10:
                                result[i] = 0
                            else:
                                val = stats.kurtosis(window_data, fisher=True)
                                result[i] = 0 if not np.isfinite(val) else val
                        except Exception:
                            result[i] = 0
                    else:
                        result[i] = 0

            return result

    @staticmethod
    def ts_wma(data, window):
        """加权移动平均（Series 级窗口缓存）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            cache = _get_cache(data)
            key = ('ts_wma', int(window))
            if cache is not None and key in cache:
                return cache[key]

            weights = np.arange(1, window + 1, dtype=np.float64)

            def weighted_mean(x):
                x = np.asarray(x, dtype=float)
                m = ~np.isnan(x)
                if not m.any():
                    return 0.0
                x = x[m]
                w = weights[:len(x)]
                w = w / w.sum()
                return float(np.dot(x, w))

            result = data.rolling(window=window, min_periods=1).apply(weighted_mean, raw=True)
            # 这里保留原本的返回（原实现已足够稳健）；如需 bfill/fillna 可按你现状追加

            if cache is not None:
                cache[key] = result
            return result
        else:
            # 原 Numpy 分支保持不变
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            full_weights = np.arange(1, window + 1, dtype=np.float64)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                if window_data.size == 0:
                    result[i] = 0.0
                    continue
                m = ~np.isnan(window_data)
                if not m.any():
                    result[i] = result[i - 1] if i > 0 and np.isfinite(result[i - 1]) else 0.0
                    continue
                valid = window_data[m]
                w = np.arange(1, len(valid) + 1, dtype=np.float64)
                w = w / w.sum()
                result[i] = float(np.dot(valid, w))
            return result

    @staticmethod
    def ts_ema(data, window):
        """指数移动平均（Series 级窗口缓存）"""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            cache = _get_cache(data)
            key = ('ts_ema', int(window))
            if cache is not None and key in cache:
                return cache[key]

            result = data.ewm(span=window, adjust=False, min_periods=1).mean()

            if cache is not None:
                cache[key] = result
            return result
        else:
            # 原 Numpy 分支保持不变
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            alpha = 2.0 / (window + 1)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

    # ================== 三元操作符（两个数据操作数 + 窗口）===========
    @staticmethod
    def corr(operand1, operand2, window):
        """相关系数"""
        window = Operators._ensure_window_int(window)

        if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
            result = operand1.rolling(window=window, min_periods=2).corr(operand2)
            return result.fillna(0)
        else:
            # NumPy实现
            operand1 = np.asarray(operand1)
            operand2 = np.asarray(operand2)
            result = np.zeros(len(operand1))

            for i in range(len(operand1)):
                start_idx = max(0, i - window + 1)
                if i - start_idx >= 1:  # 至少需要2个点
                    corr = np.corrcoef(operand1[start_idx:i + 1],
                                       operand2[start_idx:i + 1])[0, 1]
                    result[i] = corr if not np.isnan(corr) else 0

            return result

    @staticmethod
    def cov(operand1, operand2, window):
        """协方差"""
        window = Operators._ensure_window_int(window)

        if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
            result = operand1.rolling(window=window, min_periods=2).cov(operand2)
            return result.fillna(0)
        else:
            # NumPy实现
            operand1 = np.asarray(operand1)
            operand2 = np.asarray(operand2)
            result = np.zeros(len(operand1))

            for i in range(len(operand1)):
                start_idx = max(0, i - window + 1)
                if i - start_idx >= 1:  # 至少需要2个点
                    cov = np.cov(operand1[start_idx:i + 1],
                                 operand2[start_idx:i + 1])[0, 1]
                    result[i] = cov if not np.isnan(cov) else 0

            return result



    @staticmethod
    def decay_linear(x, t):
        """Decay_linear operator: 线性衰减加权移动平均"""
        if isinstance(x, pd.Series):
            weights = np.arange(1, t + 1)
            weights = weights / weights.sum()
            result = x.rolling(window=t, min_periods=1).apply(
                lambda w: np.dot(w[~np.isnan(w)], weights[:len(w[~np.isnan(w)])])
                if len(w[~np.isnan(w)]) > 0 else 0
            )
            result = result.replace([np.inf, -np.inf], np.nan)
            return result.fillna(0)
        else:
            raise TypeError("decay_linear operator requires pandas Series")
