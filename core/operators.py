"""core/operators.py

Collection of primitive operators used by the RPN evaluator.

Notes
-----
- All operators are implemented as `@staticmethod`s on the `Operators` class.
- Most operators accept pandas Series or NumPy arrays and try to preserve index/shape.
- Time-series operators follow the naming convention `ts_*` and accept a data input plus a window size.
- Numerical stability is prioritized: clipping, epsilon guards, and NaN/inf handling are used throughout.
"""
import numpy as np
import pandas as pd
import logging

from scipy import stats


def _get_cache(series):
    """
    Get or initialize a per-Series rolling-result cache dictionary.

    In high-frequency scenarios where the same *raw* Series is repeatedly rolled over
    with the same window (e.g., ts_mean/ts_std), we store computed results in
    `series.attrs['_op_cache']`. Some pandas objects or versions may not support
    `attrs` reliably; in those cases we silently disable caching by returning None.
    """
    try:
        return series.attrs.setdefault('_op_cache', {})
    except Exception:
        # Some pandas versions or objects do not support attrs; disable cache gracefully.
        return None

MAX_VALUE = 1e8   # Upper numeric bound for clipping
MIN_VALUE = -1e8  # Lower numeric bound for clipping
EPSILON = 1e-10   # Small epsilon to avoid division by zero

logger = logging.getLogger(__name__)


class Operators:
    """Static operator collection used by the evaluator."""
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def ensure_series_or_array(operand, data_length=None, data_index=None):
        """Ensure operand is a Series/array; broadcast scalars when length/index are provided."""
        if isinstance(operand, (int, float)) and data_length:
            if data_index is not None:
                return pd.Series(operand, index=data_index)
            else:
                return pd.Series([operand] * data_length)
        return operand

    # -------------------------------------------------------------------------
    # Safe arithmetic helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def safe_divide(x, y, default_value=0):
        """Division with guards against division-by-zero and infinities."""
        if isinstance(x, pd.Series):
            # Replace +/-inf and NaN with default_value
            return x.div(y).replace([np.inf, -np.inf], default_value).fillna(default_value)
        else:
            return np.divide(x, y, out=np.full_like(x, default_value, dtype=float), where=y != 0)

    # -------------------------------------------------------------------------
    # Unary operators
    # -------------------------------------------------------------------------
    @staticmethod
    def csrank(operand, data_length=None, data_index=None):
        """Cross-sectional percentile rank (Series: by index or MultiIndex level=1)."""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            if isinstance(operand.index, pd.MultiIndex):
                return operand.groupby(level=1).rank(pct=True)
            else:
                return operand.rank(pct=True)
        else:
            # NumPy array path
            return stats.rankdata(operand, method='average') / len(operand)

    @staticmethod
    def sign(operand, data_length=None, data_index=None):
        """Sign-like indicator: return 1 for positive values, else 0."""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            return (operand > 0).astype(float)
        else:
            return np.where(operand > 0, 1.0, 0.0)

    @staticmethod
    def abs(operand, data_length=None, data_index=None):
        """Absolute value."""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        return np.abs(operand)

    @staticmethod
    def log(operand, data_length=None, data_index=None):
        """Numerically safe log: log(max(|x|+1e-10, 1e-10))."""
        operand = Operators.ensure_series_or_array(operand, data_length, data_index)
        if isinstance(operand, pd.Series):
            return np.log(np.maximum(operand.abs() + 1e-10, 1e-10))
        else:
            return np.log(np.maximum(np.abs(operand) + 1e-10, 1e-10))

    # -------------------------------------------------------------------------
    # Binary operators
    # -------------------------------------------------------------------------
    @staticmethod
    def _align_operands(operand1, operand2):
        """Align scalar-vs-array/Series shapes by broadcasting scalars."""
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
        """Addition."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return operand1 + operand2

    @staticmethod
    def sub(operand1, operand2, data_length=None, data_index=None):
        """Subtraction."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return operand1 - operand2

    @staticmethod
    def mul(operand1, operand2, data_length=None, data_index=None):
        """Multiplication with result clipping to a safe numeric range."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        with np.errstate(over='ignore', invalid='ignore'):
            result = operand1 * operand2
            # Clip to a reasonable numeric range to avoid blow-ups
            if isinstance(result, pd.Series):
                result = result.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            else:
                result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result

    @staticmethod
    def div(operand1, operand2, data_length=None, data_index=None):
        """Division with epsilon guards and result clipping."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)

        # Guard tiny denominators
        if isinstance(operand2, pd.Series):
            operand2 = operand2.where(operand2.abs() > EPSILON, EPSILON)
        else:
            operand2 = np.where(np.abs(operand2) > EPSILON, operand2, EPSILON)

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            result = Operators.safe_divide(operand1, operand2)
            # Clip result to avoid extreme values
            if isinstance(result, pd.Series):
                result = result.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            else:
                result = np.clip(result, MIN_VALUE, MAX_VALUE)
        return result

    @staticmethod
    def greater(operand1, operand2, data_length=None, data_index=None):
        """Comparison: 1.0 if x > y else 0.0."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return (operand1 > operand2).astype(float)

    @staticmethod
    def less(operand1, operand2, data_length=None, data_index=None):
        """Comparison: 1.0 if x < y else 0.0."""
        operand1, operand2 = Operators._align_operands(operand1, operand2)
        return (operand1 < operand2).astype(float)

    # -------------------------------------------------------------------------
    # Time-series operators
    # -------------------------------------------------------------------------
    @staticmethod
    def _ensure_window_int(window):
        """Normalize and clamp `window` to an integer in [1, 100]."""
        if isinstance(window, (pd.Series, np.ndarray)):
            window = int(window[0]) if len(window) > 0 else 5
        else:
            window = int(window)
        return max(1, min(window, 100))  # Prevent pathological windows

    @staticmethod
    def ts_ref(data, window):
        """Shift by `window` steps (value from t-window)."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            return data.shift(window)
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            result[:window] = np.nan
            if window < len(data):
                result[window:] = data[:-window]
            return result

    @staticmethod
    def ts_rank(data, window):
        """Percentile rank of the current value within the rolling window."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            def rank_in_window(x):
                if len(x) < 2:
                    return 0.5
                return (x.iloc[-1] > x).sum() / len(x)

            result = data.rolling(window=window, min_periods=1).apply(rank_in_window, raw=False)
            return result.fillna(0.5)
        else:
            # NumPy implementation
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
        """Rolling mean (with per-Series cache to speed up repeated calls)."""
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
            # NumPy implementation (kept intentionally simple and explicit)
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.mean(window_data)
            return result

    @staticmethod
    def ts_med(data, window):
        """Rolling median."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).median()
            return result.bfill().fillna(0)
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.median(window_data)
            return result

    @staticmethod
    def ts_sum(data, window):
        """Rolling sum with clipping to avoid extreme magnitudes."""
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
        """Rolling standard deviation (with clipping and per-Series cache)."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            cache = _get_cache(data)
            key = ('ts_std', int(window))
            if cache is not None and key in cache:
                return cache[key]

            # Clip first to ensure robustness
            data_clipped = data.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            with np.errstate(all='ignore'):
                result = data_clipped.rolling(window=window, min_periods=min(3, window)).std()
            result = result.replace([np.inf, -np.inf], 0).bfill().fillna(0)

            if cache is not None:
                cache[key] = result
            return result
        else:
            # NumPy branch
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
        """Rolling variance with special handling for small windows."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            with np.errstate(over='ignore', invalid='ignore'):
                result = data.rolling(window=window, min_periods=min(3, window)).var()
            return result.replace([np.inf, -np.inf], 0).bfill().fillna(0)
        else:
            # NumPy implementation with small-window fallback
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            if window < 3:
                # For tiny windows, approximate using first-difference variance
                diff = np.diff(data, prepend=data[0])

                for i in range(len(data)):
                    start_idx = max(0, i - max(window, 2) + 1)
                    window_diff = diff[start_idx:i + 1]
                    if len(window_diff) > 0:
                        result[i] = np.var(window_diff)
                    else:
                        result[i] = 0
            else:
                # Standard variance on the window
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
        """Rolling maximum."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).max()
            return result.bfill().fillna(data.fillna(0))
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.max(window_data)
            return result

    @staticmethod
    def ts_min(data, window):
        """Rolling minimum."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            result = data.rolling(window=window, min_periods=1).min()
            return result.bfill().fillna(data.fillna(0))
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.min(window_data)
            return result

    @staticmethod
    def ts_skew(data, window):
        """Rolling skewness with a simplified proxy for very small windows."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            if window < 5:
                return pd.Series(0, index=data.index)
            else:
                min_periods = min(5, window)
                result = data.rolling(window=window, min_periods=min_periods).skew()
                return result.fillna(0)
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                if window < 5:
                    # Small-window approximation for skewness
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
                    # Standard window case
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
        """Rolling kurtosis with a simplified proxy for very small windows."""
        window = Operators._ensure_window_int(window)

        if isinstance(data, pd.Series):
            if window < 5:
                return pd.Series(0, index=data.index)
            else:
                min_periods = min(5, window)
                result = data.rolling(window=window, min_periods=min_periods).kurt()
                return result.fillna(0)
        else:
            # NumPy implementation
            data = np.asarray(data)
            result = np.zeros_like(data, dtype=np.float64)

            for i in range(len(data)):
                if window < 5:
                    # Small-window approximation for kurtosis
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
                    # Standard window case
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 4:
                        try:
                            if np.std(window_data) < 1e-_
