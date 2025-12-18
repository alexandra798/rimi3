# mcts/reward_calculator.py

from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr, pearsonr
import logging
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd
from utils.metrics import calculate_ic as _utils_ic
from sklearn.exceptions import ConvergenceWarning
import warnings


from utils.metrics import calculate_ic as _ic, calculate_ic
from core import RPNEvaluator, RPNValidator
from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Reward definition overview (as implemented in this class):

    - Intermediate reward:
        Reward_inter = IC - λ * (1/k) * Σ |mutIC_i| + diversity_bonus

      Where:
        IC combines daily RankIC and global IC (weighted).
        mutIC_i measures redundancy with existing alphas (correlation-based).
        diversity_bonus encourages non-trivial variance in the generated signal.

    - Terminal reward:
        Reward_end = composite_ic - λ_turnover * turnover - λ_regime * regime_variance

      Where:
        composite_ic is derived from a pooled alpha ensemble (e.g., Lasso-weighted).
    """

    def __init__(self, alpha_pool, lambda_param=0.1, sample_size=5000,
                 pool_size=100, min_std=1e-6, random_seed=42, cache_size=500,
                 lambda_turnover=0.02, lambda_regime=0.10):
        self.utils_metrics = None
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param
        self.sample_size = sample_size
        self.pool_size = pool_size

        # Minimum standard deviation threshold to treat a signal as "nearly constant"
        self.min_std = min_std

        # Fixed RNG seed to make sampling deterministic/reproducible
        self.random_seed = random_seed

        # Unified evaluator for formula/state execution
        self.formula_evaluator = FormulaEvaluator()

        # LRU-like cache for intermediate reward results
        self.cache_size = cache_size
        self._cache = OrderedDict()

        # Threshold used to decide whether to add a terminal alpha into the pool
        self.high_quality_ic_threshold = 0.015  # friendlier for cold start
        self.low_ic_penalty = -0.1

        # Diagnostics counter (how many times constant-like signals are penalized)
        self.constant_penalty_count = 0

        self.rng = np.random.RandomState(random_seed)

        # Penalty coefficients for additional constraints
        self.lambda_turnover = lambda_turnover
        self.lambda_regime = lambda_regime

        # Per-iteration fixed sampling control (to stabilize cache/reward)
        self.current_sample_indices = None
        self.current_data_id = None


    def set_iteration_sample(self, X_data, y_data):
        """Called at the start of each iteration to fix the sample for this iteration."""
        if len(X_data) > self.sample_size:
            self.current_sample_indices = self.rng.choice(
                len(X_data), self.sample_size, replace=False
            )
            # Mark a stable data_id to make cache keys deterministic within an iteration
            if hasattr(X_data, 'attrs'):
                X_data.attrs['data_id'] = f"iter_sample_{id(X_data)}_{self.random_seed}"
                self.current_data_id = X_data.attrs['data_id']
            else:
                self.current_data_id = f"iter_sample_{id(X_data)}_{self.random_seed}"
        else:
            self.current_sample_indices = None
            if hasattr(X_data, 'attrs'):
                X_data.attrs['data_id'] = f"full_data_{id(X_data)}"
                self.current_data_id = X_data.attrs['data_id']
            else:
                self.current_data_id = f"full_data_{id(X_data)}"

    def _manage_cache(self):
        """Maintain cache size using FIFO eviction (pop the oldest entries)."""
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _finite_series(x, index=None):
        import numpy as np
        import pandas as pd
        # Convert input into a Series and remove non-finite values (inf -> NaN -> dropped)
        s = pd.Series(x if not hasattr(x, 'values') else x.values, index=index)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s[np.isfinite(s)]

    def is_nearly_constant(self, values):
        """Check whether a vector/Series is nearly constant (low variance)."""
        if values is None:
            return True

        if hasattr(values, 'values'):
            values = values.values
        values = np.array(values).flatten()

        valid_values = values[~np.isnan(values)]
        if len(valid_values) < 2:
            return True

        std = np.std(valid_values)
        return std < self.min_std

    def _estimate_turnover(self, alpha_values, X_like):
        import pandas as pd, numpy as np
        s = alpha_values.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return 0.0
        # If MultiIndex includes 'ticker': compute mean absolute day-to-day changes per ticker, then average
        if isinstance(s.index, pd.MultiIndex) and 'ticker' in s.index.names:
            diffs = s.groupby(level='ticker').diff().abs()
            return float(np.nanmean(diffs))
        # Otherwise: mean absolute first difference along the index
        return float(np.nanmean(np.abs(s.diff())))

    def _regime_daily_rank_ic_stats(self, alpha_values, y_like, X_like):
        """
        Regime split:
          2012–2015, 2016–2019, 2020–2021, 2022–end

        For each regime, compute the mean daily RankIC (vectorized) and return:
        - per-regime means
        - variance across regime means (used as a stability penalty)
        """
        import pandas as pd, numpy as np

        dates = self._get_dates(X_like)
        if dates is None:
            return {'by_regime': [], 'var': 0.0}

        pred = alpha_values if isinstance(alpha_values, pd.Series) else pd.Series(
            getattr(alpha_values, 'values', alpha_values)
        )
        y = y_like if isinstance(y_like, pd.Series) else pd.Series(
            getattr(y_like, 'values', y_like), index=pred.index
        )

        base = pd.concat([pred.rename('pred'), y.rename('y')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if base.empty:
            return {'by_regime': [], 'var': 0.0}

        d = pd.to_datetime(dates, errors='coerce').reindex(base.index)

        def mask_span(dser, a, b):
            return (dser >= pd.Timestamp(f'{a}-01-01')) & (dser <= pd.Timestamp(f'{b}-12-31'))

        spans = [
            ('2012-2015', mask_span(d, 2012, 2015)),
            ('2016-2019', mask_span(d, 2016, 2019)),
            ('2020-2021', mask_span(d, 2020, 2021)),
            ('2022-end', d >= pd.Timestamp('2022-01-01')),
        ]

        stats = []
        for _, m in spans:
            idx = base.index[m.fillna(False)]
            if len(idx) == 0:
                stats.append(np.nan)
                continue
            ic = self._daily_rank_ic_vectorized(base.loc[idx, 'pred'], base.loc[idx, 'y'], d.loc[idx])
            stats.append(float(ic) if np.isfinite(ic) else np.nan)

        arr = np.asarray([x for x in stats if isinstance(x, (int, float)) and np.isfinite(x)], dtype=float)
        return {'by_regime': stats, 'var': float(np.var(arr)) if arr.size > 0 else 0.0}

    def calculate_intermediate_reward(self, state, X_data, y_data):
        # Cache key includes the token sequence plus a per-iteration data identifier
        cache_key = f"{' '.join([t.name for t in state.token_sequence])}_{self.current_data_id}"

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Reject syntactically invalid partial expressions early
        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return -0.1

        try:
            # Use a fixed sample for the entire iteration to stabilize reward and cache usage
            if self.current_sample_indices is not None:
                X_sample = X_data.iloc[self.current_sample_indices]
                y_sample = y_data.iloc[self.current_sample_indices]
            else:
                X_sample = X_data
                y_sample = y_data

            # Evaluate the current (partial) state into alpha values
            alpha_values = self.formula_evaluator.evaluate_state(state, X_sample)

            if alpha_values is None or alpha_values.isna().all():
                return -0.1
            else:
                # Extra: detect constant-like outputs and penalize hard
                valid_values = alpha_values.dropna()

                if len(valid_values) > 10:
                    std = valid_values.std()
                    unique_ratio = len(valid_values.unique()) / len(valid_values)

                    # Multiple constant-ness checks: low std OR extremely low unique ratio
                    if std < self.min_std or unique_ratio < 0.01:
                        self.constant_penalty_count += 1
                        logger.debug(f"Constant alpha in intermediate state (std={std:.8f}, unique={unique_ratio:.2%})")
                        return -1.0  # strong penalty

                # Compute IC (blend of daily RankIC and global IC)
                global_ic = self.calculate_ic(alpha_values, y_sample)
                daily_ic = self.calculate_daily_rank_ic(alpha_values, y_sample, X_sample)
                ic = 0.7 * daily_ic + 0.3 * global_ic

                # Diversity bonus: encourage higher variance signals (log-scaled)
                if hasattr(alpha_values, 'values'):
                    values = alpha_values.values
                else:
                    values = np.array(alpha_values)

                valid_values_for_bonus = values[~np.isnan(values)]
                if len(valid_values_for_bonus) > 0:
                    std = np.std(valid_values_for_bonus)
                    diversity_bonus = np.log(1 + std) * 0.1
                else:
                    diversity_bonus = 0

                # Compute redundancy penalty via mutual IC with a small subset of alpha_pool
                if len(self.alpha_pool) > 0:
                    mut_ic_sum = 0
                    valid_count = 0
                    for alpha in self.alpha_pool[:10]:
                        if 'values' in alpha:
                            alpha_sample_values = self.formula_evaluator.evaluate(alpha['formula'], X_sample)
                            if isinstance(alpha_sample_values, pd.Series) and isinstance(alpha_values, pd.Series):
                                alpha_sample_values = alpha_sample_values.reindex(alpha_values.index)
                            mut_ic = self._calculate_mutual_ic(alpha_values, alpha_sample_values)

                            if not np.isnan(mut_ic):
                                mut_ic_sum += abs(mut_ic)
                                valid_count += 1

                    if valid_count > 0:
                        avg_mut_ic = mut_ic_sum / valid_count
                        result = ic - self.lambda_param * avg_mut_ic + diversity_bonus
                    else:
                        result = ic + diversity_bonus
                else:
                    result = ic + diversity_bonus

            # Cache and return
            self._cache[cache_key] = result
            self._manage_cache()
            return result

        except Exception as e:
            logger.error(f"Error in intermediate reward: {e}")
            return -0.1

    def calculate_terminal_reward(self, state, X_data, y_data, evaluate_func=None):
        """Compute terminal reward (no early exit for low IC)."""
        if state.token_sequence[-1].name != 'END':
            return self.low_ic_penalty

        try:
            formula_str = ' '.join([t.name for t in state.token_sequence])
            alpha_values = self.formula_evaluator.evaluate(
                formula_str, X_data, allow_partial=False
            )

            if alpha_values is None or alpha_values.isna().all():
                return self.low_ic_penalty

            # Penalize constant-like terminal outputs
            if self.is_nearly_constant(alpha_values):
                logger.debug(f"Terminal state produces constant alpha")
                return self.low_ic_penalty

            # Compute IC (same blend as intermediate reward)
            global_ic = self.calculate_ic(alpha_values, y_data)
            daily_ic = self.calculate_daily_rank_ic(alpha_values, y_data, X_data)
            individual_ic = 0.7 * daily_ic + 0.3 * global_ic

            # Decide whether to add this alpha into the pool (must exceed quality threshold)
            readable_formula = ' '.join([t.name for t in state.token_sequence])
            if abs(individual_ic) >= self.high_quality_ic_threshold and not self.is_nearly_constant(alpha_values):
                new_alpha = {
                    'formula': readable_formula,
                    'values': alpha_values,
                    'ic': float(individual_ic),
                    'weight': 1.0
                }

                exists = any(a.get('formula') == readable_formula for a in self.alpha_pool)
                if not exists:
                    self.alpha_pool.append(new_alpha)
                    if len(self.alpha_pool) > self.pool_size:
                        # Keep only top alphas by absolute IC
                        self.alpha_pool.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)
                        self.alpha_pool = self.alpha_pool[:self.pool_size]
                    logger.info(f"Added high quality alpha: IC={individual_ic:.4f}")

            # Composite reward: ensemble IC minus turnover/regime instability penalties
            composite_ic = self._calculate_composite_ic(X_data, y_data) if len(self.alpha_pool) else individual_ic
            turnover = self._estimate_turnover(alpha_values, X_data)
            reg_stats = self._regime_daily_rank_ic_stats(alpha_values, y_data, X_data)
            reg_var = float(reg_stats.get('var', 0.0))

            reward = float(composite_ic) - self.lambda_turnover * float(turnover) - self.lambda_regime * reg_var
            return float(reward)

        except Exception as e:
            logger.error(f"Error in terminal reward: {e}")
            return self.low_ic_penalty

    def calculate_ic(self, predictions, targets):
        """Unified IC: Spearman (via utils.metrics), with automatic index alignment inside utils."""
        try:
            return float(_utils_ic(predictions, targets, method='spearman'))
        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def _get_dates(self, X_like):
        """
        Extract and cache the date series from X_like to avoid repeated to_datetime calls.

        Supported sources:
        - MultiIndex containing a 'date' level
        - DataFrame column named 'date'
        - DataFrame column named 'time' (string/numeric timestamps supported)
        """
        import pandas as pd, numpy as np

        if not hasattr(self, '_dates_cache'):
            self._dates_cache = {}

        key = id(X_like)
        cached = self._dates_cache.get(key, None)
        try:
            n = len(X_like)
        except Exception:
            n = None
        if cached is not None and (n is None or len(cached) == n):
            return cached

        dates = None
        try:
            if isinstance(X_like, pd.DataFrame):
                if 'date' in X_like.columns:
                    s = X_like['date']
                    if not np.issubdtype(s.dtype, np.datetime64):
                        s = pd.to_datetime(s, errors='coerce')
                    dates = s
                elif 'time' in X_like.columns:
                    dates = pd.to_datetime(X_like['time'], errors='coerce')
            elif isinstance(X_like.index, pd.MultiIndex) and 'date' in X_like.index.names:
                s = X_like.index.get_level_values('date')
                dates = pd.to_datetime(s, errors='coerce')
        except Exception:
            dates = None

        self._dates_cache[key] = dates
        return dates

    def _daily_rank_ic_vectorized(self, pred, tgt, dates):
        """
        Fully vectorized daily Spearman(IC) computation:

          1) groupby(date).rank() to obtain within-day ranks
          2) compute per-day Pearson correlation of ranks (equivalent to Spearman)
          3) return the mean across days

        `dates` is a pd.Series (datetime) aligned to pred/tgt.
        """
        import pandas as pd, numpy as np

        if dates is None:
            # Fallback: overall Spearman (consistent with utils.metrics)
            from utils.metrics import calculate_ic as _utils_ic
            try:
                return float(_utils_ic(pred, tgt, method='spearman'))
            except Exception:
                return 0.0

        s = pd.concat(
            [pred.rename('pred'), tgt.rename('y'), pd.Series(dates, index=pred.index, name='date')],
            axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return 0.0

        g = s['date']
        r1 = s['pred'].groupby(g).rank(method='average')
        r2 = s['y'].groupby(g).rank(method='average')

        # Demean ranks per day
        r1c = r1 - r1.groupby(g).transform('mean')
        r2c = r2 - r2.groupby(g).transform('mean')

        # Vectorized correlation per day: num / sqrt(den1*den2)
        num = (r1c * r2c).groupby(g).sum()
        den = (r1c.pow(2).groupby(g).sum() * r2c.pow(2).groupby(g).sum()) ** 0.5
        ic_by_day = (num / den).replace([np.inf, -np.inf], np.nan).dropna()

        return float(ic_by_day.mean()) if not ic_by_day.empty else 0.0

    def calculate_daily_rank_ic(self, predictions, targets, X_like):
        """Daily RankIC (vectorized): avoids groupby.apply warnings and is faster."""
        import pandas as pd

        try:
            pred = predictions if isinstance(predictions, pd.Series) else pd.Series(
                getattr(predictions, 'values', predictions)
            )
            tgt = targets if isinstance(targets, pd.Series) else pd.Series(
                getattr(targets, 'values', targets), index=pred.index
            )
            df = pd.concat([pred.rename('pred'), tgt.rename('y')], axis=1).dropna()
            if df.empty:
                return 0.0

            # Retrieve cached dates and align to df.index
            dates = self._get_dates(X_like)
            if dates is not None:
                try:
                    dates = dates.reindex(df.index)
                except Exception:
                    dates = None

            return self._daily_rank_ic_vectorized(df['pred'], df['y'], dates)

        except Exception as e:
            logger.error(f"Error in daily rank IC (vectorized): {e}")
            return 0.0

    def _calculate_mutual_ic(self, alpha1_values, alpha2_values):
        """
        Compute mutual correlation between two alpha signals (used as redundancy penalty).

        Implementation details:
        - Normalizes types into numeric Series
        - Aligns indices (intersection preferred; position fallback only when safe)
        - Cleans NaN/Inf
        - Early-exits on near-constant signals
        - Uses Pearson correlation (can be swapped to Spearman if desired)
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import pearsonr

        try:
            def to_series(x):
                # Preserve index when possible; for DataFrame prefer single-column
                if isinstance(x, pd.Series):
                    return x.copy()
                if isinstance(x, pd.DataFrame):
                    if x.shape[1] == 1:
                        s = x.iloc[:, 0]
                        s.name = getattr(x, 'name', s.name)
                        return s
                    return pd.Series(x.squeeze())
                return pd.Series(getattr(x, 'values', x))

            s1 = to_series(alpha1_values)
            s2 = to_series(alpha2_values)

            s1 = pd.to_numeric(s1, errors="coerce")
            s2 = pd.to_numeric(s2, errors="coerce")

            # Prefer index-based alignment
            if not s1.index.equals(s2.index):
                if isinstance(s1.index, pd.MultiIndex) and isinstance(s2.index, pd.MultiIndex):
                    try:
                        s2.index = s2.index.set_names(s1.index.names)
                    except Exception:
                        pass
                common = s1.index.intersection(s2.index)
                if len(common) >= 2:
                    s1 = s1.loc[common]
                    s2 = s2.loc[common]
                else:
                    # If intersection is too small, only fallback to positional alignment when lengths match
                    if len(s1) == len(s2) and len(s1) >= 2:
                        s1 = pd.Series(s1.values)
                        s2 = pd.Series(s2.values)
                    else:
                        return 0.0

            mask = np.isfinite(s1.values) & np.isfinite(s2.values)
            if mask.sum() < 2:
                return 0.0
            v1 = s1.values[mask]
            v2 = s2.values[mask]

            if v1.std() < getattr(self, "min_std", 1e-6) or v2.std() < getattr(self, "min_std", 1e-6):
                return 0.0

            corr, _ = pearsonr(v1, v2)
            return float(corr) if np.isfinite(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating mutual IC: {e}")
            return 0.0

    def _build_design_matrix(self, feature_dict, y_series):
        """
        Build a consistent design matrix for ensemble learning.

        Steps:
        - Align features and target by index; drop NaN/Inf rows
        - Remove nearly-constant columns
        - Remove exact duplicate columns
        - Standardize features to zero-mean unit-variance
        - Standardize target as well (to stabilize regression)
        """
        import numpy as np, pandas as pd

        cols = []
        for name, series in feature_dict.items():
            s = series if isinstance(series, pd.Series) else pd.Series(series, name=name)
            cols.append(s.rename(name))

        df = pd.concat(cols + [pd.Series(y_series).rename("__y__")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            return None, None, []

        y = df.pop("__y__").astype(float)
        X = df.astype(float)

        # 1) Drop nearly-constant columns
        std = X.std(axis=0).replace(0, np.nan)
        keep = std > max(getattr(self, "min_std", 1e-6), 1e-12)
        X = X.loc[:, keep]
        if X.shape[1] == 0:
            return None, None, []

        # 2) Drop exactly duplicated columns
        X = X.loc[:, ~X.T.duplicated()]

        # 3) Standardize columns
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)

        return X.values.astype(float), y.values.astype(float), list(X.columns)

    def _calculate_composite_ic(self, X_data, y_data):
        if len(self.alpha_pool) == 0:
            return 0.0

        # Build feature dictionary from pool; reuse stored values whenever possible
        feature_dict = {}
        for alpha in self.alpha_pool:
            formula = alpha['formula']

            # Prefer precomputed values to avoid re-evaluation cost
            if 'values' in alpha and alpha['values'] is not None:
                feature_dict[formula] = alpha['values']
            else:
                # Only evaluate if values are missing
                values = self.formula_evaluator.evaluate(formula, X_data)
                if values is not None:
                    alpha['values'] = values
                    feature_dict[formula] = values

        if not feature_dict:
            return 0.0

        # Build aligned design matrix
        X_mat, y_vec, names = self._build_design_matrix(feature_dict, y_data)
        if X_mat is None:
            valid_ic = [a.get('ic', 0.0) for a in self.alpha_pool if 'ic' in a]
            return float(np.mean(valid_ic)) if valid_ic else 0.0

        # Lasso regression to obtain a sparse/stable ensemble of alphas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.linear_model = Lasso(
                alpha=0.005,          # slightly stronger regularization to resist collinearity
                fit_intercept=False,
                max_iter=20000,       # more iterations for convergence
                tol=1e-4,             # relaxed tolerance
                selection="cyclic",   # more stable under correlated features
                warm_start=True
            )
            self.linear_model.fit(X_mat, y_vec)

        # Update alpha weights from regression coefficients
        weights = self.linear_model.coef_
        for i, alpha in enumerate(self.alpha_pool):
            if i < len(weights):
                alpha['weight'] = float(weights[i])

        # Composite IC for the ensemble prediction
        pred = self.linear_model.predict(X_mat)
        return float(_utils_ic(pred, y_vec, method='spearman'))



    def _orthogonalized_gain(self, f_series, selected_mat_or_none, y_series):
        """
        Compute "marginal RankIC after orthogonalization" against already selected factors.

        f_series: pd.Series aligned to y
        selected_mat_or_none: np.ndarray of shape (n, k) or None
        y_series: pd.Series aligned to f_series
        """
        f = pd.concat([pd.Series(f_series).rename("f"), pd.Series(y_series).rename("y")], axis=1).dropna()
        if f.empty:
            return 0.0

        f_vec = f["f"].values.astype(float)
        y_vec = f["y"].values.astype(float)

        # Remove the component explained by the selected set (linear projection)
        if selected_mat_or_none is None or selected_mat_or_none.shape[1] == 0:
            resid = f_vec
        else:
            coef, *_ = np.linalg.lstsq(selected_mat_or_none, f_vec, rcond=None)
            resid = f_vec - selected_mat_or_none @ coef

        # Spearman IC between residual and y
        r = pd.Series(resid, index=f.index)
        y = pd.Series(y_vec, index=f.index)
        try:
            return float(self.utils_metrics.calculate_ic(r, y, method='spearman'))
        except Exception:
            return 0.0
