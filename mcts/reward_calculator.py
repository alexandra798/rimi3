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
    - 中间奖励: Reward_inter = IC - λ * (1/k) * Σ mutIC_i
    - 终止奖励: Reward_end = 合成alpha的IC
    """

    def __init__(self, alpha_pool, lambda_param=0.1, sample_size=5000,
                 pool_size=100, min_std=1e-6, random_seed=42, cache_size=500,
                 lambda_turnover=0.02, lambda_regime=0.10):
        self.utils_metrics = None
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param
        self.sample_size = sample_size
        self.pool_size = pool_size
        self.min_std = min_std  # 新增：最小标准差阈值
        self.random_seed = random_seed  # 保存seed
        self.formula_evaluator = FormulaEvaluator()
        self.cache_size = cache_size
        self._cache = OrderedDict()

        self.high_quality_ic_threshold = 0.015  # 冷启动更友好
        self.low_ic_penalty = -0.1

        self.constant_penalty_count = 0

        self.rng = np.random.RandomState(random_seed)

        self.lambda_turnover = lambda_turnover
        self.lambda_regime = lambda_regime

        self.current_sample_indices = None
        self.current_data_id = None


    def set_iteration_sample(self, X_data, y_data):
        """每个iteration开始时调用，固定本轮采样"""
        if len(X_data) > self.sample_size:
            self.current_sample_indices = self.rng.choice(
                len(X_data), self.sample_size, replace=False
            )
            # 标记数据ID，确保缓存稳定
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
        """管理缓存大小"""
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _finite_series(x, index=None):
        import numpy as np
        import pandas as pd
        s = pd.Series(x if not hasattr(x, 'values') else x.values, index=index)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s[np.isfinite(s)]

    def is_nearly_constant(self, values):
        """检查值是否接近常数"""
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
        # MultiIndex(date,ticker) -> 按 ticker 求 |Δ|，再日均
        if isinstance(s.index, pd.MultiIndex) and 'ticker' in s.index.names:
            diffs = s.groupby(level='ticker').diff().abs()
            return float(np.nanmean(diffs))
        # 单索引 -> 直接 |Δ| 均值
        return float(np.nanmean(np.abs(s.diff())))

    def _regime_daily_rank_ic_stats(self, alpha_values, y_like, X_like):
        """
        分期：2012–2015、2016–2019、2020–2021、2022–end
        逐段用向量化的逐日 RankIC 计算均值，并返回各段均值 + 方差
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
        cache_key = f"{' '.join([t.name for t in state.token_sequence])}_{self.current_data_id}"

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return -0.1

        try:
            # 使用固定的采样索引，不再每次随机
            if self.current_sample_indices is not None:
                X_sample = X_data.iloc[self.current_sample_indices]
                y_sample = y_data.iloc[self.current_sample_indices]
            else:
                X_sample = X_data
                y_sample = y_data

            # 评估
            alpha_values = self.formula_evaluator.evaluate_state(state, X_sample)

            if alpha_values is None or alpha_values.isna().all():
                return -0.1
            else:
                # 新增：检查是否为常数
                valid_values = alpha_values.dropna()

                if len(valid_values) > 10:
                    std = valid_values.std()
                    unique_ratio = len(valid_values.unique()) / len(valid_values)

                    # 多重检测
                    if std < self.min_std or unique_ratio < 0.01:
                        self.constant_penalty_count += 1
                        logger.debug(f"Constant alpha in intermediate state (std={std:.8f}, unique={unique_ratio:.2%})")
                        return -1.0  # 严厉惩罚

                # 计算IC
                global_ic = self.calculate_ic(alpha_values, y_sample)
                daily_ic = self.calculate_daily_rank_ic(alpha_values, y_sample, X_sample)
                ic = 0.7 * daily_ic + 0.3 * global_ic

                # 额外奖励高变异性的因子
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

                # 计算mutIC
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

            # 缓存结果
            self._cache[cache_key] = result
            self._manage_cache()
            return result

        except Exception as e:
            logger.error(f"Error in intermediate reward: {e}")
            return -0.1

    def calculate_terminal_reward(self, state, X_data, y_data, evaluate_func=None):
        """计算终止奖励 - 不再因IC低而早退"""
        if state.token_sequence[-1].name != 'END':
            return self.low_ic_penalty

        try:
            formula_str = ' '.join([t.name for t in state.token_sequence])
            alpha_values = self.formula_evaluator.evaluate(
                formula_str, X_data, allow_partial=False
            )

            if alpha_values is None or alpha_values.isna().all():
                return self.low_ic_penalty

            # 检查是否为常数
            if self.is_nearly_constant(alpha_values):
                logger.debug(f"Terminal state produces constant alpha")
                return self.low_ic_penalty

            # 计算IC
            global_ic = self.calculate_ic(alpha_values, y_data)
            daily_ic = self.calculate_daily_rank_ic(alpha_values, y_data, X_data)
            individual_ic = 0.7 * daily_ic + 0.3 * global_ic

            # 决定是否入池（需过阈值）
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
                        self.alpha_pool.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)
                        self.alpha_pool = self.alpha_pool[:self.pool_size]
                    logger.info(f"Added high quality alpha: IC={individual_ic:.4f}")

            # 计算综合奖励
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
        """统一：Spearman（utils.metrics），自动对齐索引"""
        try:
            return float(_utils_ic(predictions, targets, method='spearman'))
        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def _get_dates(self, X_like):
        """
        从 X_like 提取日期序列并缓存，避免重复 to_datetime。
        支持：index含 'date' 的 MultiIndex，或 DataFrame 的 'date'/'time' 列。
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
                    # 数字/字符串日期都可
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
        纯向量化的逐日 Spearman(IC) 计算：
          先按日 groupby.rank() 得到秩，再一次性算每日的相关系数，最后取日均。
        dates 为 pd.Series（已是 datetime），与 pred/tgt 对齐。
        """
        import pandas as pd, numpy as np

        if dates is None:
            # 回退到整体 Spearman（保持你项目 utils.metrics 的口径）
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
        # 组内秩
        r1 = s['pred'].groupby(g).rank(method='average')
        r2 = s['y'].groupby(g).rank(method='average')

        # 去中心化
        r1c = r1 - r1.groupby(g).transform('mean')
        r2c = r2 - r2.groupby(g).transform('mean')

        # 向量化求每日日内相关： num / sqrt(den1*den2)
        num = (r1c * r2c).groupby(g).sum()
        den = (r1c.pow(2).groupby(g).sum() * r2c.pow(2).groupby(g).sum()) ** 0.5
        ic_by_day = (num / den).replace([np.inf, -np.inf], np.nan).dropna()

        return float(ic_by_day.mean()) if not ic_by_day.empty else 0.0

    def calculate_daily_rank_ic(self, predictions, targets, X_like):
        """逐日 RankIC（向量化版，消除 groupby.apply 的 FutureWarning & 加速）"""
        import pandas as pd

        try:
            # 统一为 Series，并内联对齐
            pred = predictions if isinstance(predictions, pd.Series) else pd.Series(
                getattr(predictions, 'values', predictions)
            )
            tgt = targets if isinstance(targets, pd.Series) else pd.Series(
                getattr(targets, 'values', targets), index=pred.index
            )
            df = pd.concat([pred.rename('pred'), tgt.rename('y')], axis=1).dropna()
            if df.empty:
                return 0.0

            # 取/缓存日期列，并对齐到 df.index
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
        计算两个alpha的“互信息相关性”（这里按皮尔逊相关执行去冗余）。
        自动处理：类型规整、索引对齐、NaN/Inf清洗、近常数早退。
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import pearsonr

        try:
            def to_series(x):
                # 尽量保留原索引；若是DataFrame取第一列
                if isinstance(x, pd.Series):
                    return x.copy()
                if isinstance(x, pd.DataFrame):
                    if x.shape[1] == 1:
                        s = x.iloc[:, 0]
                        s.name = getattr(x, 'name', s.name)
                        return s
                    # squeeze 仍然返回Series/ndarray；保证是Series
                    return pd.Series(x.squeeze())
                return pd.Series(getattr(x, 'values', x))

            s1 = to_series(alpha1_values)
            s2 = to_series(alpha2_values)

            # 转成数值并统一清洗
            s1 = pd.to_numeric(s1, errors="coerce")
            s2 = pd.to_numeric(s2, errors="coerce")

            # ------- 索引对齐（优先按索引交集） -------
            if not s1.index.equals(s2.index):
                # MultiIndex层名不一致时先对齐层名，避免“同物不同名”
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
                    # 交集过小 -> 若长度相等则退化为按位置对齐；否则放弃计算
                    if len(s1) == len(s2) and len(s1) >= 2:
                        s1 = pd.Series(s1.values)
                        s2 = pd.Series(s2.values)
                    else:
                        return 0.0

            # ------- 清洗并早退 -------
            mask = np.isfinite(s1.values) & np.isfinite(s2.values)
            if mask.sum() < 2:
                return 0.0
            v1 = s1.values[mask]
            v2 = s2.values[mask]

            if v1.std() < getattr(self, "min_std", 1e-6) or v2.std() < getattr(self, "min_std", 1e-6):
                return 0.0

            # 皮尔逊相关（或改成 spearmanr 也可）
            corr, _ = pearsonr(v1, v2)
            return float(corr) if np.isfinite(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating mutual IC: {e}")
            return 0.0

    def _build_design_matrix(self, feature_dict, y_series):
        """
        统一构建设计矩阵：
        - 对齐并丢 NaN
        - 去掉近常数/重复列
        - 列标准化到零均值单位方差
        - 目标也标准化，回头再还原评估IC
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

        # 1) 去掉近常数列
        std = X.std(axis=0).replace(0, np.nan)
        keep = std > max(getattr(self, "min_std", 1e-6), 1e-12)
        X = X.loc[:, keep]
        if X.shape[1] == 0:
            return None, None, []

        # 2) 去掉完全重复/高度共线列（简单版：精确重复）
        X = X.loc[:, ~X.T.duplicated()]

        # 3) 标准化（逐列）
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)

        return X.values.astype(float), y.values.astype(float), list(X.columns)

    def _calculate_composite_ic(self, X_data, y_data):
        if len(self.alpha_pool) == 0:
            return 0.0

        # 构建设计矩阵，复用已有values
        feature_dict = {}
        for alpha in self.alpha_pool:
            formula = alpha['formula']

            # 优先使用已存储的values
            if 'values' in alpha and alpha['values'] is not None:
                feature_dict[formula] = alpha['values']
            else:
                # 只在没有values时才评估
                values = self.formula_evaluator.evaluate(formula, X_data)
                if values is not None:
                    alpha['values'] = values  # 存储以供后续复用
                    feature_dict[formula] = values

        if not feature_dict:
            return 0.0

        # 构建对齐的设计矩阵
        X_mat, y_vec, names = self._build_design_matrix(feature_dict, y_data)
        if X_mat is None:
            valid_ic = [a.get('ic', 0.0) for a in self.alpha_pool if 'ic' in a]
            return float(np.mean(valid_ic)) if valid_ic else 0.0

        # Lasso回归
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.linear_model = Lasso(
                alpha=0.005,  # 正则略加强以抗共线
                fit_intercept=False,
                max_iter=20000,  # 迭代更足
                tol=1e-4,  # 放宽收敛阈值
                selection="cyclic",  # 对高度相关特征更稳
                warm_start=True
            )
            self.linear_model.fit(X_mat, y_vec)

        # 更新权重
        weights = self.linear_model.coef_
        for i, alpha in enumerate(self.alpha_pool):
            if i < len(weights):
                alpha['weight'] = float(weights[i])

        # 计算合成IC
        pred = self.linear_model.predict(X_mat)
        return float(_utils_ic(pred, y_vec, method='spearman'))



    def _orthogonalized_gain(self, f_series, selected_mat_or_none, y_series):
        """
        计算“对已选集合正交后的边际 RankIC”
        f_series: pd.Series(index=y.index)
        selected_mat_or_none: np.ndarray of shape (n, k) or None
        y_series: pd.Series(index=f_series.index)
        """
        f = pd.concat([pd.Series(f_series).rename("f"), pd.Series(y_series).rename("y")], axis=1).dropna()
        if f.empty:
            return 0.0

        f_vec = f["f"].values.astype(float)
        y_vec = f["y"].values.astype(float)

        if selected_mat_or_none is None or selected_mat_or_none.shape[1] == 0:
            resid = f_vec
        else:
            coef, *_ = np.linalg.lstsq(selected_mat_or_none, f_vec, rcond=None)
            resid = f_vec - selected_mat_or_none @ coef

        # 残差与 y 的 Spearman
        r = pd.Series(resid, index=f.index)
        y = pd.Series(y_vec, index=f.index)
        try:
            return float(self.utils_metrics.calculate_ic(r, y, method='spearman'))
        except Exception:
            return 0.0
