import pandas as pd
import numpy as np
import logging
import signal
from functools import lru_cache
from collections import OrderedDict
from typing import Union, Dict, Optional, Any

from core import RPNEvaluator, RPNValidator, TOKEN_DEFINITIONS, Operators

logger = logging.getLogger(__name__)


class FormulaEvaluator:

    def __init__(self, cache_size=1000, enable_precompute=True):  # 添加缓存大小参数
        self.rpn_evaluator = RPNEvaluator
        self.operators = Operators
        # 使用有限大小的OrderedDict实现LRU缓存
        self.cache_size = cache_size
        self._result_cache = OrderedDict()  # 改为OrderedDict
        self._cache_hits = 0
        self._cache_misses = 0
        self.enable_precompute = enable_precompute
        self.precomputed_features = {}

    def _manage_cache(self):
        """管理缓存大小"""
        while len(self._result_cache) > self.cache_size:
            # 删除最旧的条目（FIFO）
            self._result_cache.popitem(last=False)

    def clear_cache(self):
        """清空缓存（供外部调用）"""
        self._result_cache.clear()
        logger.info(f"Cache cleared. Hits: {self._cache_hits}, Misses: {self._cache_misses}")
        self._cache_hits = 0
        self._cache_misses = 0

    def precompute_features(self, data):
        """预计算常用的时序特征"""
        if not self.enable_precompute:
            return

        base_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        windows = [3, 5, 10, 20, 30, 40, 50, 60]
        ops = ['ts_mean', 'ts_std', 'ts_wma', 'ts_ema', 'ts_max', 'ts_min']

        for col in base_cols:
            if col in data:
                for window in windows:
                    for op in ops:
                        feature_name = f"{op}_{col}_{window}"
                        if op == 'ts_mean':
                            self.precomputed_features[feature_name] = data[col].rolling(window).mean()
                        elif op == 'ts_std':
                            self.precomputed_features[feature_name] = data[col].rolling(window).std()
                        elif op == 'ts_max':
                            self.precomputed_features[feature_name] = data[col].rolling(window).max()
                        elif op == 'ts_min':
                            self.precomputed_features[feature_name] = data[col].rolling(window).min()
                        # ... 其他操作符

        logger.info(f"Precomputed {len(self.precomputed_features)} features")




    def evaluate(self, formula: str, data: Union[pd.DataFrame, Dict],
                 allow_partial: bool = False) -> pd.Series:
        """
        Args:
            formula: RPN公式字符串
            data: 数据（DataFrame或字典）
            allow_partial: 是否允许部分表达式
        Returns:
            评估结果的Series，失败时返回NaN Series
        """
        cache_key = self._generate_cache_key(formula, data, allow_partial)

        if cache_key in self._result_cache:
            # 移到末尾（最近使用）
            self._result_cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug(f"Cache hit for formula: {formula[:50]}...")
            return self._result_cache[cache_key].copy()

        self._cache_misses += 1

        # 执行评估
        try:
            result = self._evaluate_impl(formula, data, allow_partial)
            # 缓存结果
            if result is not None:
                self._result_cache[cache_key] = result.copy()
                self._manage_cache()  # 检查并清理过多的缓存
            return result


        except Exception as e:
            logger.error(f"Error evaluating formula '{formula[:50]}...': {str(e)}")
            return self._create_nan_series(data)

    def _evaluate_impl(self, formula: str, data: Union[pd.DataFrame, Dict],
                       allow_partial: bool) -> pd.Series:
        # 解析Token序列
        token_sequence = self._parse_tokens(formula)
        if not token_sequence:
            logger.warning(f"Failed to parse formula: {formula[:50]}...")
            return self._create_nan_series(data)

        # 验证Token序列
        if not allow_partial and not self._is_complete_expression(token_sequence):
            return self._create_nan_series(data)

        # 准备数据
        data_dict = self._prepare_data(data)
        if data_dict is None:
            return self._create_nan_series(data)

        # 评估RPN表达式
        try:
            result = self.rpn_evaluator.evaluate(
                token_sequence,
                data_dict,
                allow_partial=allow_partial
            )

            # 转换结果为Series
            series_result = self._convert_to_series(result, data)

            # ===== 新增：全局NaN和常数检测处理 =====
            if series_result is not None and not series_result.isna().all():
                # 1. 处理inf值
                series_result = series_result.replace([np.inf, -np.inf], np.nan)

                # 2. 检测常数（标准差极小）
                valid_values = series_result.dropna()
                if len(valid_values) > 10:
                    std = valid_values.std()
                    if std < 1e-6:  # 常数检测阈值
                        logger.debug(f"Formula produces constant values (std={std:.8f}): {formula[:50]}...")
                        return self._create_nan_series(data)  # 返回NaN表示无效

                # 3. 智能填充NaN
                # 先尝试前向填充（用历史值）
                series_result = series_result.ffill()
                # 再后向填充（处理开头的NaN）
                series_result = series_result.bfill()
                # 最后用0填充（如果整列都是NaN）
                series_result = series_result.fillna(0)

            return series_result

        except Exception as e:
            logger.error(f"RPN evaluation failed: {type(e).__name__}: {str(e)}")
            logger.debug(f"Token sequence: {' '.join([t.name for t in token_sequence])}")
            return self._create_nan_series(data)

    def _parse_tokens(self, formula: str) -> list:
        try:
            token_names = formula.strip().split()
            token_sequence = []
            # 自动添加 BEG 如果缺失
            if not token_names or token_names[0] != 'BEG':
                token_sequence.append(TOKEN_DEFINITIONS['BEG'])

            for name in token_names:
                if name in TOKEN_DEFINITIONS:
                    token_sequence.append(TOKEN_DEFINITIONS[name])
                else:
                    # 尝试解析动态常数（如const_3.14）
                    if name.startswith('const_'):
                        try:
                            value = float(name[6:])  # 去掉'const_'前缀
                            # 创建动态Token
                            from core.token_system import Token, TokenType
                            dynamic_token = Token(TokenType.OPERAND, name, value=value)
                            token_sequence.append(dynamic_token)
                        except ValueError:
                            logger.warning(f"Unknown token: {name}")
                            return []
                    else:
                        logger.warning(f"Unknown token: {name}")
                        return []

            # 自动添加 END 如果缺失且表达式可终止
            if token_sequence and token_sequence[-1].name != 'END':
                if RPNValidator.can_terminate(token_sequence):
                    token_sequence.append(TOKEN_DEFINITIONS['END'])
            return token_sequence

        except Exception as e:
            logger.error(f"Token parsing error: {e}")
            return []

    def _prepare_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """
        准备数据为字典格式 - 确保都是 Series，并为“RPN 快路径”做两件事：
        1) 给基础列 Series 标注 attrs：orig_name / is_base_raw
        2) 预计算常用窗口的 ts_* 特征，键名形如：ts_mean_close_5
        """
        try:
            # ------ 统一先转成 {col: Series} ------
            prepared: Dict[str, pd.Series] = {}

            if isinstance(data, pd.DataFrame):
                # DataFrame -> dict('series')
                prepared = data.to_dict('series')
            elif isinstance(data, dict):
                # 保留你原有的“参考索引 + 转 Series”的逻辑
                ref_index = None
                for value in data.values():
                    if isinstance(value, pd.Series):
                        ref_index = value.index
                        break

                for key, value in data.items():
                    if isinstance(value, pd.Series):
                        prepared[key] = value
                    elif isinstance(value, np.ndarray):
                        prepared[key] = pd.Series(value, index=ref_index)
                    else:
                        prepared[key] = pd.Series(value, index=ref_index)
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None

            # ------ 标注基础列属性（供快路径识别“原始列”）------
            base_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
            for k, s in list(prepared.items()):
                if isinstance(s, pd.Series):
                    # 标注“来源列名”
                    try:
                        s.attrs['orig_name'] = k
                        s.attrs['is_base_raw'] = (k in base_cols)
                    except Exception:
                        # 某些 Series 可能不支持 attrs，跳过即可
                        pass

            # ------ 预计算常用窗口（仅对基础列做，以控内存/时间）------
            # 说明：Operators.ts_* 内部已做 Series 级缓存（attrs['_op_cache']），
            # 这里的预计算会“热一层”，RPN 快路径可直接取。
            from core.operators import Operators  # 避免顶部循环依赖
            win_list = [3, 5, 10, 20, 30, 40, 50, 60]

            for col in base_cols:
                s = prepared.get(col, None)
                if not isinstance(s, pd.Series):
                    continue

                for w in win_list:
                    key_mean = f'ts_mean_{col}_{w}'
                    key_std = f'ts_std_{col}_{w}'
                    key_wma = f'ts_wma_{col}_{w}'
                    key_ema = f'ts_ema_{col}_{w}'

                    # 避免重复计算：如果键已存在就不覆盖
                    if key_mean not in prepared:
                        prepared[key_mean] = Operators.ts_mean(s, w)
                    if key_std not in prepared:
                        prepared[key_std] = Operators.ts_std(s, w)
                    if key_wma not in prepared:
                        prepared[key_wma] = Operators.ts_wma(s, w)
                    if key_ema not in prepared:
                        prepared[key_ema] = Operators.ts_ema(s, w)

            return prepared

        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None

    def _convert_to_series(self, result: Any, original_data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """
        将评估结果转换为Series
        """
        try:
            # 如果已经是Series，直接返回
            if isinstance(result, pd.Series):
                return result

            # 获取索引
            if isinstance(original_data, pd.DataFrame):
                index = original_data.index
            elif isinstance(original_data, dict):
                # 从字典中找一个Series获取索引
                for value in original_data.values():
                    if isinstance(value, pd.Series):
                        index = value.index
                        break
                else:
                    index = None
            else:
                index = None

            # 转换为Series
            if isinstance(result, np.ndarray):
                return pd.Series(result, index=index)
            elif isinstance(result, (int, float, np.number)):
                if index is not None:
                    return pd.Series(result, index=index)
                else:
                    return pd.Series([result])
            else:
                # 尝试直接转换
                return pd.Series(result, index=index)

        except Exception as e:
            logger.error(f"Result conversion error: {e}")
            return self._create_nan_series(original_data)

    def _create_nan_series(self, data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """创建NaN Series作为错误返回值"""
        if isinstance(data, pd.DataFrame):
            return pd.Series(np.nan, index=data.index)
        elif isinstance(data, dict):
            # 尝试从字典中找一个Series获取长度和索引
            for value in data.values():
                if isinstance(value, pd.Series):
                    return pd.Series(np.nan, index=value.index)
                elif isinstance(value, np.ndarray):
                    return pd.Series(np.nan, index=range(len(value)))
        return pd.Series(np.nan)

    def _is_complete_expression(self, token_sequence: list) -> bool:

        if not token_sequence:
            return False

        # 必须以BEG开始
        if token_sequence[0].name != 'BEG':
            return False

        # 完整表达式必须以END结束
        if len(token_sequence) <= 1 or token_sequence[-1].name != 'END':
            return False  # 没有END就不是完整表达式

        # 验证栈平衡
        stack_size = RPNValidator.calculate_stack_size(token_sequence)
        return stack_size == 1  # 完整表达式应该正好留下1个结果

    def _generate_cache_key(self, formula: str, data: Any, allow_partial: bool) -> str:
        """生成缓存键 - 使用数据内容哈希而非id"""

        import hashlib
        if isinstance(data, pd.DataFrame):
            # DataFrame: 使用值和索引的哈希
            data_hash = hashlib.md5(
                data.values.tobytes() +
                str(data.index.tolist()).encode()
            ).hexdigest()[:8]  # 只取前8位避免太长
        elif isinstance(data, dict):
            # Dict: 使用所有值的哈希
            combined = b''
            for key in sorted(data.keys()):  # 排序保证一致性
                value = data[key]
                if hasattr(value, 'values'):
                    combined += value.values.tobytes()
                elif isinstance(value, (list, np.ndarray)):
                    combined += np.array(value).tobytes()
                else:
                    combined += str(value).encode()
            data_hash = hashlib.md5(combined).hexdigest()[:8]
        else:
            # 其他类型：退回到id（向后兼容）
            data_hash = str(id(data))

        return f"{formula}_{data_hash}_{allow_partial}"


    def evaluate_state(self, state, X_data) -> Optional[pd.Series]:
        try:
            rpn_string = ' '.join([t.name for t in state.token_sequence])
            result = self.evaluate(rpn_string, X_data, allow_partial=True)
            return result  # 直接返回 Series
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None




