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
        准备数据为字典格式 - 保持Series引用不变，避免重复创建
        """
        try:
            prepared: Dict[str, pd.Series] = {}

            if isinstance(data, pd.DataFrame):
                # 关键改动：不用to_dict('series')，直接引用列
                prepared = {col: data[col] for col in data.columns}
            elif isinstance(data, dict):
                # Dict类型保持原逻辑
                ref_index = None
                for value in data.values():
                    if isinstance(value, pd.Series):
                        ref_index = value.index
                        break

                for key, value in data.items():
                    if isinstance(value, pd.Series):
                        prepared[key] = value  # 直接引用，不复制
                    elif isinstance(value, np.ndarray):
                        prepared[key] = pd.Series(value, index=ref_index)
                    else:
                        prepared[key] = pd.Series(value, index=ref_index)
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None
            # 移除预计算逻辑，改到数据加载时一次性完成
            # （预计算移到main.py的数据准备阶段）
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
        """生成缓存键 - 使用数据标识而非内容哈希"""
        # 优先使用data的attrs标识
        data_id = None
        if hasattr(data, 'attrs') and isinstance(data.attrs, dict):
            data_id = data.attrs.get('data_id')

        # 如果没有attrs标识，使用对象id
        if data_id is None:
            data_id = str(id(data))

        return f"{formula}_{data_id}_{allow_partial}"


    def evaluate_state(self, state, X_data) -> Optional[pd.Series]:
        try:
            rpn_string = ' '.join([t.name for t in state.token_sequence])
            result = self.evaluate(rpn_string, X_data, allow_partial=True)
            return result  # 直接返回 Series
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None




