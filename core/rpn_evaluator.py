"""RPN表达式求值器 - 调用统一的Operators类"""
import numpy as np
import pandas as pd
import logging
from core.token_system import TokenType, TOKEN_DEFINITIONS
from core.operators import Operators

logger = logging.getLogger(__name__)


class RPNEvaluator:
    """评估RPN表达式的值"""

    @staticmethod
    def evaluate(token_sequence, data_dict, allow_partial=True):
        """
        评估RPN表达式 支持部分表达式
        Args:
            token_sequence: Token序列
            data_dict: 数据字典
            allow_partial: 是否允许部分表达式（栈中有多个元素）
        Returns:
            评估结果（Series或数组）
        """
        stack = []

        # 获取数据长度和索引
        data_length = None
        data_index = None
        for key, value in data_dict.items():
            if isinstance(value, (pd.Series, np.ndarray)):
                if isinstance(value, pd.Series):
                    data_length = len(value)
                    data_index = value.index
                else:
                    data_length = len(value)
                break

        i = 1  # 跳过BEG
        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # 处理操作数
                if token.name in data_dict:
                    stack.append(data_dict[token.name])
                elif token.name.startswith('const_'):
                    const_value = float(token.name.split('_')[1])
                    # 始终创建 Series
                    if data_index is not None:
                        stack.append(pd.Series(const_value, index=data_index))
                    elif data_length:
                        # 即使没有索引，也创建 Series
                        stack.append(pd.Series([const_value] * data_length))
                    else:
                        stack.append(const_value)
                elif token.name.startswith('delta_'):
                    # delta不应该单独出现，跳过
                    pass

            elif token.type == TokenType.OPERATOR:
                # ================== 时序操作符处理 ==================
                if token.name.startswith('ts_'):
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    data_operand = stack.pop()
                    window = 5  # 默认窗口

                    # 检查下一个token是否是delta
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        window = int(delta_token.name.split('_')[1])
                        i += 2  # 跳到delta后面
                    else:
                        i += 1

                    # 调用Operators中对应的时序方法
                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(data_operand, window)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown time series operator: {token.name}")
                        return None
                    continue

                # ================== 相关性操作符处理 ==================
                elif token.name in ('corr', 'cov'):
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    y = stack.pop()
                    x = stack.pop()

                    window = 5  # 默认窗口
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        window = int(delta_token.name.split('_')[1])
                        i += 2  # 跳过操作符后的 delta_*
                    else:
                        i += 1

                    # 调用Operators中的corr或cov方法
                    op_method = getattr(Operators, token.name)
                    result = op_method(x, y, window)
                    stack.append(result)
                    continue

                # ================== 一元操作符处理 ==================
                elif token.arity == 1:
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand = stack.pop()

                    # 调用Operators中对应的一元方法
                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(operand, data_length, data_index)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown unary operator: {token.name}")
                        return None

                # ================== 二元操作符处理 ==================
                elif token.arity == 2:
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand2 = stack.pop()
                    operand1 = stack.pop()

                    # 调用Operators中对应的二元方法
                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(operand1, operand2, data_length, data_index)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown binary operator: {token.name}")
                        return None

                # ================== 三元操作符处理 ==================
                elif token.arity == 3:
                    if len(stack) < 3:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand3 = stack.pop()
                    operand2 = stack.pop()
                    operand1 = stack.pop()

                    # 注意：目前系统中没有真正的三元操作符
                    # corr和cov已经在上面处理了
                    logger.error(f"Unexpected ternary operator: {token.name}")
                    return None

            i += 1

        # 返回结果处理
        if len(stack) == 0:
            logger.error("Empty stack after evaluation")
            return None
        elif len(stack) == 1:
            # 完整表达式的正常情况
            result = stack[0]
            if isinstance(result, (int, float, np.number)):
                if data_length and data_index is not None:
                    return pd.Series(result, index=data_index)
                elif data_length:
                    return np.full(data_length, result)
            return result
        else:
            # 部分表达式的情况
            if allow_partial:
                # 对于部分表达式，返回栈顶元素或组合多个元素
                result = stack[-1]

                # 如果有多个元素，可以尝试组合它们
                if len(stack) > 1:
                    logger.debug(f"Partial expression with {len(stack)} stack elements")
                    # 尝试将所有栈元素平均（这是一种启发式方法）
                    try:
                        arrays = []
                        for elem in stack:
                            if isinstance(elem, pd.Series):
                                arrays.append(elem.values)
                            elif isinstance(elem, np.ndarray):
                                arrays.append(elem)
                            else:
                                # 标量，扩展为数组
                                if data_length:
                                    arrays.append(np.full(data_length, elem))
                                else:
                                    arrays.append(np.array([elem]))

                        # 计算平均值作为部分表达式的估值
                        result_array = np.mean(arrays, axis=0)

                        if data_index is not None:
                            return pd.Series(result_array, index=data_index)
                        else:
                            return result_array
                    except Exception as e:
                        logger.debug(f"Failed to combine stack elements: {e}")
                        # 失败时返回栈顶元素
                        pass

                # 返回结果
                if isinstance(result, (int, float, np.number)):
                    if data_length and data_index is not None:
                        return pd.Series(result, index=data_index)
                    elif data_length:
                        return np.full(data_length, result)
                return result
            else:
                # 不允许部分表达式时报错
                logger.error(f"Stack has {len(stack)} elements after evaluation, expected 1")
                logger.error(f"Stack content: {[type(x) for x in stack]}")
                logger.error(f"RPN expression: {' '.join([t.name for t in token_sequence])}")
                return None