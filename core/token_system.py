"""core/token_system.py"""
from enum import Enum
import numpy as np

class DataType(Enum):
    PRICE   = "price"    # open/high/low/close/vwap
    VOLUME  = "volume"   # volume
    SCALAR  = "scalar"   # 常数
    RANKED  = "ranked"   # csrank 或 ts_rank 输出
    GENERIC = "generic"  # 其他数值

class TokenType(Enum):
    SPECIAL = "special"  # BEG, END
    OPERAND = "operand"  # 操作数
    OPERATOR = "operator"  # 操作符


class Token:
    def __init__(self, token_type, name, value=None, arity=0, min_window=None, dtype=None):
        self.type = token_type
        self.name = name
        self.value = value
        self.arity = arity
        self.min_window = min_window  # 新增：最小窗口要求
        self.dtype = dtype


# Token定义字典
TOKEN_DEFINITIONS = {
    # 特殊标记
    'BEG': Token(TokenType.SPECIAL, 'BEG'),
    'END': Token(TokenType.SPECIAL, 'END'),

    # 操作数 - 股票特征
    'open': Token(TokenType.OPERAND, 'open', dtype=DataType.PRICE),
    'high': Token(TokenType.OPERAND, 'high', dtype=DataType.PRICE),
    'low': Token(TokenType.OPERAND, 'low', dtype=DataType.PRICE),
    'close': Token(TokenType.OPERAND, 'close', dtype=DataType.PRICE),
    'volume': Token(TokenType.OPERAND, 'volume', dtype=DataType.VOLUME),
    'vwap': Token(TokenType.OPERAND, 'vwap', dtype=DataType.PRICE),

    # 操作数 - 时间窗口（7个，用于时序操作）
    'delta_3': Token(TokenType.OPERAND, 'delta_3', value=3,  dtype=DataType.SCALAR),
    'delta_5': Token(TokenType.OPERAND, 'delta_5', value=5,  dtype=DataType.SCALAR),
    'delta_10': Token(TokenType.OPERAND, 'delta_10', value=10,  dtype=DataType.SCALAR),
    'delta_20': Token(TokenType.OPERAND, 'delta_20', value=20,  dtype=DataType.SCALAR),
    'delta_30': Token(TokenType.OPERAND, 'delta_30', value=30,  dtype=DataType.SCALAR),
    'delta_40': Token(TokenType.OPERAND, 'delta_40', value=40,  dtype=DataType.SCALAR),
    'delta_50': Token(TokenType.OPERAND, 'delta_50', value=50,  dtype=DataType.SCALAR),
    'delta_60': Token(TokenType.OPERAND, 'delta_60', value=60,  dtype=DataType.SCALAR),

    # 操作数 - 常数（13个，根据论文Table 3）
    'const_-30': Token(TokenType.OPERAND, 'const_-30', value=-30.0,dtype=DataType.SCALAR),
    'const_-10': Token(TokenType.OPERAND, 'const_-10', value=-10.0,dtype=DataType.SCALAR),
    'const_-5': Token(TokenType.OPERAND, 'const_-5', value=-5.0,dtype=DataType.SCALAR),
    'const_-2': Token(TokenType.OPERAND, 'const_-2', value=-2.0,dtype=DataType.SCALAR),
    'const_-1': Token(TokenType.OPERAND, 'const_-1', value=-1.0,dtype=DataType.SCALAR),
    'const_-0.5': Token(TokenType.OPERAND, 'const_-0.5', value=-0.5,dtype=DataType.SCALAR),
    'const_-0.01': Token(TokenType.OPERAND, 'const_-0.01', value=-0.01,dtype=DataType.SCALAR),
    'const_0.5': Token(TokenType.OPERAND, 'const_0.5', value=0.5,dtype=DataType.SCALAR),
    'const_1': Token(TokenType.OPERAND, 'const_1', value=1.0,dtype=DataType.SCALAR),
    'const_2': Token(TokenType.OPERAND, 'const_2', value=2.0,dtype=DataType.SCALAR),
    'const_5': Token(TokenType.OPERAND, 'const_5', value=5.0,dtype=DataType.SCALAR),
    'const_10': Token(TokenType.OPERAND, 'const_10', value=10.0,dtype=DataType.SCALAR),
    'const_30': Token(TokenType.OPERAND, 'const_30', value=30.0,dtype=DataType.SCALAR),

    # 一元操作符 - 横截面（4个）
    'sign': Token(TokenType.OPERATOR, 'sign', arity=1),
    'abs': Token(TokenType.OPERATOR, 'abs', arity=1),
    'log': Token(TokenType.OPERATOR, 'log', arity=1),
    'csrank': Token(TokenType.OPERATOR, 'csrank', arity=1),  # 横截面排名

    # 二元操作符（需要2个操作数）
    'add': Token(TokenType.OPERATOR, 'add', arity=2),  # +
    'sub': Token(TokenType.OPERATOR, 'sub', arity=2),  # -
    'mul': Token(TokenType.OPERATOR, 'mul', arity=2),  # *
    'div': Token(TokenType.OPERATOR, 'div', arity=2),  # /
    'greater': Token(TokenType.OPERATOR, 'greater', arity=2),
    'less': Token(TokenType.OPERATOR, 'less', arity=2),

    # 时序操作符 - 添加最小窗口要求
    'ts_ref': Token(TokenType.OPERATOR, 'ts_ref', arity=1, min_window=1),
    'ts_rank': Token(TokenType.OPERATOR, 'ts_rank', arity=1, min_window=2),
    'ts_mean': Token(TokenType.OPERATOR, 'ts_mean', arity=1, min_window=1),
    'ts_med': Token(TokenType.OPERATOR, 'ts_med', arity=1, min_window=1),
    'ts_sum': Token(TokenType.OPERATOR, 'ts_sum', arity=1, min_window=1),
    'ts_std': Token(TokenType.OPERATOR, 'ts_std', arity=1, min_window=3),  # 需要3个点！
    'ts_var': Token(TokenType.OPERATOR, 'ts_var', arity=1, min_window=3),  # 需要3个点！
    'ts_max': Token(TokenType.OPERATOR, 'ts_max', arity=1, min_window=1),
    'ts_min': Token(TokenType.OPERATOR, 'ts_min', arity=1, min_window=1),
    'ts_skew': Token(TokenType.OPERATOR, 'ts_skew', arity=1, min_window=5),  # 需要5个点！
    'ts_kurt': Token(TokenType.OPERATOR, 'ts_kurt', arity=1, min_window=5),  # 需要5个点！
    'ts_wma': Token(TokenType.OPERATOR, 'ts_wma', arity=1, min_window=2),
    'ts_ema': Token(TokenType.OPERATOR, 'ts_ema', arity=1, min_window=2),

    # 相关性操作符
    'corr': Token(TokenType.OPERATOR, 'corr', arity=2, min_window=3),
    'cov': Token(TokenType.OPERATOR, 'cov', arity=2, min_window=3),
}

# 创建Token索引映射
TOKEN_TO_INDEX = {name: idx for idx, name in enumerate(TOKEN_DEFINITIONS.keys())}
INDEX_TO_TOKEN = {idx: name for name, idx in TOKEN_TO_INDEX.items()}
TOTAL_TOKENS = len(TOKEN_DEFINITIONS)


class RPNValidator:
    @staticmethod
    def _infer_stack_dtypes(token_sequence):
        """
        模拟栈，但只推断 dtype（不关心数值）。
        delta_* 作为参数跳过，corr/cov 视为二元操作。
        """
        from core.token_system import TokenType, DataType
        stack = []
        i = 1  # 跳过 BEG
        while i < len(token_sequence):
            tk = token_sequence[i]
            if tk.name == 'END':
                break
            if tk.type == TokenType.OPERAND:
                if not tk.name.startswith('delta_'):
                    stack.append(tk.dtype or DataType.GENERIC)
                i += 1
                continue

            # 操作符
            needs_delta = (tk.name in ['ts_ref', 'ts_rank'] or tk.name.startswith('ts_') or tk.name in ['corr', 'cov'])
            eff_arity = 2 if tk.name in ['corr', 'cov'] else tk.arity
            if needs_delta and i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                i += 1  # 跳过 delta 参数

            if len(stack) < eff_arity:
                return stack  # 不足，直接返回当前栈

            # 出栈
            args = [stack.pop() for _ in range(eff_arity)][::-1]

            # 入栈（推断输出类型的简化规则）
            if tk.name in ['csrank', 'ts_rank']:
                stack.append(DataType.RANKED)
            elif tk.name in ['corr', 'cov']:
                stack.append(DataType.SCALAR)
            else:
                # 其他一概视为 GENERIC（简化）
                stack.append(DataType.GENERIC)
            i += 1
        return stack

    @staticmethod
    def _dtype_check(op_name, stack):
        """
        早期剪枝的最小规则：
        - corr/cov：禁止任一参数是 SCALAR（const），避免 corr(x, const)
        后续可按需扩展更多规则
        """
        from core.token_system import DataType
        if op_name in ('corr', 'cov'):
            if len(stack) < 2:
                return False
            a, b = stack[-2], stack[-1]
            if DataType.SCALAR in (a, b):
                return False
        return True

    @staticmethod
    def is_valid_partial_expression(token_sequence):
        if not token_sequence or token_sequence[0].name != 'BEG':
            return False

        stack_size = 0
        used_operator = False
        i = 1

        while i < len(token_sequence):
            tk = token_sequence[i]
            if tk.name == 'END':
                # 只有出现过至少一个运算符、且栈==1 才视作"已形成有效子式"
                return used_operator and (stack_size == 1)

            if tk.type == TokenType.OPERAND:
                if not tk.name.startswith('delta_'):
                    stack_size += 1
            elif tk.type == TokenType.OPERATOR:
                used_operator = True
                needs_delta = (tk.name in ['ts_ref', 'ts_rank'] or tk.name.startswith('ts_')
                               or tk.name in ['corr', 'cov'])
                if needs_delta:
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        i += 1
                    eff_arity = 2 if tk.name in ['corr', 'cov'] else tk.arity
                    if stack_size < eff_arity:
                        return False
                    stack_size = stack_size - eff_arity + 1
                else:
                    if stack_size < tk.arity:
                        return False
                    stack_size = stack_size - tk.arity + 1
            i += 1

        # 允许作为"部分可评估"的条件：至少用过一个运算符，且栈>=1
        return used_operator and (stack_size >= 1)

    @staticmethod
    def get_valid_next_tokens(token_sequence):
        """返回当前状态下所有合法的下一个Token"""
        if not token_sequence:
            return ['BEG']

        if len(token_sequence) >= 30:
            if RPNValidator.can_terminate(token_sequence):
                return ['END']
            else:
                return []

        last_token = token_sequence[-1] if token_sequence else None

        # 时序操作符特殊处理
        time_ops = ['ts_ref', 'ts_rank', 'ts_mean', 'ts_med', 'ts_sum', 'ts_std',
                    'ts_var', 'ts_max', 'ts_min', 'ts_skew', 'ts_kurt',
                    'ts_wma', 'ts_ema', 'corr', 'cov']

        if last_token and last_token.name in time_ops:
            valid_deltas = []
            min_window = TOKEN_DEFINITIONS[last_token.name].min_window
            for delta_name in ['delta_3', 'delta_5', 'delta_10', 'delta_20',
                               'delta_30', 'delta_40', 'delta_50', 'delta_60']:
                delta_value = TOKEN_DEFINITIONS[delta_name].value
                if delta_value >= min_window:
                    valid_deltas.append(delta_name)
            return valid_deltas

        # 计算当前栈大小
        stack_size = RPNValidator.calculate_stack_size(token_sequence)
        valid_tokens = []

        # 操作数
        if stack_size < 10:
            valid_tokens.extend(['open', 'high', 'low', 'close', 'volume', 'vwap'])
            valid_tokens.extend(['const_-30', 'const_-10', 'const_-5', 'const_-2',
                                 'const_-1', 'const_-0.5', 'const_-0.01', 'const_0.5',
                                 'const_1', 'const_2', 'const_5', 'const_10', 'const_30'])

        # 操作符
        for token_name, token in TOKEN_DEFINITIONS.items():
            if token.type == TokenType.OPERATOR:
                required = 2 if token_name in ('corr', 'cov') else token.arity
                if required <= stack_size:
                    valid_tokens.append(token_name)

        # 类型检查和约束过滤
        typed_stack = RPNValidator._infer_stack_dtypes(token_sequence)

        # 定义辅助函数
        def _is_ts_op(name):
            return (name in ['ts_ref', 'ts_rank'] or name.startswith('ts_') or name in ['corr', 'cov'])

        last_names = [t.name for t in token_sequence[-3:]] if token_sequence else []
        ts_count = sum(1 for t in token_sequence if _is_ts_op(t.name))

        # 过滤约束
        filtered = []
        for name in valid_tokens:
            # 禁止连续 ts_ref
            if name == 'ts_ref' and (last_names and last_names[-1] == 'ts_ref'):
                continue

            # 限制时序操作总数（最多 3 次）
            if _is_ts_op(name) and ts_count >= 3:
                continue

            # dtype 校验
            tk = TOKEN_DEFINITIONS[name]
            if tk.type == TokenType.OPERATOR and not RPNValidator._dtype_check(name, typed_stack):
                continue

            filtered.append(name)

        valid_tokens = filtered

        # END 仅对"已用过≥1个运算符且栈==1"开放
        used_operator = any(t.type == TokenType.OPERATOR for t in token_sequence)
        if used_operator and stack_size == 1:
            valid_tokens.append('END')

        return valid_tokens

    @staticmethod
    def calculate_stack_size(token_sequence):
        """计算当前栈中的元素数量"""
        stack_size = 0
        i = 1  # 跳过BEG

        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # delta不入栈
                if not token.name.startswith('delta_'):
                    stack_size += 1
            elif token.type == TokenType.OPERATOR:
                # 检查是否有delta参数
                time_ops = ['ts_ref', 'ts_rank'] + [f'ts_{op}' for op in
                                                    ['mean', 'med', 'sum', 'std', 'var', 'max', 'min', 'skew', 'kurt',
                                                     'wma', 'ema']]
                needs_delta = token.name in time_ops or token.name in ['corr', 'cov']
                if needs_delta:
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        i += 1
                    eff_arity = 2 if token.name in ['corr', 'cov'] else token.arity
                    stack_size = stack_size - eff_arity + 1

                else:
                    stack_size = stack_size - token.arity + 1

            i += 1

        return stack_size

    @staticmethod
    def can_terminate(token_sequence):
        """检查是否可以终止（严格版本）"""
        if not token_sequence or len(token_sequence) < 2:
            return False

        # 检查末尾是否为需要参数的操作符
        last_token = token_sequence[-1]
        time_ops_need_delta = [
            'ts_ref', 'ts_rank', 'ts_mean', 'ts_med', 'ts_sum',
            'ts_std', 'ts_var', 'ts_max', 'ts_min', 'ts_skew',
            'ts_kurt', 'ts_wma', 'ts_ema', 'corr', 'cov'
        ]

        if last_token.name in time_ops_need_delta:
            return False  # 缺少delta参数，不可终止

        # 检查栈平衡
        stack_size = RPNValidator.calculate_stack_size(token_sequence)
        return stack_size == 1


