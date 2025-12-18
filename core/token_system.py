"""core/token_system.py"""
from enum import Enum
import numpy as np


class DataType(Enum):
    PRICE   = "price"    # Price-related features (open/high/low/close/vwap)
    VOLUME  = "volume"   # Volume
    SCALAR  = "scalar"   # Constant value
    RANKED  = "ranked"   # Ranked outputs (from csrank or ts_rank)
    GENERIC = "generic"  # Other numerical values


class TokenType(Enum):
    SPECIAL = "special"   # Special tokens like BEG, END
    OPERAND = "operand"   # Operands (variables or constants)
    OPERATOR = "operator" # Operators (functions)


class Token:
    """Token structure representing an element in the RPN expression."""
    def __init__(self, token_type, name, value=None, arity=0, min_window=None, dtype=None):
        self.type = token_type
        self.name = name
        self.value = value
        self.arity = arity
        self.min_window = min_window  # Minimum window size (used for time-series operators)
        self.dtype = dtype


# ==================== Token Definition Dictionary ====================
TOKEN_DEFINITIONS = {
    # ---- Special Tokens ----
    'BEG': Token(TokenType.SPECIAL, 'BEG'),
    'END': Token(TokenType.SPECIAL, 'END'),

    # ---- Operands - Stock Features ----
    'open': Token(TokenType.OPERAND, 'open', dtype=DataType.PRICE),
    'high': Token(TokenType.OPERAND, 'high', dtype=DataType.PRICE),
    'low': Token(TokenType.OPERAND, 'low', dtype=DataType.PRICE),
    'close': Token(TokenType.OPERAND, 'close', dtype=DataType.PRICE),
    'volume': Token(TokenType.OPERAND, 'volume', dtype=DataType.VOLUME),
    'vwap': Token(TokenType.OPERAND, 'vwap', dtype=DataType.PRICE),

    # ---- Operands - Time Windows (used in time-series operations) ----
    'delta_3': Token(TokenType.OPERAND, 'delta_3', value=3, dtype=DataType.SCALAR),
    'delta_5': Token(TokenType.OPERAND, 'delta_5', value=5, dtype=DataType.SCALAR),
    'delta_10': Token(TokenType.OPERAND, 'delta_10', value=10, dtype=DataType.SCALAR),
    'delta_20': Token(TokenType.OPERAND, 'delta_20', value=20, dtype=DataType.SCALAR),
    'delta_30': Token(TokenType.OPERAND, 'delta_30', value=30, dtype=DataType.SCALAR),
    'delta_40': Token(TokenType.OPERAND, 'delta_40', value=40, dtype=DataType.SCALAR),
    'delta_50': Token(TokenType.OPERAND, 'delta_50', value=50, dtype=DataType.SCALAR),
    'delta_60': Token(TokenType.OPERAND, 'delta_60', value=60, dtype=DataType.SCALAR),

    # ---- Operands - Constants (used for numeric formulas) ----
    'const_-30': Token(TokenType.OPERAND, 'const_-30', value=-30.0, dtype=DataType.SCALAR),
    'const_-10': Token(TokenType.OPERAND, 'const_-10', value=-10.0, dtype=DataType.SCALAR),
    'const_-5': Token(TokenType.OPERAND, 'const_-5', value=-5.0, dtype=DataType.SCALAR),
    'const_-2': Token(TokenType.OPERAND, 'const_-2', value=-2.0, dtype=DataType.SCALAR),
    'const_-1': Token(TokenType.OPERAND, 'const_-1', value=-1.0, dtype=DataType.SCALAR),
    'const_-0.5': Token(TokenType.OPERAND, 'const_-0.5', value=-0.5, dtype=DataType.SCALAR),
    'const_-0.01': Token(TokenType.OPERAND, 'const_-0.01', value=-0.01, dtype=DataType.SCALAR),
    'const_0.5': Token(TokenType.OPERAND, 'const_0.5', value=0.5, dtype=DataType.SCALAR),
    'const_1': Token(TokenType.OPERAND, 'const_1', value=1.0, dtype=DataType.SCALAR),
    'const_2': Token(TokenType.OPERAND, 'const_2', value=2.0, dtype=DataType.SCALAR),
    'const_5': Token(TokenType.OPERAND, 'const_5', value=5.0, dtype=DataType.SCALAR),
    'const_10': Token(TokenType.OPERAND, 'const_10', value=10.0, dtype=DataType.SCALAR),
    'const_30': Token(TokenType.OPERAND, 'const_30', value=30.0, dtype=DataType.SCALAR),

    # ---- Unary Operators ----
    'sign': Token(TokenType.OPERATOR, 'sign', arity=1),   # Sign function
    'abs': Token(TokenType.OPERATOR, 'abs', arity=1),     # Absolute value
    'log': Token(TokenType.OPERATOR, 'log', arity=1),     # Natural logarithm
    'csrank': Token(TokenType.OPERATOR, 'csrank', arity=1),  # Cross-sectional rank

    # ---- Binary Operators ----
    'add': Token(TokenType.OPERATOR, 'add', arity=2),   # Addition
    'sub': Token(TokenType.OPERATOR, 'sub', arity=2),   # Subtraction
    'mul': Token(TokenType.OPERATOR, 'mul', arity=2),   # Multiplication
    'div': Token(TokenType.OPERATOR, 'div', arity=2),   # Division
    'greater': Token(TokenType.OPERATOR, 'greater', arity=2),  # Greater comparison
    'less': Token(TokenType.OPERATOR, 'less', arity=2),        # Less comparison

    # ---- Time-Series Operators ----
    # Each has a minimum required window length
    'ts_ref': Token(TokenType.OPERATOR, 'ts_ref', arity=1, min_window=1),
    'ts_rank': Token(TokenType.OPERATOR, 'ts_rank', arity=1, min_window=2),
    'ts_mean': Token(TokenType.OPERATOR, 'ts_mean', arity=1, min_window=1),
    'ts_med': Token(TokenType.OPERATOR, 'ts_med', arity=1, min_window=1),
    'ts_sum': Token(TokenType.OPERATOR, 'ts_sum', arity=1, min_window=1),
    'ts_std': Token(TokenType.OPERATOR, 'ts_std', arity=1, min_window=3),  # Needs ≥3 points
    'ts_var': Token(TokenType.OPERATOR, 'ts_var', arity=1, min_window=3),  # Needs ≥3 points
    'ts_max': Token(TokenType.OPERATOR, 'ts_max', arity=1, min_window=1),
    'ts_min': Token(TokenType.OPERATOR, 'ts_min', arity=1, min_window=1),
    'ts_skew': Token(TokenType.OPERATOR, 'ts_skew', arity=1, min_window=5),  # Needs ≥5 points
    'ts_kurt': Token(TokenType.OPERATOR, 'ts_kurt', arity=1, min_window=5),  # Needs ≥5 points
    'ts_wma': Token(TokenType.OPERATOR, 'ts_wma', arity=1, min_window=2),
    'ts_ema': Token(TokenType.OPERATOR, 'ts_ema', arity=1, min_window=2),

    # ---- Correlation Operators ----
    'corr': Token(TokenType.OPERATOR, 'corr', arity=2, min_window=3),
    'cov': Token(TokenType.OPERATOR, 'cov', arity=2, min_window=3),
}

# Create Token index mappings for encoder/decoder
TOKEN_TO_INDEX = {name: idx for idx, name in enumerate(TOKEN_DEFINITIONS.keys())}
INDEX_TO_TOKEN = {idx: name for name, idx in TOKEN_TO_INDEX.items()}
TOTAL_TOKENS = len(TOKEN_DEFINITIONS)


class RPNValidator:
    """Validator for checking correctness and completeness of RPN expressions."""

    @staticmethod
    def _infer_stack_dtypes(token_sequence):
        """
        Infer data types through simulated stack operations.
        This does not compute values, only propagates DataType inference.

        Notes:
        - delta_* tokens are skipped (they are parameters).
        - corr/cov are treated as binary operators.
        """
        from core.token_system import TokenType, DataType
        stack = []
        i = 1  # Skip BEG
        while i < len(token_sequence):
            tk = token_sequence[i]
            if tk.name == 'END':
                break
            if tk.type == TokenType.OPERAND:
                if not tk.name.startswith('delta_'):
                    stack.append(tk.dtype or DataType.GENERIC)
                i += 1
                continue

            # Process operator tokens
            needs_delta = (tk.name in ['ts_ref', 'ts_rank'] or tk.name.startswith('ts_') or tk.name in ['corr', 'cov'])
            eff_arity = 2 if tk.name in ['corr', 'cov'] else tk.arity
            if needs_delta and i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                i += 1  # Skip delta parameter

            if len(stack) < eff_arity:
                return stack  # Stack underflow — incomplete expression

            # Pop operands
            args = [stack.pop() for _ in range(eff_arity)][::-1]

            # Infer output dtype
            if tk.name in ['csrank', 'ts_rank']:
                stack.append(DataType.RANKED)
            elif tk.name in ['corr', 'cov']:
                stack.append(DataType.SCALAR)
            else:
                stack.append(DataType.GENERIC)
            i += 1
        return stack

    @staticmethod
    def _dtype_check(op_name, stack):
        """
        Basic pruning rule (for early rejection):
        - corr/cov cannot take scalar (constant) operands.
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
        """
        Check whether a token sequence forms a partially valid RPN expression.
        Allows early-stage formulas to be evaluated safely during generation.
        """
        if not token_sequence or token_sequence[0].name != 'BEG':
            return False

        stack_size = 0
        used_operator = False
        i = 1

        while i < len(token_sequence):
            tk = token_sequence[i]
            if tk.name == 'END':
                # Consider valid if at least one operator has been used and stack==1
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

        # Partially valid: at least one operator used and stack≥1
        return used_operator and (stack_size >= 1)

    @staticmethod
    def get_valid_next_tokens(token_sequence):
        """
        Return a list of valid next tokens based on current RPN state.
        Used during alpha formula generation and MCTS exploration.
        """
        if not token_sequence:
            return ['BEG']

        # Limit maximum toke
