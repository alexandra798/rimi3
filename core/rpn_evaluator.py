"""RPN evaluator - dispatches to the unified Operators class."""
import numpy as np
import pandas as pd
import logging
from core.token_system import TokenType, TOKEN_DEFINITIONS
from core.operators import Operators

logger = logging.getLogger(__name__)


class RPNEvaluator:
    """Evaluate RPN expressions into Series/arrays using the Operators registry."""

    @staticmethod
    def evaluate(token_sequence, data_dict, allow_partial=True):
        """
        Evaluate an RPN expression, with optional partial-expression support.

        Args:
            token_sequence: List of Token objects representing the expression.
            data_dict: Dict[str, Series/ndarray], mapping variable names to data columns.
            allow_partial: If True, return the top of the stack when expression is partial.

        Returns:
            pandas.Series, numpy.ndarray, or scalar expanded to Series/array length.
            Returns None on structural errors (e.g., insufficient operands).
        """
        stack = []

        # Determine data length and index (for broadcasting scalars into Series)
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

        i = 1  # Skip BEG
        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # Operand handling
                if token.name in data_dict:
                    stack.append(data_dict[token.name])
                elif token.name.startswith('const_'):
                    const_value = float(token.name.split('_')[1])
                    # Always create a Series if shape info is available
                    if data_index is not None:
                        stack.append(pd.Series(const_value, index=data_index))
                    elif data_length:
                        # Create Series even without explicit index
                        stack.append(pd.Series([const_value] * data_length))
                    else:
                        stack.append(const_value)
                elif token.name.startswith('delta_'):
                    # delta_* should not appear alone; it will be consumed by the next operator
                    pass

            elif token.type == TokenType.OPERATOR:
                # ================== Time-series operators ==================
                if token.name.startswith('ts_'):
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    data_operand = stack.pop()
                    window = 5  # Default window

                    # 1) Parse following delta_* as window parameter (if any)
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        try:
                            window = int(delta_token.name.split('_')[1])
                        except Exception:
                            window = 5
                        i += 2  # Skip past delta_*
                    else:
                        i += 1  # No delta_*, advance by one

                    # 2) Fast path: use precomputed columns (if Series marked as base raw)
                    try:
                        if isinstance(data_operand, pd.Series):
                            base_name = data_operand.attrs.get('orig_name', None)
                            is_base = bool(data_operand.attrs.get('is_base_raw', False))

                            if is_base and base_name:
                                fast_key = f"{token.name}_{base_name}_{int(window)}"

                                # Upstream convention: a prepared `data_dict` may contain precomputed columns
                                data_dict = locals().get('data_dict', getattr(self, 'data_dict', None))

                                if isinstance(data_dict, dict) and fast_key in data_dict:
                                    stack.append(data_dict[fast_key])
                                    continue  # Fast-path hit: reuse precomputed column
                    except Exception:
                        # Any failure on the fast path silently falls back
                        pass

                    # 3) Fallback: call Operators.ts_* at runtime
                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(data_operand, window)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown time series operator: {token.name}")
                        return None

                    continue

                # ================== Correlation-like operators ==================
                elif token.name in ('corr', 'cov'):
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    y = stack.pop()
                    x = stack.pop()

                    window = 5  # Default window
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        window = int(delta_token.name.split('_')[1])
                        i += 2  # Skip delta_* after operator
                    else:
                        i += 1

                    # Dispatch to Operators.corr or Operators.cov
                    op_method = getattr(Operators, token.name)
                    result = op_method(x, y, window)
                    stack.append(result)
                    continue

                # ================== Unary operators ==================
                elif token.arity == 1:
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand = stack.pop()

                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(operand, data_length, data_index)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown unary operator: {token.name}")
                        return None

                # ================== Binary operators ==================
                elif token.arity == 2:
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand2 = stack.pop()
                    operand1 = stack.pop()

                    op_method = getattr(Operators, token.name, None)
                    if op_method:
                        result = op_method(operand1, operand2, data_length, data_index)
                        stack.append(result)
                    else:
                        logger.error(f"Unknown binary operator: {token.name}")
                        return None

                # ================== Ternary operators (not used currently) ==================
                elif token.arity == 3:
                    if len(stack) < 3:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand3 = stack.pop()
                    operand2 = stack.pop()
                    operand1 = stack.pop()

                    # There are no true ternary operators at present (corr/cov handled above).
                    logger.error(f"Unexpected ternary operator: {token.name}")
                    return None

            i += 1

        # ---------------------- Return handling ----------------------
        if len(stack) == 0:
            logger.error("Empty stack after evaluation")
            return None
        elif len(stack) == 1:
            # Normal case for a complete expression
            result = stack[0]
            if isinstance(result, (int, float, np.number)):
                if data_length and data_index is not None:
                    return pd.Series(result, index=data_index)
                elif data_length:
                    return np.full(data_length, result)
            return result

        else:
            # Partial-expression case
            if allow_partial:
                result = stack[-1]
                # Broadcast scalar to Series/array length if possible
                if isinstance(result, (int, float, np.number)):
                    if data_length and data_index is not None:
                        return pd.Series(result, index=data_index)
                    elif data_length:
                        return np.full(data_length, result)
                return result
            else:
                # Not allowed to return partial results
                logger.error(f"Stack has {len(stack)} elements after evaluation, expected 1")
                logger.error(f"Stack content: {[type(x) for x in stack]}")
                logger.error(f"RPN expression: {' '.join([t.name for t in token_sequence])}")
                return None
