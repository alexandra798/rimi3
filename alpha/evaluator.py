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

    def __init__(self, cache_size=1000, enable_precompute=True):
        # Add parameter for cache size
        self.rpn_evaluator = RPNEvaluator
        self.operators = Operators

        # Use OrderedDict to implement LRU cache with limited size
        self.cache_size = cache_size
        self._result_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.enable_precompute = enable_precompute
        self.precomputed_features = {}

    def _manage_cache(self):
        """Maintain cache size within the specified limit."""
        while len(self._result_cache) > self.cache_size:
            # Remove the oldest entry (FIFO)
            self._result_cache.popitem(last=False)

    def clear_cache(self):
        """Clear the cache and reset statistics (for external use)."""
        self._result_cache.clear()
        logger.info(f"Cache cleared. Hits: {self._cache_hits}, Misses: {self._cache_misses}")
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, formula: str, data: Union[pd.DataFrame, Dict],
                 allow_partial: bool = False) -> pd.Series:
        """
        Evaluate an RPN formula given the input data.

        Args:
            formula: RPN formula string.
            data: Input data (DataFrame or dict).
            allow_partial: Whether to allow partial expression evaluation.
        Returns:
            A pandas Series of evaluation results; returns NaN Series if evaluation fails.
        """
        cache_key = self._generate_cache_key(formula, data, allow_partial)

        # Check cache first
        if cache_key in self._result_cache:
            # Move to end (recently used)
            self._result_cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug(f"Cache hit for formula: {formula[:50]}...")
            return self._result_cache[cache_key].copy()

        self._cache_misses += 1

        # Perform evaluation
        try:
            result = self._evaluate_impl(formula, data, allow_partial)
            # Cache the result if valid
            if result is not None:
                self._result_cache[cache_key] = result.copy()
                self._manage_cache()
            return result

        except Exception as e:
            logger.error(f"Error evaluating formula '{formula[:50]}...': {str(e)}")
            return self._create_nan_series(data)

    def _evaluate_impl(self, formula: str, data: Union[pd.DataFrame, Dict],
                       allow_partial: bool) -> pd.Series:
        # Parse the token sequence
        token_sequence = self._parse_tokens(formula)
        if not token_sequence:
            logger.warning(f"Failed to parse formula: {formula[:50]}...")
            return self._create_nan_series(data)

        # Validate token sequence completeness
        if not allow_partial and not self._is_complete_expression(token_sequence):
            return self._create_nan_series(data)

        # Prepare input data
        data_dict = self._prepare_data(data)
        if data_dict is None:
            return self._create_nan_series(data)

        # Evaluate RPN expression
        try:
            result = self.rpn_evaluator.evaluate(
                token_sequence,
                data_dict,
                allow_partial=allow_partial
            )

            # Convert result to pandas Series
            series_result = self._convert_to_series(result, data)

            # === Enhancement: Handle global NaN and constant values ===
            if series_result is not None and not series_result.isna().all():
                # 1. Replace inf values with NaN
                series_result = series_result.replace([np.inf, -np.inf], np.nan)

                # 2. Detect near-constant results (very small standard deviation)
                valid_values = series_result.dropna()
                if len(valid_values) > 10:
                    std = valid_values.std()
                    if std < 1e-6:  # Constant detection threshold
                        logger.debug(f"Formula produces constant values (std={std:.8f}): {formula[:50]}...")
                        return self._create_nan_series(data)

                # 3. Smart NaN filling:
                # Forward-fill (using historical values)
                series_result = series_result.ffill()
                # Backward-fill (handling leading NaNs)
                series_result = series_result.bfill()
                # Fill remaining NaNs with 0 (if entire column is NaN)
                series_result = series_result.fillna(0)

            return series_result

        except Exception as e:
            logger.error(f"RPN evaluation failed: {type(e).__name__}: {str(e)}")
            logger.debug(f"Token sequence: {' '.join([t.name for t in token_sequence])}")
            return self._create_nan_series(data)

    def _parse_tokens(self, formula: str) -> list:
        """Convert formula string into a list of token objects."""
        try:
            token_names = formula.strip().split()
            token_sequence = []
            # Automatically prepend BEG if missing
            if not token_names or token_names[0] != 'BEG':
                token_sequence.append(TOKEN_DEFINITIONS['BEG'])

            for name in token_names:
                if name in TOKEN_DEFINITIONS:
                    token_sequence.append(TOKEN_DEFINITIONS[name])
                else:
                    # Try parsing dynamic constants like const_3.14
                    if name.startswith('const_'):
                        try:
                            value = float(name[6:])  # Strip 'const_' prefix
                            from core.token_system import Token, TokenType
                            dynamic_token = Token(TokenType.OPERAND, name, value=value)
                            token_sequence.append(dynamic_token)
                        except ValueError:
                            logger.warning(f"Unknown token: {name}")
                            return []
                    else:
                        logger.warning(f"Unknown token: {name}")
                        return []

            # Automatically append END if missing and expression can terminate
            if token_sequence and token_sequence[-1].name != 'END':
                if RPNValidator.can_terminate(token_sequence):
                    token_sequence.append(TOKEN_DEFINITIONS['END'])
            return token_sequence

        except Exception as e:
            logger.error(f"Token parsing error: {e}")
            return []

    def _prepare_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """
        Prepare input data as a dictionary of Series objects.
        Keeps references to existing Series to avoid unnecessary copies.
        """
        try:
            prepared: Dict[str, pd.Series] = {}

            if isinstance(data, pd.DataFrame):
                # Directly reference columns instead of using to_dict('series')
                prepared = {col: data[col] for col in data.columns}
            elif isinstance(data, dict):
                # Preserve original dictionary logic
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
            # Precomputation now happens at data-loading stage
            return prepared

        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None

    def _convert_to_series(self, result: Any, original_data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """
        Convert evaluation result to a pandas Series with the correct index.
        """
        try:
            if isinstance(result, pd.Series):
                return result

            # Derive index from original data
            if isinstance(original_data, pd.DataFrame):
                index = original_data.index
            elif isinstance(original_data, dict):
                for value in original_data.values():
                    if isinstance(value, pd.Series):
                        index = value.index
                        break
                else:
                    index = None
            else:
                index = None

            # Convert based on result type
            if isinstance(result, np.ndarray):
                return pd.Series(result, index=index)
            elif isinstance(result, (int, float, np.number)):
                if index is not None:
                    return pd.Series(result, index=index)
                else:
                    return pd.Series([result])
            else:
                return pd.Series(result, index=index)

        except Exception as e:
            logger.error(f"Result conversion error: {e}")
            return self._create_nan_series(original_data)

    def _create_nan_series(self, data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """Create a NaN Series as a fallback return for failed evaluations."""
        if isinstance(data, pd.DataFrame):
            return pd.Series(np.nan, index=data.index)
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, pd.Series):
                    return pd.Series(np.nan, index=value.index)
                elif isinstance(value, np.ndarray):
                    return pd.Series(np.nan, index=range(len(value)))
        return pd.Series(np.nan)

    def _is_complete_expression(self, token_sequence: list) -> bool:
        """Check whether a token sequence forms a complete RPN expression."""
        if not token_sequence:
            return False

        if token_sequence[0].name != 'BEG':
            return False

        if len(token_sequence) <= 1 or token_sequence[-1].name != 'END':
            return False

        # Check stack balance (should result in exactly one value)
        stack_size = RPNValidator.calculate_stack_size(token_sequence)
        return stack_size == 1

    def _generate_cache_key(self, formula: str, data: Any, allow_partial: bool) -> str:
        """Generate a unique cache key based on formula and data identity (not content)."""
        data_id = None
        if hasattr(data, 'attrs') and isinstance(data.attrs, dict):
            data_id = data.attrs.get('data_id')
        if data_id is None:
            data_id = str(id(data))
        return f"{formula}_{data_id}_{allow_partial}"

    def evaluate_state(self, state, X_data) -> Optional[pd.Series]:
        """Evaluate a state object (used by MCTS or reinforcement components)."""
        try:
            rpn_string = ' '.join([t.name for t in state.token_sequence])
            result = self.evaluate(rpn_string, X_data, allow_partial=True)
            return result
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None
