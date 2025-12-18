# alpha/pool.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class AlphaPool:

    def __init__(self, pool_size=100, lambda_param=0.1, learning_rate=0.01,
                 min_std=1e-6, min_unique_ratio=0.01):
        self.pool_size = pool_size
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate

        # New: constant-value detection thresholds
        self.min_std = min_std          # Minimum standard deviation to treat as non-constant
        self.min_unique_ratio = min_unique_ratio  # Minimum ratio of unique values

        self.alphas = []
        self.model = None

        # New: bookkeeping for diagnostics
        self.rejected_constant_count = 0
        self.rejected_low_ic_count = 0

    def is_valid_alpha(self, alpha_values):
        """
        Check whether an alpha series is valid (i.e., not constant/degenerate).

        Args:
            alpha_values: Sequence/Series of alpha values.

        Returns:
            bool: True if the series is valid (non-constant), False otherwise.
        """
        if alpha_values is None:
            return False

        # Convert to numpy array when possible
        if hasattr(alpha_values, 'values'):
            values = alpha_values.values
        else:
            values = np.array(alpha_values)

        # Empty sequence check
        if len(values) == 0:
            return False

        # Ensure numeric dtype
        try:
            values = np.array(values, dtype=np.float64)
        except (ValueError, TypeError):
            return False

        # Drop NaNs
        valid_values = values[~np.isnan(values)]

        # Require a minimum number of valid observations
        if len(valid_values) < 10:
            return False

        # Check 1: standard deviation
        std = np.std(valid_values)
        if std < self.min_std:
            logger.debug(f"Alpha rejected: nearly constant (std={std:.8f})")
            self.rejected_constant_count += 1
            return False

        # Check 2: uniqueness ratio
        unique_count = len(np.unique(valid_values))
        unique_ratio = unique_count / len(valid_values)
        if unique_ratio < self.min_unique_ratio:
            logger.debug(f"Alpha rejected: too few unique values ({unique_ratio:.1%})")
            self.rejected_constant_count += 1
            return False

        # Check 3: coefficient of variation (relative variability)
        mean_val = np.mean(valid_values)
        if abs(mean_val) > 1e-10:  # avoid division by zero
            cv = std / abs(mean_val)
            if cv < 0.001:
                logger.debug(f"Alpha rejected: low coefficient of variation ({cv:.6f})")
                self.rejected_constant_count += 1
                return False

        return True

    def add_to_pool(self, alpha_info):
        """
        Add an alpha record to the pool if it passes validity checks.

        Args:
            alpha_info: dict with required 'formula', 'score'; optional 'values', 'ic'
        """
        # Skip duplicates by formula identity
        if any(a['formula'] == alpha_info['formula'] for a in self.alphas):
            return

        # Validate alpha series if present
        if 'values' in alpha_info:
            if not self.is_valid_alpha(alpha_info['values']):
                logger.info(f"Rejected constant alpha: {alpha_info['formula'][:50]}...")
                return

        # Enforce minimal IC threshold if provided
        if 'ic' in alpha_info and abs(alpha_info.get('ic', 0)) < 0.01:
            logger.debug(f"Rejected low IC alpha: IC={alpha_info['ic']:.4f}")
            self.rejected_low_ic_count += 1
            return

        # Ensure required fields exist
        if 'weight' not in alpha_info:
            alpha_info['weight'] = 1.0 / max(len(self.alphas), 1)
        if 'ic' not in alpha_info and 'score' in alpha_info:
            alpha_info['ic'] = alpha_info['score']

        self.alphas.append(alpha_info)
        logger.info(f"Added valid alpha to pool: {alpha_info['formula'][:50]}... (IC={alpha_info.get('ic', 0):.4f})")

        # If pool grows beyond capacity, remove the worst one
        if len(self.alphas) > self.pool_size:
            self._remove_worst_alpha()

    def update_pool(self, X_data, y_data, evaluate_formula):
        """
        Re-evaluate all formulas in the pool and optimize weights.

        This step:
          1) Recomputes 'values' for each formula in the current data context.
          2) Filters out invalid/degenerate alphas.
          3) Recomputes IC for valid alphas.
          4) Runs gradient descent to refresh weights.
          5) Sorts and truncates the pool.
        """
        import hashlib

        # Context identity (based on index) to avoid mixing cached values from different datasets
        if isinstance(X_data, pd.DataFrame):
            context_id = hashlib.md5(X_data.index.values.tobytes()).hexdigest()[:8]
        else:
            context_id = "unknown"

        logger.info(f"Updating alpha pool with context {context_id}, {len(self.alphas)} formulas...")

        alphas_to_remove = []

        for i, alpha in enumerate(self.alphas):
            # Always recompute if context changed or missing
            if 'context_id' not in alpha or alpha['context_id'] != context_id:
                try:
                    alpha['values'] = evaluate_formula.evaluate(
                        alpha['formula'],
                        X_data,
                        allow_partial=False
                    )
                    alpha['context_id'] = context_id

                    # Validate the computed series
                    if not self.is_valid_alpha(alpha['values']):
                        alphas_to_remove.append(i)
                        continue

                    # Compute IC against target
                    if alpha['values'] is not None and not alpha['values'].isna().all():
                        alpha['ic'] = self._calculate_ic(alpha['values'], y_data)

                        # Enforce minimal IC threshold
                        if abs(alpha['ic']) < 0.01:
                            alphas_to_remove.append(i)
                    else:
                        alphas_to_remove.append(i)

                except Exception as e:
                    logger.warning(f"Failed to evaluate formula: {alpha['formula'][:50]}...")
                    alphas_to_remove.append(i)

        # Remove invalid/low-quality alphas (order reversed for safe popping)
        for idx in reversed(alphas_to_remove):
            removed = self.alphas.pop(idx)
            logger.info(f"Removed invalid alpha: {removed['formula'][:50]}...")

        # Optimize weights using gradient descent (requires 'values')
        if len(self.alphas) > 0:
            self._optimize_weights_gradient_descent(X_data, y_data)

        # Optional: free memory by dropping 'values' after optimization
        for alpha in self.alphas:
            if 'values' in alpha:
                del alpha['values']

        # Sort by |IC * weight| to rank importance
        self.alphas.sort(key=lambda x: abs(x.get('ic', 0) * x.get('weight', 1)), reverse=True)

        # Keep pool within capacity
        if len(self.alphas) > self.pool_size:
            self.alphas = self.alphas[:self.pool_size]

    def maintain_pool(self, new_alpha, X_data, y_data):
        """
        Algorithm 1: Maintain the alpha pool (original implementation retained)
        Input : current set F, new alpha f_new, composite model c(·|F, ω)
        Output: optimal set F* and weights ω*
        """
        # Step 1: F ← F ∪ f_new
        self.alphas.append(new_alpha)

        # Step 2-4: optimize weights by gradient descent
        self._optimize_weights_gradient_descent(X_data, y_data)

        # Step 5-6: if pool exceeds capacity, remove the smallest-weight alpha, then re-optimize
        if len(self.alphas) > self.pool_size:
            self._remove_worst_alpha()
            self._optimize_weights_gradient_descent(X_data, y_data)

        return self.alphas

    def get_top_formulas(self, n=5):
        """
        Return the top-n formula strings by |IC * weight|.

        Args:
            n: number of formulas to return.

        Returns:
            List[str]: top-n formula strings.
        """
        sorted_alphas = sorted(
            self.alphas,
            key=lambda x: abs(x.get('ic', 0) * x.get('weight', 1)),
            reverse=True
        )

        top_formulas = []
        for alpha in sorted_alphas[:n]:
            formula = alpha['formula']
            ic = alpha.get('ic', 0)
            weight = alpha.get('weight', 1)
            logger.info(f"Top formula: {formula[:50]}... (IC={ic:.4f}, weight={weight:.4f})")
            top_formulas.append(formula)

        return top_formulas

    def _optimize_weights_gradient_descent(self, X_data, y_data, max_iters=100):
        """Optimize weights via gradient descent (core of the referenced method)."""
        if len(self.alphas) == 0:
            return

        # Build feature matrix from current alpha values
        feature_matrix = []
        valid_indices = []

        for i, alpha in enumerate(self.alphas):
            if 'values' in alpha and alpha['values'] is not None:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(values.flatten())
                valid_indices.append(i)

        if not feature_matrix:
            logger.warning("No valid alpha values for optimization")
            return

        X = np.column_stack(feature_matrix)
        y = y_data.values if hasattr(y_data, 'values') else y_data

        # Align lengths defensively
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Drop rows with any NaNs
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        if valid_mask.sum() < 10:
            logger.warning("Insufficient valid data for optimization")
            return

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # === New: per-column standardization to stabilize optimization ===
        with np.errstate(all='ignore'):
            col_mean = X_clean.mean(axis=0)
            col_std = X_clean.std(axis=0)
            eps = 1e-8
            col_std_safe = np.where(col_std < eps, 1.0, col_std)
            X_clean = (X_clean - col_mean) / col_std_safe
            X_clean = np.clip(X_clean, -10, 10)

        # Initialize weights from existing entries (fallback to uniform)
        weights = np.array([self.alphas[i].get('weight', 1.0 / len(valid_indices))
                            for i in valid_indices])

        # Gradient descent with L2 regularization
        best_loss = float('inf')
        best_weights = weights.copy()

        for iteration in range(max_iters):
            # Forward pass
            predictions = X_clean @ weights

            # Mean Squared Error loss
            error = predictions - y_clean
            loss = np.mean(error ** 2)

            # Track best weights seen so far
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
