"""Cross-validation module"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import logging
from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def evaluate_formula_cross_val(formula, X, y, n_splits, evaluate_formula_func=None):
    """
    Evaluate a formula using cross-validation.

    Parameters:
    - formula: Alpha formula to evaluate.
    - X: Feature data.
    - y: Target data.
    - n_splits: Number of CV splits.
    - evaluate_formula_func: Optional custom formula evaluation function.

    Returns:
    - ic_scores: List of IC scores for each fold.
    """
    evaluator = FormulaEvaluator() if evaluate_formula_func is None else None
    eval_fn = evaluator.evaluate if evaluate_formula_func is None else evaluate_formula_func

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_scores = []

    logger.info(f"Evaluating formula: {formula}")

    for train_index, test_index in tscv.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Evaluate the formula on the test fold.
        feature_test = eval_fn(formula, X_test_fold)

        # Clean and align data.
        valid_indices = ~(feature_test.isna() | y_test_fold.isna())
        feature_test_clean = feature_test[valid_indices]
        y_test_fold_clean = y_test_fold[valid_indices]

        logger.debug(f"Valid data points: {len(feature_test_clean)}")

        # Compute IC.
        if len(feature_test_clean) > 1:
            ic, _ = spearmanr(feature_test_clean, y_test_fold_clean)
            ic_scores.append(ic if not np.isnan(ic) else 0)
            logger.debug(f"IC for fold: {ic:.4f}")
        else:
            ic_scores.append(0)
            logger.warning(f"Insufficient data for IC calculation, fold skipped.")

    return ic_scores

def cross_validate_formulas(formulas, X, y, n_splits, evaluate_formula_func=None):
    """
    Cross-validate multiple formulas.

    Returns:
    - cv_results: Dict containing CV results for each formula.
    """
    if evaluate_formula_func is None:
        evaluator = FormulaEvaluator()
        evaluate_formula_func = evaluator.evaluate

    cv_results = {}

    for formula in formulas:
        ic_scores = evaluate_formula_cross_val(formula, X, y, n_splits, evaluate_formula_func)
        cv_results[formula] = {
            'IC Scores': ic_scores,
            'Mean IC': np.mean(ic_scores),
            'IC Std Dev': np.std(ic_scores)
        }

    return cv_results
