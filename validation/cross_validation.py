"""交叉验证模块 validation/cross_validation.py"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import logging
from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def evaluate_formula_cross_val(formula, X, y, n_splits, evaluate_formula_func=None):
    """
    使用交叉验证评估公式

    Parameters:
    - formula: 要评估的alpha公式
    - X: 特征数据
    - y: 目标数据
    - n_splits: 交叉验证折数
    - evaluate_formula_func: 评估公式的函数

    Returns:
    - ic_scores: 每折的IC分数列表
    """
    evaluator = FormulaEvaluator() if evaluate_formula_func is None else None
    eval_fn = evaluator.evaluate if evaluate_formula_func is None else evaluate_formula_func

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_scores = []

    logger.info(f"Evaluating formula: {formula}")

    for train_index, test_index in tscv.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # 评估测试折上的公式
        feature_test = eval_fn(formula, X_test_fold)

        # 清理数据
        valid_indices = ~(feature_test.isna() | y_test_fold.isna())
        feature_test_clean = feature_test[valid_indices]
        y_test_fold_clean = y_test_fold[valid_indices]

        logger.debug(f"Valid data points: {len(feature_test_clean)}")

        # 计算IC
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
    对多个公式进行交叉验证

    Returns:
    - cv_results: 包含每个公式CV结果的字典
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