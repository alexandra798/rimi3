"""主程序入口 - 支持Token系统和传统系统（改进版）"""
import argparse
import logging
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

from config.config import *
from data.data_loader import (
    load_user_dataset,
    check_missing_values,
    handle_missing_values,
    apply_alphas_and_return_transformed,
    clean_target_zeros,
    validate_data_quality
)
from alpha.pool import AlphaPool
from alpha.evaluator import FormulaEvaluator
from validation.cross_validation import cross_validate_formulas
from validation.backtest import backtest_formulas
from mcts.trainer import RiskMinerTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=ConstantInputWarning)

def _preprocess_for_mcts(X: pd.DataFrame) -> pd.DataFrame:
    """对每列做 log1p + z-score；仅用于 MCTS/AlphaPool 的输入，不改动原始 X"""
    Xp = X.copy()
    numeric_cols = Xp.select_dtypes(include=[np.number]).columns

    # 1. 截断极端值
    Xp[numeric_cols] = np.clip(Xp[numeric_cols], a_min=-1e6, a_max=1e6)

    # 2. log变换（对正值特征如volume）
    for col in ['volume', 'vwap']:
        if col in numeric_cols:
            Xp[col] = np.log1p(np.abs(Xp[col]))

    # 3. 标准化
    with np.errstate(all='ignore'):  # 抑制中间计算警告
        mean = Xp[numeric_cols].mean(axis=0)
        std = Xp[numeric_cols].std(axis=0)
        std_safe = std.replace(0, 1.0)
        Xp[numeric_cols] = (Xp[numeric_cols] - mean) / std_safe
        Xp[numeric_cols] = np.clip(Xp[numeric_cols], -5, 5)
    return Xp


def run_mcts_with_token_system(X_train, y_train, num_iterations=200,
                               use_policy_network=True, num_simulations=50,
                               device=None, random_seed=42):
    """
    使用新的Token系统运行MCTS
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        num_iterations: 训练迭代次数
        use_policy_network: 是否使用策略网络
        num_simulations: 每次迭代的模拟次数
        device: torch设备(cuda或cpu)

    Returns:
        (top_formulas, trainer):
        top_formulas 为 [(formula, ic/score), ...]
        trainer 为 RiskMinerTrainer 实例（含 X_train_sample / y_train 等）
    """
    logger.info("Starting MCTS with Token System")
    logger.info(f"Data size: {len(X_train)} rows")

    # 创建训练器
    trainer = RiskMinerTrainer(X_train, y_train, device=device, use_sampling=True, random_seed=random_seed)

    # 训练
    trainer.train(
        num_iterations=num_iterations,
        num_simulations_per_iteration=num_simulations
    )

    # 获取最佳公式
    top_formulas = trainer.get_top_formulas(n=5)

    # 转换为兼容格式（formula, score）
    result = []
    for formula in top_formulas:
        # 计算IC作为分数
        if trainer.alpha_pool:
            matching_alpha = next((a for a in trainer.alpha_pool if a['formula'] == formula), None)
            if matching_alpha:
                result.append((formula, matching_alpha['ic']))
            else:
                result.append((formula, 0.0))
        else:
            result.append((formula, 0.0))

    return result, trainer


def main(args):
    logger.info("Starting RiMi Algorithm")

    if args.use_token_system:
        logger.info("=== Using Token-based RPN System ===")
    else:
        logger.info("=== Using Legacy System ===")

    # 设置GPU设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU instead")

    # 第1部分：数据准备和探索
    logger.info("=== Part 1: Data Preparation & Exploration ===")

    # 加载原始数据
    X, y, all_features = load_user_dataset(args.data_path, args.target_column)
    logger.info(f"Initial data shape: X={X.shape}, y={y.shape}")

    check_missing_values(X, 'initial')

    logger.info("Step 1: Cleaning target=0 samples (removing suspensions)...")
    X, y = clean_target_zeros(X, y)

    logger.info("Step 2: Handling missing values with mixed strategy...")
    X = handle_missing_values(X, strategy='mixed')

    check_missing_values(X, 'after_handling')

    logger.info("Step 3: Validating data quality...")
    is_valid, issues = validate_data_quality(X, y)

    if not is_valid:
        logger.error("Data quality validation failed! Issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        if not args.force_continue:
            raise ValueError("Data quality check failed. Use --force_continue to proceed anyway.")
        else:
            logger.warning("Continuing despite data quality issues (--force_continue flag set)")

    logger.info(f"Final clean dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Target distribution: mean={y.mean():.4f}, std={y.std():.4f}")
    logger.info(f"Samples with target=0: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    # === 新增：仅供 MCTS/AlphaPool 使用的去量纲版本 ===
    X_train_mcts = _preprocess_for_mcts(X_train)
    X_test_mcts = _preprocess_for_mcts(X_test) if args.backtest else None

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 第2-4部分：MCTS和Alpha池管理
    logger.info("=== Parts 2-4: MCTS & Alpha Pool Management ===")

    if args.use_risk_seeking:
        logger.info("Using Token system with Risk Seeking Policy Network")
        best_formulas_quantile, trainer = run_mcts_with_token_system(
            X_train_mcts, y_train,
            num_iterations=MCTS_CONFIG['num_iterations'],
            num_simulations=50,
            device=device,
            random_seed=42
        )
    else:
        logger.info("Using Token system without Policy Network")
        best_formulas_quantile, trainer = [], None

    evaluate_formula = FormulaEvaluator()

    # 初始化Alpha池
    alpha_pool = AlphaPool(
        pool_size=ALPHA_POOL_CONFIG['pool_size'],
        lambda_param=ALPHA_POOL_CONFIG['lambda_param']
    )

    # 添加公式到池中
    if best_formulas_quantile:
        for formula, score in best_formulas_quantile:
            alpha_pool.add_to_pool({
                'formula': formula,
                'score': score,
                'ic': score
            })

        # 使用与训练相同的采样集更新池
        if args.use_risk_seeking and hasattr(trainer, 'X_train_sample'):
            # 如果MCTS使用了采样，池更新也用相同的采样集
            X_pool_update = _preprocess_for_mcts(trainer.X_train_sample)
            y_pool_update = trainer.y_train
        else:
            X_pool_update = X_train_mcts
            y_pool_update = y_train

        alpha_pool.update_pool(X_pool_update, y_pool_update, evaluate_formula)

    top_formulas = alpha_pool.get_top_formulas(5)

    if not top_formulas:
        logger.warning("No formulas found in alpha pool, using default formulas")
        top_formulas = [
            "BEG close END",
            "BEG volume END",
            "BEG close volume div END"
        ]

    logger.info(f"Top formulas from alpha pool: {len(top_formulas)} formulas")
    for i, formula in enumerate(top_formulas[:5], 1):
        logger.info(f"  {i}. {formula[:80]}...")

    # 第5部分：应用公式转换数据集
    if args.transform_data:
        logger.info("=== Part 5: Apply Formulas to Transform Dataset ===")
        transformed_X = apply_alphas_and_return_transformed(X, top_formulas, evaluate_formula)
        logger.info(f"Transformed dataset shape: {transformed_X.shape}")

        # 可选：保存转换后的数据
        if args.save_transformed:
            output_path = args.output_path or "transformed_data.csv"
            logger.info(f"Saving transformed data to {output_path}")
            transformed_X.to_csv(output_path)

    # 第6部分：交叉验证
    if args.cross_validate:
        logger.info("=== Part 6: Cross-Validation ===")
        cv_results = cross_validate_formulas(
            top_formulas,
            X,
            y,
            CV_CONFIG['n_splits'],
            evaluate_formula
        )

        logger.info("\nCross-validation results:")
        for formula, results in cv_results.items():
            logger.info(f"\nFormula: {formula}")
            logger.info(f"Mean IC: {results['Mean IC']:.4f}")
            logger.info(f"IC Std Dev: {results['IC Std Dev']:.4f}")

    # 第7部分：回测
    if args.backtest:
        logger.info("=== Part 7: Backtest ===")
        backtest_results = backtest_formulas(top_formulas, X_test, y_test)

        # 按IC值排序结果
        sorted_results = sorted(backtest_results.items(), key=lambda x: x[1], reverse=True)

        logger.info("\nSorted backtest results (by IC):")
        for formula, ic in sorted_results:
            logger.info(f"Formula: {formula}")
            logger.info(f"Information Coefficient (IC): {ic:.4f}\n")

    # 保存结果
    if args.save_results:
        results_path = args.results_path or "alpha_results.txt"
        logger.info(f"Saving results to {results_path}")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("=== Top Alpha Formulas ===\n")
            f.write(f"System: {'Token-based RPN' if args.use_token_system else 'Legacy'}\n")
            f.write(f"Risk Seeking: {args.use_risk_seeking}\n\n")

            for i, formula in enumerate(top_formulas, 1):
                f.write(f"{i}. {formula}\n")

            if args.backtest and 'sorted_results' in locals():
                f.write("\n=== Backtest Results ===\n")
                for formula, ic in sorted_results:
                    f.write(f"Formula: {formula}, IC: {ic:.4f}\n")

    logger.info("RiskMiner completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiskMiner Algorithm")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV or .pt data file"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="target",
        help="Name of the target column"
    )
    parser.add_argument(
        "--use_token_system",
        action="store_true",
        help="Use the new Token-based RPN system instead of legacy string generation"
    )
    parser.add_argument(
        "--use_risk_seeking",
        action="store_true",
        help="Use risk-seeking MCTS with policy network"
    )
    parser.add_argument(
        "--transform_data",
        action="store_true",
        help="Apply formulas to transform the dataset"
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform cross-validation"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Perform backtesting"
    )
    parser.add_argument(
        "--save_transformed",
        action="store_true",
        help="Save the transformed dataset to a file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="transformed_data.csv",
        help="Path to save the transformed dataset"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save the alpha results to a file"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="alpha_results.txt",
        help="Path to save the alpha results"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--force_continue",
        action="store_true",
        help="Force continue even if data quality check fails"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()
    main(args)