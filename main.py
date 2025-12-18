"""Main entry point"""
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

# Logging setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=ConstantInputWarning)

def _preprocess_for_mcts(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p + z-score per column.

    Used only as input for MCTS/AlphaPool. The original X is not modified.
    """
    Xp = X.copy()
    numeric_cols = Xp.select_dtypes(include=[np.number]).columns

    # 1) Clip extreme values.
    Xp[numeric_cols] = np.clip(Xp[numeric_cols], a_min=-1e6, a_max=1e6)

    # 2) Log transform (for positive-scale features like volume).
    for col in ['volume', 'vwap']:
        if col in numeric_cols:
            Xp[col] = np.log1p(np.abs(Xp[col]))

    # 3) Standardization.
    with np.errstate(all='ignore'):  # Suppress intermediate computation warnings.
        mean = Xp[numeric_cols].mean(axis=0)
        std = Xp[numeric_cols].std(axis=0)
        std_safe = std.replace(0, 1.0)
        Xp[numeric_cols] = (Xp[numeric_cols] - mean) / std_safe
        Xp[numeric_cols] = np.clip(Xp[numeric_cols], -5, 5)
    return Xp


def precompute_features(X_data):
    """Precompute features in batch to avoid DataFrame fragmentation."""
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    windows = [3, 5, 10, 20, 30, 40, 50, 60]

    # Mark base columns with metadata (for evaluator fast-path detection).
    for col in base_cols:
        if col in X_data.columns:
            try:
                X_data[col].attrs['is_base_raw'] = True
                X_data[col].attrs['orig_name'] = col
            except Exception:
                pass

    # Build new columns in a single batch.
    new_cols = {}
    for col in base_cols:
        if col not in X_data.columns:
            continue
        s = X_data[col]

        for window in windows:
            # Rolling features.
            mean_col = s.rolling(window=window, min_periods=1).mean()
            std_col = s.rolling(window=window, min_periods=min(3, window)).std()

            mean_name = f'ts_mean_{col}_{window}'
            std_name = f'ts_std_{col}_{window}'

            mean_col.name = mean_name
            std_col.name = std_name

            new_cols[mean_name] = mean_col
            new_cols[std_name] = std_col

    # Concatenate all new columns at once to avoid fragmentation.
    if new_cols:
        X_data = pd.concat([X_data] + list(new_cols.values()), axis=1, copy=False)

    # Tag data identity.
    try:
        X_data.attrs['data_id'] = 'train_data_with_features'
    except Exception:
        pass

    return X_data


def run_mcts_with_token_system(X_train, y_train, num_iterations=200,
                               use_policy_network=True, num_simulations=50,
                               device=None, random_seed=42):
    """
    Returns:
        (top_formulas, trainer):
        top_formulas is [(formula, ic/score), ...]
        trainer is a RiskMinerTrainer instance (contains X_train_sample / y_train, etc.)
    """
    logger.info("Starting MCTS with Token System")
    logger.info(f"Data size: {len(X_train)} rows")

    trainer = RiskMinerTrainer(X_train, y_train, device=device, use_sampling=True, random_seed=random_seed)

    # Train.
    trainer.train(
        num_iterations=num_iterations,
        num_simulations_per_iteration=num_simulations
    )
    # Get the best formulas.
    top_formulas = trainer.get_top_formulas(n=5)

    # Convert to a compatible (formula, score) format.
    result = []
    for formula in top_formulas:
        # Use IC as the score.
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
    logger.info("Starting Rimi3")

    # Configure GPU device.
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

    # Part 1: data preparation and exploration.
    logger.info("=== Part 1: Data Preparation & Exploration ===")

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

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_train = precompute_features(X_train)
    X_train.attrs['data_id'] = 'train_sample_v1'
    # Added: use a dimensionless/preprocessed version only for MCTS/AlphaPool.
    X_train_mcts = _preprocess_for_mcts(X_train)
    X_test_mcts = _preprocess_for_mcts(X_test) if args.backtest else None

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Parts 2-4: MCTS and alpha pool management.
    logger.info("=== Parts 2-4: MCTS & Alpha Pool Management ===")

    logger.info("Using Token system with Risk Seeking Policy Network")
    best_formulas_quantile, trainer = run_mcts_with_token_system(
        X_train_mcts, y_train,
        num_iterations=MCTS_CONFIG['num_iterations'],
        num_simulations=50,
        device=device,
        random_seed=42
    )

    evaluate_formula = FormulaEvaluator()

    # Initialize alpha pool.
    alpha_pool = AlphaPool(
        pool_size=ALPHA_POOL_CONFIG['pool_size'],
        lambda_param=ALPHA_POOL_CONFIG['lambda_param']
    )

    # Add formulas to the pool.
    if best_formulas_quantile:
        for formula, score in best_formulas_quantile:
            alpha_pool.add_to_pool({
                'formula': formula,
                'score': score,
                'ic': score
            })

        # Update the pool using the same sampled data as training (if applicable).
        if args.use_risk_seeking and hasattr(trainer, 'X_train_sample'):
            # If MCTS used sampling, pool updates must use the exact same sample and convention.
            # Note: trainer.X_train_sample is already preprocessed; do not preprocess again.
            X_pool_update = trainer.X_train_sample
            y_pool_update = trainer.y_train_sample

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

    # Part 5: apply formulas to transform the dataset.
    if args.transform_data:
        logger.info("=== Part 5: Apply Formulas to Transform Dataset ===")
        transformed_X = apply_alphas_and_return_transformed(X, top_formulas, evaluate_formula)
        logger.info(f"Transformed dataset shape: {transformed_X.shape}")

        if args.save_transformed:
            output_path = args.output_path or "transformed_data.csv"
            logger.info(f"Saving transformed data to {output_path}")
            transformed_X.to_csv(output_path)

    # Part 6: cross-validation.
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

    # Part 7: backtest.
    if args.backtest:
        logger.info("=== Part 7: Backtest ===")
        backtest_results = backtest_formulas(top_formulas, X_test, y_test)

        # Sort results by IC.
        sorted_results = sorted(backtest_results.items(), key=lambda x: x[1], reverse=True)

        logger.info("\nSorted backtest results (by IC):")
        for formula, ic in sorted_results:
            logger.info(f"Formula: {formula}")
            logger.info(f"Information Coefficient (IC): {ic:.4f}\n")

    # Save results.
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

    logger.info("Rimi3 completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rimi3")

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
