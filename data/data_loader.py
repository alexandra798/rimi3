"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_user_dataset(file_path='price_volume_target5d.csv', target_column='target'):
    """
    Load a user-provided dataset, set the target column, and prepare features.

    Parameters:
    - file_path: Path to the dataset file (CSV or .pt). Default is 'price_volume_target5d.csv'.
    - target_column: Name of the target column in the dataset. Default is 'target'.

    Returns:
    - X: Feature DataFrame
    - y: Target Series
    - all_features: List of feature column names
    """
    logger.info(f"Loading dataset from {file_path}")

    # Determine file type
    if file_path.endswith('.pt'):
        # Load PyTorch serialized file
        data_dict = torch.load(file_path, weights_only=False)

        # Extract tensors
        X_tensor = data_dict['X']
        y_tensor = data_dict['y']
        all_features = data_dict['feature_columns']

        # Convert tensors to pandas structures for compatibility
        X = pd.DataFrame(X_tensor.numpy(), columns=all_features)
        y = pd.Series(y_tensor.numpy(), name=target_column)

        # Rebuild MultiIndex if date and ticker information is available
        if data_dict.get('has_date') and data_dict.get('has_ticker'):
            if 'dates' in data_dict and 'tickers' in data_dict:
                # Create a MultiIndex: (ticker, date)
                index = pd.MultiIndex.from_arrays(
                    [data_dict['tickers'], pd.to_datetime(data_dict['dates'])],
                    names=['ticker', 'date']
                )
                X.index = index
                y.index = index

    else:
        # Original CSV loading logic
        user_dataset = pd.read_csv(file_path)

        # Convert date column to datetime format
        if 'date' in user_dataset.columns:
            user_dataset['date'] = pd.to_datetime(user_dataset['date'], errors='coerce')
            user_dataset.dropna(subset=['date'], inplace=True)

            # Set MultiIndex if both ticker and date exist
            if 'ticker' in user_dataset.columns:
                user_dataset.set_index(['ticker', 'date'], inplace=True)

        # Ensure the target column exists
        if target_column not in user_dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Separate features and target
        X = user_dataset.drop(columns=[target_column])
        y = user_dataset[target_column]

        # Collect feature names
        all_features = X.columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, all_features


def detect_suspension_periods(df, price_columns=['close']):
    """
    Detect suspension periods where prices remain unchanged for 5 consecutive days.

    Parameters:
    - df: DataFrame containing data for a single ticker
    - price_columns: Price columns used to detect suspension

    Returns:
    - suspension_mask: Boolean Series, True indicates suspension period
    """
    suspension_mask = pd.Series(False, index=df.index)

    for col in price_columns:
        if col in df.columns:
            # Compute rolling standard deviation over a 5-day window
            rolling_std = df[col].rolling(window=5, min_periods=5).std()

            # Near-zero standard deviation implies no price movement
            suspension_mask |= (rolling_std < 1e-10)

    return suspension_mask


def clean_target_zeros(X, y):
    """
    Clean samples where target equals zero, distinguishing between suspension and normal trading.

    Parameters:
    - X: Feature DataFrame
    - y: Target Series

    Returns:
    - X_clean: Cleaned feature DataFrame
    - y_clean: Cleaned target Series
    """
    logger.info(f"Cleaning target=0 samples. Initial shape: {len(y)}")

    # Identify samples with target equal to 0 or NaN
    zero_mask = (y == 0) | y.isna()
    logger.info(f"Found {zero_mask.sum()} samples with target=0 or NaN")

    # Case: MultiIndex (ticker, date)
    if isinstance(X.index, pd.MultiIndex):
        valid_mask = pd.Series(True, index=X.index)

        # Process each ticker independently
        for ticker in X.index.get_level_values(0).unique():
            ticker_mask = X.index.get_level_values(0) == ticker
            ticker_X = X[ticker_mask]
            ticker_y = y[ticker_mask]
            ticker_zero_mask = zero_mask[ticker_mask]

            if ticker_zero_mask.any():
                # Detect suspension periods for this ticker
                suspension = detect_suspension_periods(ticker_X)

                # Remove samples where target=0 during suspension
                to_remove = ticker_zero_mask & suspension
                valid_mask[ticker_mask] = ~to_remove

                if to_remove.any():
                    logger.info(f"Ticker {ticker}: Removing {to_remove.sum()} suspended samples")
    else:
        # Case: single ticker or no ticker index
        suspension = detect_suspension_periods(X)
        to_remove = zero_mask & suspension
        valid_mask = ~to_remove

        logger.info(f"Removing {to_remove.sum()} suspended samples")

    # Always remove samples with missing target values
    valid_mask = valid_mask & ~y.isna()

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    logger.info(f"After cleaning: {len(y_clean)} samples remaining")
    logger.info(f"Kept {(y_clean == 0).sum()} normal samples with target=0")

    return X_clean, y_clean


def handle_missing_values(dataset, strategy='mixed'):
    """
    Handle missing values in the dataset.

    Parameters:
    - dataset: Input DataFrame
    - strategy: Missing value handling strategy
        - 'mixed': Apply different strategies based on column type (recommended)
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - Other legacy strategies

    Returns:
    - dataset: Processed DataFrame
    """
    dataset = dataset.copy()

    if strategy == 'mixed':
        # Price-related columns: backward fill first, then forward fill
        price_cols = ['open', 'high', 'low', 'close', 'vwap']
        for col in price_cols:
            if col in dataset.columns:
                dataset[col] = dataset[col].bfill()
                dataset[col] = dataset[col].ffill()

                # Handle columns still containing NaN (e.g., entirely missing)
                if dataset[col].isna().any():
                    logger.warning(f"Column {col} still has {dataset[col].isna().sum()} NaN values after filling")
                    median_val = dataset[col].median()
                    if pd.isna(median_val):
                        dataset[col] = dataset[col].fillna(0)
                    else:
                        dataset[col] = dataset[col].fillna(median_val)

        # Volume: fill missing values with 0
        if 'volume' in dataset.columns:
            dataset['volume'] = dataset['volume'].fillna(0)

        # Other columns: forward fill, then fill remaining NaN with 0
        other_cols = [col for col in dataset.columns
                      if col not in price_cols + ['volume']]
        for col in other_cols:
            dataset[col] = dataset[col].ffill().fillna(0)

    elif strategy == 'forward_fill':
        dataset = dataset.ffill().fillna(0)
    elif strategy == 'backward_fill':
        dataset = dataset.bfill().fillna(0)
    elif strategy == 'mean':
        dataset = dataset.fillna(dataset.mean()).fillna(0)
    elif strategy == 'median':
        dataset = dataset.fillna(dataset.median()).fillna(0)
    elif strategy == 'zero':
        dataset = dataset.fillna(0)
    elif strategy == 'drop':
        dataset = dataset.dropna()
    else:
        logger.warning(f"Unknown strategy '{strategy}', using forward fill")
        dataset = dataset.ffill().fillna(0)

    # Final safety check: ensure no NaN or infinite values remain
    if dataset.isna().any().any():
        nan_cols = dataset.columns[dataset.isna().any()].tolist()
        logger.error(f"Still have NaN values in columns: {nan_cols}")
        dataset = dataset.fillna(0)

    # Replace infinite values
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(dataset[numeric_cols].values)
    if inf_mask.any():
        logger.warning("Found inf values, replacing with 0")
        dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], 0)

    return dataset


def validate_data_quality(X, y):
    """
    Validate data quality to ensure downstream algorithms (e.g., MCTS) can run safely.

    Parameters:
    - X: Feature DataFrame
    - y: Target Series

    Returns:
    - is_valid: Boolean indicating whether data quality is acceptable
    - issues: List of detected issues
    """
    issues = []

    # Check for NaN values
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        issues.append(f"NaN values found in features: {nan_cols}")

    if y.isna().any():
        issues.append(f"NaN values found in target: {y.isna().sum()} samples")

    # Check for infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        if np.isinf(X[numeric_cols].values).any():
            issues.append("Inf values found in features")

    if np.isinf(y.values).any():
        issues.append("Inf values found in target")

    # Check sample size
    if len(X) < 100:
        issues.append(f"Too few samples: {len(X)}")

    # Check for constant (non-informative) features
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols}")

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning("Data quality issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data quality check passed")

    return is_valid, issues


def check_missing_values(dataset, dataset_name):
    """
    Check for missing values in a dataset (legacy interface preserved).

    Parameters:
    - dataset: DataFrame to inspect
    - dataset_name: Name of the dataset (used for logging)
    """
    missing_values = dataset.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        logger.warning(f'Missing values in {dataset_name} dataset:')
        logger.warning(missing_columns)
    else:
        logger.info(f'No missing values in {dataset_name} dataset.')


def apply_alphas_and_return_transformed(X, alpha_formulas, evaluate_formula_func):
    """
    Apply alpha formulas to the dataset and return the transformed feature set.

    Parameters:
    - X: Original feature DataFrame
    - alpha_formulas: List of alpha formulas to apply
    - evaluate_formula_func: Function used to evaluate each formula

    Returns:
    - transformed_X: DataFrame containing original and alpha-generated features
    """
    transformed_X = X.copy()

    for formula in alpha_formulas:
        result = evaluate_formula_func(formula, X)

        # Replace invalid values
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)

        transformed_X[formula] = result

    return transformed_X


def prepare_stock_features(raw_data):
    """
    Prepare the six core features required by the paper.
    """
    features = pd.DataFrame()

    # Basic price features
    features['open'] = raw_data['open']
    features['high'] = raw_data['high']
    features['low'] = raw_data['low']
    features['close'] = raw_data['close']
    features['volume'] = raw_data['volume']

    # Compute VWAP (Volume Weighted Average Price)
    # VWAP = sum(Price * Volume) / sum(Volume)
    typical_price = (raw_data['high'] + raw_data['low'] + raw_data['close']) / 3
    features['vwap'] = (typical_price * raw_data['volume']).rolling(window=1).sum() / \
                       raw_data['volume'].rolling(window=1).sum()

    # Compute future return targets
    returns_5d = raw_data['close'].pct_change(5).shift(-5)    # 5-day forward return
    returns_10d = raw_data['close'].pct_change(10).shift(-10) # 10-day forward return

    return features, returns_5d, returns_10d
