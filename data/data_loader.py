"""数据加载和预处理模块 - 改进版"""
import pandas as pd
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_user_dataset(file_path='price_volume_target5d.csv', target_column='target'):
    """
    加载用户数据集，设置目标列，并准备特征。

    Parameters:
    - file_path: 数据集文件路径 (CSV或.pt文件), 默认为 price_volume_target5d.csv
    - target_column: 数据集中的目标列名称, 默认为 'target'

    Returns:
    - X (特征), y (目标), all_features (特征名称列表)
    """
    logger.info(f"Loading dataset from {file_path}")

    # 判断文件类型
    if file_path.endswith('.pt'):
        # 加载PyTorch二进制文件
        data_dict = torch.load(file_path, weights_only=False)

        # 提取数据
        X_tensor = data_dict['X']
        y_tensor = data_dict['y']
        all_features = data_dict['feature_columns']

        # 转换为pandas DataFrame以保持与原代码的兼容性
        X = pd.DataFrame(X_tensor.numpy(), columns=all_features)
        y = pd.Series(y_tensor.numpy(), name=target_column)

        # 如果有date和ticker信息，重建索引
        if data_dict.get('has_date') and data_dict.get('has_ticker'):
            if 'dates' in data_dict and 'tickers' in data_dict:
                # 创建多级索引
                index = pd.MultiIndex.from_arrays(
                    [data_dict['tickers'], pd.to_datetime(data_dict['dates'])],
                    names=['ticker', 'date']
                )
                X.index = index
                y.index = index

    else:
        # 原始CSV加载逻辑
        user_dataset = pd.read_csv(file_path)

        # 转换日期列为datetime格式
        if 'date' in user_dataset.columns:
            user_dataset['date'] = pd.to_datetime(user_dataset['date'], errors='coerce')
            user_dataset.dropna(subset=['date'], inplace=True)
            # 如果存在ticker和date，设置多级索引
            if 'ticker' in user_dataset.columns:
                user_dataset.set_index(['ticker', 'date'], inplace=True)

        # 确保目标列存在
        if target_column not in user_dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # 分离特征和目标
        X = user_dataset.drop(columns=[target_column])
        y = user_dataset[target_column]

        # 获取特征名称列表
        all_features = X.columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, all_features


def detect_suspension_periods(df, price_columns=['close']):
    """
    检测停牌期间（价格连续5天不变）

    Parameters:
    - df: DataFrame，应该是单个ticker的数据
    - price_columns: 用于检测停牌的价格列

    Returns:
    - suspension_mask: bool Series，True表示停牌期
    """
    suspension_mask = pd.Series(False, index=df.index)

    for col in price_columns:
        if col in df.columns:
            # 检测连续5天价格不变
            rolling_std = df[col].rolling(window=5, min_periods=5).std()
            # 标准差为0或接近0表示价格无变化
            suspension_mask |= (rolling_std < 1e-10)

    return suspension_mask


def clean_target_zeros(X, y):
    """
    清理target=0的数据，区分停牌和正常交易

    Parameters:
    - X: 特征DataFrame
    - y: 目标Series

    Returns:
    - X_clean: 清理后的特征
    - y_clean: 清理后的目标
    """
    logger.info(f"Cleaning target=0 samples. Initial shape: {len(y)}")

    # 找出target=0的样本
    zero_mask = (y == 0) | y.isna()
    logger.info(f"Found {zero_mask.sum()} samples with target=0 or NaN")

    # 如果数据有多级索引（ticker, date）
    if isinstance(X.index, pd.MultiIndex):
        valid_mask = pd.Series(True, index=X.index)

        # 按ticker分组处理
        for ticker in X.index.get_level_values(0).unique():
            ticker_mask = X.index.get_level_values(0) == ticker
            ticker_X = X[ticker_mask]
            ticker_y = y[ticker_mask]
            ticker_zero_mask = zero_mask[ticker_mask]

            if ticker_zero_mask.any():
                # 检测停牌期
                suspension = detect_suspension_periods(ticker_X)

                # target=0且处于停牌期的要删除
                to_remove = ticker_zero_mask & suspension
                valid_mask[ticker_mask] = ~to_remove

                if to_remove.any():
                    logger.info(f"Ticker {ticker}: Removing {to_remove.sum()} suspended samples")
    else:
        # 单ticker或无ticker索引的情况
        suspension = detect_suspension_periods(X)
        # 只删除target=0且停牌的样本
        to_remove = zero_mask & suspension
        valid_mask = ~to_remove

        logger.info(f"Removing {to_remove.sum()} suspended samples")

    # 删除target缺失的行（无论是否停牌）
    valid_mask = valid_mask & ~y.isna()

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    logger.info(f"After cleaning: {len(y_clean)} samples remaining")
    logger.info(f"Kept {(y_clean == 0).sum()} normal samples with target=0")

    return X_clean, y_clean


def handle_missing_values(dataset, strategy='mixed'):
    """
    处理数据集中的缺失值

    Parameters:
    - dataset: 要处理的DataFrame
    - strategy: 处理策略
      - 'mixed': 根据列类型采用不同策略（推荐）
      - 'forward_fill': 前向填充
      - 'backward_fill': 后向填充
      - 其他原有策略

    Returns:
    - dataset: 处理后的DataFrame
    """
    dataset = dataset.copy()

    if strategy == 'mixed':
        # 价格类列：先后向填充，再前向填充
        price_cols = ['open', 'high', 'low', 'close', 'vwap']
        for col in price_cols:
            if col in dataset.columns:
                # 先后向填充（处理开始时的缺失）
                dataset[col] = dataset[col].bfill()
                # 再前向填充（处理结尾的缺失）
                dataset[col] = dataset[col].ffill()

                # 检查是否还有缺失（整列都是NaN的情况）
                if dataset[col].isna().any():
                    logger.warning(f"Column {col} still has {dataset[col].isna().sum()} NaN values after filling")
                    # 用中位数填充剩余的NaN
                    median_val = dataset[col].median()
                    if pd.isna(median_val):
                        # 如果中位数也是NaN，用0填充
                        dataset[col] = dataset[col].fillna(0)
                    else:
                        dataset[col] = dataset[col].fillna(median_val)

        # 成交量：缺失填0
        if 'volume' in dataset.columns:
            dataset['volume'] = dataset['volume'].fillna(0)

        # 其他列：前向填充后用0填充
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

    # 最终检查：确保没有NaN和inf
    if dataset.isna().any().any():
        nan_cols = dataset.columns[dataset.isna().any()].tolist()
        logger.error(f"Still have NaN values in columns: {nan_cols}")
        # 强制填充为0
        dataset = dataset.fillna(0)

    # 检查inf值
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(dataset[numeric_cols].values)
    if inf_mask.any():
        logger.warning("Found inf values, replacing with 0")
        dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], 0)

    return dataset


def validate_data_quality(X, y):
    """
    验证数据质量，确保MCTS可以正常运行

    Parameters:
    - X: 特征DataFrame
    - y: 目标Series

    Returns:
    - is_valid: bool，数据是否有效
    - issues: list，发现的问题列表
    """
    issues = []

    # 检查NaN
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        issues.append(f"NaN values found in features: {nan_cols}")

    if y.isna().any():
        issues.append(f"NaN values found in target: {y.isna().sum()} samples")

    # 检查inf
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_mask = np.isinf(X[numeric_cols].values)
        if inf_mask.any():
            issues.append("Inf values found in features")

    if np.isinf(y.values).any():
        issues.append("Inf values found in target")

    # 检查数据量
    if len(X) < 100:
        issues.append(f"Too few samples: {len(X)}")

    # 检查特征变化
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() == 1:
            constant_cols.append(col)
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
    检查数据集中的缺失值（保持原有接口）

    Parameters:
    - dataset: 要检查缺失值的DataFrame
    - dataset_name: 数据集名称（用于打印）
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
    应用顶级alpha公式到数据集，返回包含原始特征和新alpha特征的转换数据集

    Parameters:
    - X: 原始特征数据集
    - alpha_formulas: 要应用的alpha公式列表
    - evaluate_formula_func: 评估公式的函数

    Returns:
    - transformed_X: 包含原始特征和新alpha特征的数据集
    """
    transformed_X = X.copy()

    for formula in alpha_formulas:
        result = evaluate_formula_func(formula, X)
        # 处理结果中的NaN值
        result = result.fillna(0)
        # 处理inf值
        result = result.replace([np.inf, -np.inf], 0)
        transformed_X[formula] = result

    return transformed_X


def prepare_stock_features(raw_data):
    """
    准备论文要求的6个特征
    """
    features = pd.DataFrame()

    # 基础价格特征
    features['open'] = raw_data['open']
    features['high'] = raw_data['high']
    features['low'] = raw_data['low']
    features['close'] = raw_data['close']
    features['volume'] = raw_data['volume']

    # 计算VWAP (Volume Weighted Average Price)
    # VWAP = Σ(Price * Volume) / Σ(Volume)
    typical_price = (raw_data['high'] + raw_data['low'] + raw_data['close']) / 3
    features['vwap'] = (typical_price * raw_data['volume']).rolling(window=1).sum() / \
                       raw_data['volume'].rolling(window=1).sum()

    # 计算收益率目标
    returns_5d = raw_data['close'].pct_change(5).shift(-5)  # 未来5天收益率
    returns_10d = raw_data['close'].pct_change(10).shift(-10)  # 未来10天收益率

    return features, returns_5d, returns_10d