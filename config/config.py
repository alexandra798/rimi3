"""配置文件"""

# MCTS参数
MCTS_CONFIG = {
    "num_iterations": 200,  # MCTS search cycle执行200次
    "risk_seeking_exploration": 2.0,
    "max_episode_length": 30,
    "num_simulations": 200,

    "gamma": 1.0,
    "c_puct": 1.414,  # sqrt(2) 是常用默认值，可根据实验调整

    'exploration_constant': 1.414,
    'min_window_size': 3,  # 新增：最小窗口大小
    'constant_penalty': -1.0,  # 新增：常数惩罚
    'diversity_bonus': 0.1,  # 新增：多样性奖励
    'max_constants_per_formula': 2,  # 新增：每个公式最多常数数量
}

# 风险寻求优化配置
RISK_SEEKING_CONFIG = {
    "quantile_threshold": 0.85,  # 优化top 15%
    "learning_rate_beta": 0.01,  # quantile regression学习率
    "learning_rate_gamma": 0.001,  # 网络参数更新学习率
    "gradient_clip": 0.5,
}

# Alpha池参数
ALPHA_POOL_CONFIG = {
    "pool_size": 100,  # K=100
    "lambda_param": 0.1,  # λ=0.1 (reward-dense MDP)
    "gradient_descent_lr": 0.01,
    "gradient_descent_iters": 100,
    'min_std': 1e-5,  # 添加：最小标准差阈值
    'min_unique_ratio': 0.01,  # 添加：最小唯一值比例
    'min_ic_threshold': 0.01,  # 添加：最小IC阈值
    'constant_check_sample_size': 1000,
}

# GRU特征提取器参数
GRU_CONFIG = {
    "num_layers": 4,  # 4层结构
    "hidden_dim": 64  # 隐藏层维度64
}

# Policy头参数
POLICY_CONFIG = {
    "hidden_layers": 2,  # 两个隐藏层
    "hidden_neurons": 32,  # 每层32个神经元
    "gru_layers": 4,  # 4层GRU
    "gru_hidden_dim": 64,
    "policy_hidden_layers": [32, 32],  # 两层MLP，每层32神经元
    "dropout_rate": 0.1,
}

# 交叉验证参数
CV_CONFIG = {
    "n_splits": 8
}

# 数据路径
DATA_CONFIG = {
    "default_data_path": "/path/to/data.csv",
    "target_column": "label_shifted",
    "features": ["open", "high", "low", "close", "volume", "vwap"],
    "target_windows": [5, 10],  # 5天和10天收益率
    "train_period": "2012-01-01 to 2021-12-31",
    "val_period": "2022-01-01 to 2022-12-31",
    "test_period": "2023-01-01 to 2024-12-31",
}

# 回测配置
BACKTEST_CONFIG = {
    "top_k": 40,  # 选择前40只股票
    "rebalance_freq": 5,  # 每5天重新平衡
    "transaction_cost": 0.001,  # 交易成本
    "initial_capital": 1000000,
}
# 验证配置
def validate_config():
    """验证配置的合理性"""
    assert MCTS_CONFIG["gamma"] == 1.0, "论文要求γ=1鼓励长表达式"
    assert ALPHA_POOL_CONFIG["lambda_param"] == 0.1, "论文指定λ=0.1"
    assert ALPHA_POOL_CONFIG["pool_size"] == 100, "论文指定K=100"
    assert POLICY_CONFIG["gru_layers"] == 4, "论文指定4层GRU"
    assert POLICY_CONFIG["gru_hidden_dim"] == 64, "论文指定隐藏维度64"
    print("Configuration validated successfully!")