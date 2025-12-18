"""Configuration file"""

# MCTS parameters
MCTS_CONFIG = {
    "num_iterations": 200,  # Run 200 MCTS search cycles
    "risk_seeking_exploration": 2.0,
    "max_episode_length": 30,
    "num_simulations": 200,

    "gamma": 1.0,
    "c_puct": 1.414,  # sqrt(2) is a common default, tunable

    'exploration_constant': 1.414,
    'min_window_size': 3,          # New: minimum rolling-window size
    'constant_penalty': -1.0,      # New: penalty for constant expressions
    'diversity_bonus': 0.1,        # New: reward for expression diversity
    'max_constants_per_formula': 2,# New: max allowed constants per formula
}

# Risk-seeking optimization config
RISK_SEEKING_CONFIG = {
    "quantile_threshold": 0.85,  # Optimize top 15%
    "learning_rate_beta": 0.01,  # LR for quantile regression
    "learning_rate_gamma": 0.001,# LR for network parameters
    "gradient_clip": 0.5,
}

# Alpha pool parameters
ALPHA_POOL_CONFIG = {
    "pool_size": 100,            # K = 100
    "lambda_param": 0.1,         # λ = 0.1 (reward-dense MDP)
    "gradient_descent_lr": 0.01,
    "gradient_descent_iters": 100,
    'min_std': 1e-5,             # Minimum std threshold for constant check
    'min_unique_ratio': 0.01,    # Minimum unique ratio threshold
    'min_ic_threshold': 0.01,    # Minimum IC threshold
    'constant_check_sample_size': 1000,
}

# GRU feature extractor parameters
GRU_CONFIG = {
    "num_layers": 4,   # 4-layer structure
    "hidden_dim": 64   # Hidden size = 64
}

# Policy head parameters
POLICY_CONFIG = {
    "hidden_layers": 2,              # Two hidden layers
    "hidden_neurons": 32,            # 32 neurons per layer
    "gru_layers": 4,                 # 4-layer GRU
    "gru_hidden_dim": 64,
    "policy_hidden_layers": [32, 32],# Two-layer MLP, 32 neurons each
    "dropout_rate": 0.1,
}

# Cross-validation parameters
CV_CONFIG = {
    "n_splits": 8
}

# Data paths
DATA_CONFIG = {
    "default_data_path": "/path/to/data.csv",
    "target_column": "label_shifted",
    "features": ["open", "high", "low", "close", "volume", "vwap"],
    "target_windows": [5, 10],                 # 5-day and 10-day returns
    "train_period": "2012-01-01 to 2021-12-31",
    "val_period":   "2022-01-01 to 2022-12-31",
    "test_period":  "2023-01-01 to 2024-12-31",
}

# Backtest configuration
BACKTEST_CONFIG = {
    "top_k": 40,               # Select top 40 stocks
    "rebalance_freq": 5,       # Rebalance every 5 days
    "transaction_cost": 0.001, # Trading cost
    "initial_capital": 1000000,
}

# Validation config
def validate_config():
    """Validate configuration consistency with referenced paper settings."""
    assert MCTS_CONFIG["gamma"] == 1.0, "论文要求γ=1鼓励长表达式"
    assert ALPHA_POOL_CONFIG["lambda_param"] == 0.1, "论文指定λ=0.1"
    assert ALPHA_POOL_CONFIG["pool_size"] == 100, "论文指定K=100"
    assert POLICY_CONFIG["gru_layers"] == 4, "论文指定4层GRU"
    assert POLICY_CONFIG["gru_hidden_dim"] == 64, "论文指定隐藏维度64"
    print("Configuration validated successfully!")
