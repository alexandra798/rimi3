# Rimi3: Automated Alpha Factor Mining System with MCTS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

Rimi3 is an **automated alpha factor discovery system** that leverages Monte Carlo Tree Search (MCTS) and deep reinforcement learning to automatically discover effective quantitative trading signals. The system uses Reverse Polish Notation (RPN) to represent factors, supporting a rich set of technical indicators and statistical operators.

### Key Features

- 🎯 **Intelligent Search**: MCTS-based formula space exploration guided by policy networks
- 🧮 **RPN Syntax**: Uses Reverse Polish Notation to construct factors, ensuring grammatical correctness
- 📊 **Rich Operators**: Supports 70+ operators (time-series, cross-sectional, statistical, etc.)
- 🎓 **Reinforcement Learning**: Risk-seeking optimizer focuses on improving top-quantile factor performance
- 💾 **Alpha Pool Management**: Automatically maintains high-quality factor pool with ensemble optimization
- ⚡ **Performance Optimization**: Multi-level caching, precomputation, GPU acceleration
- 🔬 **Complete Validation**: Cross-validation, backtesting, multi-dimensional performance evaluation

---

## 🏗️ System Architecture

```
Rimi3/
├── core/                  # Core components
│   ├── token_system.py    # Token definitions and RPN validator
│   ├── rpn_evaluator.py   # RPN expression evaluation engine
│   └── operators.py       # 70+ operator implementations
├── alpha/                 # Alpha factor management
│   ├── evaluator.py       # Formula evaluator (with caching)
│   └── pool.py           # Alpha pool management and weight optimization
├── mcts/                  # Monte Carlo Tree Search
│   ├── node.py           # MCTS nodes (PUCT selection)
│   ├── searcher.py       # MCTS searcher
│   ├── environment.py    # MDP environment definition
│   ├── reward_calculator.py # Reward computation (IC, diversity, etc.)
│   └── trainer.py        # Training orchestration
├── policy/                # Policy network
│   ├── network.py        # GRU-based policy network
│   └── optimizer.py      # Risk-seeking optimizer
├── data/                  # Data processing
│   └── data_loader.py    # Data loading, cleaning, preprocessing
├── validation/            # Validation modules
│   ├── cross_validation.py # Time-series cross-validation
│   └── backtest.py       # Backtesting and trading simulation
├── utils/                 # Utility functions
│   └── metrics.py        # Evaluation metrics (IC, Sharpe, etc.)
├── config/                # Configuration
│   └── config.py         # Global parameter settings
└── main.py               # Main entry point
```

---

## 🚀 Quick Start

### Installation

```bash
pip install torch pandas numpy scikit-learn scipy
```

### Data Preparation

The system supports two data formats:

**Format 1: CSV** (Recommended for beginners)
```csv
date,ticker,open,high,low,close,volume,vwap,target
2020-01-02,AAPL,75.0,76.0,74.5,75.8,100000,75.5,0.02
2020-01-02,MSFT,160.0,162.0,159.0,161.5,50000,160.8,0.01
...
```

**Format 2: PyTorch** (Recommended for large-scale data)
```python
torch.save({
    'X': features_tensor,           # shape: (N, num_features)
    'y': targets_tensor,            # shape: (N,)
    'feature_columns': ['open', 'high', ...],
    'dates': dates_array,           # Date sequence
    'tickers': tickers_array,       # Ticker symbols
    'has_date': True,
    'has_ticker': True
}, 'data.pt')
```

### Usage Examples

```bash
# Basic training
python main.py --data_path your_data.csv --target_column target

# Full pipeline (training + cross-validation + backtest)
python main.py \
    --data_path your_data.csv \
    --target_column target \
    --cross_validate \
    --backtest \
    --save_results \
    --results_path results.txt

# GPU acceleration
python main.py --data_path your_data.csv --gpu_id 0

# Save transformed dataset
python main.py \
    --data_path your_data.csv \
    --transform_data \
    --save_transformed \
    --output_path transformed_data.csv
```

---

## 🧠 Core Mechanisms

### 1. Token System & RPN Expressions

The system uses **Reverse Polish Notation** (RPN) to represent alpha factors, ensuring grammatical correctness and efficient evaluation.

**Examples**:
```python
# Traditional: (close - open) / open
# RPN:        BEG close open sub open div END

# Cross-sectional rank of 5-day close mean
# RPN:        BEG close ts_mean delta_5 csrank END
```

**Token Types**:
- **Operands**:
  - Base features: `open`, `high`, `low`, `close`, `volume`, `vwap`
  - Time windows: `delta_3`, `delta_5`, `delta_10`, ..., `delta_60`
  - Constants: `const_-30`, ..., `const_30`
  
- **Operators**:
  - Unary: `sign`, `abs`, `log`, `csrank` (cross-sectional rank)
  - Binary: `add`, `sub`, `mul`, `div`, `greater`, `less`
  - Time-series: `ts_mean`, `ts_std`, `ts_rank`, `ts_max`, `ts_min`, `ts_skew`, `ts_kurt`, etc.
  - Correlation: `corr`, `cov`

### 2. MCTS Search Mechanism

The search process consists of four phases:

```
Selection
    ↓
Expansion
    ↓
Rollout (Simulation)
    ↓
Backpropagation
```

**PUCT Formula** (with diversity penalty):
$$\text{PUCT}(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)} - \beta \cdot \log(1 + \text{freq}(s))$$

Where:
- $Q(s,a)$: Action value estimate
- $P(s,a)$: Prior probability from policy network
- $N(s,a)$: Visit count
- $\text{freq}(s)$: Subtree frequency (prevents mode collapse)

### 3. Reward Function Design

**Intermediate Reward** (per step):
$$R_{\text{inter}} = \text{IC} - \lambda \cdot \frac{1}{k} \sum_{i=1}^k |\text{mutIC}_i| + \text{diversity\_bonus}$$

- **IC**: Information Coefficient (70% daily RankIC + 30% global IC)
- **mutIC**: Mutual correlation with existing alphas (reduces redundancy)
- **diversity_bonus**: $0.1 \times \log(1 + \text{std}(\alpha))$

**Terminal Reward** (complete formula):
$$R_{\text{end}} = \text{composite\_IC} - \lambda_{\text{turnover}} \cdot \text{turnover} - \lambda_{\text{regime}} \cdot \text{var}(\text{IC}_{\text{regime}})$$

Includes turnover penalty and cross-regime stability considerations.

### 4. Risk-Seeking Optimization

Unlike traditional RL that optimizes expected (average) returns, this system uses **quantile regression** to focus on improving top-15% trajectory performance.

**Quantile Update**:
$$q_{t+1} = q_t + \beta \cdot (1 - \alpha - \mathbb{1}_{R(\tau_t) \leq q_t})$$

Negative gradients are applied only when trajectory return $R(\tau) \leq q$, suppressing low-quality formulas.

### 5. Alpha Pool Management

- **Admission Criteria**:
  - IC threshold: 0.005 during cold-start, 0.01 during stable phase
  - Constant detection: std < $10^{-6}$ or unique value ratio < 1%
  
- **Weight Optimization**:
  - Uses Lasso regression for sparse weights
  - Regularization parameter $\alpha = 0.005$
  
- **Capacity Management**:
  - Pool size K = 100
  - Sorted by $|\text{IC} \times \text{weight}|$

---

## 📊 Evaluation Metrics

### Information Coefficient (IC)
$$\text{IC} = \text{Corr}(\text{Prediction}, \text{Actual Returns})$$

Uses Spearman correlation (more robust to outliers)

### Sharpe Ratio
$$\text{Sharpe} = \frac{\sqrt{252} \cdot \mathbb{E}[R - R_f]}{\text{std}(R)}$$

### Maximum Drawdown
$$\text{MaxDD} = \max_{t} \left( \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t} \right)$$

### ICIR (IC Information Ratio)
$$\text{ICIR} = \frac{\mathbb{E}[\text{IC}]}{\text{std}(\text{IC})}$$

Measures IC stability

---

## ⚙️ Configuration

Modify settings in `config/config.py`:

```python
# MCTS parameters
MCTS_CONFIG = {
    "num_iterations": 200,           # Search iterations
    "num_simulations": 200,          # Simulations per iteration
    "c_puct": 1.414,                 # Exploration coefficient (√2)
    "max_episode_length": 30,        # Maximum formula length
}

# Alpha pool parameters
ALPHA_POOL_CONFIG = {
    "pool_size": 100,                # Pool capacity
    "lambda_param": 0.1,             # Redundancy penalty coefficient
    "min_ic_threshold": 0.01,        # IC admission threshold
}

# Policy network parameters
POLICY_CONFIG = {
    "gru_layers": 4,                 # Number of GRU layers
    "gru_hidden_dim": 64,            # Hidden dimension
    "dropout_rate": 0.1,             # Dropout rate
}

# Backtest parameters
BACKTEST_CONFIG = {
    "top_k": 40,                     # Number of stocks to select
    "rebalance_freq": 5,             # Rebalancing frequency (days)
    "transaction_cost": 0.001,       # Transaction cost
}
```

---

## 📈 Output Examples

### Training Log
```
=== Iteration 50/200 ===
Policy network loss: 0.2341
Supervised distillation loss: 0.1876
New formula added: IC=0.0234

=== Training Statistics ===
Alpha pool size: 47
IC distribution: mean=0.0187, std=0.0089
IC range: [0.0051, 0.0421]

Top 5 Alphas by |IC|:
  1. IC=+0.0421 | BEG close ts_mean delta_10 volume div csrank END
  2. IC=+0.0389 | BEG high low sub close div ts_std delta_20 mul END
  3. IC=+0.0356 | BEG vwap ts_rank delta_5 close sub abs END
  ...

Cache hit rate: 78.3% (15672/20000)
Constants filtered: 234
Quantile estimate: 0.0198
```

### Backtest Results
```
=== Backtest Results ===
Cumulative Return: 45.2%
Sharpe Ratio: 1.87
Max Drawdown: 12.3%
ICIR: 0.89

Top Formulas:
  1. IC=0.0421, Weight=0.182 | BEG close ts_mean delta_10 ...
  2. IC=0.0389, Weight=0.156 | BEG high low sub close div ...
```

---

## 🔧 Advanced Usage

### Custom Operators

Add to `core/operators.py`:

```python
@staticmethod
def my_custom_operator(operand, data_length=None, data_index=None):
    """Custom operator description"""
    # Implement your operator logic
    result = operand * 2 + 1
    return result
```

Register in `core/token_system.py`:

```python
TOKEN_DEFINITIONS = {
    ...
    'my_op': Token(TokenType.OPERATOR, 'my_op', arity=1),
}
```

### Using Pretrained Models

```python
from policy.network import PolicyNetwork
import torch

# Load model
model = PolicyNetwork()
model.load_state_dict(torch.load('pretrained_model.pth'))
model.eval()

# Use in trainer
trainer = RiskMinerTrainer(X_data, y_data, device=device)
trainer.policy_network = model
```

### Distributed Training

```python
# Using PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[local_rank])

# Training code remains unchanged
trainer.train(num_iterations=200)
```

---

## 🐛 Troubleshooting

### Q1: High memory usage?
**A**: Enable sampling mode and adjust cache size:
```python
trainer = RiskMinerTrainer(
    X_data, y_data, 
    use_sampling=True,      # Enable sampling
    sample_size=50000       # Sample size
)
# Set cache size in evaluator
evaluator = FormulaEvaluator(cache_size=500)
```

### Q2: How to handle trading halts/missing values?
**A**: The system automatically detects and filters:
- Uses `detect_suspension_periods()` to identify suspensions
- Uses `clean_target_zeros()` to clean anomalous zero-target samples
- Uses `handle_missing_values(strategy='mixed')` for missing values

### Q3: GPU out of memory?
**A**: Reduce batch size or use CPU:
```bash
# Use CPU
python main.py --data_path data.csv

# Or reduce MCTS simulations
# Set num_simulations=50 in config.py
```

### Q4: Discovered formulas are all constants?
**A**: The system has multiple built-in safeguards:
- Variance threshold: `min_std = 1e-6`
- Unique value ratio: `min_unique_ratio = 0.01`
- Coefficient of variation: `cv < 0.001`

If issues persist, adjust thresholds in `ALPHA_POOL_CONFIG`.

---

## 📚 References

This project is inspired by the following papers:

1. **AlphaGo Zero**: Silver, D., et al. (2017). "Mastering the game of Go without human knowledge."
2. **Risk-Seeking RL**: Ahmadi, M., et al. (2021). "Risk-aware reinforcement learning."
3. **Alpha Mining**: Kakushadze, Z., & Yu, W. (2017). "101 formulaic alphas."

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Pandas and NumPy communities for data processing tools
- All contributors and users

---

**⚠️ Disclaimer**: This project is for educational and research purposes only. It does not constitute investment advice. Quantitative trading involves risks; please exercise caution in decision-making.
