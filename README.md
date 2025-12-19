# Rimi3: Automated Alpha Factor Discovery via Distributional Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **An end-to-end factor mining system combining Monte Carlo Tree Search (MCTS) with Risk-Seeking Reinforcement Learning to discover statistically significant alpha signals.**

---

## 🎯 Research Objective

This project addresses two fundamental challenges in systematic alpha research:

| Challenge | Traditional Approach | Rimi3 Solution |
|-----------|---------------------|----------------|
| **Formula Space Explosion** | Genetic programming generates syntactically invalid expressions, wasting 40-60% of computational budget | RPN-based DSL with stack validation guarantees **100% syntactic validity** |
| **Mediocre Factor Problem** | Standard RL optimizes $\mathbb{E}[R]$, producing average-quality factors | Quantile Regression targets the **top-15% of IR distribution**, focusing on tail-alpha |

### Why This Matters

In quantitative finance, the difference between a mediocre factor (IC ≈ 0.02) and a strong factor (IC ≈ 0.05) can translate to **hundreds of basis points** in annual alpha. By biasing the search toward extreme-quality outcomes, Rimi3 discovers factors that would be missed by expected-value optimization.

---

## 📊 Methodology: Evaluation Framework

### Why Cross-Sectional IC Instead of P&L Backtest?

> *"In factor research, premature backtesting conflates signal quality with portfolio construction choices."*

| Evaluation Method | Pros | Cons |
|-------------------|------|------|
| **P&L Backtest** | Intuitive, end-to-end | Sensitive to weighting, rebalancing, transaction costs; prone to overfitting |
| **Cross-Sectional IC** | Isolates signal quality; robust to implementation details | Doesn't capture execution costs |

**Our Approach**: Use IC/ICIR as the primary optimization target during factor discovery, reserving backtesting for final validation. This separation follows industry best practices at leading quantitative firms.

### Information Coefficient (IC) Definition

$$\text{IC}_t = \text{Spearman}\big(\text{Factor}_t, \text{Return}_{t+1:t+k}\big)$$

- **Daily Rank IC**: Cross-sectional correlation on each trading day
- **ICIR (Information Ratio)**: $\frac{\mathbb{E}[\text{IC}]}{\text{std}(\text{IC})}$ — measures signal consistency

---

## 🔬 Key Results

### Performance Summary

| Metric | Value | Benchmark | Interpretation |
|--------|-------|-----------|----------------|
| **Mean OOS Rank IC** | 0.045 | >0.02 tradeable | Strong predictive signal |
| **ICIR** | 1.15 | >1.0 significant | Consistent across time periods |
| **Hit Rate** | 87-90% | >50% random | Positive IC on most trading days |
| **t-statistic** | 23-28 | >2.0 significant | Highly statistically significant |
| **Factor Redundancy** | -65% | — | Via Lasso orthogonalization |
| **Evaluation Throughput** | 100k+ evals | — | 50× speedup from baseline |

### Factor Library Statistics


<p align="center">
  <img width="4819" height="1468" alt="7_summary_stats" src="https://github.com/user-attachments/assets/6d443930-9d9b-4512-a633-e9c2f43742d6" />
</p>
---

## 📈 Visualizations

### Comprehensive Analysis Dashboard


<p align="center">
  <img width="5341" height="4931" alt="8_rimi3_dashboard" src="https://github.com/user-attachments/assets/5e64d7e0-e237-47ef-b04e-373c932ca1d6" />
</p>

*Dashboard includes: (A) Cumulative IC trajectory, (B) Factor correlation matrix, (C) IC decay analysis, (D) ICIR comparison, (E) Hit rate analysis, (F) Quintile return spreads.*

### IC Time Series Analysis

The following visualization shows the out-of-sample Information Coefficient for each discovered factor over the 2023-2025 period. All factors maintain stable positive IC with **ICIR > 1.0**.


<p align="center">
  <img width="4164" height="4605" alt="1_ic_time_series" src="https://github.com/user-attachments/assets/09eba5da-dabc-4d2c-b028-26ac389d4ac5" />
</p>

**Key Observations:**
- Rolling 21-day IC remains consistently positive across market regimes
- Confidence bands (±1σ) stay above zero for >85% of the evaluation period
- Regime changes (e.g., 2024-Q1) show temporary IC compression but rapid recovery

### Factor Orthogonality


<p align="center">
  <img width="2633" height="2378" alt="2_factor_correlation" src="https://github.com/user-attachments/assets/51dc6245-67f6-45a3-89b4-39fdc54f36c8" />
</p>

Factor correlation analysis demonstrates **low inter-factor correlation** (average |ρ| = 0.125), ensuring the ensemble provides incremental alpha rather than redundant signals.



---

## 🏗️ System Architecture

```
Rimi3/
├── core/                      # Core Components
│   ├── token_system.py        # 58-token DSL with RPN validation
│   ├── rpn_evaluator.py       # Stack-based expression evaluator
│   └── operators.py           # 70+ operator implementations
│
├── mcts/                      # Monte Carlo Tree Search
│   ├── node.py                # PUCT selection with diversity penalty
│   ├── searcher.py            # Parallel MCTS with virtual loss
│   └── environment.py         # MDP state transitions
│
├── policy/                    # Policy Network
│   ├── network.py             # 4-layer GRU with attention
│   └── optimizer.py           # Risk-seeking quantile optimizer
│
└── alpha/                     # Alpha Management
    ├── evaluator.py           # LRU-cached formula evaluation
    └── pool.py                # Lasso-weighted ensemble

```

---

## 🧠 Core Mechanisms

### 1. RPN-Based Domain Specific Language

The system uses **Reverse Polish Notation** to represent alpha factors, ensuring grammatical correctness through stack-based validation.

```python
# Traditional Infix:  (close - open) / open
# Rimi3 RPN:          BEG close open sub open div END

# Volume-Adjusted Momentum
# RPN: BEG close close delta_5 delay div rank volume ts_std delta_20 neg mul END
```

**Why RPN?**
1. **O(n) Evaluation**: Stack-based computation, no recursion
2. **Incremental Validation**: At each MCTS step, only valid tokens are available
3. **No Parentheses**: Eliminates ambiguity and parsing overhead

### 2. Risk-Seeking Optimization via Quantile Regression

Unlike standard RL that optimizes expected returns, we use **Distributional RL** to focus on the right tail of the outcome distribution.

**Standard RL Objective:**
$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

**Risk-Seeking Objective (Ours):**
$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau) \mid R(\tau) \geq q_\alpha\big]$$

Where $q_\alpha$ is the $(1-\alpha)$-quantile of returns, updated online:

$$q_{t+1} = q_t + \beta \cdot \big(1 - \alpha - \mathbb{1}_{R(\tau_t) \leq q_t}\big)$$

**Effect**: Policy gradients are only applied when trajectory return exceeds the quantile threshold, suppressing mediocre factors and amplifying exceptional ones.

### 3. PUCT Selection with Diversity Penalty

To prevent mode collapse (discovering the same factor repeatedly), we augment the standard PUCT formula:

$$\text{PUCT}(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)} - \beta \cdot \log(1 + \text{freq}(s))$$

Where:
- $Q(s,a)$: Action value estimate from rollouts
- $P(s,a)$: Prior probability from policy network
- $\text{freq}(s)$: Subtree visit frequency (novelty penalty)

### 4. Reward Function Design

**Multi-Objective Reward:**

$$R = \underbrace{\frac{\text{mean}(\text{IC})}{\text{std}(\text{IC})}}_{\text{ICIR}} \times \underbrace{(1 - \lambda \cdot |\rho_{\text{pool}}|)}_{\text{Orthogonality}} + \underbrace{\gamma \cdot \log(1 + \sigma_\alpha)}_{\text{Diversity Bonus}}$$

| Component | Purpose | Weight |
|-----------|---------|--------|
| ICIR | Reward consistent predictive power | Primary |
| Orthogonality | Penalize correlation with existing factors | λ = 0.1 |
| Diversity Bonus | Encourage varied factor distributions | γ = 0.1 |

---

## 🛠️ Technical Highlights

### Numerical Stability Protocol

Quantitative systems require extreme numerical care. Our protocol:

```python
# Division safety
def safe_div(a, b, eps=1e-10):
    return np.where(np.abs(b) > eps, a / b, 0.0)

# Log safety  
def safe_log(x, eps=1e-10):
    return np.log(np.maximum(np.abs(x), eps))

# Outlier clipping
def clip_extreme(x, n_sigma=3):
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return np.clip(x, mu - n_sigma*sigma, mu + n_sigma*sigma)
```

**Results**: 100k+ factor evaluations with **zero NaN propagation**.

### Performance Optimization

| Optimization | Speedup | Implementation |
|--------------|---------|----------------|
| LRU Caching | 5× | Memoize intermediate computations |
| Vectorized Operators | 10× | NumPy broadcasting over pandas loops |
| Precomputed Rolling | 3× | Cache `ts_mean`, `ts_std` for common windows |
| **Total** | **50×** | 15s → 0.3s per factor evaluation |



---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/rimi3.git
cd rimi3
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train factor discovery system
python main.py --data_path data/sp500.csv --target_column fwd_ret_5d

# Run with cross-validation
python main.py --data_path data/sp500.csv --cross_validate --n_folds 5

# GPU acceleration
python main.py --data_path data/sp500.csv --gpu_id 0
```

### Configuration

Key parameters in `config/config.py`:

```python
MCTS_CONFIG = {
    "num_iterations": 200,        # Search iterations
    "num_simulations": 200,       # Rollouts per iteration  
    "c_puct": 1.414,              # Exploration coefficient
    "max_formula_length": 30,     # Maximum RPN tokens
}

RISK_SEEKING_CONFIG = {
    "quantile_alpha": 0.85,       # Target top-15% of outcomes
    "quantile_lr": 0.01,          # Quantile update rate
}

ALPHA_POOL_CONFIG = {
    "pool_size": 100,             # Maximum factors to retain
    "min_ic_threshold": 0.01,     # Admission criteria
    "lasso_alpha": 0.005,         # Ensemble regularization
}
```

---

## 📚 References

This project builds upon foundational work in:

**Reinforcement Learning:**
- Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." *Nature*.
- Dabney, W., et al. (2018). "Distributional Reinforcement Learning with Quantile Regression." *AAAI*.

**Quantitative Finance:**
- Kakushadze, Z., & Yu, W. (2017). "101 Formulaic Alphas." *SSRN*.
- Tulchinsky, I. (2019). *Finding Alphas: A Quantitative Approach to Building Trading Strategies*. Wiley.

**Risk-Aware Learning:**
- Greenberg, I., et al. (2022). "Efficient Risk-Averse Reinforcement Learning." *NeurIPS*.

---

## 🔮 Future Work

- [ ] **Multi-Asset Extension**: Extend DSL to support cross-asset signals
- [ ] **Transaction Cost Modeling**: Integrate market impact models into reward
- [ ] **Online Learning**: Adapt factors to regime changes in real-time
- [ ] **Interpretability Module**: SHAP-based factor attribution

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute investment advice. Quantitative trading involves substantial risk of loss. Past performance (including simulated results) does not guarantee future results.

---

<p align="center">
  <i>Built with ❤️ for the quantitative finance research community</i>
</p>
