# Machine Learning for Stock Return Prediction

A leakage-aware, time-series-aware comparison of four machine-learning model families: **Linear Regression, Support Vector Machines, Random Forests, and a feed-forward Neural Network** for predicting **next-day stock returns** on five S&P 500 equities.

> **Headline result:** under a strict, leakage-controlled workflow, **no model beats the naive baselines**. Out-of-sample R² is negative for every model and directional accuracy sits near (or below) 0.5. This is the *expected and honest* outcome for daily equity return prediction, and the project is built to demonstrate the methodology rather than to manufacture a spurious edge.

---

## Table of contents

- [Motivation](#motivation)
- [Data](#data)
- [Method](#method)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility & leakage controls](#reproducibility--leakage-controls)
- [Limitations](#limitations)
- [References](#references)
- [License](#license)

---

## Motivation

Predicting the direction or magnitude of next-day equity returns is a classic and famously difficult problem. Daily returns are dominated by a near-unpredictable component, so the practical question is rarely *whether a model can fit returns*, but **whether any apparent predictive signal survives time-aware validation, leakage controls, and comparison against trivial baselines.**

A model reporting a high R² on this task is almost always a symptom of a methodological error (look-ahead leakage, a mis-shifted target, or a scaler fitted on the full dataset) rather than genuine skill. The whole pipeline is therefore designed around **not** producing that false positive.

## Data

- **Assets:** AAPL, MSFT, GOOGL (technology), JPM (financials), XOM (energy) five sector-diverse S&P 500 names.
- **Market factor:** the S&P 500 index (`^GSPC`), used as a market-state feature.
- **Period:** 2021-01-04 → 2024-12-30 (~1,004 trading days), spanning the 2022 bear market and subsequent recovery.
- **Source:** downloaded live via [`yfinance`](https://pypi.org/project/yfinance/), auto-adjusted for splits and dividends.

**Target:** next-day simple return, `y_t = r_{t+1}`, built by shifting the return series *before* the split so features at time `t` only use information available up to and including day `t`.

**Features (7, all backward-looking):**

| Feature | Description |
|---|---|
| `return` | current-day return `r_t` |
| `ma10`, `ma50`, `ma200` | 10 / 50 / 200-day price moving averages (trend) |
| `vol10` | 10-day rolling std of returns (volatility) |
| `rsi14` | 14-day Relative Strength Index (Wilder smoothing) |
| `mkt_return` | daily S&P 500 return (systematic market factor) |

The 200-day moving average introduces 200 leading NaN rows per asset; after `dropna()` each asset retains **804 usable observations (603 train / 201 test)**.

## Method

- **Per-asset modelling** : each ticker is trained and evaluated separately (no cross-asset leakage).
- **Chronological 75/25 split** : first 75% of dates for training, final 25% for test, no shuffling.
- **Winsorisation** of `return` and `mkt_return` at the 1st/99th percentiles, thresholds fit on **train only**.
- **Standardisation** with `StandardScaler`, fit on **train only**.
- **Hyperparameter tuning** via `GridSearchCV` with `TimeSeriesSplit(n_splits=5)` expanding-window CV, so every validation fold sits strictly in the future of its training window.
- **Baselines:** constant-zero and training-mean forecasts.
- **Metrics:** MSE, RMSE, MAE, R², Directional Accuracy (DA), and a cross-sectional information coefficient (IC).

### Models

| Model | Notes |
|---|---|
| **Linear Regression** | OLS via scikit-learn for prediction + statsmodels for coefficient significance (t-stats, p-values). |
| **SVR** | Linear/RBF kernel, `C` and `ε` tuned by grid search. Every asset selected a linear kernel. |
| **Random Forest** | Tuned over tree count, depth, min-leaf. Consistently converged to `max_depth = 3` (heavy regularisation). |
| **Neural Network** | Keras: `Input(7) → Dense(64,ReLU) → Dropout(0.2) → Dense(32,ReLU) → Dropout(0.2) → Dense(1,linear)`, Adam (lr=1e-3), MSE loss, early stopping on a chronological 15% validation slice. |

## Results

Aggregate test-set performance, averaged across the five assets:

| Model | MSE | RMSE | MAE | R² | DA |
|---|---|---|---|---|---|
| **Baseline: Train-mean** | 0.000220 | 0.01468 | 0.01054 | −0.0070 | **0.563** |
| Baseline: Zero | 0.000220 | 0.01468 | 0.01057 | −0.0073 | 0.003 |
| Random Forest | 0.000238 | 0.01524 | 0.01118 | −0.0840 | 0.469 |
| SVM | 0.000243 | 0.01540 | 0.01132 | −0.1058 | 0.484 |
| Linear Regression | 0.000274 | 0.01624 | 0.01234 | −0.2193 | 0.424 |
| Neural Network | 0.000361 | 0.01850 | 0.01414 | −0.6926 | 0.453 |

**Takeaways**

- Both **naive baselines achieve the lowest error**; no ML model improves on them.
- Every learned model has **negative out-of-sample R²** and directional accuracy at or below chance.
- Random Forest is the strongest ML model; the Neural Network is the weakest (most parameters relative to a small sample).
- Random Forest predictions **collapse toward ≈0**, the rational response to a near-zero signal-to-noise ratio (see `figures/fig3_actual_vs_predicted.png`).
- OLS p-values are non-significant for almost every feature/asset pair, consistent with the Efficient Market Hypothesis.

### Figures

All figures generated by the pipeline:

| File | Content |
|---|---|
| `figures/fig1_rf_feature_importance.png` | RF feature importance (mean across assets) |
| `figures/fig2_nn_loss_curves.png` | NN train/validation loss curves |
| `figures/fig3_actual_vs_predicted.png` | RF actual-vs-predicted scatter, per asset |
| `figures/fig4_ols_pvalues.png` | OLS coefficient p-value heatmap |

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── pipeline.py              # end-to-end pipeline
└── figures/                     # generated plots
```

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/hameedsoyemi/ml-stock-return-prediction.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt
```

## Usage

```bash
python src/pipeline.py
```

The script downloads the data, trains all models, prints metrics to the console, and displays PNGs. A full run takes a few minutes on CPU (the grid searches and NN training dominate).

To change the assets or period, edit the constants near the top of `src/cf969_sp_pipeline.py`:

```python
ASSETS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
INDEX  = "^GSPC"
START  = "2021-01-01"
END    = "2024-12-31"
```

## Reproducibility & leakage controls

Five deliberate design choices prevent look-ahead leakage:

1. Every rolling feature uses **past observations only**; no future price enters any calculation.
2. The target `y_t = r_{t+1}` is built by shifting the return series **before** the split.
3. The train/test split is **chronological with no shuffling**.
4. `StandardScaler` and winsorisation thresholds are **fit on training data only**, then applied to test with fixed parameters.
5. `TimeSeriesSplit` guarantees every validation fold is **temporally after** its training window.

Random seeds are fixed for NumPy, TensorFlow, and every scikit-learn estimator so runs reproduce. (Note: `yfinance` data can be revised over time, so tiny numeric drift on a future re-download is possible.)

## Limitations

- Five US large-caps over a single 2021–2024 window; other sectors, cap tiers, or regimes may differ.
- A single 75/25 split tests one regime pair, a walk-forward / rolling out-of-sample evaluation would be more robust.
- Purely technical features; no fundamentals, macro (VIX, rates), or sentiment.
- **No trading simulation.** A favourable error metric is not a tradeable signal, any apparent edge would have to survive transaction costs and turnover first.

## References

1. E. F. Fama, "Efficient capital markets: A review of theory and empirical work," *Journal of Finance*, 25(2), 383–417, 1970.
2. S. Gu, B. Kelly, D. Xiu, "Empirical asset pricing via machine learning," *Review of Financial Studies*, 33(5), 2223–2273, 2020.
3. T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.
4. M. Lopez de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

