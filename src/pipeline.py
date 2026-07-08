import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf

from scipy.stats import mstats, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

ASSETS   = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
INDEX    = '^GSPC'
START    = '2021-01-01'
END      = '2024-12-31'
SPLIT    = 0.75
print("Libraries loaded")

# downloading the data
raw = yf.download(ASSETS + [INDEX], start=START, end=END, auto_adjust=True, progress=False)
prices = raw['Close']
prices.columns.name = None
print(f"Downloaded {len(prices)} trading days, {prices.shape[1]} tickers")
print(f"Date range: {prices.index[0].date()} - {prices.index[-1].date()}")
print("\nMissing values per ticker:")
print(prices.isnull().sum())

# feature engineering
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

mkt_ret = prices[INDEX].pct_change().rename('mkt_return')

asset_data = {}
nan_report = {}

for ticker in ASSETS:
    p   = prices[ticker]
    ret = p.pct_change().rename('return')

    df = pd.DataFrame({
        'return'    : ret,
        'ma10'      : p.rolling(10).mean(),
        'ma50'      : p.rolling(50).mean(),
        'ma200'     : p.rolling(200).mean(),
        'vol10'     : ret.rolling(10).std(),
        'rsi14'     : compute_rsi(p, 14),
        'mkt_return': mkt_ret,
    })

    # Target: next-day return
    df['target'] = df['return'].shift(-1)

    # NaN handling
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after  = len(df)

    nan_report[ticker] = {'rows_before_dropna': rows_before,
                          'rows_dropped'      : rows_before - rows_after,
                          'rows_kept'         : rows_after}
    asset_data[ticker] = df

nan_df = pd.DataFrame(nan_report).T
print("NaN handling summary:")
print(nan_df.to_string())

FEATURES = ['return', 'ma10', 'ma50', 'ma200', 'vol10', 'rsi14', 'mkt_return']
WINSORISE = ['return', 'mkt_return']

# Preprocessing pipeline
def preprocess(df):
    n        = len(df)
    cut      = int(n * SPLIT)
    train    = df.iloc[:cut].copy()
    test     = df.iloc[cut:].copy()

    # Winsorise using training set only
    for col in WINSORISE:
        lo, hi = np.percentile(train[col], [1, 99])
        train[col] = train[col].clip(lo, hi)
        test[col]  = test[col].clip(lo, hi)

    # Standardise using training set only
    scaler = StandardScaler()
    train[FEATURES] = scaler.fit_transform(train[FEATURES])
    test[FEATURES]  = scaler.transform(test[FEATURES])

    X_train, y_train = train[FEATURES].values, train['target'].values
    X_test,  y_test  = test[FEATURES].values,  test['target'].values

    return X_train, y_train, X_test, y_test, scaler, train.index, test.index

splits = {ticker: preprocess(asset_data[ticker]) for ticker in ASSETS}

# Printing out split sizes using APPLE asset as an example
print("Train/test split sizes:")
for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, _, _ = splits[ticker]
    print(f"  {ticker:6s} train: {len(X_tr)} obs | test: {len(X_te)} obs")

# evaluation metrics
def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def information_coefficient(y_true, y_pred):
    ic, _ = spearmanr(y_true, y_pred)
    return ic

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE' : mse,
        'RMSE': np.sqrt(mse),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'R2'  : r2_score(y_true, y_pred),
        'DA'  : directional_accuracy(y_true, y_pred),
        'IC'  : information_coefficient(y_true, y_pred),
    }

def baseline_metrics(y_train, y_test):
    zero_pred  = np.zeros_like(y_test)
    mean_pred  = np.full_like(y_test, y_train.mean())
    return {
        'Baseline: Zero'     : evaluate(y_test, zero_pred),
        'Baseline: TrainMean': evaluate(y_test, mean_pred),
    }

# model 1 - Linear Regression (OLS)
# This also includes statsmodels coefficient significance table

ols_summaries = {}
lr_preds      = {}

for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, tr_idx, te_idx = splits[ticker]

    # sklearn for predictions
    lr = LinearRegression().fit(X_tr, y_tr)
    lr_preds[ticker] = (y_te, lr.predict(X_te), te_idx)

    # statsmodels for significance
    X_tr_sm = sm.add_constant(X_tr)
    ols     = sm.OLS(y_tr, X_tr_sm).fit()
    ols_summaries[ticker] = ols

# Print coefficient table for AAPL
print("OLS Coefficient Table - AAPL ")
coef_df = pd.DataFrame({
    'Feature'    : ['const'] + FEATURES,
    'Coef'       : ols_summaries['AAPL'].params,
    'Std Err'    : ols_summaries['AAPL'].bse,
    't-stat'     : ols_summaries['AAPL'].tvalues,
    'p-value'    : ols_summaries['AAPL'].pvalues,
    'Significant': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
                    for p in ols_summaries['AAPL'].pvalues]
})
print(coef_df.to_string(index=False))
print(f"\nAdj. R2: {ols_summaries['AAPL'].rsquared_adj:.4f}")

# model 2 - Support Vector Machine
# using GridSearch with TimeSeriesSplit (time-aware, no random k-fold)

svm_preds        = {}
svm_best_params  = {}
tscv             = TimeSeriesSplit(n_splits=5)

param_grid_svm = {
    'kernel' : ['linear', 'rbf'],
    'C'      : [0.1, 1, 10],
    'epsilon': [0.001, 0.01, 0.1],
}

for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, tr_idx, te_idx = splits[ticker]
    gs = GridSearchCV(SVR(), param_grid_svm, cv=tscv,
                      scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    svm_preds[ticker]       = (y_te, gs.best_estimator_.predict(X_te), te_idx)
    svm_best_params[ticker] = gs.best_params_
    print(f"{ticker}: best params = {gs.best_params_}  | CV MSE = {-gs.best_score_:.6f}")

# model 3 - Random Forest
# hyperparam tuning via TimeSeriesSplit

rf_preds        = {}
rf_best_params  = {}
rf_importances  = {}

param_grid_rf = {
    'n_estimators'    : [100, 200],
    'max_depth'       : [3, 5, None],
    'min_samples_leaf': [5, 10],
}

for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, tr_idx, te_idx = splits[ticker]
    gs = GridSearchCV(RandomForestRegressor(random_state=42),
                      param_grid_rf, cv=tscv,
                      scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    rf_preds[ticker]       = (y_te, gs.best_estimator_.predict(X_te), te_idx)
    rf_best_params[ticker] = gs.best_params_
    rf_importances[ticker] = gs.best_estimator_.feature_importances_
    print(f"{ticker}: best params = {gs.best_params_}  | CV MSE = {-gs.best_score_:.6f}")

# extracting feature importances
avg_imp = pd.Series(
    np.mean(list(rf_importances.values()), axis=0),
    index=FEATURES
).sort_values(ascending=False)
print("\nAverage feature importances (across all assets):")
print(avg_imp.round(4).to_string())

# model 4 - neural network (simple feed-forward baseline)

nn_preds    = {}
nn_histories= {}

def build_nn(input_dim=7):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, tr_idx, te_idx = splits[ticker]

    # Chronological 15% validation slice (no shuffle)
    val_cut    = int(len(X_tr) * 0.85)
    X_t, X_v   = X_tr[:val_cut], X_tr[val_cut:]
    y_t, y_v   = y_tr[:val_cut], y_tr[val_cut:]

    tf.random.set_seed(42)
    model = build_nn(X_tr.shape[1])
    hist  = model.fit(X_t, y_t, validation_data=(X_v, y_v),
                      epochs=200, batch_size=32,
                      callbacks=[es], verbose=0)
    nn_preds[ticker]     = (y_te, model.predict(X_te, verbose=0).flatten(), te_idx)
    nn_histories[ticker] = hist.history
    stopped = len(hist.history['loss'])
    print(f"{ticker}: stopped at epoch {stopped} | val_loss = {hist.history['val_loss'][-1]:.6f}")

# cross-validation fold results (TimeSeriesSplit k=5) on the training set only.
cv_results = {}

models_for_cv = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=5,
                                               min_samples_leaf=5, random_state=42)
}

for ticker in ASSETS:
    X_tr, y_tr, _, _, _, _, _ = splits[ticker]
    cv_results[ticker] = {}
    for name, model in models_for_cv.items():
        fold_mses = []
        for tr_idx, val_idx in tscv.split(X_tr):
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            pred = model.predict(X_tr[val_idx])
            fold_mses.append(mean_squared_error(y_tr[val_idx], pred))
        cv_results[ticker][name] = {'mean_MSE': np.mean(fold_mses),
                                    'std_MSE' : np.std(fold_mses)}

# Summary
print("Cross-validation MSE (mean +/- std) per asset:")
for ticker in ASSETS:
    for name in models_for_cv:
        m = cv_results[ticker][name]['mean_MSE']
        s = cv_results[ticker][name]['std_MSE']
        print(f"  {ticker} | {name:25s}: {m:.6f} +/- {s:.6f}")

# test set results: baselines + four models
all_preds = {
    'Linear Regression': lr_preds,
    'SVM'              : svm_preds,
    'Random Forest'    : rf_preds,
    'Neural Network'   : nn_preds,
}

per_asset_rows = []

for ticker in ASSETS:
    X_tr, y_tr, X_te, y_te, _, _, _ = splits[ticker]
    bm = baseline_metrics(y_tr, y_te)
    for bname, bmet in bm.items():
        per_asset_rows.append({'Asset': ticker, 'Model': bname, **bmet})
    for mname, preds in all_preds.items():
        y_true, y_pred, _ = preds[ticker]
        per_asset_rows.append({'Asset': ticker, 'Model': mname,
                                **evaluate(y_true, y_pred)})

per_asset_df = pd.DataFrame(per_asset_rows)

# Aggregate (mean across assets), sorted by MSE
agg = per_asset_df.groupby('Model')[['MSE','RMSE','MAE','R2','DA','IC']].mean().sort_values('MSE')
print("Aggregate (mean across 5 assets)")
print(agg.round(6).to_string())

print("\nPer-Asset MSE Matrix")
mse_pivot = per_asset_df.pivot_table(index='Asset', columns='Model', values='MSE')
print(mse_pivot.round(6).to_string())

print("\nPer-Asset Directional Accuracy Matrix")
da_pivot = per_asset_df.pivot_table(index='Asset', columns='Model', values='DA')
print(da_pivot.round(3).to_string())

# Some visualizations

# Fig 1: Random Forest Feature Importance (average across assets)
fig, ax = plt.subplots(figsize=(8, 4))
imp_df = pd.DataFrame(rf_importances, index=FEATURES).T
mean_imp = imp_df.mean().sort_values(ascending=True)
mean_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_xlabel('Mean Impurity Decrease')
ax.set_title('RF Feature Importance - Average Across All Assets')
plt.tight_layout()
plt.savefig('fig1_rf_feature_importance.png', dpi=150)
plt.show()
print("Saved fig1_rf_feature_importance.png")

# Fig 2: NN Training vs Validation Loss - AAPL & JPM
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, ticker in enumerate(['AAPL', 'JPM']):
    hist = nn_histories[ticker]
    ax = axes[i]
    ax.plot(hist['loss'],     label='Train loss', linewidth=1.5)
    ax.plot(hist['val_loss'], label='Val loss',   linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title(f'NN Loss Curve - {ticker}')
    ax.legend()
    ax.set_ylim(bottom=0)
plt.suptitle('Neural Network Training vs Validation Loss', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig2_nn_loss_curves.png', dpi=150)
plt.show()
print("Saved fig2_nn_loss_curves.png")

# Fig 3: Actual vs Predicted - Best Model (Random Forest) per asset
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=False)
for i, ticker in enumerate(ASSETS):
    y_true, y_pred, te_idx = rf_preds[ticker]
    ax = axes[i]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color='steelblue')
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='Perfect fit')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted')
    ax.set_title(ticker)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f'R2={r2:.3f}', transform=ax.transAxes, fontsize=8)
plt.suptitle('Random Forest: Actual vs Predicted Returns (Test Set)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('fig3_actual_vs_predicted.png', dpi=150)
plt.show()
print("Saved fig3_actual_vs_predicted.png")

# Fig 4: OLS Coefficient p-values across all assets
pval_data = {ticker: ols_summaries[ticker].pvalues[1:]
             for ticker in ASSETS}
pval_df = pd.DataFrame(pval_data, index=FEATURES)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(pval_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'p-value'})
ax.set_title('OLS Coefficient p-values by Feature and Asset', fontsize=13)
ax.set_xlabel('Asset')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('fig4_ols_pvalues.png', dpi=150)
plt.show()
print("Saved fig4_ols_pvalues.png")

# best hyperparams summary
print("Best SVM Hyperparameters ")
svm_hp_df = pd.DataFrame(svm_best_params).T
print(svm_hp_df.to_string())

print("\nBest Random Forest Hyperparameters")
rf_hp_df = pd.DataFrame(rf_best_params).T
print(rf_hp_df.to_string())

print("\nCross-validation MSE table (LR vs RF)")
cv_rows = []
for ticker in ASSETS:
    row = {'Asset': ticker}
    for name in models_for_cv:
        row[name] = f"{cv_results[ticker][name]['mean_MSE']:.6f} +/- {cv_results[ticker][name]['std_MSE']:.6f}"
    cv_rows.append(row)
print(pd.DataFrame(cv_rows).to_string(index=False))

