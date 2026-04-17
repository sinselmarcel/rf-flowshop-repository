import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path("out") / "ml_dataset_labeled.parquet"
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)


# ---- Laden
P_PARQ = Path("out/ml_dataset_labeled.parquet")
P_CSV  = Path("out/ml_dataset_labeled.csv")
df = pd.read_parquet(P_PARQ) if P_PARQ.exists() else pd.read_csv(P_CSV)

# --- Grundreinigung: NaNs/Inf raus
target = "y_rct"
features = [
    "machine","p_i","rest_from_i","rest_downstream","slack","age",
    "queue_len_m","q_len_total","on_floor","wip_cap","tau","phi"
]

# nur komplette Zeilen verwenden
df = df.replace([np.inf, -np.inf], np.nan)
before = len(df)
df = df.dropna(subset=[target] + features).copy()
print(f"[clean] dropped {before - len(df)} rows (NaN/Inf in target/features)")


# ---- Features/Target
target = "y_rct"
features = [
    "machine","p_i","rest_from_i","rest_downstream","slack","age",
    "queue_len_m","q_len_total","on_floor","wip_cap","tau","phi",
    # Szenariofeatures:
    "due_tight","dist_high","rho_set","K","bn_workload_now"
]
X = df[features].copy()
y = df[target].astype("float32")
groups = df["run_id"]

# ---- Train/Test-Split nach run_id (keine Leaks über gleiche Runs)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

# ---- Modell + grobes Tuning
base = RandomForestRegressor(
    n_estimators=400, max_depth=12, min_samples_leaf=8, n_jobs=-1, random_state=42
)
param_dist = {
    "n_estimators": [300, 400, 500, 700],
    "max_depth": [8, 10, 12, 14, 16],
    "min_samples_leaf": [3, 5, 8, 12, 20],
    "max_features": ["sqrt", 0.6, 0.8, 1.0],
}
search = RandomizedSearchCV(
    base, param_distributions=param_dist, n_iter=20, cv=3, random_state=42,
    n_jobs=-1, verbose=0
)
search.fit(Xtr, ytr)
model = search.best_estimator_

# ---- Bewertung
pred = model.predict(Xte)
mae = mean_absolute_error(yte, pred)
r2  = r2_score(yte, pred)
print(f"[y_rct] MAE={mae:.2f}  R2={r2:.3f}  (best_params={search.best_params_})")

# ---- Speichern (+ Featureliste)
joblib.dump({"model": model, "features": features}, MODELS / "model_rct.pkl")
print("[OK] models/model_rct.pkl")
