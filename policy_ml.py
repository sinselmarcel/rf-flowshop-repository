import numpy as np
import pandas as pd
from pathlib import Path


_RCT_BUNDLE = None
_TARD_BUNDLE = None

def _load_models():
    """
    Loads model_rct.pkl and model_tard.pkl exactly once
    and caches them in _RCT_BUNDLE / _TARD_BUNDLE.
    """    
    global _RCT_BUNDLE, _TARD_BUNDLE
    try:
        import joblib
    except ModuleNotFoundError as e:
        raise RuntimeError from e

    if _RCT_BUNDLE is None:
        b_rct = joblib.load(Path("models/model_rct.pkl"))
        b_tard = joblib.load(Path("models/model_tard.pkl"))
        _RCT_BUNDLE = (b_rct["model"], b_rct["features"])
        _TARD_BUNDLE = (b_tard["model"], b_tard["features"])


def build_candidates(waiting_jobs, now, scenario_state):
    """
    waiting_jobs: List of dictionaries with the following fields:
      job_id, machine, p_i, p_i_rest_from_i, slack, age,
      queue_len_m_on_floor, wip_cap, tau
    """
    df = pd.DataFrame(waiting_jobs)

    # Ensure required fields are filled in
    required = [
        "job_id", "machine", "p_i", "p_i_rest_from_i",
        "slack", "age", "queue_len_m_on_floor", "wip_cap"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = 0

    # tau-Fallback
    if "tau" not in df.columns:
        df["tau"] = max(df["p_i"].mean(), 1e-6)

    # Feature-Engineering
    df = df.rename(columns={
        "p_i_rest_from_i": "rest_from_i",
        "queue_len_m_on_floor": "queue_len_m",
    })
    df["rest_downstream"] = df["rest_from_i"] - df["p_i"]
    df["q_len_total"] = df["queue_len_m"]  

    slack_pos = df["slack"].clip(lower=0).astype("float32")
    tau_safe = df["tau"].clip(lower=1e-6).astype("float32")
    k_atc = 2.0
    df["phi"] = np.exp(-(slack_pos / (k_atc * tau_safe)))

    # Add time and scenario status
    df["time"] = now
    scen = scenario_state or {}

    # Retrieve raw scenario values from scenario_state
    scen_due  = scen.get("scenario_due", None)      # "eng (tight)"/"weit (loose)"
    scen_dist = scen.get("scenario_dist", None)     # "high"/"low"
    rho_set   = scen.get("rho_set", 0.0)
    K_val     = scen.get("K", 0.0)
    bn_work   = scen.get("bn_workload_now", 0.0)

    # Same coding as in training:
    df["due_tight"]  = 1 if scen_due  == "eng"  else 0
    df["dist_high"]  = 1 if scen_dist == "high" else 0
    df["rho_set"]    = float(rho_set)
    df["K"]          = float(K_val)
    df["bn_workload_now"] = float(bn_work)

    # Include only the columns relevant to the model
    cols = [
        "job_id", "time", "machine", "p_i",
        "rest_from_i", "rest_downstream",
        "slack", "age",
        "queue_len_m", "q_len_total",
        "on_floor", "wip_cap", "tau", "phi",
        # neue Features:
        "due_tight","dist_high","rho_set","K","bn_workload_now",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    return df[cols]


def choose_job(model_path, df_candidates):
    """
    Uses two models:
      - model_rct.pkl  → ŷ_rct (Rest-CT)
      - model_tard.pkl → ŷ_tard (Tardiness)

    Lexicographic rule:
      1) If there are jobs with ŷ_tard > 0: choose min(ŷ_tard); in case of a tie, choose min(ŷ_rct)
      2) Otherwise: choose min(ŷ_rct)
    """
    _load_models()
    model_rct, feats = _RCT_BUNDLE
    model_tard, _    = _TARD_BUNDLE   # same list of features

    # Fill in the missing feature columns
    for c in feats:
        if c not in df_candidates.columns:
            df_candidates[c] = 0

    X = df_candidates[feats]

    # Predictions
    yhat_rct  = model_rct.predict(X)
    yhat_tard = model_tard.predict(X)

    job_ids = df_candidates["job_id"].to_numpy()

    # Jobs that are expected to be delayed
    late_mask  = yhat_tard > 0
    idx_late   = np.where(late_mask)[0]
    idx_normal = np.where(~late_mask)[0]

    if len(idx_late) > 0:
        # 1) delayed Jobs: minimal ŷ_tard, Tie-Breaker ŷ_rct
        late_pairs = list(zip(idx_late,
                              yhat_tard[idx_late],
                              yhat_rct[idx_late]))
        late_pairs.sort(key=lambda t: (t[1], t[2]))
        best_idx = late_pairs[0][0]
    else:
        # 2) no one is in danger of being late -> min ŷ_rct
        best_idx = idx_normal[np.argmin(yhat_rct[idx_normal])]


    # --- Return logging/score (for sim_core / policy_scores.csv) ---
    df_out = df_candidates.copy()
    df_out["yhat_rct"] = yhat_rct
    df_out["yhat_tard"] = yhat_tard
    df_out["late_pred"] = (yhat_tard > 0).astype(int)

    # Define score (numerical, for logging):
    # - if “late”: tardiness is the dominant factor, then use RCT as a tiebreaker
    # - otherwise: use only RCT
    SCALE = 10000.0  
    df_out["ml_score"] = np.where(
        yhat_tard > 0,
        -(yhat_tard * SCALE + yhat_rct),
        -yhat_rct
    )

    return int(job_ids[best_idx]), df_out
