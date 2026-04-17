import pandas as pd
from pathlib import Path
import numpy as np

def find_first(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

def main():
    out_dir = Path("out"); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inputs finden ----
    ds_path = find_first(["out/ml_dataset.csv"])
    if not ds_path:
        raise FileNotFoundError("ml_dataset*.csv nicht gefunden in out/")
    dfx = pd.read_csv(ds_path)

    job_path = find_first(["out/job.csv", "out/jobs.csv", "out/job_log.csv"])
    if not job_path:
        raise FileNotFoundError("Job-Log (job*.csv) nicht gefunden in out/")
    dfj = pd.read_csv(job_path)

    # --- Kompatibilität: completion/release/job_id angleichen ---
    if "t_done" not in dfj.columns and "completion" in dfj.columns:
        dfj = dfj.rename(columns={"completion": "t_done"})
    if "t_release" not in dfj.columns and "release" in dfj.columns:
        dfj = dfj.rename(columns={"release": "t_release"})
    if "job_id" not in dfj.columns and "job" in dfj.columns:
        dfj = dfj.rename(columns={"job": "job_id"})

    # ---- Label y_rct, y_tard, y_late ----

    # falls die Spalte anders heißt (z.B. 'due_date'), hier anpassen:
    if "due" not in dfj.columns and "due_date" in dfj.columns:
        dfj = dfj.rename(columns={"due_date": "due"})

    # WICHTIG: run_id + job_id, sonst Zeilen-Vervielfachung
    comp = dfj[["run_id", "job_id", "t_done"]].drop_duplicates()

    # Job-ID an ml_dataset angleichen
    dfx = dfx.rename(columns={"job": "job_id"}) if "job_id" not in dfx.columns and "job" in dfx.columns else dfx
    if "time" not in dfx.columns:
        raise KeyError("Spalte 'time' (Entscheidungszeit) fehlt in ml_dataset.csv")
    if "due_minus_now" not in dfx.columns:
        raise KeyError("Spalte 'due_minus_now' fehlt in ml_dataset.csv")

    # Merge job-Fertigstellungszeiten hinein
    dfx = dfx.merge(comp, on=["run_id", "job_id"], how="left")

    # Restdurchlaufzeit (ab Entscheidungszeit)
    dfx["y_rct"] = (dfx["t_done"] - dfx["time"]).astype("float32")

    # Absolutes Due Date aus time + due_minus_now
    dfx["due"] = dfx["time"] + dfx["due_minus_now"]

    # Tardiness = max(0, Fertigstellung - Due Date)
    dfx["y_tard"] = (dfx["t_done"] - dfx["due"]).clip(lower=0.0).astype("float32")

    # Hilfsspalten wieder weg
    dfx = dfx.drop(columns=["t_done", "due"])

    # --- Szenario-Features numerisch kodieren ---
    if "scenario_due" in dfx.columns:
        dfx["due_tight"] = dfx["scenario_due"].map({"eng": 1, "weit": 0}).astype("int8")

    if "scenario_dist" in dfx.columns:
        dfx["dist_high"] = dfx["scenario_dist"].map({"high": 1, "low": 0}).astype("int8")

    for col in ["rho_set", "K", "bn_workload_now"]:
        if col in dfx.columns:
            dfx[col] = dfx[col].astype("float32")

    # --- phi & Rest-Infos wie bisher ---
    k_atc = 2.0

    if "tau" not in dfx.columns:
        dfx["tau"] = max(dfx["p_i"].mean(), 1e-6)

    slack_pos = dfx["slack"].clip(lower=0).astype("float32")
    tau_safe  = dfx["tau"].clip(lower=1e-6).astype("float32")
    dfx["phi"] = np.exp(-(slack_pos / (k_atc * tau_safe)))

    dfx["q_len_total"]     = dfx["queue_len_m"]
    dfx["rest_downstream"] = dfx["rest_from_i"] - dfx["p_i"]


    # ---- Spaltenauswahl (schlank!) ----
    keep = [
        "run_id","decision_id","time","job_id","machine",
        "p_i","rest_from_i","rest_downstream","slack","age",
        "queue_len_m","q_len_total","on_floor","wip_cap","tau",
        "phi",
        # neue Szenariofeatures:
        "due_tight","dist_high","rho_set","K","bn_workload_now",
        # Targets / Meta:
        "selected","y_rct","y_tard"
    ]
    keep = [c for c in keep if c in dfx.columns]
    dfx = dfx[keep]


    # ---- Downcast für kleinere Dateien ----
    dfx["machine"] = dfx["machine"].astype("int8")
    for c in ["p_i","rest_from_i","rest_downstream","slack","age",
          "tau","phi","y_rct","y_tard","rho_set","K","bn_workload_now"]:
        if c in dfx.columns:
            dfx[c] = dfx[c].astype("float32")
    for c in ["queue_len_m","q_len_total","on_floor","wip_cap",
          "selected","due_tight","dist_high"]:
        if c in dfx.columns:
            dfx[c] = dfx[c].astype("int16")

    # ---- Speichern + Log ----
    out_path_parq = out_dir / "ml_dataset_labeled.parquet"
    out_path_csv  = out_dir / "ml_dataset_labeled.csv"

    try:
        import pyarrow  # nur um klaren Fehler zu bekommen, falls nicht installiert
        dfx.to_parquet(out_path_parq, index=False)
        print(f"[OK] {out_path_parq} rows={len(dfx):,}")
    except Exception as e:
        # Fallback auf CSV (kein Parquet-Engine installiert o.ä.)
        dfx.to_csv(out_path_csv, index=False)
        print(f"[OK] {out_path_csv} rows={len(dfx):,}  [Parquet fallback: {e.__class__.__name__}]")

if __name__ == "__main__":
    main()
