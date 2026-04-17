# scripts/quick_summary.py
"""
Extended quick_summary:
- reads all runs/**/kpi_summary.json
- exports:
  1) out/kpi_runs_<subset>.csv                 (run-level: scenario×policy×seed)
  2) out/quick_summary_<subset>.csv            (backwards-compatible means for main KPIs)
  3) out/kpi_stats_<subset>.csv                (mean/median/std/q25/q75/min/max per scenario×policy)
  4) out/wins_ranks_<subset>.csv               (wins + avg rank per KPI like your Table 6-2)
  5) out/ml_vs_best_heur_<subset>.csv          (ML vs best heuristic per scenario, incl. Δ% “improvement”)
  6) out/factor_delta_<subset>.csv             (Δ% aggregated by due/dist/K for sensitivity)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
KPI_FILE = "kpi_summary.json"
POLICIES = {"SPT", "EDD", "ATC", "ML"}

TRAIN_SEEDS = {"42", "1337", "2025", "2601", "31415", "2718", "73", "99", "808", "111"}
TEST_SEEDS  = {"222", "333", "444", "555", "666", "777", "888", "999", "1212", "1717"}

# KPIs you currently use in Chapter 6 tables (keep names consistent with kpi_summary.json)
MAIN_KPIS = {
    "on_time_rate": "ontime",
    "tardiness_p95": "tard_p95",
    "ct_mean": "ct_mean",
    "ct_p50": "ct_p50",
    "ct_p95": "ct_p95",
    "wip_avg": "wip_avg",
    "wip_max": "wip_max",
    "throughput_per_min": "thr_per_min",
}

# Define for ranking/delta which KPIs are "maximize" vs "minimize"
# (positive Δ% should mean "ML better" in both cases)
MAXIMIZE_KPIS = {"on_time_rate", "throughput_per_min"}
MINIMIZE_KPIS = {"tardiness_p95", "ct_mean", "ct_p95", "wip_avg", "wip_max"}  # ct_p50 not used for ranks by default

RANK_KPIS = ["on_time_rate", "tardiness_p95", "ct_mean", "ct_p95", "wip_avg", "wip_max", "throughput_per_min"]


# -----------------------------
# Helpers
# -----------------------------
SCEN_RE = re.compile(
    r"util(?P<util>\d+)_due(?P<due>eng|weit)_K(?P<K>\d+)_dist(?P<dist>low|high)",
    re.IGNORECASE,
)

def parse_scenario_factors(scenario: str) -> Dict[str, Optional[str]]:
    """
    Extract factors from scenario name like:
      util95_dueeng_K3_disthigh
    """
    m = SCEN_RE.search(scenario)
    if not m:
        return {"util": None, "due": None, "K": None, "dist": None}
    d = m.groupdict()
    return {
        "util": d["util"],
        "due": d["due"].lower(),
        "K": d["K"],
        "dist": d["dist"].lower(),
    }

def safe_seed_from_part(seed_part: str) -> str:
    # accepts "seed999", "Seed999", etc.
    s = seed_part.lower()
    if s.startswith("seed"):
        return s.replace("seed", "").strip()
    return s.strip()

def find_policy(parts: Tuple[str, ...]) -> Optional[str]:
    return next((p for p in parts if p in POLICIES), None)

def find_seed_part(parts: Tuple[str, ...]) -> Optional[str]:
    return next((p for p in parts if p.lower().startswith("seed")), None)

def ci95_from_std(n: int, std: float) -> float:
    if n <= 1 or pd.isna(std):
        return float("nan")
    return 1.96 * (std / (n ** 0.5))

def to_excel_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=";", decimal=",")

def quantile_agg(q: float):
    return lambda x: x.quantile(q)


# -----------------------------
# Core computations
# -----------------------------
def compute_wins_ranks(df_mean: pd.DataFrame) -> pd.DataFrame:
    """
    df_mean: rows are scenario×policy with mean KPI columns (original KPI names).
    Output: one row per KPI with wins and avg rank per policy.
    """
    policies_present = sorted(df_mean["policy"].unique().tolist(), key=lambda x: ["SPT","EDD","ATC","ML"].index(x) if x in ["SPT","EDD","ATC","ML"] else 999)

    rows = []
    for kpi in RANK_KPIS:
        # build scenario×policy matrix
        pivot = df_mean.pivot_table(index="scenario", columns="policy", values=kpi, aggfunc="mean")

        # rank per scenario (lower rank=better)
        ascending = (kpi in MINIMIZE_KPIS)  # minimize => ascending=True
        # optional: stabilisiert Throughput-Ties (empfohlen)
        if kpi == "throughput_per_min":
            pivot = pivot.round(5)

        ranks = pivot.rank(axis=1, method="min", ascending=ascending)


        # wins: count best in each scenario (handle ties by counting all with rank==1)
        wins = (ranks == 1).sum(axis=0)

        # avg rank
        avg_rank = ranks.mean(axis=0)

        row = {"KPI": kpi}
        for p in policies_present:
            row[f"Wins_{p}"] = float(wins.get(p, 0))
        for p in policies_present:
            row[f"AvgRank_{p}"] = float(avg_rank.get(p, float("nan")))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_ml_vs_best(df_mean: pd.DataFrame) -> pd.DataFrame:
    """
    For each scenario and KPI:
      - find best heuristic among {SPT, EDD, ATC} (per scenario) using df_mean values
      - compute Δ% improvement of ML vs best heuristic:
          maximize KPI:  (ML - best) / best
          minimize KPI:  (best - ML) / best
      => positive means ML better.
    Output is wide (scenario rows) with blocks per KPI (best_policy, best_value, ml_value, delta_pct).
    """
    heuristics = ["SPT", "EDD", "ATC"]

    out_rows = []
    scenarios = sorted(df_mean["scenario"].unique().tolist())
    for sc in scenarios:
        row = {"scenario": sc}
        sc_df = df_mean[df_mean["scenario"] == sc].set_index("policy")

        # If ML missing in scenario, skip (still output NaNs)
        for kpi in RANK_KPIS:
            # heuristic best
            best_policy = None
            best_val = None

            vals = {}
            for h in heuristics:
                if h in sc_df.index and kpi in sc_df.columns:
                    vals[h] = sc_df.loc[h, kpi]

            if vals:
                if kpi in MAXIMIZE_KPIS:
                    best_policy = max(vals, key=lambda p: vals[p])
                else:
                    best_policy = min(vals, key=lambda p: vals[p])
                best_val = vals[best_policy]

            ml_val = sc_df.loc["ML", kpi] if "ML" in sc_df.index and kpi in sc_df.columns else float("nan")

            # delta
            if best_val is None or pd.isna(ml_val) or best_val == 0 or pd.isna(best_val):
                delta = float("nan")
            else:
                if kpi in MAXIMIZE_KPIS:
                    delta = (ml_val - best_val) / best_val
                else:
                    delta = (best_val - ml_val) / best_val

            row[f"{kpi}__best_policy"] = best_policy
            row[f"{kpi}__best"] = best_val
            row[f"{kpi}__ml"] = ml_val
            row[f"{kpi}__delta_pct"] = delta * 100 if pd.notna(delta) else float("nan")

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_factor_delta(delta_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Δ% by factors (due, dist, K) based on scenario parsing.
    Uses columns *_delta_pct from compute_ml_vs_best.
    """
    df = delta_wide.copy()
    # add factors
    facts = df["scenario"].apply(parse_scenario_factors).apply(pd.Series)
    df = pd.concat([df, facts], axis=1)

    delta_cols = [c for c in df.columns if c.endswith("__delta_pct")]

    # long format for easy groupby
    long = df.melt(
        id_vars=["scenario", "util", "due", "K", "dist"],
        value_vars=delta_cols,
        var_name="kpi",
        value_name="delta_pct",
    )
    long["kpi"] = long["kpi"].str.replace("__delta_pct", "", regex=False)

    # group by each factor separately (plus combined if you want)
    by_due = long.groupby(["kpi", "due"]).agg(
        n=("delta_pct", "count"),
        mean=("delta_pct", "mean"),
        median=("delta_pct", "median"),
        q25=("delta_pct", quantile_agg(0.25)),
        q75=("delta_pct", quantile_agg(0.75)),
    ).reset_index()
    by_due["factor"] = "due"

    by_dist = long.groupby(["kpi", "dist"]).agg(
        n=("delta_pct", "count"),
        mean=("delta_pct", "mean"),
        median=("delta_pct", "median"),
        q25=("delta_pct", quantile_agg(0.25)),
        q75=("delta_pct", quantile_agg(0.75)),
    ).reset_index()
    by_dist["factor"] = "dist"

    by_K = long.groupby(["kpi", "K"]).agg(
        n=("delta_pct", "count"),
        mean=("delta_pct", "mean"),
        median=("delta_pct", "median"),
        q25=("delta_pct", quantile_agg(0.25)),
        q75=("delta_pct", quantile_agg(0.75)),
    ).reset_index()
    by_K["factor"] = "K"

    # normalize column name of level
    by_due = by_due.rename(columns={"due": "level"})
    by_dist = by_dist.rename(columns={"dist": "level"})
    by_K = by_K.rename(columns={"K": "level"})

    return pd.concat([by_due, by_dist, by_K], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs", help="Basisordner mit Runs (default: runs)")
    parser.add_argument("--out_dir", default="out", help="Output-Ordner (default: out)")
    parser.add_argument(
        "--subset",
        choices=["all", "train", "test"],
        default="all",
        help="Welche Seeds sollen in die Auswertung einfließen?"
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    rows = []

    for kpi_path in runs_dir.rglob(KPI_FILE):
        parts = kpi_path.parts

        # Scenario: folder directly under runs_dir (relative)
        # We keep your old logic, but more robust:
        try:
            rel = kpi_path.relative_to(runs_dir)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        scenario = rel.parts[0]

        # policy/seed
        policy = find_policy(parts)
        if policy is None:
            # fallback: try in relative parts
            policy = find_policy(rel.parts)
        if policy is None:
            continue

        seed_part = find_seed_part(parts) or find_seed_part(rel.parts)
        if seed_part is None:
            continue
        seed = safe_seed_from_part(seed_part)

        # subset filtering
        if args.subset == "train" and seed not in TRAIN_SEEDS:
            continue
        if args.subset == "test" and seed not in TEST_SEEDS:
            continue

        # read KPI json
        with open(kpi_path, "r") as f:
            kpi = json.load(f)

        row = dict(kpi)
        row.update({
            "scenario": scenario,
            "policy": policy,
            "seed": seed,
        })
        row.update(parse_scenario_factors(scenario))
        rows.append(row)

    if not rows:
        raise SystemExit("Keine KPI-Daten gefunden – stimmt runs_dir / Dateiname (kpi_summary.json)?")

    df = pd.DataFrame(rows)

    # convert numeric columns where possible
    for c in df.columns:
        if c in {"scenario", "policy", "seed", "due", "dist", "util", "K", "dist_level"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # ------------------------------------------------------------
    # 1) Run-level export
    # ------------------------------------------------------------
    subset_suffix = args.subset
    runs_path = out_dir / f"kpi_runs_{subset_suffix}.csv"
    to_excel_csv(df, runs_path)

    # ------------------------------------------------------------
    # 2) Backwards-compatible mean summary (your old quick_summary.csv)
    # ------------------------------------------------------------
    # ensure all columns exist
    agg_dict = {}
    for src, outname in MAIN_KPIS.items():
        if src in df.columns:
            agg_dict[outname] = (src, "mean")
        else:
            # keep missing cols as NaN later
            pass

    # Keep n as mean (as before)
    if "n" in df.columns:
        agg_dict["n"] = ("n", "mean")

    summary = df.groupby(["scenario", "policy"]).agg(**agg_dict).reset_index()

    # if some MAIN_KPIS missing, add them as NaN columns for stable downstream Excel formulas
    for outname in MAIN_KPIS.values():
        if outname not in summary.columns:
            summary[outname] = float("nan")

    quick_path = out_dir / f"quick_summary_{subset_suffix}.csv"
    to_excel_csv(summary, quick_path)

    # ------------------------------------------------------------
    # 3) Rich stats per scenario×policy (mean/median/std/q25/q75/min/max + CI95)
    # ------------------------------------------------------------
    # numeric KPI columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    meta_cols = {"scenario", "policy", "seed"}
    kpi_num_cols = [c for c in num_cols if c not in meta_cols]

    grouped = df.groupby(["scenario", "policy"])
    stats = grouped[kpi_num_cols].agg(
        n_seeds=("ct_mean", "count") if "ct_mean" in df.columns else (kpi_num_cols[0], "count"),
        mean=("ct_mean", "mean") if "ct_mean" in df.columns else (kpi_num_cols[0], "mean"),
    )
    # Replace placeholder mean later by expanding manually for each col
    # We'll build a multi-stat table in long form first (cleaner)

    stats_rows = []
    for (sc, pol), g in grouped:
        n = len(g)
        for col in kpi_num_cols:
            s = g[col].dropna()
            if s.empty:
                continue
            std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
            stats_rows.append({
                "scenario": sc,
                "policy": pol,
                "kpi": col,
                "n_seeds": n,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": std,
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
                "min": float(s.min()),
                "max": float(s.max()),
                "ci95": ci95_from_std(n, std),
            })

    kpi_stats = pd.DataFrame(stats_rows)
    stats_path = out_dir / f"kpi_stats_{subset_suffix}.csv"
    to_excel_csv(kpi_stats, stats_path)

    # ------------------------------------------------------------
    # 4) Wins & Average Rank (like your Table 6-2)
    #    Use scenario×policy means in ORIGINAL KPI names (not renamed)
    # ------------------------------------------------------------
    # build df_mean with original KPI names (means over seeds)
    mean_cols = [c for c in RANK_KPIS if c in df.columns]
    df_mean = df.groupby(["scenario", "policy"])[mean_cols].mean().reset_index()

    wins_ranks = compute_wins_ranks(df_mean)
    wins_path = out_dir / f"wins_ranks_{subset_suffix}.csv"
    to_excel_csv(wins_ranks, wins_path)

    # ------------------------------------------------------------
    # 5) ML vs best heuristic (scenario-level means; Δ% positive = ML better)
    # ------------------------------------------------------------
    ml_vs_best = compute_ml_vs_best(df_mean)
    ml_vs_path = out_dir / f"ml_vs_best_heur_{subset_suffix}.csv"
    to_excel_csv(ml_vs_best, ml_vs_path)

    # ------------------------------------------------------------
    # 6) Factor sensitivity on Δ% (due/dist/K)
    # ------------------------------------------------------------
    factor_delta = compute_factor_delta(ml_vs_best)
    factor_path = out_dir / f"factor_delta_{subset_suffix}.csv"
    to_excel_csv(factor_delta, factor_path)

    # Console info
    print(f"[OK] rows collected: {len(df)}")
    print(f"[OK] wrote: {runs_path}")
    print(f"[OK] wrote: {quick_path}")
    print(f"[OK] wrote: {stats_path}")
    print(f"[OK] wrote: {wins_path}")
    print(f"[OK] wrote: {ml_vs_path}")
    print(f"[OK] wrote: {factor_path}")


if __name__ == "__main__":
    main()
