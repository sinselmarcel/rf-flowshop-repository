#!/usr/bin/env python3
"""
collect_runs.py
Sammelt rekursiv alle Runs unter --runs-dir, merged ml_dataset.csv + job_log.csv,
fügt run_id hinzu und schreibt nach out/ml_dataset.csv und out/job.csv.
"""
import argparse, sys
from pathlib import Path
import pandas as pd

def find_files(root: Path, name: str):
    return list(root.rglob(name))

def run_id_from_path(p: Path, root: Path) -> str:
    # z.B. "util95_dueeng_K3_distlow/ATC_capBN177min/seed42/20251114_103144"
    rel = p.relative_to(root)
    return str(rel.parent).replace("\\", "/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs", help="Root der Runs (rekursiv)")
    ap.add_argument("--min-rows", type=int, default=5, help="Mindestzeilen je Datei")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    out_dir = Path("out"); out_dir.mkdir(parents=True, exist_ok=True)

    ds_paths = find_files(root, "ml_dataset.csv")
    jl_paths = find_files(root, "job_log.csv")

    ds_index = {run_id_from_path(p, root): p for p in ds_paths}
    jl_index = {run_id_from_path(p, root): p for p in jl_paths}

    keys = sorted(set(ds_index) & set(jl_index))
    if not keys:
        print("[WARN] Keine Paare (ml_dataset.csv + job_log.csv) gefunden.", file=sys.stderr)
        sys.exit(1)

    ds_all, jl_all = [], []
    added = 0
    for k in keys:
        p_ds, p_jl = ds_index[k], jl_index[k]
        try:
            df_ds = pd.read_csv(p_ds)
            df_jl = pd.read_csv(p_jl)
        except Exception as e:
            print(f"[SKIP] {k}: CSV-Fehler -> {e}", file=sys.stderr)
            continue
        if len(df_ds) < args.min_rows or len(df_jl) < args.min_rows:
            print(f"[SKIP] {k}: zu wenige Zeilen (ds={len(df_ds)}, job={len(df_jl)})", file=sys.stderr)
            continue
        df_ds["run_id"] = k
        df_jl["run_id"] = k
        ds_all.append(df_ds); jl_all.append(df_jl); added += 1

    if added == 0:
        print("[ERR] Keine geeigneten Runs.", file=sys.stderr); sys.exit(2)

    big_ds = pd.concat(ds_all, ignore_index=True).drop_duplicates()
    big_jl = pd.concat(jl_all, ignore_index=True).drop_duplicates()

    if "time" in big_ds.columns:
        big_ds = big_ds.sort_values(["run_id","time"], kind="mergesort")
    if "t_done" in big_jl.columns:
        big_jl = big_jl.sort_values(["run_id","t_done"], kind="mergesort")

    big_ds.to_csv(out_dir/"ml_dataset.csv", index=False)
    big_jl.to_csv(out_dir/"job.csv", index=False)
    print(f"[OK] out/ml_dataset.csv rows={len(big_ds):,}")
    print(f"[OK] out/job.csv       rows={len(big_jl):,}")
    print(f"[OK] Runs gemerged: {added}")

if __name__ == "__main__":
    main()
