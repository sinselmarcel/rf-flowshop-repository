# scripts/run_grid.py
import argparse, itertools, subprocess, sys
from pathlib import Path
import yaml

TRAIN_SEEDS = [42, 73, 99, 111, 808, 1337, 2025, 2601, 2718, 31415]
TEST_SEEDS  = [222, 333, 444, 555, 666, 777, 888, 999, 1212, 1717]

def parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/base.yaml")
    ap.add_argument("--policies", default="SPT,EDD,ATC,ML")
    ap.add_argument("--due", default="eng,weit")
    ap.add_argument("--dist", default="low,high")      # ggf: off,low,high
    ap.add_argument("--K", default="3,5")
    ap.add_argument("--rho", default="0.95")
    ap.add_argument("--wlc", default="on,off")         # on,off oder nur on
    ap.add_argument("--seeds", default="test")         # train|test|all oder "1,2,3"
    ap.add_argument("--out-on", default="runs")
    ap.add_argument("--out-off", default="runs_nowlc") # trennt WLC-off sauber ab
    ap.add_argument("--quiet", action="store_true")    # verbose/max_print aus
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    policies = parse_csv_list(args.policies)
    dues     = parse_csv_list(args.due)
    dists    = parse_csv_list(args.dist)
    Ks       = [float(x) for x in parse_csv_list(args.K)]
    rhos     = [float(x) for x in parse_csv_list(args.rho)]
    wlcs     = parse_csv_list(args.wlc)

    if args.seeds.lower() == "train":
        seeds = TRAIN_SEEDS
    elif args.seeds.lower() == "test":
        seeds = TEST_SEEDS
    elif args.seeds.lower() == "all":
        seeds = TRAIN_SEEDS + TEST_SEEDS
    else:
        seeds = [int(x) for x in parse_csv_list(args.seeds)]

    # ML-Modelle checken (sonst ML überspringen)
    have_ml = Path("models/model_rct.pkl").exists() and Path("models/model_tard.pkl").exists()
    if "ML" in policies and not have_ml:
        print("[WARN] ML gewählt, aber models/model_rct.pkl oder model_tard.pkl fehlt -> ML wird übersprungen.")
        policies = [p for p in policies if p != "ML"]

    base_cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))

    tmp = Path("cfg/_tmp_grid.yaml")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(wlcs, rhos, dues, Ks, dists, policies, seeds))
    print(f"[PLAN] Runs: {len(combos)}")

    for (wlc, rho, due, K, dist, pol, seed) in combos:
        cfg = dict(base_cfg)  # shallow copy ok, wir setzen darunter neu
        cfg.setdefault("arrival", {})["rho_levels"] = [rho]
        cfg.setdefault("scenario", {})["due"] = due
        cfg["scenario"]["dist"] = dist
        cfg.setdefault("wlc", {}).setdefault("norm", {})["K"] = K
        cfg["wlc"]["enabled"] = (wlc.lower() == "on")

        cfg.setdefault("logging", {})["out_root"] = (args.out_on if wlc.lower() == "on" else args.out_off)

        if args.quiet:
            cfg.setdefault("logging", {})["verbose"] = False
            cfg["logging"]["max_print"] = 0

        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        cmd = [sys.executable, "src/run_pilot.py", "--cfg", str(tmp), "--policy", pol, "--seed", str(seed)]
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

    print("[OK] Fertig.")

if __name__ == "__main__":
    main()
