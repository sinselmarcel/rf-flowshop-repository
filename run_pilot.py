# src/run_pilot.py
import argparse, yaml
from pathlib import Path
from sim_core import run_pilot

def main():
    ap = argparse.ArgumentParser(description="Run FlowShop pilot with CLI overrides")
    ap.add_argument("--cfg", default="cfg/base.yaml", help="Pfad zur YAML-Config")
    ap.add_argument("--policy", choices={"SPT", "EDD", "ATC", "ML"}, help="Regel überschreiben")
    ap.add_argument("--seed", type=int, default=42, help="Seed")
    ap.add_argument("--verbose", action="store_true", help="Trace ins Terminal")
    ap.add_argument("--max-print", type=int, default=None, help="Anzahl Entscheidungsprints")
    ap.add_argument("--update-ml", action="store_true", help="Nach dem Run: collect -> label -> train")
    args = ap.parse_args()

    # YAML laden
    cfg_path = Path(args.cfg)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Overrides anwenden (ohne Original-Config zu verändern)
    if args.policy:
        cfg.setdefault("policies", {}).setdefault("heuristics", [args.policy])
        cfg["policies"]["heuristics"] = [args.policy]
    if args.verbose:
        cfg.setdefault("logging", {})["verbose"] = True
        if args.max_print is None:
            cfg["logging"]["max_print"] = 30
    if args.max_print is not None:
        cfg.setdefault("logging", {})["max_print"] = args.max_print


    # Temporäre Config schreiben und starten
    tmp = Path("cfg/_tmp_cli.yaml")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    res = run_pilot(str(tmp), seed=args.seed)
    print("Pilot-Result:", res)
    print(f"(cfg used: {tmp})")

    if args.update_ml:
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/collect_runs.py", "--runs-dir", "runs"], check=True)
        subprocess.run([sys.executable, "scripts/label_make.py"], check=True)

if __name__ == "__main__":
    main()
