"""Run the main conditional-comparison and completion analyses in dependency order."""
from pathlib import Path
import argparse, os, subprocess, sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--plan", action="store_true", help="Print the execution order without fitting.")
args = parser.parse_args()
stages = [
    "06_sentinel_context.py",
    "13_sentinel_stack_nested.py",
    "17_imputation_baselines.py",
    "19_completion_atlas.py",
]
env = os.environ.copy()
env["BASE_LEARNER"] = "RF"
env["SENT_LEARNER"] = "RF"
for name in stages:
    print(name, flush=True)
    if not args.plan:
        subprocess.run([sys.executable, str(ROOT/"analysis/scripts"/name)],
                       cwd=ROOT, env=env, check=True)
