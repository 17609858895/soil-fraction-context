"""Exp 16: pseudo-study grouped CV for the 20-model benchmark and the element representations.

Uses the pseudo-study map from exp15 (results/exp15_pseudostudy_map.csv).
  bench   all 20 learners (default settings), StratifiedGroupKFold(5) on pseudo-study, seed 42
  repr    ExtraTrees with the four element representations (exp3) under pseudo-study CV
Outputs: exp16_pstudy20_folds.csv / _summary.csv, ET/exp16_pstudy_repr.csv
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import importlib.util
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import ELEM, FEATS, RES, SOIL, TOTAL, load, ordinal_metrics
from hm_geochem import GEO_COLS, add_geo

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("b1", os.path.join(HERE, "01_benchmark20.py"))
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
NJ = int(os.environ.get("E16_JOBS", "2"))


def main():
    df = add_geo(load())
    pmap = pd.read_csv(RES / "exp15_pseudostudy_map.csv").set_index("soil_id")["pstudy"]
    ps = pmap.loc[df["soil_id"]].values
    y = df["y"].values
    X = df[FEATS].values.astype(float)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, ps))
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "bench":
        M = b1.models()
        tasks = [(n, M[n], X, y, tr, te, "pstudy", k) for n in M for k, (tr, te) in enumerate(folds)]
        out = Parallel(n_jobs=NJ, verbose=2)(delayed(b1.run_fold)(*t) for t in tasks)
        f = pd.DataFrame([o[0] for o in out])
        f.to_csv(RES / "exp16_pstudy20_folds.csv", index=False)
        s = f.drop(columns=["fold", "scheme"]).groupby("model").agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        s = s.reset_index().sort_values("MacroF1_mean", ascending=False)
        s.to_csv(RES / "exp16_pstudy20_summary.csv", index=False)
        print(s[["model", "MacroF1_mean", "Accuracy_mean", "LogLoss_mean"]].round(4).to_string(index=False))
    else:
        from sklearn.ensemble import ExtraTreesClassifier
        base = SOIL + [TOTAL]
        onehot = pd.get_dummies(df["element"], prefix="el").astype(float).values
        reps = {"NoElement": df[base].values, "Magpie7": df[base + ELEM].values,
                "Geo9": df[base + GEO_COLS].values,
                "OneHot": np.column_stack([df[base].values, onehot])}
        rows = []
        for r, Xr in reps.items():
            P = np.zeros((len(y), 4))
            for tr, te in folds:
                m = ExtraTreesClassifier(n_estimators=500, max_features="sqrt", n_jobs=NJ,
                                         random_state=42).fit(Xr[tr], y[tr])
                P[te] = m.predict_proba(Xr[te])
            met = ordinal_metrics(y, P)
            met.update(rep=r, scheme="pstudy")
            rows.append(met)
            print(r, round(met["MacroF1"], 4), flush=True)
        (RES / "ET").mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(RES / "ET" / "exp16_pstudy_repr.csv", index=False)


if __name__ == "__main__":
    main()
