"""Exp 11: is the sentinel gain model-agnostic? All 20 benchmark learners, soil-grouped 5-fold CV,
same folds as exp1, with sentinel context added in a NaN-free encoding so every learner can use it:
  class distribution of co-measured elements (4) + their count + has_context flag
  + one-hot class of each of 15 sentinel elements (15 x 4; all-zero when absent).
Outputs bench20ctx_metrics_folds.csv / _summary.csv / _oof_soilgroup.npz.
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

from hm_data import FEATS, RES, load

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b1 = _mod("b1", "01_benchmark20.py")
e6 = _mod("e6", "06_sentinel_context.py")


def context_matrix(df, y):
    dist, _, sent = e6.build_context(df, y)
    has = (~np.isnan(dist[:, 0])).astype(float)
    d = np.nan_to_num(dist, nan=0.0)
    oh = np.zeros((len(y), sent.shape[1] * 4))
    for j in range(sent.shape[1]):
        ok = ~np.isnan(sent[:, j])
        oh[np.where(ok)[0], j * 4 + sent[ok, j].astype(int)] = 1.0
    return np.column_stack([d, has, oh])


def main():
    df = load()
    y, g = df["y"].values, df["soil_id"].values
    X = np.column_stack([df[FEATS].values.astype(float), context_matrix(df, y)])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=b1.SEED).split(X, y, g))
    M = b1.models()
    only = sys.argv[1:]
    tasks = [(n, M[n], "soilgroup_ctx", k, tr, te) for n in M if (not only or n in only)
             for k, (tr, te) in enumerate(folds)]
    out = Parallel(n_jobs=3, verbose=5)(
        delayed(b1.run_fold)(n, e, X, y, tr, te, s, k) for n, e, s, k, tr, te in tasks)
    folds_df = pd.DataFrame([o[0] for o in out])
    folds_df.to_csv(RES / "bench20ctx_metrics_folds.csv", index=False)
    summ = folds_df.drop(columns=["fold", "scheme"]).groupby("model").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().sort_values("MacroF1_mean", ascending=False).to_csv(
        RES / "bench20ctx_metrics_summary.csv", index=False)
    oof = {}
    for _, n, s, te, p in out:
        oof.setdefault(n, np.zeros((len(y), 4)))[te] = p
    np.savez_compressed(RES / "bench20ctx_oof_soilgroup.npz", y=y, **oof)
    print(summ[["Accuracy_mean", "MacroF1_mean", "QWK_mean", "LogLoss_mean"]]
          .sort_values("MacroF1_mean", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
