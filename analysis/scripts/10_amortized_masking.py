"""Exp 10: one budget-agnostic (amortized) sentinel model vs budget-specific models.

Training augmentation: every training row is replicated K times; each copy sees a random subset
of the co-measured elements in its soil (subset size m ~ Uniform{0..n_other}). One LightGBM is
fitted on the augmented table and evaluated at test budgets m in {0,1,2,3,5,all} (3 random test
masks each). Budget-specific reference = exp6B (a separate model per m).
Ablation: K=1 (no augmentation, full context only) evaluated at the same budgets.
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
_spec = importlib.util.spec_from_file_location("e6", os.path.join(os.path.dirname(__file__),
                                                                    "06_sentinel_context.py"))
e6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e6)


def random_visibility(df, rng, m):
    """Row-level visibility mask. m=None: everything visible; m='rand': per soil a uniform
    random number of visible rows; int: exactly m+1 visible rows per soil (target excluded)."""
    vis = np.zeros(len(df), bool)
    for s, idx in df.groupby("soil_id").indices.items():
        if m is None:
            vis[idx] = True
            continue
        k = rng.integers(0, len(idx) + 1) if m == "rand" else min(len(idx), m + 1)
        if k > 0:
            vis[rng.choice(idx, k, replace=False)] = True
    return vis


def feats(df, y, X0, vis):
    d, _, s = e6.build_context(df, y, vis)
    return np.column_stack([X0, d, s])


def main():
    df = load()
    y = df["y"].values
    X0 = df[FEATS].values.astype(float)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X0, y, df["soil_id"]))
    rng = np.random.default_rng(0)
    K = 4
    train_views = [feats(df, y, X0, random_visibility(df, rng, "rand")) for _ in range(K)]
    full_view = feats(df, y, X0, None)
    budgets = [0, 1, 2, 3, 5, None]
    test_views = {m: [feats(df, y, X0, random_visibility(df, np.random.default_rng(100 + r), m))
                      if m != 0 else np.column_stack([X0, np.full((len(y), 5 + len(e6.SENTINELS)), np.nan)])
                      for r in range(3)] for m in budgets}
    preds = {(v, m, r): np.zeros((len(y), 4)) for v in ["amortized", "full-only"]
             for m in budgets for r in range(3)}
    for k, (tr, te) in enumerate(folds):
        Xa = np.vstack([tv[tr] for tv in train_views])
        ya = np.concatenate([y[tr]] * K)
        mods = {"amortized": e6.lgb().fit(Xa, ya), "full-only": e6.lgb().fit(full_view[tr], y[tr])}
        for v, mod in mods.items():
            for m in budgets:
                for r in range(3):
                    preds[(v, m, r)][te] = mod.predict_proba(test_views[m][r][te])
        print("fold", k, flush=True)
    has = ~np.isnan(full_view[:, X0.shape[1]])
    rows = []
    for (v, m, r), P in preds.items():
        met = ordinal_metrics(y[has], P[has])
        met.update(variant=v, m="all" if m is None else m, rep=r)
        rows.append(met)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp10_amortized_masking.csv", index=False)
    print(out.groupby(["variant", "m"])[["Accuracy", "MacroF1", "LogLoss"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
