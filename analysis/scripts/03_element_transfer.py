"""Exp 3: element representation x validation scheme.

Representations (all include soil Clay/pH/CEC/OC + total content):
  NoElement   no element information
  Magpie7     the 7 published Magpie-style descriptors (original paper)
  Geo9        aqueous-geochemistry descriptors (hm_geochem.py)
  Magpie+Geo  both
  OneHot      element identity one-hot (cannot extrapolate to unseen elements)
Schemes: random 5-fold, soil-group 5-fold, leave-one-element-out (LOEO).
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from hm_data import ELEM, RES, SOIL, TOTAL, load, ordinal_metrics
from hm_geochem import GEO_COLS, add_geo

warnings.filterwarnings("ignore")
SEED = 42


def lgb():
    if os.environ.get("EXP3_LEARNER") == "ET":  # best context-free learner from exp1
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", n_jobs=1,
                                    random_state=SEED)
    return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                          n_jobs=1, random_state=SEED, verbose=-1)


def fit_pred(X, y, tr, te):
    m = lgb().fit(X[tr], y[tr])
    p = np.zeros((len(te), 4))
    p[:, m.classes_] = m.predict_proba(X[te])
    return p


def main():
    df = add_geo(load())
    base = SOIL + [TOTAL]
    onehot = pd.get_dummies(df["element"], prefix="el").astype(float)
    reps = {
        "NoElement": df[base].values,
        "Magpie7": df[base + ELEM].values,
        "Geo9": df[base + GEO_COLS].values,
        "Magpie+Geo": df[base + ELEM + GEO_COLS].values,
        "OneHot": np.column_stack([df[base].values, onehot.values]),
    }
    y, g, e = df["y"].values, df["soil_id"].values, df["element"].values
    Xd = df[base].values
    splits = {
        "random": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, y)),
        "soilgroup": list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xd, y, g)),
        "LOEO": list(LeaveOneGroupOut().split(Xd, y, e)),
    }
    tasks = [(r, s, k, tr, te) for r in reps for s, sp in splits.items()
             for k, (tr, te) in enumerate(sp)]
    out = Parallel(n_jobs=2, verbose=2)(delayed(fit_pred)(reps[r], y, tr, te)
                                         for r, s, k, tr, te in tasks)
    oof = {}
    for (r, s, k, tr, te), p in zip(tasks, out):
        oof.setdefault((r, s), np.zeros((len(y), 4)))[te] = p
    rows, per_el = [], []
    for (r, s), p in oof.items():
        met = ordinal_metrics(y, p)
        met.update(rep=r, scheme=s)
        rows.append(met)
        for el in np.unique(e):
            m = e == el
            yhat = p[m].argmax(1)
            per_el.append({"rep": r, "scheme": s, "element": el, "n": int(m.sum()),
                           "acc": (yhat == y[m]).mean(),
                           "mae": np.abs(yhat - y[m]).mean(),
                           "obs_labile": (y[m] <= 1).mean(),
                           "pred_labile": p[m][:, :2].sum(1).mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "exp3_element_transfer_summary.csv", index=False)
    pe = pd.DataFrame(per_el)
    pe.to_csv(RES / "exp3_element_transfer_per_element.csv", index=False)
    np.savez_compressed(RES / "exp3_oof.npz", y=y, element=e,
                        **{f"{r}__{s}": v for (r, s), v in oof.items()})
    cols = ["scheme", "rep", "Accuracy", "BalancedAcc", "MacroF1", "QWK", "OrdinalMAE",
            "SevereErr", "LogLoss"]
    print(summ[cols].sort_values(["scheme", "MacroF1"], ascending=[True, False])
          .round(4).to_string(index=False))
    # element-macro view for LOEO: mean |obs - pred| labile share across elements
    lo = pe[pe.scheme == "LOEO"].copy()
    lo["abs_labile_err"] = (lo.obs_labile - lo.pred_labile).abs()
    print(lo.groupby("rep")[["acc", "mae", "abs_labile_err"]].mean().round(4))
    for r in reps:
        sub = lo[lo.rep == r]
        print(r, "Spearman(obs,pred labile share) =",
              round(sub[["obs_labile", "pred_labile"]].corr("spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    main()
