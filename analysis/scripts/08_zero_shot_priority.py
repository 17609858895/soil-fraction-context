"""Exp 8: two realistic deployment scenarios for sentinel-conditioned prediction.

S1 zero-shot element: leave-one-element-out (target element absent from training), with and
   without co-measured sentinel context; element descriptors = Magpie7 + Geo9.
S2 priority-sentinel inference: only Cd/Pb/Zn/Cu fractions are observable as context (the metals
   most often reported); targets = all other elements; soil-group 5-fold CV.
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from hm_data import ELEM, RES, SOIL, TOTAL, load, ordinal_metrics
from hm_geochem import GEO_COLS, add_geo

warnings.filterwarnings("ignore")
SEED = 42
_spec = importlib.util.spec_from_file_location("e6", os.path.join(os.path.dirname(__file__),
                                                                    "06_sentinel_context.py"))
e6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e6)
PRIORITY4 = ["Cd", "Pb", "Zn", "Cu"]


def fit_pred(X, y, tr, te):
    m = e6.lgb().fit(X[tr], y[tr])
    p = np.zeros((len(te), 4))
    p[:, m.classes_] = m.predict_proba(X[te])
    return p


def main():
    df = add_geo(load())
    y, els = df["y"].values, df["element"].values
    base = df[SOIL + [TOTAL] + ELEM + GEO_COLS].values.astype(float)
    dist, _, sent = e6.build_context(df, y)
    X_ctx = np.column_stack([base, dist, sent])
    has = ~np.isnan(dist[:, 0])

    # S1 zero-shot (LOEO). The held-out element's labels are hidden from every context
    # (training rows included), so no label of the target element enters training.
    loeo = list(LeaveOneGroupOut().split(base, y, els))

    def loeo_ctx(tr, te):
        e_out = els[te][0]
        d, _, s = e6.build_context(df, y, els != e_out)
        return fit_pred(np.column_stack([base, d, s]), y, tr, te)

    tasks = [(name, tr, te) for name in ["LOEO-none", "LOEO-sentinel"] for tr, te in loeo]
    res = Parallel(n_jobs=2)(delayed(fit_pred)(base, y, tr, te) if name == "LOEO-none"
                             else delayed(loeo_ctx)(tr, te) for name, tr, te in tasks)
    tasks = [(name, None, tr, te) for name, tr, te in tasks]
    oof = {}
    for (name, _, tr, te), p in zip(tasks, res):
        oof.setdefault(name, np.zeros((len(y), 4)))[te] = p
    rows, pe = [], []
    for name, P in oof.items():
        for sub, m in [("all", np.ones(len(y), bool)), ("with_context", has)]:
            met = ordinal_metrics(y[m], P[m])
            met.update(scenario=name, subset=sub)
            rows.append(met)
        for e in np.unique(els):
            m = (els == e) & has
            if m.sum() >= 20:
                pe.append(dict(scenario=name, element=e, n=int(m.sum()),
                               acc=(P[m].argmax(1) == y[m]).mean(),
                               obs_labile=(y[m] <= 1).mean(), pred_labile=P[m][:, :2].sum(1).mean()))

    # S2 priority sentinels -> other elements
    vis = np.isin(els, PRIORITY4)
    dist4, _, sent4 = e6.build_context(df, y, vis)
    keep = [e6.SENTINELS.index(e) for e in PRIORITY4]
    X_p = np.column_stack([base, dist4, sent4[:, keep]])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(base, y, df["soil_id"]))
    P0 = np.zeros((len(y), 4))
    P1 = np.zeros((len(y), 4))
    for tr, te in folds:
        P0[te] = fit_pred(base, y, tr, te)
        P1[te] = fit_pred(X_p, y, tr, te)
    tgt = (~np.isin(els, PRIORITY4)) & (~np.isnan(dist4[:, 0]))
    for name, P in [("Priority4-none", P0), ("Priority4-sentinel", P1)]:
        met = ordinal_metrics(y[tgt], P[tgt])
        met.update(scenario=name, subset=f"non-priority targets with >=1 priority sentinel (n={tgt.sum()})")
        rows.append(met)
        for e in np.unique(els[tgt]):
            m = tgt & (els == e)
            if m.sum() >= 20:
                pe.append(dict(scenario=name, element=e, n=int(m.sum()),
                               acc=(P[m].argmax(1) == y[m]).mean(),
                               obs_labile=(y[m] <= 1).mean(), pred_labile=P[m][:, :2].sum(1).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp8_zero_shot_priority.csv", index=False)
    pd.DataFrame(pe).to_csv(RES / "exp8_per_element.csv", index=False)
    np.savez_compressed(RES / "exp8_oof.npz", y=y, element=els, **oof, P0=P0, P1=P1)
    print(out[["scenario", "subset", "Accuracy", "BalancedAcc", "MacroF1", "QWK", "SevereErr",
               "LogLoss"]].round(4).to_string(index=False))
    pe = pd.DataFrame(pe)
    w = pe.pivot_table(index="element", columns="scenario", values="acc")
    print(w.round(3).to_string())


if __name__ == "__main__":
    main()
