"""Exp 5: conformal prediction sets on the BCR mobility ladder.

Outer: soil-group 5-fold CV. Inside each outer training set, 25% of soil groups form a
calibration set (never used for fitting). Base learners: nominal LightGBM and the
DTS-PP discrete-hazard booster (from 02_ordinal_survival.py).
Set constructors at miscoverage alpha:
  LAC        {k : p_k >= 1 - q}, score 1 - p_y
  APS        adaptive prediction sets (randomised=False)
  OrdAPS     contiguous interval grown from the mode toward the larger neighbour
  Mondrian   class-conditional LAC (per-class quantiles)
Reported: marginal / class-conditional / element-conditional coverage, mean set size,
singleton rate, contiguity, and 'potentially mobile' flag (set contains F1 or F2).
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from hm_data import FEATS, RES, load

warnings.filterwarnings("ignore")
SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
_spec = importlib.util.spec_from_file_location("e2", os.path.join(os.path.dirname(__file__),
                                                                    "02_ordinal_survival.py"))
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)
_spec6 = importlib.util.spec_from_file_location("e6", os.path.join(os.path.dirname(__file__),
                                                                     "06_sentinel_context.py"))
e6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(e6)
BASES = ["Nominal", "DTS-PP", "Sentinel"]


def qhat(scores, alpha):
    n = len(scores)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, lvl, method="higher")


def ord_interval(p, mass):
    """Grow a contiguous interval from the mode until cumulative prob >= mass."""
    lo = hi = int(p.argmax())
    tot = p[lo]
    while tot < mass - 1e-12 and (lo > 0 or hi < len(p) - 1):
        left = p[lo - 1] if lo > 0 else -1
        right = p[hi + 1] if hi < len(p) - 1 else -1
        if left >= right:
            lo -= 1
            tot += p[lo]
        else:
            hi += 1
            tot += p[hi]
    return lo, hi


def ord_score(p, y):
    """Smallest interval mass (from ord_interval growth path) that contains y."""
    lo = hi = int(p.argmax())
    tot = p[lo]
    while not (lo <= y <= hi):
        left = p[lo - 1] if lo > 0 else -1
        right = p[hi + 1] if hi < len(p) - 1 else -1
        if left >= right:
            lo -= 1
            tot += p[lo]
        else:
            hi += 1
            tot += p[hi]
    return tot


def aps_score(p, y):
    o = np.argsort(-p)
    c = np.cumsum(p[o])
    return c[np.where(o == y)[0][0]]


def build_sets(pc, yc, pt, alpha):
    sets = {}
    q = qhat(1 - pc[np.arange(len(yc)), yc], alpha)
    sets["LAC"] = pt >= 1 - q
    qa = qhat(np.array([aps_score(p, y) for p, y in zip(pc, yc)]), alpha)
    S = np.zeros_like(pt, bool)
    for i, p in enumerate(pt):
        o = np.argsort(-p)
        c = np.cumsum(p[o])
        k = np.searchsorted(c, qa - 1e-12) + 1
        S[i, o[:k]] = True
    sets["APS"] = S
    qo = qhat(np.array([ord_score(p, y) for p, y in zip(pc, yc)]), alpha)
    S = np.zeros_like(pt, bool)
    for i, p in enumerate(pt):
        lo, hi = ord_interval(p, qo)
        S[i, lo:hi + 1] = True
    sets["OrdAPS"] = S
    S = np.zeros_like(pt, bool)
    for k in range(4):
        sc = 1 - pc[yc == k, k]
        qk = qhat(sc, alpha) if len(sc) else 1.0
        S[:, k] = pt[:, k] >= 1 - qk
    sets["Mondrian"] = S
    for v in sets.values():  # never return an empty set
        empty = ~v.any(1)
        v[empty, pt[empty].argmax(1)] = True
    return sets


def fold_job(X, Xc, y, groups, tr, te, k):
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=SEED + k)
    fit_i, cal_i = next(gss.split(X[tr], y[tr], groups[tr]))
    fit_i, cal_i = tr[fit_i], tr[cal_i]
    out = {}
    for base in BASES:
        if base == "Sentinel":
            m = e6.lgb().fit(Xc[fit_i], y[fit_i])
            p = m.predict_proba(np.vstack([Xc[cal_i], Xc[te]]))
        else:
            f = e2.fit_nominal if base == "Nominal" else e2.fit_dts
            p = f(X[fit_i], y[fit_i], np.vstack([X[cal_i], X[te]]))
        out[base] = (p[:len(cal_i)], p[len(cal_i):])
    return k, cal_i, te, out


def main():
    df = load()
    X, y, g, el = df[FEATS].values.astype(float), df["y"].values, df["soil_id"].values, df["element"].values
    dist, _, sent = e6.build_context(df, y)
    Xc = np.column_stack([X, dist, sent])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
    res = Parallel(n_jobs=2)(delayed(fold_job)(X, Xc, y, g, tr, te, k) for k, (tr, te) in enumerate(folds))
    rows, per_el = [], []
    store = {}
    for alpha in ALPHAS:
        for base in BASES:
            allS = {m: np.zeros((len(y), 4), bool) for m in ["LAC", "APS", "OrdAPS", "Mondrian"]}
            for k, cal_i, te, out in res:
                pc, pt = out[base]
                S = build_sets(pc, y[cal_i], pt, alpha)
                for m in S:
                    allS[m][te] = S[m]
            for m, S in allS.items():
                cov = S[np.arange(len(y)), y]
                size = S.sum(1)
                idx = np.arange(4)
                contig = np.array([(r.any() and (idx[r].max() - idx[r].min() + 1 == r.sum())) for r in S])
                mobile_flag = S[:, :2].any(1)
                true_mob = y <= 1
                elcov = pd.Series(cov).groupby(el).mean()
                elcnt = pd.Series(cov).groupby(el).size()
                big = elcov[elcnt >= 50]
                rows.append(dict(alpha=alpha, base=base, method=m, coverage=cov.mean(),
                                 min_class_cov=min(cov[y == c].mean() for c in range(4)),
                                 **{f"cov_F{c+1}": cov[y == c].mean() for c in range(4)},
                                 min_elem_cov_n50=big.min(), frac_elem_below=(big < 1 - alpha - 0.05).mean(),
                                 mean_size=size.mean(), singleton=(size == 1).mean(),
                                 contiguous=contig.mean(),
                                 mobile_sensitivity=mobile_flag[true_mob].mean(),
                                 mobile_flag_rate=mobile_flag.mean()))
                for e_, c_ in elcov.items():
                    per_el.append(dict(alpha=alpha, base=base, method=m, element=e_, n=int(elcnt[e_]),
                                       coverage=c_, mean_size=pd.Series(size).groupby(el).mean()[e_]))
                store[f"{alpha}_{base}_{m}"] = S
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp5_conformal_summary.csv", index=False)
    pd.DataFrame(per_el).to_csv(RES / "exp5_conformal_per_element.csv", index=False)
    np.savez_compressed(RES / "exp5_conformal_sets.npz", y=y, element=el, **store)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
