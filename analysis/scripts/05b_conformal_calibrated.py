"""Exp 5b: conformal sets after temperature scaling, with contiguous (ordinal) variants.

Per outer soil-group fold: fit base on 75% of training soils; the remaining 25% of soils are split
again by soil into a temperature set (fit one scalar T by NLL) and a conformal calibration set,
so the conformal quantile never sees data used for T.
Bases: context-free LightGBM ('Nominal') and sentinel LightGBM ('Sentinel').
Methods: LAC, APS, OrdAPS, Mondrian, and '+contig' versions (fill any gap between the lowest and
highest class in the set so every set is an interval on the F1-F4 ladder).
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from hm_data import FEATS, RES, load

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(__file__)


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c5 = _mod("c5", "05_conformal.py")
e6 = _mod("e6", "06_sentinel_context.py")


def temp(P, T):
    L = np.log(np.clip(P, 1e-9, 1)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_T(P, y):
    f = lambda T: -np.log(np.clip(temp(P, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
    return minimize_scalar(f, bounds=(0.3, 6), method="bounded").x


def contig(S):
    S = S.copy()
    for i in range(len(S)):
        k = np.where(S[i])[0]
        S[i, k.min():k.max() + 1] = True
    return S


def main():
    df = load()
    y, g, el = df["y"].values, df["soil_id"].values, df["element"].values
    X = df[FEATS].values.astype(float)
    dist, _, sent = e6.build_context(df, y)
    Xc = np.column_stack([X, dist, sent])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
    store = {}
    for k, (tr, te) in enumerate(folds):
        fit_i, rest = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED + k).split(X[tr], y[tr], g[tr]))
        fit_i, rest = tr[fit_i], tr[rest]
        t_i, c_i = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED + 10 + k).split(X[rest], y[rest], g[rest]))
        t_i, c_i = rest[t_i], rest[c_i]
        for base, M in [("Nominal", X), ("Sentinel", Xc)]:
            m = e6.lgb().fit(M[fit_i], y[fit_i])
            pt, pc, pe = m.predict_proba(M[t_i]), m.predict_proba(M[c_i]), m.predict_proba(M[te])
            T = fit_T(pt, y[t_i])
            store.setdefault(base, []).append((c_i, te, temp(pc, T), temp(pe, T), T))
    rows, per_el, keep = [], [], {}
    for alpha in [0.05, 0.10, 0.20]:
        for base, parts in store.items():
            S_all = {}
            for c_i, te, pc, pe, T in parts:
                S = c5.build_sets(pc, y[c_i], pe, alpha)
                S["Mondrian+contig"] = contig(S["Mondrian"])
                S["LAC+contig"] = contig(S["LAC"])
                for mth, s in S.items():
                    S_all.setdefault(mth, np.zeros((len(y), 4), bool))[te] = s
            for mth, S in S_all.items():
                keep[f"{alpha}_{base}_{mth}"] = S
                cov = S[np.arange(len(y)), y]
                size = S.sum(1)
                elcov = pd.Series(cov).groupby(el).mean()
                elcnt = pd.Series(cov).groupby(el).size()
                big = elcov[elcnt >= 50]
                rows.append(dict(alpha=alpha, base=base, method=mth, coverage=cov.mean(),
                                 min_class_cov=min(cov[y == c].mean() for c in range(4)),
                                 **{f"cov_F{c+1}": cov[y == c].mean() for c in range(4)},
                                 min_elem_cov_n50=big.min(), mean_size=size.mean(),
                                 singleton=(size == 1).mean(),
                                 mobile_sensitivity=S[y <= 1][:, :2].any(1).mean(),
                                 mobile_flag_rate=S[:, :2].any(1).mean(),
                                 T_mean=np.mean([p[4] for p in store[base]])))
                if alpha == 0.10:
                    for e_, c_ in elcov.items():
                        per_el.append(dict(base=base, method=mth, element=e_, n=int(elcnt[e_]),
                                           coverage=c_, mean_size=pd.Series(size).groupby(el).mean()[e_]))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp5b_conformal_calibrated.csv", index=False)
    np.savez_compressed(RES / "exp5b_sets.npz", y=y, element=el, **keep)
    pd.DataFrame(per_el).to_csv(RES / "exp5b_conformal_per_element.csv", index=False)
    cols = ["alpha", "base", "method", "coverage", "min_class_cov", "min_elem_cov_n50", "mean_size",
            "singleton", "mobile_sensitivity", "mobile_flag_rate", "T_mean"]
    print(out[out.alpha == 0.10][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
