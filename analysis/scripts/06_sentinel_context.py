"""Exp 6: co-measured ('sentinel') element fractions as context for the target element.

Scenario: BCR fractionation is known for some elements in a soil; predict the dominant fraction
of another element in the same soil. All evaluation uses soil-group 5-fold CV (test soils never
seen in training). Context for row i = observed classes of OTHER elements in the same soil
(rows of the same element in that soil are excluded, so replicates cannot leak).

Parts
  A  context representations: none / mean class / class distribution / element-specific sentinels
     / distribution + sentinels; plus a shuffled-context control (context permuted among test
     rows of the same element -> keeps marginals, breaks soil pairing)
  B  measurement-budget curve: reveal m randomly chosen co-measured elements (train with the same
     random-m masking), m in {0,1,2,3,5,all}
  C  single-sentinel informativeness: reveal only element s; gain on other elements in soils where
     s was measured (vs. no-context model on the same rows)
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
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
SENTINELS = ["Pb", "Zn", "Cu", "Cr", "Ni", "Cd", "As", "Mn", "Fe", "Co", "V", "Sb", "Ca", "Sr", "Ba"]


def lgb(seed=SEED):
    """Sentinel-analysis learner: LightGBM by default; SENT_LEARNER=RF uses the random forest
    selected by the 20-model sentinel benchmark (exp11); sklearn>=1.4 RF accepts NaN sentinels."""
    if os.environ.get("SENT_LEARNER") == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=1,
                                      random_state=seed)
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1,
                          random_state=seed, verbose=-1)


def build_context(df, y, reveal=None):
    """reveal: optional boolean array (row-level) of which rows' labels are observable as context.
    Returns dist(4)+count, mean class, sentinel ordinal matrix (NaN if absent/hidden)."""
    n = len(df)
    oh = np.eye(4)[y]
    dist = np.full((n, 5), np.nan)
    mean = np.full((n, 1), np.nan)
    sent = np.full((n, len(SENTINELS)), np.nan)
    sidx = {e: k for k, e in enumerate(SENTINELS)}
    els = df["element"].values
    vis = np.ones(n, bool) if reveal is None else reveal
    for s, idx in df.groupby("soil_id").indices.items():
        e_ = els[idx]
        for j, i in enumerate(idx):
            m = (e_ != e_[j]) & vis[idx]
            if not m.any():
                continue
            o = idx[m]
            c = oh[o].sum(0)
            dist[i, :4], dist[i, 4] = c / c.sum(), m.sum()
            mean[i, 0] = y[o].mean()
            for r in o:
                if els[r] in sidx:
                    sent[i, sidx[els[r]]] = y[r]
    return dist, mean, sent


def cv_predict(X, y, folds, seed=SEED):
    P = np.zeros((len(y), 4))
    for tr, te in folds:
        m = lgb(seed).fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])
    return P


def part_a(df, y, X0, folds):
    dist, mean, sent = build_context(df, y)
    reps = {"None": X0,
            "MeanClass": np.column_stack([X0, mean, dist[:, 4]]),
            "Distribution": np.column_stack([X0, dist]),
            "Sentinels": np.column_stack([X0, sent]),
            "Dist+Sentinels": np.column_stack([X0, dist, sent])}
    out = {k: cv_predict(v, y, folds) for k, v in reps.items()}
    # shuffled control: permute context rows among same-element rows within each test fold
    rng = np.random.default_rng(0)
    ctx = np.column_stack([dist, sent])
    Psh = np.zeros((len(y), 4))
    for tr, te in folds:
        m = lgb().fit(np.column_stack([X0, ctx])[tr], y[tr])
        c_te = ctx[te].copy()
        els = df["element"].values[te]
        for e in np.unique(els):
            k = np.where(els == e)[0]
            c_te[k] = c_te[rng.permutation(k)]
        Psh[te] = m.predict_proba(np.column_stack([X0[te], c_te]))
    out["Shuffled-context"] = Psh
    return out, dist


def masked_features(df, y, X0, rng, m):
    """Per soil keep m random co-measured rows visible (m=None -> all); return features."""
    vis = np.zeros(len(y), bool)
    for s, idx in df.groupby("soil_id").indices.items():
        if m is None or len(idx) <= m + 1:
            vis[idx] = True
        else:
            vis[rng.choice(idx, m + 1, replace=False)] = True  # +1: target itself is excluded anyway
    dist, _, sent = build_context(df, y, vis)
    return np.column_stack([X0, dist, sent])


def part_b_one(df, y, X0, folds, m, rep):
    rng = np.random.default_rng(100 + rep)
    X = X0 if m == 0 else masked_features(df, y, X0, rng, m)
    return m, rep, cv_predict(X, y, folds, seed=SEED + rep)


def part_c_one(df, y, X0, folds, s, P0):
    els = df["element"].values
    vis = els == s
    dist, _, sent = build_context(df, y, vis)
    X = np.column_stack([X0, dist[:, :4]])
    P = cv_predict(X, y, folds)
    has = (~np.isnan(dist[:, 0])) & (els != s)
    if has.sum() < 50:
        return None
    a, b = ordinal_metrics(y[has], P0[has]), ordinal_metrics(y[has], P[has])
    return dict(sentinel=s, n_rows=int(has.sum()), n_soils=int(df.loc[has, "soil_id"].nunique()),
                macroF1_base=a["MacroF1"], macroF1_ctx=b["MacroF1"], acc_base=a["Accuracy"],
                acc_ctx=b["Accuracy"], logloss_base=a["LogLoss"], logloss_ctx=b["LogLoss"],
                dMacroF1=b["MacroF1"] - a["MacroF1"], dLogLoss=b["LogLoss"] - a["LogLoss"])


def main():
    import sys
    df = load()
    y = df["y"].values
    X0 = df[FEATS].values.astype(float)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X0, y, df["soil_id"]))
    if sys.argv[1:] == ["C"]:
        P0 = np.load(RES / "exp6A_oof.npz")["None"]
        resC = Parallel(n_jobs=2)(delayed(part_c_one)(df, y, X0, folds, s, P0) for s in SENTINELS)
        C = pd.DataFrame([r for r in resC if r is not None]).sort_values("dMacroF1", ascending=False)
        C.to_csv(RES / "exp6C_single_sentinel.csv", index=False)
        print(C.round(4).to_string(index=False))
        return
    outA, dist = part_a(df, y, X0, folds)
    has = ~np.isnan(dist[:, 0])
    rows = []
    for k, P in outA.items():
        for sub, msk in [("all", np.ones(len(y), bool)), ("with_context", has), ("no_context", ~has)]:
            met = ordinal_metrics(y[msk], P[msk])
            met.update(rep=k, subset=sub, n=int(msk.sum()))
            rows.append(met)
    A = pd.DataFrame(rows)
    A.to_csv(RES / "exp6A_context_representations.csv", index=False)
    np.savez_compressed(RES / "exp6A_oof.npz", y=y, **outA)
    print(A[A.subset == "all"][["rep", "Accuracy", "BalancedAcc", "MacroF1", "QWK", "SevereErr",
                                "F1asF4", "LogLoss", "ECE"]].round(4).to_string(index=False))
    print(A[A.subset == "with_context"][["rep", "n", "Accuracy", "MacroF1", "LogLoss"]].round(4)
          .to_string(index=False))
    # per-element gain (Distribution+Sentinels vs None)
    els = df["element"].values
    pe = []
    for e in np.unique(els):
        m = (els == e) & has
        if m.sum() < 30:
            continue
        a = (outA["None"][m].argmax(1) == y[m]).mean()
        b = (outA["Dist+Sentinels"][m].argmax(1) == y[m]).mean()
        pe.append(dict(element=e, n=int(m.sum()), acc_none=a, acc_ctx=b, gain=b - a))
    pd.DataFrame(pe).sort_values("gain").to_csv(RES / "exp6A_per_element_gain.csv", index=False)

    # Part B budget curve
    jobs = [(m, r) for m in [0, 1, 2, 3, 5, None] for r in range(3)]
    resB = Parallel(n_jobs=2)(delayed(part_b_one)(df, y, X0, folds, m, r) for m, r in jobs)
    rowsB = []
    for m, r, P in resB:
        for sub, msk in [("all", np.ones(len(y), bool)), ("with_context", has)]:
            met = ordinal_metrics(y[msk], P[msk])
            met.update(m="all" if m is None else m, rep=r, subset=sub)
            rowsB.append(met)
    B = pd.DataFrame(rowsB)
    B.to_csv(RES / "exp6B_budget_curve.csv", index=False)
    print(B[B.subset == "with_context"].groupby("m")[["Accuracy", "MacroF1", "LogLoss"]]
          .agg(["mean", "std"]).round(4))

    # Part C single-sentinel ranking
    P0 = outA["None"]
    resC = Parallel(n_jobs=2)(delayed(part_c_one)(df, y, X0, folds, s, P0) for s in SENTINELS)
    C = pd.DataFrame([r for r in resC if r is not None]).sort_values("dMacroF1", ascending=False)
    C.to_csv(RES / "exp6C_single_sentinel.csv", index=False)
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
