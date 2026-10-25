"""Exp 18: which co-measured elements should be analysed first? (measurement design)

For a target row (soil s, element e), the candidates are the other elements measured in soil s
that belong to the 15 commonly co-measured sentinel elements. A policy reveals k of them
(k = 0..5) and the amortized sentinel RF (exp10 design: one model trained on random visibility
subsets) predicts the target's dominant fraction. All policies draw from the same candidates.

Policies
  learned priority  global ranking of the 15 commonly co-measured sentinel elements by
                    single-sentinel informativeness (mean change in log-loss when only that
                    element is visible), estimated inside each outer training set on a 20%
                    validation split of its soils (cross-fitted: never sees outer test soils).
                    Rare elements are excluded from the ranking: they occur almost only in a few
                    multi-element studies, so their apparent gain reflects study homogeneity
                    (a first run that ranked all elements put U, Sc, Rb, Lu, Tb first).
  frequency-first   elements most often measured in the training soils first
  random            30 random orders per target (median and 2.5-97.5% range reported)
Evaluation rows: outer-test target rows with >= 5 candidate sentinel elements (every k defined).
Same soil-grouped 5-fold split (seed 42).
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
KMAX = 5
NRAND = 30
HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e6 = _mod("e6", "06_sentinel_context.py")
e10 = _mod("e10", "10_amortized_masking.py")
SIDX = {e: k for k, e in enumerate(e6.SENTINELS)}


def rf(seed=SEED):
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=2, random_state=seed)


def fit_amortized(df, y, X0, rows, K=4, seed=0):
    rng = np.random.default_rng(seed)
    views = [e10.feats(df, y, X0, e10.random_visibility(df, rng, "rand"))[rows] for _ in range(K)]
    return rf().fit(np.vstack(views), np.concatenate([y[rows]] * K))


class Context:
    """Fast per-target context: class shares + count of visible other elements, sentinel ordinals."""

    def __init__(self, df, y):
        self.y = y
        self.els = df["element"].values
        self.soil_rows = df.groupby("soil_id").indices
        self.soil = df["soil_id"].values

    def pool(self, i):
        idx = self.soil_rows[self.soil[i]]
        return sorted((set(self.els[idx]) - {self.els[i]}) & set(SIDX))

    def feats(self, i, visible):
        idx = self.soil_rows[self.soil[i]]
        d = np.full(5, np.nan)
        s = np.full(len(SIDX), np.nan)
        if visible:
            o = idx[np.isin(self.els[idx], list(visible))]
            c = np.bincount(self.y[o], minlength=4).astype(float)
            d[:4], d[4] = c / c.sum(), len(o)
            for r in o:
                if self.els[r] in SIDX:
                    s[SIDX[self.els[r]]] = self.y[r]
        return np.concatenate([d, s])


def rank_elements(model, df, y, X0, ctx, rows):
    """Mean change in log-loss on `rows` when only element c is visible (vs nothing visible)."""
    base = np.column_stack([X0[rows], np.full((len(rows), 5 + len(SIDX)), np.nan)])
    p0 = model.predict_proba(base)
    ll0 = -np.log(np.clip(p0[np.arange(len(rows)), y[rows]], 1e-9, 1))
    out = []
    for c in e6.SENTINELS:
        sel = [k for k, i in enumerate(rows) if c in ctx.pool(i)]
        if len(sel) < 30:
            continue
        r = rows[sel]
        F = np.array([ctx.feats(i, {c}) for i in r])
        p = model.predict_proba(np.column_stack([X0[r], F]))
        ll = -np.log(np.clip(p[np.arange(len(r)), y[r]], 1e-9, 1))
        out.append(dict(element=c, dLogLoss=float((ll - ll0[sel]).mean()), n=len(sel)))
    return pd.DataFrame(out).sort_values("dLogLoss").reset_index(drop=True)


def choose(pool, order, k):
    ranked = [e for e in order if e in pool] + [e for e in pool if e not in order]
    return set(ranked[:k])


def main():
    t0 = time.time()
    df = load()
    y = df["y"].values
    soil = df["soil_id"].values
    X0 = df[FEATS].values.astype(float)
    ctx = Context(df, y)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X0, y, soil))
    curve, ranks = [], []
    for f, (tr, te) in enumerate(folds):
        # 1) cross-fitted priority ranking inside the outer training set
        rng = np.random.default_rng(f)
        ts = np.unique(soil[tr])
        val = np.isin(soil[tr], rng.choice(ts, len(ts) // 5, replace=False))
        m_fit = fit_amortized(df, y, X0, tr[~val], seed=f)
        rk = rank_elements(m_fit, df, y, X0, ctx, tr[val])
        rk["fold"] = f
        ranks.append(rk)
        priority = rk["element"].tolist()
        freq = [e for e in pd.Series(df["element"].values[tr]).value_counts().index
                if e in SIDX]
        del m_fit
        # 2) final amortized model on the whole outer training set
        model = fit_amortized(df, y, X0, tr, seed=100 + f)
        rows = np.array([i for i in te if len(ctx.pool(i)) >= KMAX])
        pools = {i: ctx.pool(i) for i in rows}
        policies = {"learned priority": [priority], "frequency-first": [freq]}
        rr = np.random.default_rng(1000 + f)
        for k in range(KMAX + 1):
            for pol, orders in list(policies.items()) + [("random", [None] * NRAND)]:
                for rep, order in enumerate(orders):
                    F = []
                    for i in rows:
                        if order is None:
                            vis = set(rr.permutation(pools[i])[:k])
                        else:
                            vis = choose(pools[i], order, k)
                        F.append(ctx.feats(i, vis))
                    P = model.predict_proba(np.column_stack([X0[rows], np.array(F)]))
                    met = ordinal_metrics(y[rows], P)
                    curve.append(dict(fold=f, policy=pol, k=k, rep=rep, n=len(rows),
                                      MacroF1=met["MacroF1"], Accuracy=met["Accuracy"],
                                      LogLoss=met["LogLoss"]))
            print(f"fold {f} k={k} done {time.time()-t0:.0f}s", flush=True)
        print("fold", f, "top-8 priority:", priority[:8], flush=True)
        del model
    cv = pd.DataFrame(curve)
    cv.to_csv(RES / "exp18_priority_curve.csv", index=False)
    pd.concat(ranks).to_csv(RES / "exp18_priority_rank.csv", index=False)
    # summary: per policy and k, fold-mean of the metric; random -> median and 95% range over reps
    rows = []
    for (pol, k), g in cv.groupby(["policy", "k"]):
        per_rep = g.groupby("rep")[["MacroF1", "Accuracy", "LogLoss"]].mean()
        rows.append(dict(policy=pol, k=k, MacroF1=per_rep["MacroF1"].median(),
                         MacroF1_lo=per_rep["MacroF1"].quantile(0.025),
                         MacroF1_hi=per_rep["MacroF1"].quantile(0.975),
                         LogLoss=per_rep["LogLoss"].median(), Accuracy=per_rep["Accuracy"].median()))
    sm = pd.DataFrame(rows)
    sm.to_csv(RES / "exp18_priority_summary.csv", index=False)
    print(sm.round(4).to_string(index=False))
    print(pd.concat(ranks).groupby("element")["dLogLoss"].agg(["mean", "count"]).sort_values("mean")
          .head(12).round(4).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
