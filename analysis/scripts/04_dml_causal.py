"""Exp 4: double/debiased ML (partially linear model) for adjusted effects of soil
properties on fraction dominance, with element-specific effects.

  Y = theta * T + g(W) + e,   T = m(W) + v
  T in {pH, OC, CEC, Clay}; Y in {F1 dominant, labile (F1|F2) dominant, F3 dominant, F4 dominant}
  W = remaining soil properties, log total content, element one-hot + Magpie descriptors.
Cross-fitting: 5-fold StratifiedGroupKFold on soil_id (rows sharing soil properties stay together).
Pooled theta: residual-on-residual; element theta_e: same estimator restricted to element e
(group-constant CATE, R-learner final stage). SEs: cluster-robust by soil_id.
Refutations: placebo (T permuted within element), random common cause, 80% subsets,
alternative nuisance learner (random forest).
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import ELEM, RES, SOIL, TOTAL, load
from hm_geochem import geo_table

warnings.filterwarnings("ignore")
SEED = 42
TREATS = ["pH", "OC", "CEC", "Clay"]
OUTCOMES = {"F1": lambda y: (y == 0).astype(float), "F2": lambda y: (y == 1).astype(float),
            "Labile": lambda y: (y <= 1).astype(float),
            "F3": lambda y: (y == 2).astype(float), "F4": lambda y: (y == 3).astype(float)}
MIN_N = 60


def learners(kind):
    if kind == "lgbm":
        return (LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, n_jobs=1, random_state=SEED, verbose=-1),
                LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=1, random_state=SEED, verbose=-1))
    return (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=1, random_state=SEED),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=1, random_state=SEED))


def crossfit(W, T, Y, folds, kind="lgbm"):
    vt, uy = np.zeros(len(T)), np.zeros(len(T))
    for tr, te in folds:
        mreg, lclf = learners(kind)
        mreg.fit(W[tr], T[tr])
        vt[te] = T[te] - mreg.predict(W[te])
        if len(np.unique(Y[tr])) < 2:
            uy[te] = Y[te] - Y[tr].mean()
        else:
            lclf.fit(W[tr], Y[tr])
            uy[te] = Y[te] - lclf.predict_proba(W[te])[:, 1]
    return vt, uy


def theta_cluster(v, u, cl):
    d = (v * v).sum()
    th = (v * u).sum() / d
    psi = v * (u - th * v)
    s = pd.Series(psi).groupby(cl).sum().values
    se = np.sqrt((s ** 2).sum()) / d
    return th, se


def one_job(df, W, folds, t, oname, kind="lgbm", placebo=False, rcc=False, subset=None, seed=0):
    rng = np.random.default_rng(seed)
    T = df[t].values.astype(float).copy()
    if t in ("OC", "CEC"):
        T = np.log1p(T)  # per log-unit effects for right-skewed properties
    Y = OUTCOMES[oname](df["y"].values)
    Wm = W.copy()
    if placebo:  # permute treatment within element -> true effect 0
        for el in df["element"].unique():
            m = (df["element"] == el).values
            T[m] = rng.permutation(T[m])
    if rcc:
        Wm = np.column_stack([Wm, rng.normal(size=len(T))])
    idx = np.arange(len(T)) if subset is None else subset
    v, u = np.full(len(T), np.nan), np.full(len(T), np.nan)
    sub_folds = folds if subset is None else _refold(df.iloc[idx], seed)
    vv, uu = crossfit(Wm[idx], T[idx], Y[idx], sub_folds, kind)
    v[idx], u[idx] = vv, uu
    rows = []
    cl = df["soil_id"].values
    ok = ~np.isnan(v)
    th, se = theta_cluster(v[ok], u[ok], cl[ok])
    rows.append(dict(treatment=t, outcome=oname, element="ALL", n=int(ok.sum()), theta=th, se=se,
                     sd_v=v[ok].std()))
    for el, sub in df[ok].groupby("element"):
        if len(sub) < MIN_N:
            continue
        m = (df["element"] == el).values & ok
        if v[m].std() < 1e-3:
            continue
        th, se = theta_cluster(v[m], u[m], cl[m])
        rows.append(dict(treatment=t, outcome=oname, element=el, n=int(m.sum()), theta=th, se=se,
                         sd_v=v[m].std()))
    tag = ("placebo" if placebo else "rcc" if rcc else "subset" if subset is not None else kind)
    for r in rows:
        r.update(test=tag, rep=seed)
    return rows


def _refold(sub, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + seed)
                .split(sub, sub["y"].values, sub["soil_id"].values))


def main():
    import sys
    df = load()
    if sys.argv[1:] == ["dose"]:
        onehot = pd.get_dummies(df["element"], prefix="el").astype(float).values
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                     .split(df, df["y"].values, df["soil_id"].values))
        Wo = df[["OC", "CEC", "Clay"]].values.astype(float)
        Wo[:, :2] = np.log1p(Wo[:, :2])
        W = np.column_stack([Wo, np.log1p(df[TOTAL].values), df[ELEM].values, onehot])
        dose_response(df, W, folds)
        return
    onehot = pd.get_dummies(df["element"], prefix="el").astype(float).values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                 .split(df, df["y"].values, df["soil_id"].values))
    jobs = []
    Wcache = {}
    for t in TREATS:
        others = [c for c in SOIL if c != t]
        Wo = df[others].values.astype(float)
        Wo[:, [i for i, c in enumerate(others) if c in ("OC", "CEC")]] = np.log1p(
            Wo[:, [i for i, c in enumerate(others) if c in ("OC", "CEC")]])
        Wcache[t] = np.column_stack([Wo, np.log1p(df[TOTAL].values), df[ELEM].values, onehot])
    rng = np.random.default_rng(1)
    for t in TREATS:
        for o in OUTCOMES:
            jobs.append(dict(t=t, oname=o, kind="lgbm"))
    for o in ["F1", "Labile"]:  # refutation & sensitivity on the headline treatment pH
        jobs.append(dict(t="pH", oname=o, kind="rf"))
        for s in range(3):
            jobs.append(dict(t="pH", oname=o, placebo=True, seed=s))
            jobs.append(dict(t="pH", oname=o, rcc=True, seed=s))
            jobs.append(dict(t="pH", oname=o, seed=s,
                             subset=np.sort(rng.choice(len(df), int(0.8 * len(df)), replace=False))))
    res = Parallel(n_jobs=2, verbose=5)(
        delayed(one_job)(df, Wcache[j["t"]], folds, **j) for j in jobs)
    out = pd.DataFrame([r for rr in res for r in rr])
    out["zstat"] = out["theta"] / out["se"]
    out["ci_lo"], out["ci_hi"] = out["theta"] - 1.96 * out["se"], out["theta"] + 1.96 * out["se"]
    gt = geo_table()
    out = out.merge(gt, left_on="element", right_index=True, how="left")
    out.to_csv(RES / "exp4_dml_effects.csv", index=False)
    main_ = out[out.test == "lgbm"]
    print(main_[main_.element == "ALL"][["treatment", "outcome", "n", "theta", "se", "zstat"]]
          .round(4).to_string(index=False))
    ph = main_[(main_.treatment == "pH") & (main_.outcome.isin(["F1", "Labile"])) & (main_.element != "ALL")]
    print(ph.pivot_table(index=["element", "oxy", "hsab"], columns="outcome", values=["theta", "zstat"])
          .round(3).sort_values(("theta", "F1")).to_string())
    from scipy.stats import spearmanr, mannwhitneyu
    for o in ["F1", "Labile"]:
        s = ph[ph.outcome == o]
        print(o, "Spearman(theta, IP)=", spearmanr(s.theta, s.IP), "\n   cation vs oxy MWU:",
              mannwhitneyu(s.theta[s.oxy == 0], s.theta[s.oxy == 1]) if (s.oxy == 1).sum() > 1 else "n/a")
    ref = out[(out.element == "ALL") & (out.treatment == "pH")]
    print(ref.groupby(["outcome", "test"])[["theta", "se"]].agg(["mean", "std"]).round(4))
    dose_response(df, Wcache["pH"], folds)


def dose_response(df, W, folds, outcomes=("F1", "F2", "F4"), min_n=150):
    """Partially linear additive DML: Y = f(pH) + g(W) + e with f in a cubic B-spline basis.
    Each basis column and Y are residualised on W with cross-fitting; f is recovered by
    least squares of the Y-residual on the basis residuals (pooled and per element).
    Curves are centred at pH 6.5; 95% bands from soil-cluster bootstrap of the final stage."""
    from sklearn.preprocessing import SplineTransformer
    T = df["pH"].values.astype(float)
    spl = SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False).fit(T[:, None])
    B = spl.transform(T[:, None])
    grid = np.linspace(4.0, 9.0, 51)
    Bg = spl.transform(grid[:, None]) - spl.transform(np.array([[6.5]]))
    Vb = np.zeros_like(B)
    for j in range(B.shape[1]):
        for tr, te in folds:
            mreg, _ = learners("lgbm")
            mreg.fit(W[tr], B[tr, j])
            Vb[te, j] = B[te, j] - mreg.predict(W[te])
    rng = np.random.default_rng(7)
    rows = []
    els_all = df["element"].values
    cl = df["soil_id"].values
    for o in outcomes:
        Y = OUTCOMES[o](df["y"].values)
        _, U = crossfit(W, T, Y, folds)
        groups = [("ALL", np.ones(len(T), bool))] + [
            (e, els_all == e) for e in pd.Series(els_all).value_counts().index
            if (els_all == e).sum() >= min_n]
        for name, m in groups:
            beta = np.linalg.lstsq(Vb[m], U[m], rcond=None)[0]
            curve = Bg @ beta
            by_cl = pd.Series(np.where(m)[0]).groupby(cl[m]).apply(np.array).to_dict()
            uc = np.array(list(by_cl))
            boots = []
            for b in range(200):
                pick = rng.choice(uc, len(uc), replace=True)
                idx = np.concatenate([by_cl[c] for c in pick])
                bb = np.linalg.lstsq(Vb[idx], U[idx], rcond=None)[0]
                boots.append(Bg @ bb)
            boots = np.array(boots)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for g_, c_, l_, h_ in zip(grid, curve, lo, hi):
                rows.append(dict(outcome=o, element=name, pH=g_, effect=c_, lo=l_, hi=h_,
                                 n=int(m.sum())))
    pd.DataFrame(rows).to_csv(RES / "exp4_dml_pH_dose_response.csv", index=False)
    dr = pd.DataFrame(rows)
    print(dr[dr.pH.isin([4.0, 5.0, 6.5, 8.0, 9.0])].pivot_table(
        index=["outcome", "element"], columns="pH", values="effect").round(3).to_string())


if __name__ == "__main__":
    main()
