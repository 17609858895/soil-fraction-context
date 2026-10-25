"""Exp 14: statistical robustness of the benchmark and of the sentinel gain.

Modes (python 14_robustness.py <mode>):
  sent   RF / ET / LightGBM, without vs with sentinel context (NaN-free encoding of exp11),
         3 repeats x soil-grouped 5-fold (seeds 42, 7, 2026) -> 15 paired folds per learner;
         Wilcoxon signed-rank on paired fold differences; soil-cluster bootstrap 95% CI of
         the pooled OOF difference (repeat seed 42, 1000 resamples of soils).
  rep20  all 20 learners (default settings), seeds 7 and 2026; merged with exp1 (seed 42)
         -> 15 blocks; Friedman test and Nemenyi critical difference on macro-F1.
  tune   nested randomized search for ET, RF, LightGBM, XGBoost and CatBoost: outer soil-
         grouped 5-fold (seed 42), inner soil-grouped 3-fold, scoring macro-F1.
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
SEEDS = [42, 7, 2026]
NJ = int(os.environ.get("E14_JOBS", "2"))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b1 = _mod("b1", "01_benchmark20.py")
e11 = _mod("e11", "11_models_with_sentinels.py")


def folds_for(X, y, g, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g))


def fit_fold(name, est, X, y, tr, te, tag, seed, k):
    e = clone(est).fit(X[tr], y[tr])
    p = e.predict_proba(X[te])
    m = ordinal_metrics(y[te], p)
    m.update(model=name, setting=tag, seed=seed, fold=k)
    return m, te, p


def mode_sent(df, y, g):
    X0 = df[FEATS].values.astype(float)
    Xc = np.column_stack([X0, e11.context_matrix(df, y)])
    M = b1.models()
    learners = ["RF", "ET", "LightGBM"]
    tasks = []
    for seed in SEEDS:
        for k, (tr, te) in enumerate(folds_for(X0, y, g, seed)):
            for n in learners:
                tasks += [(n, M[n], X0, y, tr, te, "none", seed, k), (n, M[n], Xc, y, tr, te, "sentinel", seed, k)]
    out = Parallel(n_jobs=NJ, verbose=2)(delayed(fit_fold)(*t) for t in tasks)
    rows = pd.DataFrame([o[0] for o in out])
    rows.to_csv(RES / "exp14_sent_folds.csv", index=False)
    # paired fold-level tests
    summ = []
    for n in learners:
        a = rows[(rows.model == n) & (rows.setting == "none")].sort_values(["seed", "fold"])
        b = rows[(rows.model == n) & (rows.setting == "sentinel")].sort_values(["seed", "fold"])
        for met in ["MacroF1", "Accuracy", "LogLoss", "QWK"]:
            d = b[met].values - a[met].values
            w = stats.wilcoxon(d)
            summ.append(dict(model=n, metric=met, none_mean=a[met].mean(), none_sd=a[met].std(),
                             sent_mean=b[met].mean(), sent_sd=b[met].std(), diff_mean=d.mean(),
                             diff_min=d.min(), diff_max=d.max(), n_pairs=len(d),
                             n_improved=int((d < 0).sum() if met == "LogLoss" else (d > 0).sum()),
                             wilcoxon_p=w.pvalue))
    # soil-cluster bootstrap on pooled OOF of seed 42
    oof = {}
    for (m, te, p) in out:
        if m["seed"] == 42:
            oof.setdefault((m["model"], m["setting"]), np.zeros((len(y), 4)))[te] = p
    rng = np.random.default_rng(0)
    by_soil = pd.Series(np.arange(len(y))).groupby(g).apply(np.array).to_dict()
    soils = np.array(list(by_soil))
    boot = []
    for n in learners:
        P0, P1 = oof[(n, "none")], oof[(n, "sentinel")]
        for _ in range(1000):
            idx = np.concatenate([by_soil[s] for s in rng.choice(soils, len(soils), replace=True)])
            f0 = f1_score(y[idx], P0[idx].argmax(1), average="macro")
            f1 = f1_score(y[idx], P1[idx].argmax(1), average="macro")
            l0 = -np.log(np.clip(P0[idx, y[idx]], 1e-12, 1)).mean()
            l1 = -np.log(np.clip(P1[idx, y[idx]], 1e-12, 1)).mean()
            boot.append(dict(model=n, dF1=f1 - f0, dLL=l1 - l0))
    B = pd.DataFrame(boot)
    ci = B.groupby("model").agg(dF1_mean=("dF1", "mean"), dF1_lo=("dF1", lambda s: s.quantile(0.025)),
                                dF1_hi=("dF1", lambda s: s.quantile(0.975)),
                                dLL_mean=("dLL", "mean"), dLL_lo=("dLL", lambda s: s.quantile(0.025)),
                                dLL_hi=("dLL", lambda s: s.quantile(0.975))).reset_index()
    S = pd.DataFrame(summ)
    S.to_csv(RES / "exp14_sent_paired_tests.csv", index=False)
    ci.to_csv(RES / "exp14_sent_bootstrap_ci.csv", index=False)
    print(S.round(4).to_string(index=False))
    print(ci.round(4).to_string(index=False))


def mode_rep20(df, y, g):
    X = df[FEATS].values.astype(float)
    M = b1.models()
    tasks = [(n, M[n], X, y, tr, te, "none", seed, k) for seed in SEEDS[1:]
             for k, (tr, te) in enumerate(folds_for(X, y, g, seed)) for n in M]
    out = Parallel(n_jobs=NJ, verbose=2)(delayed(fit_fold)(*t) for t in tasks)
    new = pd.DataFrame([o[0] for o in out])
    e1 = pd.read_csv(RES / "bench20_metrics_folds.csv")
    e1 = e1[e1.scheme == "soilgroup"].assign(seed=42, setting="none")
    allf = pd.concat([e1[new.columns.intersection(e1.columns)], new], ignore_index=True)
    allf.to_csv(RES / "exp14_rep20_folds.csv", index=False)
    friedman_nemenyi(allf)


def friedman_nemenyi(allf, metric="MacroF1"):
    piv = allf.pivot_table(index=["seed", "fold"], columns="model", values=metric)
    ranks = piv.rank(axis=1, ascending=False)
    k, N = piv.shape[1], piv.shape[0]
    fr = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    out = pd.DataFrame({"mean_rank": ranks.mean(), "mean": piv.mean(), "sd": piv.std()}).sort_values("mean_rank")
    out["friedman_chi2"], out["friedman_p"], out["CD"], out["N_blocks"] = fr.statistic, fr.pvalue, cd, N
    out.to_csv(RES / "exp14_rep20_friedman.csv")
    print(out.round(4).to_string())


SPACES = {
    "ET": {"n_estimators": [300, 500, 800], "max_features": ["sqrt", 0.5, 0.8],
           "min_samples_leaf": [1, 2, 4], "max_depth": [None, 20, 40],
           "class_weight": [None, "balanced_subsample"]},
    "RF": {"n_estimators": [300, 500, 800], "max_features": ["sqrt", 0.5, 0.8],
           "min_samples_leaf": [1, 2, 4], "max_depth": [None, 20, 40],
           "class_weight": [None, "balanced_subsample"]},
    "LightGBM": {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.02, 0.05, 0.1],
                 "n_estimators": [300, 600, 1000], "min_child_samples": [5, 10, 20, 40],
                 "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 5],
                 "class_weight": [None, "balanced"]},
    "XGBoost": {"max_depth": [4, 6, 8, 10], "learning_rate": [0.02, 0.05, 0.1],
                "n_estimators": [300, 600, 1000], "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.5, 1, 5]},
    "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.08, 0.15],
                 "iterations": [500, 1000], "l2_leaf_reg": [1, 3, 9]},
}
N_ITER = {"ET": 15, "RF": 15, "LightGBM": 15, "XGBoost": 15, "CatBoost": 8}


def mode_tune(df, y, g):
    X = df[FEATS].values.astype(float)
    M = b1.models()
    outer = folds_for(X, y, g, 42)
    which = sys.argv[2:] or list(SPACES)
    rows, best = [], []
    for n in which:
        pipe = M[n]
        step = pipe.steps[-1][0]
        space = {f"{step}__{k}": v for k, v in SPACES[n].items()}
        for k, (tr, te) in enumerate(outer):
            t0 = time.time()
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=100 + k)
            rs = RandomizedSearchCV(clone(pipe), space, n_iter=N_ITER[n], scoring="f1_macro",
                                    cv=inner, n_jobs=NJ, random_state=k, refit=True)
            rs.fit(X[tr], y[tr], groups=g[tr])
            p = rs.predict_proba(X[te])
            m = ordinal_metrics(y[te], p)
            m.update(model=n, fold=k, inner_best=rs.best_score_, seconds=time.time() - t0)
            rows.append(m)
            best.append(dict(model=n, fold=k, **{kk.split("__")[1]: str(v) for kk, v in rs.best_params_.items()}))
            print(n, k, round(m["MacroF1"], 4), round(time.time() - t0), "s", flush=True)
        pd.DataFrame(rows).to_csv(RES / "exp14_tune_folds.csv", index=False)
        pd.DataFrame(best).to_csv(RES / "exp14_tune_best_params.csv", index=False)
    T = pd.DataFrame(rows).groupby("model")[["Accuracy", "BalancedAcc", "MacroF1", "QWK", "LogLoss", "ECE"]].agg(["mean", "std"])
    T.columns = [f"{a}_{b}" for a, b in T.columns]
    T.reset_index().to_csv(RES / "exp14_tune_summary.csv", index=False)
    print(T.round(4).to_string())


if __name__ == "__main__":
    df = load()
    y, g = df["y"].values, df["soil_id"].values
    {"sent": mode_sent, "rep20": mode_rep20, "tune": mode_tune}[sys.argv[1]](df, y, g)
