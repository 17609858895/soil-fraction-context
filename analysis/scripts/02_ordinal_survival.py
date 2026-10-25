"""Exp 2: treat the BCR sequence as a discrete 'extraction time' and compare
nominal, ordinal and survival formulations on the same base learner.

BCR steps: step 1 acetic acid (F1), step 2 hydroxylammonium (F2), step 3 H2O2/NH4OAc (F3);
F4 (residual) = not released by the three reagent steps -> right-censored at step 3.

Formulations (all output a 4-class probability vector):
  Nominal      multiclass booster
  Cumulative   Frank & Hall P(y>k) binaries
  CR-forward   continuation ratio h_k = P(y=k | y>=k) in BCR order (separate binaries)
  DTS-PP       discrete-time survival: person-period expansion, one shared binary booster
               with step index as covariate
  AFT          XGBoost accelerated-failure-time on interval-censored latent extraction time
               (F1:(0,1], F2:(1,2], F3:(2,3], F4:(3,inf))
  CR-reverse   continuation ratio starting from F4 (direction falsification)
  CR-random    continuation ratio in 5 random non-BCR class orders (order falsification)
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import xgboost as xgb

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
K = 4


def lgb_bin(seed=SEED):
    return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                          n_jobs=1, random_state=seed, verbose=-1)


def fit_nominal(Xtr, ytr, Xte):
    m = lgb_bin().fit(Xtr, ytr)
    return m.predict_proba(Xte)


def fit_cumulative(Xtr, ytr, Xte):
    gt = np.column_stack([lgb_bin().fit(Xtr, (ytr > k).astype(int)).predict_proba(Xte)[:, 1]
                          for k in range(K - 1)])
    gt = np.minimum.accumulate(gt, axis=1)  # enforce P(y>0)>=P(y>1)>=P(y>2)
    p = np.column_stack([1 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1] - gt[:, 2], gt[:, 2]])
    return np.clip(p, 1e-6, 1)


def cr_probs(h):
    """hazards h (n x K-1) -> class probabilities in the given order."""
    n = h.shape[0]
    p = np.zeros((n, K))
    surv = np.ones(n)
    for k in range(K - 1):
        p[:, k] = surv * h[:, k]
        surv = surv * (1 - h[:, k])
    p[:, K - 1] = surv
    return p


def fit_cr(Xtr, ytr, Xte, order=(0, 1, 2, 3)):
    rank = np.empty(K, int)
    rank[list(order)] = np.arange(K)
    r = rank[ytr]
    h = np.zeros((len(Xte), K - 1))
    for k in range(K - 1):
        m = r >= k
        h[:, k] = lgb_bin().fit(Xtr[m], (r[m] == k).astype(int)).predict_proba(Xte)[:, 1]
    p_ord = cr_probs(h)
    p = np.zeros_like(p_ord)
    p[:, list(order)] = p_ord
    return p


def person_period(X, y):
    rows, ev = [], []
    for j in range(K - 1):
        m = y >= j
        rows.append(np.column_stack([X[m], np.full(m.sum(), j)]))
        ev.append((y[m] == j).astype(int))
    return np.vstack(rows), np.concatenate(ev)


def fit_dts(Xtr, ytr, Xte):
    Xpp, ev = person_period(Xtr, ytr)
    m = lgb_bin().fit(Xpp, ev)
    h = np.column_stack([m.predict_proba(np.column_stack([Xte, np.full(len(Xte), j)]))[:, 1]
                         for j in range(K - 1)])
    return cr_probs(h)


def fit_aft(Xtr, ytr, Xte, dist="normal", sigma=0.6):
    lo = np.array([0.0, 1.0, 2.0, 3.0])[ytr]
    hi = np.array([1.0, 2.0, 3.0, np.inf])[ytr]
    d = xgb.DMatrix(Xtr)
    d.set_float_info("label_lower_bound", lo)
    d.set_float_info("label_upper_bound", hi)
    params = {"objective": "survival:aft", "aft_loss_distribution": dist,
              "aft_loss_distribution_scale": sigma, "eta": 0.05, "max_depth": 7,
              "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
              "nthread": 1, "seed": SEED, "verbosity": 0}
    b = xgb.train(params, d, num_boost_round=600)
    mu = b.predict(xgb.DMatrix(Xte), output_margin=True)
    cdf = norm.cdf if dist == "normal" else (lambda z: 1 / (1 + np.exp(-z)))
    edges = np.log(np.array([1.0, 2.0, 3.0]))
    F = np.column_stack([cdf((e - mu) / sigma) for e in edges])
    p = np.column_stack([F[:, 0], F[:, 1] - F[:, 0], F[:, 2] - F[:, 1], 1 - F[:, 2]])
    return np.clip(p, 1e-6, 1), mu


def random_orders(n=5, seed=0):
    rng = np.random.default_rng(seed)
    perms = [p for p in itertools.permutations(range(K)) if p not in [(0, 1, 2, 3), (3, 2, 1, 0)]]
    idx = rng.choice(len(perms), n, replace=False)
    return [perms[i] for i in idx]


def run_fold(X, y, tr, te, scheme, fold):
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    out = {"Nominal": fit_nominal(Xtr, ytr, Xte),
           "Cumulative": fit_cumulative(Xtr, ytr, Xte),
           "CR-forward": fit_cr(Xtr, ytr, Xte),
           "DTS-PP": fit_dts(Xtr, ytr, Xte),
           "CR-reverse": fit_cr(Xtr, ytr, Xte, order=(3, 2, 1, 0))}
    p_aft, mu = fit_aft(Xtr, ytr, Xte)
    out["AFT"] = p_aft
    for i, o in enumerate(random_orders()):
        out[f"CR-random{i+1}"] = fit_cr(Xtr, ytr, Xte, order=o)
    out["Hybrid(Nom+DTS)"] = (out["Nominal"] + out["DTS-PP"]) / 2
    out["Hybrid(Nom+DTS+AFT)"] = (out["Nominal"] + out["DTS-PP"] + out["AFT"]) / 3
    return scheme, fold, te, out, mu


def main():
    df = load()
    X, y, g = df[FEATS].values.astype(float), df["y"].values, df["soil_id"].values
    splits = {
        "random": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)),
        "soilgroup": list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g)),
    }
    res = Parallel(n_jobs=5)(delayed(run_fold)(X, y, tr, te, s, k)
                              for s, sp in splits.items() for k, (tr, te) in enumerate(sp))
    rows, oof = [], {}
    for s, k, te, out, mu in res:
        for name, p in out.items():
            met = ordinal_metrics(y[te], p)
            met.update(model=name, scheme=s, fold=k)
            rows.append(met)
            oof.setdefault((s, name), np.zeros((len(y), K)))[te] = p
        oof.setdefault((s, "AFT_mu"), np.zeros((len(y), 1)))[te, 0] = mu
    folds = pd.DataFrame(rows)
    folds.to_csv(RES / "exp2_ordinal_survival_folds.csv", index=False)
    summ = folds.drop(columns="fold").groupby(["scheme", "model"]).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(RES / "exp2_ordinal_survival_summary.csv", index=False)
    for s in splits:
        np.savez_compressed(RES / f"exp2_oof_{s}.npz", y=y,
                            **{n: v for (ss, n), v in oof.items() if ss == s})
    cols = ["scheme", "model", "Accuracy_mean", "BalancedAcc_mean", "MacroF1_mean", "QWK_mean",
            "OrdinalMAE_mean", "SevereErr_mean", "F1asF4_mean", "LogLoss_mean", "ECE_mean"]
    print(summ[cols].sort_values(["scheme", "MacroF1_mean"], ascending=[True, False])
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
