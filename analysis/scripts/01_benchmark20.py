"""Exp 1: 20-model benchmark under random-stratified and soil-grouped 5-fold CV.

Outputs (analysis/results):
  bench20_metrics_folds.csv   per model x scheme x fold metrics
  bench20_metrics_summary.csv mean/sd per model x scheme
  bench20_oof_<scheme>.npz    out-of-fold probabilities (for figures)
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from hm_data import FEATS, RES, TOTAL, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
LOGCOLS = [FEATS.index(c) for c in ["CEC", "OC", TOTAL]]


def log_tf(X):
    X = np.array(X, float, copy=True)
    X[:, LOGCOLS] = np.log1p(np.clip(X[:, LOGCOLS], 0, None))
    return X


def scaled(est):
    return make_pipeline(FunctionTransformer(log_tf), StandardScaler(), est)


def raw(est):
    return make_pipeline(FunctionTransformer(log_tf), est)


def xgb():
    return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
                         tree_method="hist", n_jobs=1, random_state=SEED, verbosity=0)


def lgbm():
    return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                          n_jobs=1, random_state=SEED, verbose=-1)


def cat():
    return CatBoostClassifier(iterations=1000, learning_rate=0.08, depth=6, l2_leaf_reg=3,
                              loss_function="MultiClass", thread_count=1, random_seed=SEED, allow_writing_files=False,
                              verbose=0)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  n_jobs=1, random_state=SEED)


def et():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                n_jobs=1, random_state=SEED)


def models():
    m = {
        "LR": scaled(LogisticRegression(max_iter=3000, C=1.0)),
        "Ridge": scaled(CalibratedClassifierCV(RidgeClassifier(alpha=1.0),
                                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED))),
        "LDA": scaled(LinearDiscriminantAnalysis()),
        "QDA": scaled(QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "GNB": scaled(GaussianNB()),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVM": scaled(SVC(C=10, gamma="scale", probability=True, random_state=SEED)),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                                    early_stopping=True, random_state=SEED)),
        "DT": raw(DecisionTreeClassifier(min_samples_leaf=3, random_state=SEED)),
        "RF": raw(rf()),
        "ET": raw(et()),
        "BRF": raw(BalancedRandomForestClassifier(n_estimators=500, sampling_strategy="all",
                                                  replacement=True, bootstrap=False,
                                                  n_jobs=1, random_state=SEED)),
        "Bagging": raw(BaggingClassifier(DecisionTreeClassifier(), n_estimators=300,
                                         n_jobs=1, random_state=SEED)),
        "AdaBoost": raw(AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=300,
                                           learning_rate=0.3, algorithm="SAMME",
                                           random_state=SEED)),
        "GBDT": raw(GradientBoostingClassifier(n_estimators=400, learning_rate=0.08, max_depth=5,
                                               subsample=0.8, random_state=SEED)),
        "HGB": raw(HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06,
                                                  max_leaf_nodes=63, l2_regularization=0.1,
                                                  random_state=SEED)),
        "XGBoost": raw(xgb()),
        "LightGBM": raw(lgbm()),
        "CatBoost": raw(cat()),
    }
    base = [("rf", rf()), ("et", et()), ("xgb", xgb()), ("lgbm", lgbm()), ("cat", cat())]
    m["Stacking"] = raw(StackingClassifier(base, final_estimator=LogisticRegression(max_iter=3000),
                                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                           stack_method="predict_proba", n_jobs=1))
    # NB: rows are stored in element blocks; unshuffled inner folds would hold out whole
    # elements and corrupt the stacking/calibration features, hence shuffled inner CV.
    return m


def run_fold(name, est, X, y, tr, te, scheme, fold):
    t0 = time.time()
    e = clone(est)
    e.fit(X[tr], y[tr])
    p = e.predict_proba(X[te])
    met = ordinal_metrics(y[te], p)
    met.update(model=name, scheme=scheme, fold=fold, seconds=time.time() - t0)
    return met, name, scheme, te, p


def main():
    df = load()
    X, y, g = df[FEATS].values.astype(float), df["y"].values, df["soil_id"].values
    splits = {
        "random": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)),
        "soilgroup": list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g)),
    }
    M = models()
    only = sys.argv[1:]  # optional subset of model names
    tasks = [(n, M[n], sch, k, tr, te) for n in M if (not only or n in only)
             for sch, sp in splits.items() for k, (tr, te) in enumerate(sp)]
    out = Parallel(n_jobs=3, verbose=5)(
        delayed(run_fold)(n, e, X, y, tr, te, sch, k) for n, e, sch, k, tr, te in tasks)
    rows = [o[0] for o in out]
    folds = pd.DataFrame(rows)
    tag = "" if not only else "_" + "_".join(only)
    folds.to_csv(RES / f"bench20_metrics_folds{tag}.csv", index=False)
    summ = folds.drop(columns=["fold"]).groupby(["scheme", "model"]).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index().sort_values(["scheme", "MacroF1_mean"], ascending=[True, False])
    summ.to_csv(RES / f"bench20_metrics_summary{tag}.csv", index=False)
    for sch in splits:
        oof = {}
        for _, n, s, te, p in out:
            if s != sch:
                continue
            oof.setdefault(n, np.zeros((len(y), 4)))[te] = p
        np.savez_compressed(RES / f"bench20_oof_{sch}{tag}.npz", y=y, **oof)
    cols = ["scheme", "model", "Accuracy_mean", "BalancedAcc_mean", "MacroF1_mean", "QWK_mean",
            "SevereErr_mean", "LogLoss_mean", "ECE_mean"]
    print(summ[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
