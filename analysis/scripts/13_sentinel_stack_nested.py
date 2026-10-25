"""Exp 13: nested soil-grouped evaluation of the final Sentinel-Stack.

Outer: soil-group 5-fold (seed 42, same folds as all other experiments).
Inner (inside each outer-training set): soil-group 5-fold to create out-of-fold base
predictions for the meta-learner, so no outer-test soil influences any stacking weight.
Base learners:
  GBM-S   gradient-boosting classifier on soil + element + total + sentinel context
          (learner chosen from the 20-model benchmark; set BASE below)
  SetTF   masked co-fractionation set transformer, inference with all co-measured labels visible
Meta: multinomial logistic regression on log-probabilities of both bases.
Also reported: each base alone and the plain average. Usage: python 13_... [epochs]
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import ELEM, FEATS, RES, SOIL, TOTAL, load, ordinal_metrics
from hm_geochem import GEO_COLS, add_geo

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
HERE = os.path.dirname(__file__)
SEED = 42
BASE = os.environ.get("BASE_LEARNER", "auto")


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e6 = _mod("e6", "06_sentinel_context.py")
e11 = _mod("e11", "11_models_with_sentinels.py")
if BASE == "auto":  # best single learner (not Stacking) with sentinel context, from exp11
    _s = pd.read_csv(RES / "bench20ctx_metrics_summary.csv")
    BASE = _s[_s.model != "Stacking"].sort_values("MacroF1_mean", ascending=False).iloc[0]["model"]
e7 = _mod("e7", "07_masked_set_transformer.py")
b1 = _mod("b1", "01_benchmark20.py")


def gbm():
    from sklearn.base import clone
    return clone(b1.models()[BASE])


def settf_fit_predict(df, y, tr, te, el_raw, tot, soil_raw, idx_by_soil, els, epochs, seed):
    Xel_tr, Xel_all = e7.standardize(np.hstack([el_raw, tot])[tr], np.hstack([el_raw, tot]))
    Xs_tr, Xs_all = e7.standardize(soil_raw[tr], soil_raw)
    sid = df["soil_id"].values
    tr_rows = [idx_by_soil[s][np.isin(idx_by_soil[s], tr)] for s in np.unique(sid[tr])]
    te_rows = [idx_by_soil[s][np.isin(idx_by_soil[s], te)] for s in np.unique(sid[te])]
    freq = np.bincount(y[tr], minlength=4)
    w = (len(tr) / (4 * freq)) ** 0.5
    torch.manual_seed(seed)
    model = e7.SetModel(Xel_all.shape[1] - 1, Xs_all.shape[1])
    e7.fit(model, tr_rows, Xel_all.astype(np.float32), Xs_all.astype(np.float32), y, w,
           epochs=epochs, seed=seed)
    P = e7.predict(model, te_rows, Xel_all.astype(np.float32), Xs_all.astype(np.float32), y, els,
                   len(y), "all")
    return P[te]


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = add_geo(load())
    y, els, sid = df["y"].values, df["element"].values, df["soil_id"].values
    # NaN-free sentinel encoding (same as exp11) so any learner can be the GBM-S base
    Xg = np.column_stack([df[FEATS].values.astype(float), e11.context_matrix(df, y)])
    el_raw = df[ELEM + GEO_COLS].values.astype(float)
    tot = np.log1p(df[TOTAL].values.astype(float))[:, None]
    soil_raw = df[SOIL].values.astype(float)
    soil_raw[:, 2:] = np.log1p(soil_raw[:, 2:])
    idx_by_soil = {s: np.array(v) for s, v in df.groupby("soil_id").indices.items()}
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xg, y, sid))
    P = {k: np.zeros((len(y), 4)) for k in ["GBM-S", "SetTF", "Average", "Sentinel-Stack"]}
    W = []
    for k, (tr, te) in enumerate(outer):
        t0 = time.time()
        inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1 + k)
                     .split(Xg[tr], y[tr], sid[tr]))
        oof_g, oof_t = np.zeros((len(tr), 4)), np.zeros((len(tr), 4))
        for j, (itr, ite) in enumerate(inner):
            a, b = tr[itr], tr[ite]
            oof_g[ite] = gbm().fit(Xg[a], y[a]).predict_proba(Xg[b])
            oof_t[ite] = settf_fit_predict(df, y, a, b, el_raw, tot, soil_raw, idx_by_soil, els,
                                           epochs, seed=100 * k + j)
        Z = np.column_stack([np.log(np.clip(oof_g, 1e-6, 1)), np.log(np.clip(oof_t, 1e-6, 1))])
        meta = LogisticRegression(max_iter=3000).fit(Z, y[tr])
        W.append(meta.coef_.ravel())
        pg = gbm().fit(Xg[tr], y[tr]).predict_proba(Xg[te])
        pt = settf_fit_predict(df, y, tr, te, el_raw, tot, soil_raw, idx_by_soil, els, epochs,
                               seed=1000 + k)
        P["GBM-S"][te], P["SetTF"][te], P["Average"][te] = pg, pt, (pg + pt) / 2
        P["Sentinel-Stack"][te] = meta.predict_proba(
            np.column_stack([np.log(np.clip(pg, 1e-6, 1)), np.log(np.clip(pt, 1e-6, 1))]))
        print(f"outer fold {k} done {time.time() - t0:.0f}s", flush=True)
    rows = []
    for n, p in P.items():
        m = ordinal_metrics(y, p)
        m.update(model=n, base=BASE)
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"exp13_sentinel_stack_nested_{BASE}.csv", index=False)
    np.savez_compressed(RES / f"exp13_oof_{BASE}.npz", y=y, **{k.replace('-', '_'): v for k, v in P.items()})
    print(out[["model", "Accuracy", "BalancedAcc", "MacroF1", "QWK", "SevereErr", "F1asF4", "LogLoss",
               "ECE"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
