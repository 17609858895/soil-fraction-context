"""Exp 17: sentinel-conditioned RF vs generic imputation / multi-output baselines.

All methods see the same information for a test row (soil s, element e): soil properties, the
element's identity/descriptors (and its total content where the method can use it), and the
dominant fractions of the OTHER elements measured in soil s. Rows of element e in soil s are
never visible. Same soil-grouped 5-fold split as every other experiment (seed 42).

Methods
  RF-none          RF on soil + element features, no co-measured information (reference)
  kNN-property     per element: distance-weighted kNN over training soils that measured e
                   (soil properties + log total)
  kNN-context      kNN-property distance + lambda * ordinal disagreement on shared co-measured
                   elements (collaborative filtering); (k, lambda) chosen by inner CV
  SoftImpute       low-rank completion of the soil x (element x class) indicator matrix with soil
                   properties as side columns; (rank, shrinkage) chosen on inner masked cells
  PerElement-RF    one RF per element; features = soil properties, log total, and the observed
                   class of each of the other 51 elements (NaN if not measured)
  (missForest-style iterative filling was not used: filling a test soil's missing cells from its
   observed cells would route the target label back into its own features.)
  SetTF / RF+S     masked set transformer and RF + sentinels: OOF from exp13 (same folds)
Also: RF-none and RF+S without the total-content feature (deployment for unmeasured elements).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.utils.extmath import randomized_svd

from hm_data import FEATS, RES, SOIL, TOTAL, load, ordinal_metrics

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rf(seed=SEED, n=300):
    return RandomForestClassifier(n_estimators=n, max_features="sqrt", n_jobs=2, random_state=seed)


def soil_matrix(df, y):
    """soils x elements ordinal class (mode over replicates); NaN if not measured."""
    els = sorted(df["element"].unique())
    S = df["soil_id"].max() + 1
    C = np.full((S, len(els)), np.nan)
    ix = {e: k for k, e in enumerate(els)}
    t = pd.DataFrame({"s": df["soil_id"], "e": df["element"].map(ix), "y": y})
    mode = t.groupby(["s", "e"])["y"].agg(lambda v: v.value_counts().index[0])
    C[mode.index.get_level_values(0), mode.index.get_level_values(1)] = mode.values
    return C, els, ix


def soil_props(df):
    P = df.groupby("soil_id")[SOIL].first().sort_index().values.astype(float)
    return (P - P.mean(0)) / P.std(0)


def smooth(counts, a=0.5):
    c = counts + a
    return c / c.sum(1, keepdims=True)


# --------------------------------------------------------------------------- kNN baselines
def knn_predict(df, y, C, ix, Pz, ltot, tr, te, k, lam, use_ctx):
    P = np.zeros((len(te), 4))
    els = df["element"].values
    soil = df["soil_id"].values
    for e in np.unique(els[te]):
        ti = te[els[te] == e]
        ri = tr[els[tr] == e]
        if len(ri) == 0:
            P[np.searchsorted(te, ti)] = 0.25
            continue
        fa = np.column_stack([Pz[soil[ti]], ltot[ti]])
        fb = np.column_stack([Pz[soil[ri]], ltot[ri]])
        d = np.sqrt(((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1) / fa.shape[1])
        if use_ctx:
            A = C[soil[ti]].copy()
            B = C[soil[ri]].copy()
            A[:, ix[e]] = np.nan
            B[:, ix[e]] = np.nan
            shared = ~np.isnan(A)[:, None, :] & ~np.isnan(B)[None, :, :]
            diff = np.abs(np.nan_to_num(A)[:, None, :] - np.nan_to_num(B)[None, :, :]) / 3.0
            ns = shared.sum(-1)
            dis = np.where(ns > 0, (diff * shared).sum(-1) / np.maximum(ns, 1), 0.5)
            d = d + lam * dis
        kk = min(k, len(ri))
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        w = 1.0 / (np.take_along_axis(d, nn, 1) + 0.05)
        oh = np.eye(4)[y[ri]][nn]
        P[np.searchsorted(te, ti)] = smooth((oh * w[..., None]).sum(1), a=0.2)
    return P


def knn_cv(df, y, C, ix, Pz, ltot, folds, use_ctx):
    grid = [(k, lam) for k in (15, 30) for lam in ((0.5, 1.0, 2.0) if use_ctx else (0.0,))]
    out = np.zeros((len(y), 4))
    chosen = []
    g = df["soil_id"].values
    for tr, te in folds:
        best, bs = None, -1
        inner = list(GroupKFold(3).split(tr, groups=g[tr]))
        for k, lam in grid:
            f1 = []
            for a, b in inner:
                Pi = knn_predict(df, y, C, ix, Pz, ltot, np.sort(tr[a]), np.sort(tr[b]), k, lam, use_ctx)
                f1.append(ordinal_metrics(y[np.sort(tr[b])], Pi)["MacroF1"])
            if np.mean(f1) > bs:
                best, bs = (k, lam), np.mean(f1)
        chosen.append(best)
        out[te] = knn_predict(df, y, C, ix, Pz, ltot, np.sort(tr), np.sort(te), *best, use_ctx)[
            np.argsort(np.argsort(te))]
    return out, chosen


# --------------------------------------------------------------------------- SoftImpute
def soft_impute(M, mask, rank, lam, iters=40):
    """M: matrix with arbitrary values where mask==False; returns completed matrix."""
    Z = np.where(mask, M, 0.0)
    X = Z.copy()
    for _ in range(iters):
        U, s, Vt = randomized_svd(X, rank, n_iter=3, random_state=0)
        s = np.maximum(s - lam, 0)
        L = (U * s) @ Vt
        Xn = np.where(mask, M, L)
        if np.linalg.norm(Xn - X) / (np.linalg.norm(X) + 1e-9) < 1e-3:
            X = Xn
            break
        X = Xn
    return L


def indicator(C, Pz, w_side=1.0):
    S, E = C.shape
    M = np.zeros((S, E * 4))
    obs = np.zeros((S, E * 4), bool)
    for e in range(E):
        ok = ~np.isnan(C[:, e])
        M[np.where(ok)[0], e * 4 + C[ok, e].astype(int)] = 1.0
        obs[ok, e * 4:(e + 1) * 4] = True
    return np.column_stack([M, w_side * Pz]), np.column_stack([obs, np.ones_like(Pz, bool)])


def si_scores(L, s_idx, e):
    v = np.clip(L[s_idx, e * 4:(e + 1) * 4], 1e-3, None)
    return v / v.sum(1, keepdims=True)


def softimpute_fold(df, y, C, ix, Pz, tr, te, rank, lam):
    """Hide the target cell's element for all test soils at once, per element."""
    els = df["element"].values
    soil = df["soil_id"].values
    test_soils = np.unique(soil[te])
    Ctr = C.copy()
    # test soils keep all their cells as context; only the predicted element is hidden below
    M, obs = indicator(Ctr, Pz)
    P = np.zeros((len(te), 4))
    for e in np.unique(els[te]):
        k = ix[e]
        o = obs.copy()
        o[test_soils, k * 4:(k + 1) * 4] = False
        L = soft_impute(M, o, rank, lam)
        ti = te[els[te] == e]
        P[np.searchsorted(te, ti)] = si_scores(L, soil[ti], k)
    return P


def softimpute_cv(df, y, C, ix, Pz, folds):
    grid = [(20, 2.0), (40, 2.0), (40, 5.0), (60, 5.0)]
    out = np.zeros((len(y), 4))
    chosen = []
    soil = df["soil_id"].values
    for tr, te in folds:
        # inner selection: hold out 25% of the training soils as pseudo-test soils
        rng = np.random.default_rng(1)
        ts = np.unique(soil[tr])
        val_soils = set(rng.choice(ts, len(ts) // 4, replace=False))
        b = np.sort(tr[np.isin(soil[tr], list(val_soils))])
        a = np.sort(tr[~np.isin(soil[tr], list(val_soils))])
        Cin = C.copy()
        Cin[np.unique(soil[te])] = np.nan  # outer test soils invisible during selection
        best, bs = None, -1
        for r, l in grid:
            Pi = softimpute_fold(df, y, Cin, ix, Pz, a, b, r, l)
            f = ordinal_metrics(y[b], Pi)["MacroF1"]
            if f > bs:
                best, bs = (r, l), f
        chosen.append(best)
        Cout = C.copy()
        ste = np.sort(te)
        out[ste] = softimpute_fold(df, y, Cout, ix, Pz, np.sort(tr), ste, *best)
        print("  softimpute fold", best, round(bs, 3), flush=True)
    return out, chosen


# --------------------------------------------------------------------------- per-element RF
def per_element_cv(df, y, C, ix, Pz, ltot, folds, P_fallback):
    """Target column k is deleted, so a test soil's own label never enters its features."""
    els = df["element"].values
    soil = df["soil_id"].values
    out = np.zeros((len(y), 4))
    Cv = C
    for f, (tr, te) in enumerate(folds):
        for e in np.unique(els[te]):
            k = ix[e]
            ri = tr[els[tr] == e]
            ti = te[els[te] == e]
            if len(ri) < 30 or len(np.unique(y[ri])) < 2:
                out[ti] = P_fallback[ti]
                continue
            Xc = np.delete(Cv, k, axis=1)
            Xa = np.column_stack([Pz[soil[ri]], ltot[ri], Xc[soil[ri]]])
            Xb = np.column_stack([Pz[soil[ti]], ltot[ti], Xc[soil[ti]]])
            mod = rf(n=300).fit(Xa, y[ri])
            pr = np.zeros((len(ti), 4))
            pr[:, mod.classes_] = mod.predict_proba(Xb)
            out[ti] = pr
        print("  per-element fold", f, flush=True)
    return out


def main():
    t0 = time.time()
    df = load()
    y = df["y"].values
    soil = df["soil_id"].values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df[FEATS], y, soil))
    C, els, ix = soil_matrix(df, y)
    Pz = soil_props(df)
    lt = np.log10(df[TOTAL].values.astype(float) + 1e-3)
    ltot = (lt - lt.mean()) / lt.std()
    e11 = _mod("e11", "11_models_with_sentinels.py")
    ctx = e11.context_matrix(df, y)
    has = ctx[:, 4] > 0  # rows with >= 1 co-measured element

    preds = {}
    X0 = df[FEATS].values.astype(float)
    X0nt = df[[c for c in FEATS if c != TOTAL]].values.astype(float)
    for name, X in [("RF-none", X0), ("RF-none (no total)", X0nt),
                    ("RF+S (no total)", np.column_stack([X0nt, ctx]))]:
        P = np.zeros((len(y), 4))
        for tr, te in folds:
            P[te] = rf(n=500).fit(X[tr], y[tr]).predict_proba(X[te])
        preds[name] = P
        print(name, round(ordinal_metrics(y, P)["MacroF1"], 4), f"{time.time()-t0:.0f}s", flush=True)

    preds["kNN-property"], ch1 = knn_cv(df, y, C, ix, Pz, ltot, folds, use_ctx=False)
    print("kNN-property", ch1, round(ordinal_metrics(y, preds["kNN-property"])["MacroF1"], 4), flush=True)
    preds["kNN-context"], ch2 = knn_cv(df, y, C, ix, Pz, ltot, folds, use_ctx=True)
    print("kNN-context", ch2, round(ordinal_metrics(y, preds["kNN-context"])["MacroF1"], 4), flush=True)
    preds["SoftImpute"], ch3 = softimpute_cv(df, y, C, ix, Pz, folds)
    print("SoftImpute", ch3, round(ordinal_metrics(y, preds["SoftImpute"])["MacroF1"], 4),
          f"{time.time()-t0:.0f}s", flush=True)
    preds["PerElement-RF"] = per_element_cv(df, y, C, ix, Pz, ltot, folds, preds["RF-none"])
    print("PerElement-RF", round(ordinal_metrics(y, preds["PerElement-RF"])["MacroF1"], 4), flush=True)

    e13 = np.load(RES / "exp13_oof_RF.npz")
    assert (e13["y"] == y).all()
    preds["SetTF (masked)"] = e13["SetTF"]
    preds["RF+S (ours)"] = e13["GBM_S"]
    preds["Sentinel stack (ours)"] = e13["Sentinel_Stack"]

    rows = []
    for name, P in preds.items():
        for subset, m in [("all", np.ones(len(y), bool)), ("co-measured", has)]:
            met = ordinal_metrics(y[m], P[m])
            met.update(method=name, subset=subset, n=int(m.sum()))
            rows.append(met)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp17_imputation_baselines.csv", index=False)
    np.savez_compressed(RES / "exp17_oof.npz", y=y, **{k.replace(" ", "_").replace("(", "").replace(")", "")
                                                        .replace("+", "p"): v for k, v in preds.items()})
    # paired fold-level comparison against RF+S
    fold_rows = []
    for name, P in preds.items():
        for k, (tr, te) in enumerate(folds):
            fold_rows.append(dict(method=name, fold=k, MacroF1=ordinal_metrics(y[te], P[te])["MacroF1"]))
    pd.DataFrame(fold_rows).to_csv(RES / "exp17_folds.csv", index=False)
    pd.DataFrame(dict(method=["kNN-property", "kNN-context", "SoftImpute"],
                      chosen=[str(ch1), str(ch2), str(ch3)])).to_csv(RES / "exp17_chosen_params.csv",
                                                                     index=False)
    print(out[out.subset == "all"][["method", "MacroF1", "Accuracy", "LogLoss", "ECE"]]
          .sort_values("MacroF1", ascending=False).round(4).to_string(index=False))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
