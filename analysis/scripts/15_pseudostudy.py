"""Exp 15: pseudo-study reconstruction and study-level controls for the sentinel gain.

Row order inside each element block follows the source sequence (98% of adjacent soil pairs in
the Pb block are also adjacent in the Zn block). Pseudo-studies = connected components of soils
that are adjacent in any element block AND have the identical set of measured elements.
(Approximation: adjacent studies with identical element panels merge.)

A  validation by pseudo-study: StratifiedGroupKFold(5) on pseudo-study, RF without vs with
   sentinel context (NaN-free encoding of exp11).
B  context-swap controls (soil-grouped folds, seed 42; model trained on true contexts):
   true      the soil's own co-measured fractions (standard)
   same-study  fractions of another soil in the same pseudo-study with the same element set
   other-study fractions of a soil from another pseudo-study with the same element set
   evaluated on test rows for which both donors exist.
   Soil-specific share of the gain = (true - same-study) / (true - none).
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import FEATS, RES, load, ordinal_metrics

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("b1", os.path.join(HERE, "01_benchmark20.py"))
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
spec6 = importlib.util.spec_from_file_location("e6", os.path.join(HERE, "06_sentinel_context.py"))
e6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(e6)
SENT = e6.SENTINELS


def pseudo_studies(df):
    df = df.assign(row=np.arange(len(df)))
    sig = df.groupby("soil_id")["element"].apply(frozenset)
    n = df["soil_id"].max() + 1
    edges = []
    for _, g in df.groupby("element"):
        s = g.sort_values("row")["soil_id"].values
        edges += [(a, b) for a, b in zip(s[:-1], s[1:]) if a != b and sig[a] == sig[b]]
    E = np.array(edges)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return lab, sig


def encode(labels_by_el, exclude):
    """NaN-free context identical to exp11: class shares (4) + count + has + 15x4 one-hot."""
    other = {e: c for e, c in labels_by_el.items() if e != exclude}
    v = np.zeros(4 + 1 + 1 + 4 * len(SENT))
    if other:
        cls = np.array(list(other.values()))
        v[:4] = np.bincount(cls, minlength=4) / len(cls)
        v[4], v[5] = len(cls), 1.0
        for e, c in other.items():
            if e in SENT:
                v[6 + 4 * SENT.index(e) + c] = 1.0
    return v


def soil_labels(df):
    """soil -> {element: modal class} (replicate rows of an element collapse to their mode)."""
    out = {}
    for (s, e), yy in df.groupby(["soil_id", "element"])["y"]:
        out.setdefault(s, {})[e] = int(yy.mode().iloc[0])
    return out


def main():
    df = load()
    y, els, soil = df["y"].values, df["element"].values, df["soil_id"].values
    lab, sig = pseudo_studies(df)
    df["pstudy"] = lab[soil]
    df[["soil_id", "pstudy"]].drop_duplicates().to_csv(RES / "exp15_pseudostudy_map.csv", index=False)
    SL = soil_labels(df)
    X0 = df[FEATS].values.astype(float)
    Xc = np.column_stack([X0, np.array([encode(SL[s], e) for s, e in zip(soil, els)])])
    rf = b1.models()["RF"]
    rows = []
    # ---------- A: pseudo-study grouped CV
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X0, y, df["pstudy"]))
    for name, X in [("none", X0), ("sentinel", Xc)]:
        P = np.zeros((len(y), 4))
        for tr, te in folds:
            P[te] = clone(rf).fit(X[tr], y[tr]).predict_proba(X[te])
        m = ordinal_metrics(y, P)
        m.update(part="A_pseudostudy_CV", setting=name)
        rows.append(m)
        print("A", name, round(m["MacroF1"], 4), flush=True)
    # ---------- B: context-swap controls under soil-grouped CV
    rng = np.random.default_rng(0)
    soils_by_sig_study = {}
    for s in np.unique(soil):
        soils_by_sig_study.setdefault((sig[s], lab[s]), []).append(s)
    soils_by_sig = {}
    for s in np.unique(soil):
        soils_by_sig.setdefault(sig[s], []).append(s)
    donor_same, donor_other = {}, {}
    for s in np.unique(soil):
        same = [t for t in soils_by_sig_study[(sig[s], lab[s])] if t != s]
        other = [t for t in soils_by_sig[sig[s]] if lab[t] != lab[s]]
        if same and other and len(sig[s]) > 1:
            donor_same[s] = rng.choice(same)
            donor_other[s] = rng.choice(other)
    elig = np.array([s in donor_same for s in soil])
    Xs = X0.copy()
    Xsame = np.column_stack([X0, np.array([encode(SL[donor_same[s]], e) if s in donor_same
                                           else encode(SL[s], e) for s, e in zip(soil, els)])])
    Xother = np.column_stack([X0, np.array([encode(SL[donor_other[s]], e) if s in donor_other
                                            else encode(SL[s], e) for s, e in zip(soil, els)])])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X0, y, soil))
    P = {k: np.zeros((len(y), 4)) for k in ["none", "true", "same-study", "other-study"]}
    for tr, te in folds:
        m0 = clone(rf).fit(X0[tr], y[tr])
        m1 = clone(rf).fit(Xc[tr], y[tr])
        P["none"][te] = m0.predict_proba(Xs[te])
        P["true"][te] = m1.predict_proba(Xc[te])
        P["same-study"][te] = m1.predict_proba(Xsame[te])
        P["other-study"][te] = m1.predict_proba(Xother[te])
    for k, p in P.items():
        m = ordinal_metrics(y[elig], p[elig])
        m.update(part="B_context_swap", setting=k, n=int(elig.sum()),
                 n_soils=int(len(set(soil[elig]))), n_pstudies=int(len(set(lab[soil[elig]]))))
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp15_pseudostudy.csv", index=False)
    b = out[out.part == "B_context_swap"].set_index("setting")
    for met in ["MacroF1", "Accuracy", "LogLoss"]:
        g_true = b.loc["true", met] - b.loc["none", met]
        g_same = b.loc["same-study", met] - b.loc["none", met]
        print(f"{met}: none {b.loc['none', met]:.4f} true {b.loc['true', met]:.4f} same-study "
              f"{b.loc['same-study', met]:.4f} other-study {b.loc['other-study', met]:.4f} | "
              f"soil-specific share {(g_true - g_same) / g_true:.2f}")
    print(out[["part", "setting", "MacroF1", "Accuracy", "LogLoss", "ECE"]].round(4).to_string(index=False))
    print("eligible rows", int(elig.sum()), "pseudo-studies", len(set(lab)))


if __name__ == "__main__":
    main()
