"""Exp 19: completing the soil x priority-metal speciation matrix.

Deployment model: RF + sentinels WITHOUT the total-content feature (the total of an unmeasured
element is unknown too). Its soil-grouped 5-fold OOF probabilities come from exp17
("RF+S (no total)"); those cross-fitted scores calibrate class-conditional (Mondrian) conformal
sets at alpha = 0.10, with contiguity (an F1 + F3 set also includes F2). The final model is
refitted on all rows and predicts every unmeasured (soil, metal) cell for the eight priority
metal(loid)s with >= 300 observed records in soils that have at least one measured element
(its context). Hg (71 records from few studies) is not completed: a first run filled 2,338 Hg
cells from 71 observations and shifted the labile share from 0.06 to 0.25.
Thresholds are element x class conditional (Mondrian over (metal, class) cells, n >= 30, else the
class-level threshold); their held-out coverage is checked by two-fold soil splits of the OOF
scores (calibrate on one half, evaluate on the other).

Applicability domain per cell: distance from the soil (standardised Clay, pH, CEC, OC) to the
nearest training soil that measured that metal, compared with the 95th percentile of the same
nearest-neighbour distance among the metal's own training soils (leave-one-soil-out).

Caveat recorded with the outputs: unmeasured cells are not a random sample of cells, so the
conformal guarantee (exchangeability) is not assured for them; it is checked on held-out rows.
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from hm_data import ELEM, FEATS, PRIORITY, RES, SOIL, TOTAL, load

warnings.filterwarnings("ignore")
ALPHA = 0.10
HERE = os.path.dirname(os.path.abspath(__file__))
METALS = ["Cd", "Pb", "Zn", "Cu", "Ni", "Cr", "As"]
assert set(METALS) | {"Hg"} == PRIORITY


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e6 = _mod("e6", "06_sentinel_context.py")


def encode(dist, sent):
    has = (~np.isnan(dist[:, 0])).astype(float)
    d = np.nan_to_num(dist, nan=0.0)
    oh = np.zeros((len(dist), sent.shape[1] * 4))
    for j in range(sent.shape[1]):
        ok = ~np.isnan(sent[:, j])
        oh[np.where(ok)[0], j * 4 + sent[ok, j].astype(int)] = 1.0
    return np.column_stack([d, has, oh])


def mondrian_q(P, y, alpha):
    q = np.zeros(4)
    for c in range(4):
        s = np.sort(1 - P[y == c, c])
        n = len(s)
        q[c] = s[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)]
    return q


def mondrian_q_el(P, y, els, alpha, nmin=30):
    """Thresholds per (element, class); fall back to the class-level threshold."""
    q0 = mondrian_q(P, y, alpha)
    Q = {}
    for e in np.unique(els):
        m = els == e
        q = q0.copy()
        for c in range(4):
            s = np.sort(1 - P[m & (y == c), c])
            if len(s) >= nmin:
                q[c] = s[min(len(s) - 1, int(np.ceil((len(s) + 1) * (1 - alpha))) - 1)]
        Q[e] = q
    return Q, q0


def sets_el(P, els, Q, q0):
    qq = np.array([Q.get(e, q0) for e in els])
    return sets_from(P, qq)


def sets_from(P, q):
    q = np.atleast_2d(q)
    S = (1 - P) <= q
    S[S.sum(1) == 0, P[S.sum(1) == 0].argmax(1)] = True  # never empty
    for i in np.where(S.sum(1) > 1)[0]:  # contiguity: fill gaps between included classes
        k = np.where(S[i])[0]
        S[i, k.min():k.max() + 1] = True
    return S


def main():
    df = load()
    y = df["y"].values
    noT = [c for c in FEATS if c != TOTAL]
    # --- calibration from cross-fitted OOF (exp17) -------------------------------------
    oof = np.load(RES / "exp17_oof.npz")
    P_oof = oof["RFpS_no_total"]
    els_all = df["element"].values
    Q, q = mondrian_q_el(P_oof, y, els_all, ALPHA)
    # honest check: calibrate on one half of the soils, evaluate on the other (both directions)
    rng = np.random.default_rng(0)
    soils = df["soil_id"].unique()
    half = np.isin(df["soil_id"].values, rng.choice(soils, len(soils) // 2, replace=False))
    S_oof = np.zeros_like(P_oof, bool)
    for a, b in [(half, ~half), (~half, half)]:
        Qa, qa = mondrian_q_el(P_oof[a], y[a], els_all[a], ALPHA)
        S_oof[b] = sets_el(P_oof[b], els_all[b], Qa, qa)
    cov = S_oof[np.arange(len(y)), y]
    check = (pd.DataFrame(dict(element=df["element"], covered=cov, size=S_oof.sum(1)))
             .groupby("element").agg(n=("covered", "size"), coverage=("covered", "mean"),
                                     mean_size=("size", "mean")))
    check.loc["ALL"] = [len(y), cov.mean(), S_oof.sum(1).mean()]
    check.to_csv(RES / "exp19_oof_coverage.csv")
    print("held-out coverage (split-half)", round(cov.mean(), 3), "mean size",
          round(S_oof.sum(1).mean(), 2), "class q", q.round(3), flush=True)

    # --- pseudo-rows for unmeasured (soil, metal) cells ---------------------------------
    soils = df.groupby("soil_id")
    measured = soils["element"].apply(set)
    props = soils[SOIL].first()
    desc = df.groupby("element")[ELEM].first()
    cells = [(s, m) for s in measured.index for m in METALS if m not in measured[s]]
    aug = pd.DataFrame({"soil_id": [c[0] for c in cells], "element": [c[1] for c in cells]})
    aug = aug.join(props, on="soil_id").join(desc, on="element")
    full = pd.concat([df[["soil_id", "element"] + SOIL + ELEM], aug], ignore_index=True)
    yy = np.concatenate([y, np.zeros(len(aug), int)])
    real = np.r_[np.ones(len(df), bool), np.zeros(len(aug), bool)]
    dist, _, sent = e6.build_context(full, yy, reveal=real)  # pseudo-rows never act as context
    Xc = encode(dist, sent)
    X = np.column_stack([full[noT].values.astype(float), Xc])
    model = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=2, random_state=42)
    model.fit(X[real], y)
    P = model.predict_proba(X[~real])
    S = sets_el(P, aug["element"].values, Q, q)
    out = aug[["soil_id", "element"]].copy()
    out[["pF1", "pF2", "pF3", "pF4"]] = P
    out["pred"] = P.argmax(1) + 1
    out["set"] = ["".join(str(k + 1) for k in np.where(r)[0]) for r in S]
    out["set_size"] = S.sum(1)
    out["n_context"] = dist[~real, 4]
    # --- applicability domain --------------------------------------------------------
    Z = (props - props.mean()) / props.std()
    dom = np.zeros(len(out), bool)
    dnn = np.zeros(len(out))
    for m in METALS:
        tr_s = np.array(sorted(df.loc[df.element == m, "soil_id"].unique()))
        nn = NearestNeighbors(n_neighbors=2).fit(Z.loc[tr_s].values)
        dself = nn.kneighbors(Z.loc[tr_s].values)[0][:, 1]
        thr = np.quantile(dself, 0.95)
        k = (out.element == m).values
        d = nn.kneighbors(Z.loc[out.loc[k, "soil_id"]].values, n_neighbors=1)[0][:, 0]
        dnn[k], dom[k] = d, d <= thr
    out["nn_dist"] = dnn
    out["in_domain"] = dom
    out = out[out.n_context > 0].reset_index(drop=True)
    out.to_csv(RES / "exp19_atlas_cells.csv", index=False)

    # --- summary per metal ---------------------------------------------------------------
    rows = []
    for m in METALS:
        o = out[out.element == m]
        obs = df[df.element == m]
        rows.append(dict(element=m, observed_rows=len(obs), observed_soils=obs.soil_id.nunique(),
                         completed=len(o), singleton=int((o.set_size == 1).sum()),
                         in_domain=int(o.in_domain.sum()),
                         singleton_in_domain=int(((o.set_size == 1) & o.in_domain).sum()),
                         obs_labile=float((obs.y <= 1).mean()),
                         pred_labile=float((o.pF1 + o.pF2).mean()),
                         oof_coverage=float(check.loc[m, "coverage"]),
                         oof_size=float(check.loc[m, "mean_size"])))
    sm = pd.DataFrame(rows)
    sm.to_csv(RES / "exp19_atlas_summary.csv", index=False)
    print(sm.round(3).to_string(index=False))
    print("cells", len(out), "singleton", int((out.set_size == 1).sum()),
          "in-domain", int(out.in_domain.sum()))


if __name__ == "__main__":
    main()
