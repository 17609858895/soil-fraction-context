"""Exp 9: residual co-fractionation network between co-measured elements.

Baseline = context-free soil-group OOF probabilities (exp6A 'None').
Ordinal residual per row: r = y - E[y | soil, element, total]   (y in 0..3, F1..F4)
  r > 0: dominant pool less labile than the baseline expects; r < 0: more labile.
For each element pair (a, b) with >= MIN_SHARED shared soils: Spearman correlation of the
soil-level mean residuals. Two-sided permutation p-values (soil labels of b shuffled),
Benjamini-Hochberg FDR across pairs. Hierarchical clustering of the correlation matrix
-> candidate co-fractionation guilds. Also reports raw (unadjusted) class agreement.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

from hm_data import RES, load

MIN_SHARED = 15
NPERM = 2000


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    df = load()
    y = df["y"].values
    P = np.load(RES / "exp6A_oof.npz", allow_pickle=True)["None"]
    df["r"] = y - P @ np.arange(4)
    soil_el = df.groupby(["soil_id", "element"])["r"].mean().unstack()  # soils x elements
    cls = df.groupby(["soil_id", "element"])["y"].agg(lambda s: s.mode().iloc[0]).unstack()
    keep = [e for e in soil_el.columns if soil_el[e].notna().sum() >= 30]
    rng = np.random.default_rng(0)
    rows = []
    for a, b in itertools.combinations(keep, 2):
        m = soil_el[a].notna() & soil_el[b].notna()
        if m.sum() < MIN_SHARED:
            continue
        xa, xb = soil_el.loc[m, a].values, soil_el.loc[m, b].values
        if xa.std() < 1e-9 or xb.std() < 1e-9:
            continue
        rho = spearmanr(xa, xb).correlation
        # vectorised permutation null: Spearman = Pearson on average ranks
        ra, rb = rankdata(xa), rankdata(xb)
        ra = (ra - ra.mean()) / ra.std()
        rb = (rb - rb.mean()) / rb.std()
        perm_idx = np.argsort(rng.random((NPERM, len(rb))), axis=1)
        perm = (rb[perm_idx] * ra).mean(1)
        p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (NPERM + 1)
        agree = (cls.loc[m, a].values == cls.loc[m, b].values).mean()
        rows.append(dict(a=a, b=b, n_shared=int(m.sum()), rho=rho, p=p, raw_agreement=agree))
    E = pd.DataFrame(rows)
    E["q"] = bh(E["p"].values)
    E.to_csv(RES / "exp9_pair_residual_corr.csv", index=False)
    els = sorted(set(E.a) | set(E.b))
    M = pd.DataFrame(np.nan, index=els, columns=els)
    for _, r in E.iterrows():
        M.loc[r.a, r.b] = M.loc[r.b, r.a] = r.rho
    np.fill_diagonal(M.values, 1.0)
    M.to_csv(RES / "exp9_residual_corr_matrix.csv")
    # clustering on elements with enough partners
    cov = M.notna().sum()
    core = cov[cov >= max(5, int(0.3 * len(els)))].index.tolist()
    D = 1 - M.loc[core, core].fillna(0).values
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip(D, 0, 2), checks=False), "average")
    guild = fcluster(Z, t=0.85, criterion="distance")
    G = pd.DataFrame({"element": core, "guild": guild}).sort_values("guild")
    G.to_csv(RES / "exp9_guilds.csv", index=False)
    print(f"pairs tested: {len(E)}, q<0.05: {(E.q < 0.05).sum()}, positive: {(E[E.q < 0.05].rho > 0).sum()}")
    print(E.sort_values("rho", ascending=False).head(25).round(3).to_string(index=False))
    print(E.sort_values("rho").head(10).round(3).to_string(index=False))
    print(G.groupby("guild")["element"].apply(list))


if __name__ == "__main__":
    main()
