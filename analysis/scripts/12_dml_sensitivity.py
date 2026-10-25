"""Exp 12: omitted-variable-bias sensitivity for the partially linear DML estimates
(Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, 'Long story short').

For theta from residual-on-residual regression:
  S^2 = E[(u - theta v)^2] / E[v^2]
  |bias| <= sqrt(cf_y * cf_d / (1 - cf_d)) * S        (rho = 1, worst case)
  cf_y: share of residual outcome variance explained by the omitted confounder
  cf_d: share of residual treatment variance explained by it (Riesz-representer gain)
Robustness value RV (cf_y = cf_d = c): smallest c with bound = |theta|; RV_a uses
|theta| - 1.96 SE. Benchmarks: leave each observed covariate out of W and measure the implied
cf_y, cf_d from the change in cross-fitted residual variances.
Targets: pH -> F1 / F2 / F4, pooled and for elements with n >= 300.
"""
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import ELEM, RES, TOTAL, load

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("e4", os.path.join(os.path.dirname(__file__),
                                                                    "04_dml_causal.py"))
e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4)


def build_W(df, drop=None):
    cols = {"OC": np.log1p(df["OC"].values), "CEC": np.log1p(df["CEC"].values),
            "Clay": df["Clay"].values, "Total": np.log1p(df[TOTAL].values)}
    keep = [v for k, v in cols.items() if k != drop]
    onehot = pd.get_dummies(df["element"], prefix="el").astype(float).values
    return np.column_stack(keep + [df[ELEM].values, onehot])


def rv(theta, se, S):
    out = []
    for target in [abs(theta), max(abs(theta) - 1.96 * se, 0)]:
        lo, hi = 0.0, 0.999
        if target == 0:
            out.append(0.0)
            continue
        for _ in range(60):
            c = (lo + hi) / 2
            if c / np.sqrt(1 - c) * S < target:
                lo = c
            else:
                hi = c
        out.append((lo + hi) / 2)
    return out


def main():
    df = load()
    T = df["pH"].values.astype(float)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=e4.SEED)
                 .split(df, df["y"].values, df["soil_id"].values))
    els = df["element"].values
    cl = df["soil_id"].values
    groups = [("ALL", np.ones(len(T), bool))] + [
        (e, els == e) for e in pd.Series(els).value_counts().index if (els == e).sum() >= 300]
    rows = []
    for o in ["F1", "F2", "F4"]:
        Y = e4.OUTCOMES[o](df["y"].values)
        v, u = e4.crossfit(build_W(df), T, Y, folds)
        bench = {}
        for z in ["OC", "CEC", "Clay", "Total"]:
            bench[z] = e4.crossfit(build_W(df, drop=z), T, Y, folds)
        for name, m in groups:
            th, se = e4.theta_cluster(v[m], u[m], cl[m])
            S = np.sqrt(np.mean((u[m] - th * v[m]) ** 2) / np.mean(v[m] ** 2))
            r, ra = rv(th, se, S)
            row = dict(outcome=o, element=name, n=int(m.sum()), theta=th, se=se, S=S, RV=r, RV_a=ra)
            for z, (vz, uz) in bench.items():
                # residual variance shrinkage when z is added back = implied partial R2
                cf_d = max(0.0, 1 - np.mean(v[m] ** 2) / np.mean(vz[m] ** 2))
                cf_y = max(0.0, 1 - np.mean((u[m] - th * v[m]) ** 2) / np.mean((uz[m] - th * vz[m]) ** 2))
                row[f"cfd_{z}"], row[f"cfy_{z}"] = cf_d, cf_y
                row[f"bias_{z}x1"] = np.sqrt(cf_y * cf_d / (1 - cf_d + 1e-12)) * S
            rows.append(row)
        print(o, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp12_dml_sensitivity.csv", index=False)
    print(out[["outcome", "element", "n", "theta", "se", "RV", "RV_a", "cfd_OC", "cfy_OC",
               "bias_OCx1", "bias_CECx1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
