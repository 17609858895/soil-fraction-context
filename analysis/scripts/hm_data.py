"""Shared data loading, element identification and evaluation metrics.

Data: data-figshare.csv (Figshare 10.6084/m9.figshare.29554202.v5, CC BY 4.0).
class 1-4 = dominant BCR fraction F1 (acid-soluble), F2 (reducible),
F3 (oxidizable), F4 (residual); mapping verified against the author's
Source data.xlsx Fig6 proportions (F1 0.0924 = 877/9489, etc.).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data-figshare.csv"
# HM_RES lets sentinel experiments with another learner write to results/<learner>/
RES = Path(os.environ.get("HM_RES", ROOT / "analysis" / "results"))
RES.mkdir(parents=True, exist_ok=True)

SOIL = ["Clay", "pH", "CEC", "OC"]
ELEM = ["CovalentRadius", "ElectronAffinity", "GSmagmom", "MeltingT",
        "NfUnfilled", "NfValence", "SpaceGroupNumber"]
TOTAL = "Total metal(loid)s content"
FEATS = SOIL + ELEM + [TOTAL]
CLASS_NAMES = ["F1 acid-soluble", "F2 reducible", "F3 oxidizable", "F4 residual"]

# Element identification from (CovalentRadius, MeltingT[K]); Magpie/matminer tables
# (Cordero covalent radius in pm; melting point in K). Checked against EA/NfValence.
ELEMENT_KEY = {
    (84, 2348.0): "B", (96, 1560.0): "Be", (111, 1687.0): "Si", (119, 1090.0): "As",
    (121, 933.47): "Al", (122, 692.68): "Zn", (124, 1728.0): "Ni", (126, 1768.0): "Co",
    (128, 453.69): "Li", (132, 234.32): "Hg", (132, 1811.0): "Fe", (132, 1357.77): "Cu",
    (139, 1519.0): "Mn", (139, 2180.0): "Cr", (139, 903.78): "Sb", (139, 505.08): "Sn",
    (141, 923.0): "Mg", (144, 594.22): "Cd", (145, 577.0): "Tl", (145, 1234.93): "Ag",
    (146, 600.61): "Pb", (148, 544.4): "Bi", (153, 2183.0): "V", (154, 2896.0): "Mo",
    (160, 1941.0): "Ti", (162, 3695.0): "W", (166, 370.87): "Na", (170, 1814.0): "Sc",
    (175, 2128.0): "Zr", (176, 1115.0): "Ca", (187, 1092.0): "Yb", (187, 1936.0): "Lu",
    (189, 1770.0): "Tm", (190, 1799.0): "Y", (190, 1818.0): "Er", (192, 1685.0): "Dy",
    (192, 1747.0): "Ho", (194, 1629.0): "Tb", (195, 1050.0): "Sr", (196, 1408.0): "U",
    (196, 1586.0): "Gd", (198, 1095.0): "Eu", (198, 1345.0): "Sm", (201, 1294.0): "Nd",
    (203, 336.53): "K", (203, 1204.0): "Pr", (204, 1071.0): "Ce", (206, 2023.0): "Th",
    (207, 1193.0): "La", (215, 1000.0): "Ba", (220, 312.46): "Rb", (244, 301.59): "Cs",
}

REE = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}
PRIORITY = {"Cd", "Pb", "Zn", "Cu", "Ni", "Cr", "As", "Hg"}  # common regulated metal(loid)s


def load(dedup=False):
    df = pd.read_csv(DATA)
    df["y"] = df["class"].astype(int) - 1
    key = list(zip(df["CovalentRadius"].astype(int), df["MeltingT"].round(2)))
    df["element"] = [ELEMENT_KEY.get((r, round(m, 2)), "?") for r, m in key]
    assert (df["element"] != "?").all(), "unidentified element descriptor set"
    df["soil_id"] = df.groupby(SOIL).ngroup()
    df["el_id"] = df.groupby("element").ngroup()
    if dedup:
        df = df.drop_duplicates(subset=FEATS + ["class"]).reset_index(drop=True)
    return df


def ordinal_metrics(y, proba, prefix=""):
    """Metrics for 4-class ordinal target (0..3). proba: n x 4."""
    proba = np.clip(np.asarray(proba, float), 1e-9, 1)
    proba = proba / proba.sum(1, keepdims=True)
    yhat = proba.argmax(1)
    err = np.abs(yhat - y)
    conf = proba.max(1)
    # 15-bin top-label expected calibration error
    bins = np.linspace(0, 1, 16)
    idx = np.digitize(conf, bins) - 1
    ece = 0.0
    for b in range(15):
        m = idx == b
        if m.any():
            ece += m.mean() * abs((yhat[m] == y[m]).mean() - conf[m].mean())
    out = {
        "Accuracy": accuracy_score(y, yhat),
        "BalancedAcc": balanced_accuracy_score(y, yhat),
        "MacroF1": f1_score(y, yhat, average="macro"),
        "WeightedF1": f1_score(y, yhat, average="weighted"),
        "QWK": cohen_kappa_score(y, yhat, weights="quadratic"),
        "OrdinalMAE": err.mean(),
        "SevereErr": (err >= 2).mean(),           # >=2 steps on the BCR ladder
        "F1asF4": ((y == 0) & (yhat == 3)).sum() / max((y == 0).sum(), 1),  # mobile missed as residual
        "LogLoss": log_loss(y, proba, labels=[0, 1, 2, 3]),
        "ECE": ece,
    }
    for k in range(4):
        out[f"Recall_F{k+1}"] = ((yhat == k) & (y == k)).sum() / max((y == k).sum(), 1)
    return {prefix + k: v for k, v in out.items()}
