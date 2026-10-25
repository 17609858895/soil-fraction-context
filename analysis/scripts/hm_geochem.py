"""Aqueous/soil-geochemistry element descriptors (candidate representation).

Values are textbook constants and must be re-verified against the cited sources
before manuscript use (SI table):
  z      dominant dissolved/exchangeable oxidation state under oxic soil conditions
         (definition chosen so Mn = +2; Ce(IV), Cr(VI), V(IV) noted as alternatives in the SI)
  r_ion  Shannon (1976) effective ionic radius, CN6, Angstrom, for that state (high-spin)
  chi    element-level revised Pauling electronegativity (Allred 1961, as compiled by
         WebElements); Eu, Tb, Yb have no authoritative value and are estimates
  hsab   Pearson (1963, 1968) acid class for the stated oxidation state: 0 hard, 1 borderline,
         2 soft (Sn(IV), As(V), Sb(V) hard)
  oxy    1 if the dominant soil species is an oxyanion/neutral oxyacid (B, Si, As, Sb, V, Mo, W)
  redox  1 if redox-sensitive in soils (Fe, Mn, Cr, As, Sb, V, Mo, U, Ce, Tl, Co, Hg)
  gold   Goldschmidt class: 0 lithophile, 1 chalcophile, 2 siderophile
Verified 2026-10-22 against the Shannon database and WebElements
(analysis/literature/地球化学描述符核对_2026-10-22.md); pre-verification copy archived.
Derived: ionic potential IP = z / r_ion; z^2/r (cation field strength proxy).
"""
import numpy as np
import pandas as pd

#        z   r_ion   chi  hsab oxy redox gold
GEO = {
    "Li": (1, 0.76, 0.98, 0, 0, 0, 0), "Na": (1, 1.02, 0.93, 0, 0, 0, 0),
    "K": (1, 1.38, 0.82, 0, 0, 0, 0), "Rb": (1, 1.52, 0.82, 0, 0, 0, 0),
    "Cs": (1, 1.67, 0.79, 0, 0, 0, 0), "Be": (2, 0.45, 1.57, 0, 0, 0, 0),
    "Mg": (2, 0.72, 1.31, 0, 0, 0, 0), "Ca": (2, 1.00, 1.00, 0, 0, 0, 0),
    "Sr": (2, 1.18, 0.95, 0, 0, 0, 0), "Ba": (2, 1.35, 0.89, 0, 0, 0, 0),
    "B": (3, 0.27, 2.04, 0, 1, 0, 0), "Al": (3, 0.535, 1.61, 0, 0, 0, 0),
    "Si": (4, 0.40, 1.90, 0, 1, 0, 0), "Sc": (3, 0.745, 1.36, 0, 0, 0, 0),
    "Ti": (4, 0.605, 1.54, 0, 0, 0, 0), "V": (5, 0.54, 1.63, 0, 1, 1, 0),
    "Cr": (3, 0.615, 1.66, 0, 0, 1, 0), "Mn": (2, 0.83, 1.55, 0, 0, 1, 0),
    "Fe": (3, 0.645, 1.83, 0, 0, 1, 2), "Co": (2, 0.745, 1.88, 1, 0, 1, 2),
    "Ni": (2, 0.69, 1.91, 1, 0, 0, 2), "Cu": (2, 0.73, 1.90, 1, 0, 0, 1),
    "Zn": (2, 0.74, 1.65, 1, 0, 0, 1), "As": (5, 0.46, 2.18, 0, 1, 1, 1),
    "Y": (3, 0.90, 1.22, 0, 0, 0, 0), "Zr": (4, 0.72, 1.33, 0, 0, 0, 0),
    "Mo": (6, 0.59, 2.16, 0, 1, 1, 2), "Ag": (1, 1.15, 1.93, 2, 0, 0, 1),
    "Cd": (2, 0.95, 1.69, 2, 0, 0, 1), "Sn": (4, 0.69, 1.96, 0, 0, 0, 1),
    "Sb": (5, 0.60, 2.05, 0, 1, 1, 1), "La": (3, 1.032, 1.10, 0, 0, 0, 0),
    "Ce": (3, 1.01, 1.12, 0, 0, 1, 0), "Pr": (3, 0.99, 1.13, 0, 0, 0, 0),
    "Nd": (3, 0.983, 1.14, 0, 0, 0, 0), "Sm": (3, 0.958, 1.17, 0, 0, 0, 0),
    "Eu": (3, 0.947, 1.20, 0, 0, 0, 0), "Gd": (3, 0.938, 1.20, 0, 0, 0, 0),
    "Tb": (3, 0.923, 1.20, 0, 0, 0, 0), "Dy": (3, 0.912, 1.22, 0, 0, 0, 0),
    "Ho": (3, 0.901, 1.23, 0, 0, 0, 0), "Er": (3, 0.89, 1.24, 0, 0, 0, 0),
    "Tm": (3, 0.88, 1.25, 0, 0, 0, 0), "Yb": (3, 0.868, 1.10, 0, 0, 0, 0),
    "Lu": (3, 0.861, 1.27, 0, 0, 0, 0), "W": (6, 0.60, 2.36, 0, 1, 0, 2),
    "Hg": (2, 1.02, 2.00, 2, 0, 1, 1), "Tl": (1, 1.50, 1.62, 2, 0, 1, 1),
    "Pb": (2, 1.19, 2.33, 1, 0, 0, 1), "Bi": (3, 1.03, 2.02, 1, 0, 0, 1),
    "Th": (4, 0.94, 1.30, 0, 0, 0, 0), "U": (6, 0.73, 1.38, 0, 0, 1, 0),
}
GEO_COLS = ["z", "r_ion", "chi", "hsab", "oxy", "redox", "gold", "IP", "z2r"]


def geo_table():
    t = pd.DataFrame.from_dict(GEO, orient="index",
                               columns=["z", "r_ion", "chi", "hsab", "oxy", "redox", "gold"])
    t["IP"] = t["z"] / t["r_ion"]
    t["z2r"] = t["z"] ** 2 / t["r_ion"]
    return t


def add_geo(df):
    t = geo_table()
    g = t.loc[df["element"].values, GEO_COLS].reset_index(drop=True)
    g.index = df.index
    return pd.concat([df, g], axis=1)
