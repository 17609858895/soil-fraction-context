"""Exp 7: masked co-fractionation set transformer (candidate method).

Each soil (soil_id) is a set of element tokens. Token = element descriptor embedding
(Magpie 7 + geochemistry 9, through an MLP; no element ID so unseen elements remain encodable)
+ log total content + label embedding (one of F1..F4 or [MASK]). Soil properties enter as a
conditioning token. A transformer encoder lets every masked token attend to the visible
co-measured elements; loss = cross-entropy on masked tokens only.
Training: per soil, mask a random subset (>=1 token; singletons always masked).
Evaluation (soil-group 5-fold CV, same folds as other experiments):
  ctx=none   every token masked  (context-free prediction)
  ctx=all    only the target element masked (all other elements visible)
Output: OOF probabilities for both modes; attention-derived element-pair weights.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

from hm_data import ELEM, RES, SOIL, TOTAL, load, ordinal_metrics
from hm_geochem import GEO_COLS, add_geo

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
SEED = 42
MASK = 4


def standardize(tr, *arrs):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return [(a - mu) / sd for a in (tr,) + arrs]


class SetModel(nn.Module):
    def __init__(self, d_el, d_soil, d=64, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.el = nn.Sequential(nn.Linear(d_el + 1, d), nn.GELU(), nn.Linear(d, d))
        self.lab = nn.Embedding(5, d)
        self.soil = nn.Sequential(nn.Linear(d_soil, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, xel, lab, xsoil, pad):
        # xel: B x L x d_el+1, lab: B x L, xsoil: B x d_soil, pad: B x L (True = padding)
        s = self.soil(xsoil).unsqueeze(1)
        tok = self.el(xel) + self.lab(lab) + s
        tok = torch.cat([s, tok], 1)
        pad2 = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        h = self.tr(tok, src_key_padding_mask=pad2)[:, 1:]
        return self.head(torch.cat([h, self.el(xel)], -1))


def soil_batches(soils, idx_by_soil, maxlen):
    out = []
    for s in soils:
        out.append(idx_by_soil[s][:maxlen])
    return out


def collate(rows_list, Xel, Xsoil, y, mask_fn, rng):
    B = len(rows_list)
    L = max(len(r) for r in rows_list)
    xel = np.zeros((B, L, Xel.shape[1]), np.float32)
    lab = np.full((B, L), MASK, np.int64)
    tgt = np.full((B, L), -100, np.int64)
    pad = np.ones((B, L), bool)
    xs = np.zeros((B, Xsoil.shape[1]), np.float32)
    for b, r in enumerate(rows_list):
        n = len(r)
        xel[b, :n] = Xel[r]
        pad[b, :n] = False
        xs[b] = Xsoil[r[0]]
        m = mask_fn(n, rng)
        lab[b, :n] = np.where(m, MASK, y[r])
        tgt[b, :n] = np.where(m, y[r], -100)
    return (torch.tensor(xel), torch.tensor(lab), torch.tensor(xs), torch.tensor(pad), torch.tensor(tgt))


def train_mask(n, rng):
    if n == 1:
        return np.array([True])
    k = rng.integers(1, n + 1)  # 1..n masked
    m = np.zeros(n, bool)
    m[rng.choice(n, k, replace=False)] = True
    return m


def fit(model, train_rows, Xel, Xsoil, y, w, epochs=60, bs=64, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(train_rows) // bs + 1))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), ignore_index=-100)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), bs):
            batch = [train_rows[j] for j in order[i:i + bs]]
            xel, lab, xs, pad, tgt = collate(batch, Xel, Xsoil, y, train_mask, rng)
            out = model(xel, lab, xs, pad)
            loss = lossf(out.reshape(-1, 4), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, test_rows, Xel, Xsoil, y, els, n_total, mode):
    model.eval()
    P = np.zeros((n_total, 4))
    for r in test_rows:
        r = np.asarray(r)
        if mode == "none":
            xel, lab, xs, pad, _ = collate([r], Xel, Xsoil, y, lambda n, g: np.ones(n, bool), None)
            P[r] = torch.softmax(model(xel, lab, xs, pad)[0], -1).numpy()
            continue
        # ctx=all: one pass per distinct element in the soil; mask all rows of that element
        for e in np.unique(els[r]):
            m = els[r] == e
            xel, lab, xs, pad, _ = collate([r], Xel, Xsoil, y, lambda n, g, m=m: m, None)
            p = torch.softmax(model(xel, lab, xs, pad)[0], -1).numpy()
            P[r[m]] = p[m]
    return P


def main():
    torch.manual_seed(SEED)
    df = add_geo(load())
    y = df["y"].values
    els = df["element"].values
    el_raw = df[ELEM + GEO_COLS].values.astype(float)
    tot = np.log1p(df[TOTAL].values.astype(float))[:, None]
    soil_raw = df[SOIL].values.astype(float)
    soil_raw[:, 2:] = np.log1p(soil_raw[:, 2:])  # CEC, OC
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, df["soil_id"]))
    idx_by_soil = {s: np.array(v) for s, v in df.groupby("soil_id").indices.items()}
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    Pn, Pa = np.zeros((len(y), 4)), np.zeros((len(y), 4))
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        Xel_tr, Xel_all = standardize(np.hstack([el_raw, tot])[tr], np.hstack([el_raw, tot]))
        Xs_tr, Xs_all = standardize(soil_raw[tr], soil_raw)
        tr_soils = np.unique(df["soil_id"].values[tr])
        te_soils = np.unique(df["soil_id"].values[te])
        train_rows = [idx_by_soil[s] for s in tr_soils]
        test_rows = [idx_by_soil[s] for s in te_soils]
        freq = np.bincount(y[tr], minlength=4)
        w = (len(tr) / (4 * freq)) ** 0.5  # tempered inverse-frequency weights
        model = SetModel(Xel_all.shape[1] - 1, Xs_all.shape[1])
        fit(model, train_rows, Xel_all.astype(np.float32), Xs_all.astype(np.float32), y, w,
            epochs=epochs, seed=k)
        Pn += predict(model, test_rows, Xel_all.astype(np.float32), Xs_all.astype(np.float32), y, els, len(y), "none")
        Pa += predict(model, test_rows, Xel_all.astype(np.float32), Xs_all.astype(np.float32), y, els, len(y), "all")
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(RES / "exp7_settransformer_oof.npz", y=y, ctx_none=Pn, ctx_all=Pa)
    rows = []
    for name, P in [("SetTF ctx=none", Pn), ("SetTF ctx=all", Pa)]:
        met = ordinal_metrics(y, P)
        met.update(model=name)
        rows.append(met)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exp7_settransformer_summary.csv", index=False)
    print(out[["model", "Accuracy", "BalancedAcc", "MacroF1", "QWK", "SevereErr", "LogLoss", "ECE"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
