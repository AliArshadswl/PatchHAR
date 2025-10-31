# PatchTST-HAR (Channel-Independent, Learnable PosEmb, Vanilla MHA) + HMM smoothing
# - Per-channel instance norm
# - Univariate patching per channel
# - Learnable additive positional embeddings
# - Shared Transformer encoder across channels
# - Concat pooled channel embeddings -> classifier
# - HMM post-processing (transition matrix learned from train set; Viterbi per subject)
# - Expanded metrics: Macro/Weighted F1, F2, Precision/Recall (macro), Balanced Acc,
#   Cohen's Kappa, Multiclass MCC, Pearson–Yule φ (OvR macro), Yule's Q (OvR macro),
#   Macro AUROC (OvR) using softmax probabilities.
#
# UPDATED: 5-fold cross-validation with 7:1:2 subject-level split per fold.

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import nullcontext
import math, random, json, os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.metrics import (
    f1_score, confusion_matrix, precision_recall_fscore_support,
    balanced_accuracy_score, roc_auc_score, precision_score, recall_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed_minimal/")

    SIGNAL_RATE = 100
    WINDOW_SIZE = 1000
    PATCH_LEN   = 100
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN   # 10
    CHANNELS    = 3

    # PatchTST-like encoder
    D_MODEL   = 128
    N_HEADS   = 4
    N_LAYERS  = 3
    DROPOUT   = 0.2
    FFN_DIM   = 256   # transformer feedforward hidden size

    BATCH_SIZE    = 32
    EPOCHS        = 30
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 8

    # HMM settings
    HMM_LAPLACE = 1.0       # Laplace smoothing for priors/transitions
    HMM_MIN_LOG = -20.0     # clamp log probs
    SOFTMAX_T   = 1.0       # temperature for emissions (1.0 = none)

    SEED = 42

cfg = Config()

# =========================
# Reproducibility / Device
# =========================
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print(f"🚀 Device: {device}")
if GPU:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

def amp_autocast():
    if GPU:
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()

# =========================
# Subject-level 5-fold CV splits (7:1:2 per fold)
# =========================
def make_5fold_splits_712(pids: list[str], seed: int = 42):
    """
    Returns list of 5 tuples: (train_pids, val_pids, test_pids)
    Policy:
      - Shuffle subjects once (seeded)
      - Split into 5 contiguous test folds (~20% each)
      - For each fold: validation is a deterministic sliding window (~10% of total)
        chosen from the remaining subjects (not in test) with fold-dependent offset.
      - Train is the remainder (~70%).
    """
    rng = np.random.default_rng(seed)
    pids_shuf = pids.copy()
    rng.shuffle(pids_shuf)
    N = len(pids_shuf)
    # contiguous 5 test folds
    test_chunks = np.array_split(np.arange(N), 5)

    # overall counts
    val_n = max(1, int(round(0.10 * N)))   # ~10% of ALL subjects
    folds = []
    for k in range(5):
        test_idx = test_chunks[k]
        test_set = [pids_shuf[i] for i in test_idx]

        remain_idx = np.setdiff1d(np.arange(N), test_idx, assume_unique=False)
        if len(remain_idx) == 0:
            val_set = []
            train_set = []
        else:
            # fold-dependent offset so validation rotates
            offset = (k * val_n) % len(remain_idx)
            roll_idx = np.roll(remain_idx, -offset)
            val_take = min(val_n, len(remain_idx))
            val_idx = roll_idx[:val_take]
            val_set = [pids_shuf[i] for i in val_idx]
            # train are the remaining (not in test nor val)
            train_idx = np.setdiff1d(remain_idx, val_idx, assume_unique=False)
            train_set = [pids_shuf[i] for i in train_idx]

        folds.append((train_set, val_set, test_set))
    return folds

# =========================
# Load processed metadata
# =========================
classes = json.loads((cfg.PROC_DIR / "classes.json").read_text())
label_encoder = json.loads((cfg.PROC_DIR / "label_encoder.json").read_text())
class_to_idx = {c: int(i) for c, i in label_encoder.items()}
idx_to_class = {int(i): c for c, i in label_encoder.items()}
num_classes = len(classes)
print(f"📝 Classes ({num_classes}): {classes}")

manifest = pd.read_csv(cfg.PROC_DIR / "manifest.csv")
manifest = manifest[(manifest["status"] == "ok") & (manifest["outfile"].astype(str).str.len() > 0)]
pids_all = manifest["participant"].astype(str).sort_values().tolist()

fold_splits = make_5fold_splits_712(pids_all, seed=cfg.SEED)
for i, (tr, va, te) in enumerate(fold_splits, 1):
    print(f"Fold {i}: train={len(tr)} | val={len(va)} | test={len(te)} | total={len(set(tr)|set(va)|set(te))}")

# =========================
# Dataset (PatchTST-style inputs)
# =========================
class ProcessedDataset(Dataset):
    """
    Returns:
      patches: (3, N_PATCHES, PATCH_LEN) float32  # univariate patch tokens per channel
      label:   ()                                  # long
      pid:     str
      first_ns:()                                  # long (timestamp; used for ordering in HMM)
    """
    def __init__(self, pid_list, proc_dir: Path, class_to_idx: dict):
        self.proc_dir = proc_dir
        self.class_to_idx = class_to_idx
        self.entries = []  # tuples: (pid, window, label_idx, first_ns)
        for pid in pid_list:
            npz_path = proc_dir / f"{pid}.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path, allow_pickle=True)
            W = npz["windows"].astype(np.float32)          # (N,1000,3)
            L = npz["labels_str"].astype(str)              # (N,)
            F = npz["first_ts_epoch_ns"].astype(np.int64)  # (N,)

            order = np.argsort(F)
            W, L, F = W[order], L[order], F[order]
            for w, lab, f in zip(W, L, F):
                if lab in class_to_idx:
                    self.entries.append((pid, w, int(class_to_idx[lab]), int(f)))
        self.len = len(self.entries)

    def __len__(self): return self.len

    def __getitem__(self, idx):
        pid, window, label, first_ns = self.entries[idx]

        # Expect (WINDOW_SIZE, CHANNELS)
        assert window.shape == (cfg.WINDOW_SIZE, cfg.CHANNELS), \
               f"Got {window.shape}, expected ({cfg.WINDOW_SIZE},{cfg.CHANNELS})"

        # Instance norm per channel (PatchTST)
        normed = np.zeros_like(window, dtype=np.float32)
        for c in range(cfg.CHANNELS):
            ch = window[:, c]
            mu, sd = float(ch.mean()), float(ch.std())
            normed[:, c] = (ch - mu) / (sd + 1e-8)
        normed = np.clip(normed, -10, 10)

        # Univariate patching PER CHANNEL: (C, NP, PL)
        patches = normed.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS).transpose(2,0,1).astype(np.float32)

        return (
            torch.from_numpy(patches),                      # (C, NP, PL)
            torch.tensor(label, dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
        )

def make_loader(pid_list, batch_size=32, shuffle=False):
    ds = ProcessedDataset(pid_list, cfg.PROC_DIR, class_to_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=GPU)
    return ds, dl

# =========================
# Model (PatchTST-style)
# =========================
class PatchEmbed1D(nn.Module):
    """Linear projection from patch length (PL) -> d_model (shared across channels)."""
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: (B, NP, PL)
        return self.proj(x)  # (B, NP, D)

class PositionalEmbedding(nn.Module):
    """Learnable additive positional embedding per patch index (shared across channels)."""
    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):  # x: (B, NP, D)
        return x + self.pos

def make_transformer_encoder(d_model: int, n_heads: int, n_layers: int, ffn_dim: int, dropout: float):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
        dropout=dropout, activation="relu", batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)

class PatchTSTBackbone(nn.Module):
    """
    Shared Transformer encoder (channel-independent forward passes).
    - Input per channel: (B, NP, PL) -> proj -> add pos -> encoder -> pooled (B, D)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = PatchEmbed1D(cfg.PATCH_LEN, cfg.D_MODEL)
        self.pos   = PositionalEmbedding(cfg.N_PATCHES, cfg.D_MODEL)
        self.enc   = make_transformer_encoder(cfg.D_MODEL, cfg.N_HEADS, cfg.N_LAYERS, cfg.FFN_DIM, cfg.DROPOUT)
        self.norm  = nn.LayerNorm(cfg.D_MODEL)

    def forward_one_channel(self, x_c):  # x_c: (B, NP, PL)
        tok = self.embed(x_c)            # (B, NP, D)
        tok = self.pos(tok)              # (B, NP, D)
        tok = self.enc(tok)              # (B, NP, D)
        tok = self.norm(tok)             # (B, NP, D)
        return tok.mean(dim=1)           # (B, D)  (mean-pool over patches)

    def forward(self, patches):          # patches: (B, C, NP, PL)
        B, C, NP, PL = patches.shape
        outs = []
        for c in range(C):
            x_c = patches[:, c]          # (B, NP, PL)
            outs.append(self.forward_one_channel(x_c))  # (B, D)
        return torch.cat(outs, dim=1)    # (B, C*D)

class PatchTSTClassifier(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.backbone = PatchTSTBackbone(cfg)  # shared encoder
        self.cls = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg.CHANNELS * cfg.D_MODEL, cfg.D_MODEL),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.D_MODEL, num_classes),
        )

    def forward(self, patches):  # (B, C, NP, PL)
        z = self.backbone(patches)  # (B, C*D)
        return self.cls(z)          # (B, K)

# =========================
# Metrics (core + requested)
# =========================
def cohen_kappa_standard(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=None)
    n = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm)/n
    pe = np.dot(cm.sum(1), cm.sum(0))/(n*n)
    return (po - pe) / (1 - pe) if abs(1-pe) > 1e-12 else 0.0

def multiclass_mcc_gorodkin(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    if n == 0: return 0.0
    s = np.trace(cm); t = cm.sum(1); p = cm.sum(0)
    num = s*n - np.sum(t*p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.0) * max(n**2 - np.sum(p**2), 0.0))
    return num/den if den > 0 else 0.0

def phi_yule_ovr_macro(y_true, y_pred, K):
    """
    Pearson–Yule's φ for binary is identical to MCC. For multiclass we do OvR and macro-average.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    phis = []
    for k in range(K):
        t = (y_true == k)
        p = (y_pred == k)
        TP = np.sum(t & p)
        TN = np.sum(~t & ~p)
        FP = np.sum(~t & p)
        FN = np.sum(t & ~p)
        num = TP*TN - FP*FN
        den = math.sqrt(max((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN), 0.0))
        phis.append(num/den if den > 0 else 0.0)
    return float(np.mean(phis))

def yules_q_ovr_macro(y_true, y_pred, K):
    """
    Yule's Q = (ad - bc) / (ad + bc) on 2x2; here OvR macro-average.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    Qs = []
    for k in range(K):
        t = (y_true == k); p = (y_pred == k)
        a = np.sum(t & p)         # TP
        d = np.sum(~t & ~p)       # TN
        b = np.sum(~t & p)        # FP
        c = np.sum(t & ~p)        # FN
        num = a*d - b*c
        den = a*d + b*c
        Qs.append(num/den if den > 0 else 0.0)
    return float(np.mean(Qs))

def macro_fbeta(y_true, y_pred, beta=2.0):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro", beta=beta, zero_division=0)
    return f

def compute_all_metrics(y_true, y_pred, K, proba=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    out = {}
    out["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    out["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    out["f2_macro"] = macro_fbeta(y_true, y_pred, beta=2.0)
    out["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    out["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    out["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    out["kappa"] = cohen_kappa_standard(y_true, y_pred)
    out["mcc"] = multiclass_mcc_gorodkin(y_true, y_pred)
    out["phi_yule_macro"] = phi_yule_ovr_macro(y_true, y_pred, K)
    out["yules_q_macro"] = yules_q_ovr_macro(y_true, y_pred, K)
    # AUROC (macro OvR) if probabilities are available and valid
    if proba is not None:
        try:
            out["auroc_macro_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except Exception:
            out["auroc_macro_ovr"] = float("nan")
    else:
        out["auroc_macro_ovr"] = float("nan")
    return out

# =========================
# HMM utilities (estimate from train, Viterbi decode on test)
# =========================
def estimate_hmm_from_dataset(train_ds: ProcessedDataset, K: int, laplace: float = 1.0):
    """
    Estimate initial distribution π and transition matrix A from ordered label sequences per subject.
    Laplace smoothing applied to avoid zeros.
    """
    pi_counts = np.full(K, laplace, dtype=np.float64)
    A_counts  = np.full((K, K), laplace, dtype=np.float64)

    # Gather labels per PID (entries are already time-sorted within each PID)
    by_pid: dict[str, list[int]] = {}
    for pid, _w, lab, _ts in train_ds.entries:
        by_pid.setdefault(pid, []).append(int(lab))

    for pid, labs in by_pid.items():
        if len(labs) == 0: continue
        pi_counts[labs[0]] += 1
        for a, b in zip(labs[:-1], labs[1:]):
            A_counts[a, b] += 1

    pi = pi_counts / pi_counts.sum()
    A  = A_counts / A_counts.sum(axis=1, keepdims=True)
    return pi.astype(np.float64), A.astype(np.float64)

def viterbi_log(emissions_log: np.ndarray, A_log: np.ndarray, pi_log: np.ndarray) -> np.ndarray:
    """
    Viterbi in log-space.
    emissions_log: (T, K)
    A_log: (K, K)
    pi_log: (K,)
    Returns: best path (T,)
    """
    T, K = emissions_log.shape
    delta = np.full((T, K), -np.inf, dtype=np.float64)
    psi   = np.zeros((T, K), dtype=np.int32)

    delta[0] = pi_log + emissions_log[0]
    psi[0] = 0

    for t in range(1, T):
        for j in range(K):
            vals = delta[t-1] + A_log[:, j]
            psi[t, j] = int(np.argmax(vals))
            delta[t, j] = vals[psi[t, j]] + emissions_log[t, j]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path

def apply_hmm_viterbi_over_subjects(proba: np.ndarray, pids: np.ndarray, first_ns: np.ndarray, pi: np.ndarray, A: np.ndarray):
    """
    proba: (N, K) softmax probabilities (emissions)
    pids:  (N,) subject ids (dtype object)
    first_ns: (N,) timestamps for ordering
    Returns array of length N with Viterbi-decoded labels, aligned to input order.
    """
    K = proba.shape[1]
    # Prepare logs with clamping for numerical stability
    eps = 1e-12
    emissions_log_all = np.log(np.clip(proba, eps, 1.0))
    emissions_log_all = np.maximum(emissions_log_all, cfg.HMM_MIN_LOG)
    A_log  = np.log(np.clip(A, eps, 1.0))
    pi_log = np.log(np.clip(pi, eps, 1.0))

    decoded = np.zeros(proba.shape[0], dtype=np.int32)
    # Process per subject
    unique_pids = pd.Index(pids).unique().tolist()
    for pid in unique_pids:
        idx = np.where(pids == pid)[0]
        # sort by time within subject
        order = idx[np.argsort(first_ns[idx])]
        em = emissions_log_all[order]
        path = viterbi_log(em, A_log, pi_log)
        decoded[order] = path
    return decoded

# =========================
# Training utils
# =========================
def compute_class_weights(train_ds, K):
    counts = np.zeros(K, dtype=np.int64)
    for pid, w, lab, f in train_ds.entries:
        counts[lab] += 1
    weights = counts.max() / np.clip(counts, 1, None)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * K  # normalize around 1.0

def train_model(model, train_loader, val_loader, class_w: torch.Tensor | None = None, patience: int = 8, save_path: Path | None = None):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)
    scaler = GradScaler(enabled=GPU)

    best_score = -1e9
    best_epoch, patience_ctr = -1, 0
    if save_path is None:
        save_path = cfg.PROC_DIR / "patchtst_ci_best.pth"

    for epoch in range(cfg.EPOCHS):
        # ---- Train ----
        model.train()
        total = 0.0
        for patches, labels, pids, first_ns in train_loader:
            patches = patches.to(device)             # (B,C,NP,PL)
            labels  = labels.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                logits = model(patches)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item())
        scheduler.step()

        # ---- Validation
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for patches, labels, pids, first_ns in val_loader:
                patches = patches.to(device)
                labels  = labels.to(device).view(-1)
                pred    = model(patches).argmax(1)
                val_preds.extend(pred.cpu().numpy().tolist())
                val_truths.extend(labels.cpu().numpy().tolist())

        val_preds, val_truths = np.array(val_preds), np.array(val_truths)
        f1 = f1_score(val_truths, val_preds, average="macro")
        kappa = cohen_kappa_standard(val_truths, val_preds)
        mcc = multiclass_mcc_gorodkin(val_truths, val_preds)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR {current_lr:.2e} | "
              f"Loss {total/max(1,len(train_loader)):.4f} | F1 {f1:.4f} | Kappa {kappa:.4f} | MCC {mcc:.4f}")

        score = f1 + kappa
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model (F1={f1:.4f}, Kappa={kappa:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}; best epoch {best_epoch+1}")
                break

# -------------------------
@torch.no_grad()
def get_prob_predictions(model, loader, temperature: float = 1.0):
    """
    Returns per-sample softmax probabilities (emissions), argmax predictions, truths, PIDs, timestamps.
    """
    model.eval()
    all_probs, preds, truths, pids, first = [], [], [], [], []
    softmax = nn.Softmax(dim=1)
    for patches, labels, sub_ids, firsts in loader:
        patches = patches.to(device)
        logits  = model(patches)
        if temperature != 1.0:
            logits = logits / temperature
        prob    = softmax(logits).cpu().numpy()
        pred    = np.argmax(prob, axis=1)
        all_probs.append(prob)
        preds.extend(pred.tolist())
        truths.extend(labels.numpy().tolist())
        pids.extend(list(sub_ids))
        first.extend(firsts.numpy().tolist())
    proba = np.vstack(all_probs) if len(all_probs) else np.zeros((0, num_classes), dtype=np.float32)
    return proba, np.array(preds), np.array(truths), np.array(pids, dtype=object), np.array(first, dtype=np.int64)

# =========================
# Plotting helpers
# =========================
def save_confusion(cm, classes, title, outpath):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, square=True)
    plt.title(title)
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()
    print(f"🖼️ Saved: {outpath}")

# =========================
# Run (5-fold CV)
# =========================
def main():
    cfg.PROC_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    keys = ["f1_macro","f1_weighted","f2_macro","precision_macro","recall_macro",
            "balanced_acc","kappa","mcc","phi_yule_macro","yules_q_macro","auroc_macro_ovr"]

    for fold_idx, (train_pids, val_pids, test_pids) in enumerate(fold_splits, start=1):
        print("\n" + "="*80)
        print(f"🔁 Fold {fold_idx} — train={len(train_pids)} | val={len(val_pids)} | test={len(test_pids)}")
        print("="*80)

        # Loaders
        train_ds, train_dl = make_loader(train_pids, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_ds,   val_dl   = make_loader(val_pids,   batch_size=cfg.BATCH_SIZE, shuffle=False)
        test_ds,  test_dl  = make_loader(test_pids,  batch_size=cfg.BATCH_SIZE, shuffle=False)

        print("🧩 Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

        # Model
        model = PatchTSTClassifier(cfg, num_classes).to(device)

        # Class weights (from this fold's train set)
        class_w = compute_class_weights(train_ds, num_classes).to(device)

        # Train
        fold_best_path = cfg.PROC_DIR / f"patchtst_ci_best_fold{fold_idx}.pth"
        print("\n🎯 Training (PatchTST: channel-independent, learnable pos-emb)...")
        train_model(model, train_dl, val_dl, class_w=class_w, patience=cfg.EARLY_STOP_PATIENCE, save_path=fold_best_path)

        # Load best weights for this fold
        if fold_best_path.exists():
            model.load_state_dict(torch.load(fold_best_path, map_location=device))

        # === Estimate HMM from training sequences ===
        pi, A = estimate_hmm_from_dataset(train_ds, num_classes, laplace=cfg.HMM_LAPLACE)
        print("\n📈 HMM priors (π):", np.round(pi, 4))
        print("📊 HMM transitions (A) row-normalized:")
        with np.printoptions(precision=3, suppress=True):
            print(A)

        # === Test predictions (probabilities for AUROC + HMM emissions) ===
        print("\n🎯 Testing...")
        proba_raw, pred_raw, true_raw, pid_raw, first_raw = get_prob_predictions(model, test_dl, temperature=cfg.SOFTMAX_T)

        # --- Raw metrics ---
        metrics_raw = compute_all_metrics(true_raw, pred_raw, num_classes, proba=proba_raw)
        cm_raw = confusion_matrix(true_raw, pred_raw, labels=np.arange(num_classes))
        save_confusion(cm_raw, classes,
                       f"[Fold {fold_idx}] Confusion Matrix (Raw) — MacroF1 {metrics_raw['f1_macro']:.3f}",
                       cfg.PROC_DIR / f"cm_patchtst_ci_raw_fold{fold_idx}.png")

        # --- HMM Viterbi decode per subject ---
        pred_hmm = apply_hmm_viterbi_over_subjects(proba_raw, pid_raw, first_raw, pi, A)
        metrics_hmm = compute_all_metrics(true_raw, pred_hmm, num_classes, proba=proba_raw)
        cm_hmm = confusion_matrix(true_raw, pred_hmm, labels=np.arange(num_classes))
        save_confusion(cm_hmm, classes,
                       f"[Fold {fold_idx}] Confusion Matrix (HMM) — MacroF1 {metrics_hmm['f1_macro']:.3f}",
                       cfg.PROC_DIR / f"cm_patchtst_ci_hmm_fold{fold_idx}.png")

        # === Per-fold printout ===
        def fmt(x):
            return "nan" if (x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.3f}"
        print("\n📋 Test Metrics (RAW vs HMM-smoothed) — Fold", fold_idx)
        header = "{:>18} | {:>8} | {:>8}".format("Metric","RAW","HMM")
        print(header); print("-"*len(header))
        for k in keys:
            print("{:>18} | {:>8} | {:>8}".format(k, fmt(metrics_raw[k]), fmt(metrics_hmm[k])))

        # Keep results for aggregation
        row = {"fold": fold_idx}
        for k in keys:
            row[f"{k}_raw"] = float(metrics_raw[k])
            row[f"{k}_hmm"] = float(metrics_hmm[k])
        all_fold_results.append(row)

        # Save fold bundle
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "metrics_raw": metrics_raw,
            "metrics_hmm": metrics_hmm,
            "hmm": {"pi": pi, "A": A},
            "config": vars(cfg),
            "fold": fold_idx,
            "splits": {"train_pids": train_pids, "val_pids": val_pids, "test_pids": test_pids},
        }, cfg.PROC_DIR / f"patchtst_ci_hmm_artifacts_fold{fold_idx}.pth")
        print(f"💾 Saved fold bundle: {cfg.PROC_DIR / f'patchtst_ci_hmm_artifacts_fold{fold_idx}.pth'}")

    # === Aggregate across folds ===
    print("\n" + "="*80)
    print("📈 Cross-Validation Summary (5 folds, 7:1:2 per fold)")
    print("="*80)
    df = pd.DataFrame(all_fold_results).sort_values("fold")
    # Mean ± Std for each metric (RAW/HMM)
    for suffix in ["raw","hmm"]:
        print(f"\n➡️  {suffix.UPPER()} metrics (mean ± std):")
        for k in keys:
            m = df[f"{k}_{suffix}"].mean()
            s = df[f"{k}_{suffix}"].std(ddof=1) if len(df) > 1 else 0.0
            print(f"{k:>18}: {m:.4f} ± {s:.4f}")

    # Save CSV
    out_csv = cfg.PROC_DIR / "cv5_patchtst_ci_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved metrics table: {out_csv}")

if __name__ == "__main__":
    main()
