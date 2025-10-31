# CT-PatchTST-HAR (Classification) + HMM smoothing — NO STRIDE (non-overlapping patches)
# - Same HAR pipeline, HMM postproc, and metrics as your original
# - CT-PatchTST backbone (RevIN, Channel→Time attention) with NON-OVERLAPPING patching
# - If WINDOW_SIZE % PATCH_LEN != 0, the tail is dropped (no padding, no overlap)

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
# Config (NO STRIDE)
# =========================
class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed_minimal/")
    TRAIN_N    = 80
    VAL_N      = 20

    SIGNAL_RATE = 100
    WINDOW_SIZE = 1000
    CHANNELS    = 3

    # ---- Non-overlapping patching ----
    # Use PATCH_LEN that divides WINDOW_SIZE to keep all samples.
    # If it doesn't divide, tail is dropped (no overlap).
    PATCH_LEN   = 100  # matches your original (1000 // 100 = 10 patches)
    # N = WINDOW_SIZE // PATCH_LEN  (computed in-model)

    # CT-Encoder (paper-aligned sizes)
    D_MODEL   = 256
    N_BLOCKS  = 4
    HEADS_CH  = 1
    HEADS_TM  = 16
    DROPOUT   = 0.1
    FFN_DIM   = 512

    # Training
    BATCH_SIZE    = 128
    EPOCHS        = 50
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 8

    # HMM
    HMM_LAPLACE = 1.0
    HMM_MIN_LOG = -20.0
    SOFTMAX_T   = 1.0

    SEED = 42

cfg = Config()

# =========================
# Repro / Device
# =========================
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print(f"🚀 Device: {device}")
if GPU: print(f"   GPU: {torch.cuda.get_device_name(0)}")

def amp_autocast():
    if GPU:
        try: return torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()

# =========================
# Data (unchanged I/O; model does RevIN+patching)
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
n_train = min(cfg.TRAIN_N, len(pids_all)); n_val = min(cfg.VAL_N, max(0, len(pids_all) - n_train))
train_pids, val_pids, test_pids = pids_all[:n_train], pids_all[n_train:n_train+n_val], pids_all[n_train+n_val:]
print(f"Subject split: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")

class ProcessedDataset(Dataset):
    def __init__(self, pid_list, proc_dir: Path, class_to_idx: dict):
        self.entries = []
        for pid in pid_list:
            p = proc_dir / f"{pid}.npz"
            if not p.exists(): continue
            z = np.load(p, allow_pickle=True)
            W = z["windows"].astype(np.float32)          # (N,1000,3)
            L = z["labels_str"].astype(str)              # (N,)
            F = z["first_ts_epoch_ns"].astype(np.int64)  # (N,)
            order = np.argsort(F); W, L, F = W[order], L[order], F[order]
            for w, lab, f in zip(W, L, F):
                if lab in class_to_idx and w.shape == (cfg.WINDOW_SIZE, cfg.CHANNELS):
                    self.entries.append((pid, w, int(class_to_idx[lab]), int(f)))
        self.len = len(self.entries)

    def __len__(self): return self.len
    def __getitem__(self, idx):
        pid, window, label, first_ns = self.entries[idx]
        return (
            torch.from_numpy(window.astype(np.float32)),   # raw (W,C)
            torch.tensor(label, dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
        )

def make_loader(pid_list, batch_size=32, shuffle=False):
    ds = ProcessedDataset(pid_list, cfg.PROC_DIR, class_to_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=GPU)
    return ds, dl

# =========================
# CT-PatchTST backbone (NO STRIDE)
# =========================
def compute_num_patches_nostride(L: int, P: int) -> int:
    # Non-overlapping: N = floor(L / P)
    return L // P

def patchify_no_stride(x: torch.Tensor, P: int) -> torch.Tensor:
    """
    x: (B, L, M) -> (B, M, N, P) with NON-OVERLAPPING patches.
    If L % P != 0, tail is dropped so that N = floor(L/P).
    """
    B, L, M = x.shape
    N = compute_num_patches_nostride(L, P)
    L_used = N * P
    if L_used < L:
        x = x[:, :L_used, :]  # drop tail (no overlap)
    x = x.view(B, N, P, M)         # (B, N, P, M)
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, M, N, P)
    return x

class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps; self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_channels))
            self.bias   = nn.Parameter(torch.zeros(1, 1, num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)

    def forward(self, x, mode: str, stats=None):
        if mode == 'norm':
            mu = x.mean(dim=1, keepdim=True); sigma = x.std(dim=1, keepdim=True) + self.eps
            xn = (x - mu) / sigma
            if self.affine: xn = xn * self.weight + self.bias
            return xn, (mu, sigma)
        elif mode == 'denorm':
            mu, sigma = stats
            if self.affine: x = (x - self.bias) / (self.weight + 1e-8)
            return x * sigma + mu
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, x):  # x: (B,M,N,D)
        return x + self.pos.unsqueeze(1)

class MHABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, axis: str):
        B, M, N, D = x.shape
        if axis == 'channel':
            x = x.permute(0,2,1,3).contiguous().view(B*N, M, D)
            r = x; o,_ = self.mha(x,x,x,need_weights=False); x = self.ln1(r + self.drop(o))
            r = x; x = self.ln2(r + self.drop(self.ffn(x)))
            return x.view(B,N,M,D).permute(0,2,1,3).contiguous()
        else:  # 'time'
            x = x.view(B*M, N, D)
            r = x; o,_ = self.mha(x,x,x,need_weights=False); x = self.ln1(r + self.drop(o))
            r = x; x = self.ln2(r + self.drop(self.ffn(x)))
            return x.view(B,M,N,D)

class CTBlock(nn.Module):
    def __init__(self, d_model, heads_ch, heads_tm, ffn_dim, dropout):
        super().__init__()
        self.ch = MHABlock(d_model, heads_ch, ffn_dim, dropout)
        self.tm = MHABlock(d_model, heads_tm, ffn_dim, dropout)
    def forward(self, x):
        x = self.ch(x, 'channel')
        x = self.tm(x, 'time')
        return x

class CTPatchTSTBackbone(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.P = cfg.PATCH_LEN
        self.L = cfg.WINDOW_SIZE
        self.M = cfg.CHANNELS
        self.N = compute_num_patches_nostride(self.L, self.P)  # <-- NO STRIDE
        if self.N * self.P != self.L:
            print(f"⚠️ Non-divisible window: using only first {self.N*self.P}/{self.L} samples per window (no overlap).")
        self.revin = RevIN(self.M, eps=1e-5, affine=False)
        self.patch_proj = nn.Linear(self.P, cfg.D_MODEL)
        self.pos = PositionalEmbedding(self.N, cfg.D_MODEL)
        self.blocks = nn.ModuleList([CTBlock(cfg.D_MODEL, cfg.HEADS_CH, cfg.HEADS_TM, cfg.FFN_DIM, cfg.DROPOUT)
                                     for _ in range(cfg.N_BLOCKS)])
        self.norm = nn.LayerNorm(cfg.D_MODEL)
    def forward(self, window):  # (B,W,C)
        x,_ = self.revin(window, mode='norm')           # (B,W,M)
        patches = patchify_no_stride(x, self.P)         # (B,M,N,P) non-overlap
        tok = self.patch_proj(patches)                  # (B,M,N,D)
        tok = self.pos(tok)                             # (B,M,N,D)
        z = tok
        for blk in self.blocks: z = blk(z)
        z = self.norm(z)
        z = z.mean(dim=2)                               # (B,M,D) mean over patches
        return z.view(z.size(0), -1)                    # (B,M*D)

class PatchTSTClassifier(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.backbone = CTPatchTSTBackbone(cfg)
        self.cls = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg.CHANNELS * cfg.D_MODEL, cfg.D_MODEL),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.D_MODEL, num_classes),
        )
    def forward(self, window):
        return self.cls(self.backbone(window))

# =========================
# Metrics (same)
# =========================
def cohen_kappa_standard(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=None)
    n = cm.sum(); 
    if n == 0: return 0.0
    po = np.trace(cm)/n; pe = np.dot(cm.sum(1), cm.sum(0))/(n*n)
    return (po - pe) / (1 - pe) if abs(1-pe) > 1e-12 else 0.0

def multiclass_mcc_gorodkin(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum(); 
    if n == 0: return 0.0
    s = np.trace(cm); t = cm.sum(1); p = cm.sum(0)
    num = s*n - np.sum(t*p)
    den = math.sqrt(max(n**2 - np.sum(t**2), 0.0) * max(n**2 - np.sum(p**2), 0.0))
    return num/den if den > 0 else 0.0

def phi_yule_ovr_macro(y_true, y_pred, K):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); phis = []
    for k in range(K):
        t = (y_true == k); p = (y_pred == k)
        TP = np.sum(t & p); TN = np.sum(~t & ~p); FP = np.sum(~t & p); FN = np.sum(t & ~p)
        num = TP*TN - FP*FN
        den = math.sqrt(max((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN), 0.0))
        phis.append(num/den if den > 0 else 0.0)
    return float(np.mean(phis))

def yules_q_ovr_macro(y_true, y_pred, K):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); Qs = []
    for k in range(K):
        t = (y_true == k); p = (y_pred == k)
        a = np.sum(t & p); d = np.sum(~t & ~p); b = np.sum(~t & p); c = np.sum(t & ~p)
        num = a*d - b*c; den = a*d + b*c
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
    if proba is not None:
        try: out["auroc_macro_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except Exception: out["auroc_macro_ovr"] = float("nan")
    else:
        out["auroc_macro_ovr"] = float("nan")
    return out

# =========================
# HMM utils (same)
# =========================
def estimate_hmm_from_dataset(train_ds: ProcessedDataset, K: int, laplace: float = 1.0):
    pi_counts = np.full(K, laplace, dtype=np.float64)
    A_counts  = np.full((K, K), laplace, dtype=np.float64)
    by_pid: dict[str, list[int]] = {}
    for pid, _w, lab, _ts in train_ds.entries:
        by_pid.setdefault(pid, []).append(int(lab))
    for labs in by_pid.values():
        if not labs: continue
        pi_counts[labs[0]] += 1
        for a, b in zip(labs[:-1], labs[1:]): A_counts[a, b] += 1
    pi = pi_counts / pi_counts.sum()
    A  = A_counts / A_counts.sum(axis=1, keepdims=True)
    return pi.astype(np.float64), A.astype(np.float64)

def viterbi_log(emissions_log: np.ndarray, A_log: np.ndarray, pi_log: np.ndarray) -> np.ndarray:
    T, K = emissions_log.shape
    delta = np.full((T, K), -np.inf, dtype=np.float64); psi = np.zeros((T, K), dtype=np.int32)
    delta[0] = pi_log + emissions_log[0]
    for t in range(1, T):
        for j in range(K):
            vals = delta[t-1] + A_log[:, j]
            psi[t, j] = int(np.argmax(vals))
            delta[t, j] = vals[psi[t, j]] + emissions_log[t, j]
    path = np.zeros(T, dtype=np.int32); path[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1): path[t] = psi[t+1, path[t+1]]
    return path

def apply_hmm_viterbi_over_subjects(proba: np.ndarray, pids: np.ndarray, first_ns: np.ndarray, pi: np.ndarray, A: np.ndarray):
    K = proba.shape[1]; eps = 1e-12
    em_log = np.log(np.clip(proba, eps, 1.0)); em_log = np.maximum(em_log, cfg.HMM_MIN_LOG)
    A_log  = np.log(np.clip(A, eps, 1.0));    pi_log = np.log(np.clip(pi, eps, 1.0))
    decoded = np.zeros(proba.shape[0], dtype=np.int32)
    for pid in pd.Index(pids).unique().tolist():
        idx = np.where(pids == pid)[0]; order = idx[np.argsort(first_ns[idx])]
        decoded[order] = viterbi_log(em_log[order], A_log, pi_log)
    return decoded

# =========================
# Train / Eval (same)
# =========================
def compute_class_weights(train_ds, K):
    counts = np.zeros(K, dtype=np.int64)
    for _,_,lab,_ in train_ds.entries: counts[lab] += 1
    weights = counts.max() / np.clip(counts, 1, None)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * K

def train_model(model, train_loader, val_loader, class_w: torch.Tensor | None = None, patience: int = 8):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)
    scaler = GradScaler(enabled=GPU)

    best_score = -1e9; best_path = cfg.PROC_DIR / "ct_patchtst_har_best.pth"
    best_epoch, patience_ctr = -1, 0

    for epoch in range(cfg.EPOCHS):
        model.train(); total = 0.0; steps = 0
        for windows, labels, pids, first_ns in train_loader:
            windows = windows.to(device); labels = labels.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                logits = model(windows)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item()); steps += 1
        scheduler.step()

        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for windows, labels, pids, first_ns in val_loader:
                pred = model(windows.to(device)).argmax(1)
                val_preds.extend(pred.cpu().numpy().tolist())
                val_truths.extend(labels.numpy().tolist())
        val_preds, val_truths = np.array(val_preds), np.array(val_truths)
        f1 = f1_score(val_truths, val_preds, average="macro")
        kappa = cohen_kappa_standard(val_truths, val_preds)
        mcc = multiclass_mcc_gorodkin(val_truths, val_preds)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR {optimizer.param_groups[0]['lr']:.2e} | "
              f"Loss {total/max(1,steps):.4f} | F1 {f1:.4f} | Kappa {kappa:.4f} | MCC {mcc:.4f}")

        score = f1 + kappa
        if score > best_score + 1e-6:
            best_score = score; best_epoch = epoch; patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model (F1={f1:.4f}, Kappa={kappa:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}; best epoch {best_epoch+1}")
                break

@torch.no_grad()
def get_prob_predictions(model, loader, temperature: float = 1.0):
    model.eval()
    all_probs, preds, truths, pids, first = [], [], [], [], []
    softmax = nn.Softmax(dim=1)
    for windows, labels, sub_ids, firsts in loader:
        logits  = model(windows.to(device))
        if temperature != 1.0: logits = logits / temperature
        prob    = softmax(logits).cpu().numpy(); pred = np.argmax(prob, axis=1)
        all_probs.append(prob)
        preds.extend(pred.tolist()); truths.extend(labels.numpy().tolist())
        pids.extend(list(sub_ids)); first.extend(firsts.numpy().tolist())
    proba = np.vstack(all_probs) if len(all_probs) else np.zeros((0, num_classes), dtype=np.float32)
    return proba, np.array(preds), np.array(truths), np.array(pids, dtype=object), np.array(first, dtype=np.int64)

def save_confusion(cm, classes, title, outpath):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, square=True)
    plt.title(title); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()
    print(f"🖼️ Saved: {outpath}")

# =========================
# Run
# =========================
def main():
    cfg.PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, train_dl = make_loader(train_pids, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ds,   val_dl   = make_loader(val_pids,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_ds,  test_dl  = make_loader(test_pids,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("\n🧩 Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
    print(f"Backbone: CT-PatchTST (NO STRIDE) | W={cfg.WINDOW_SIZE}, C={cfg.CHANNELS}, "
          f"P={cfg.PATCH_LEN}, N={cfg.WINDOW_SIZE // cfg.PATCH_LEN}, "
          f"D={cfg.D_MODEL}, blocks={cfg.N_BLOCKS}, heads_ch={cfg.HEADS_CH}, heads_tm={cfg.HEADS_TM}")

    model = PatchTSTClassifier(cfg, num_classes).to(device)
    class_w = compute_class_weights(train_ds, num_classes).to(device)

    print("\n🎯 Training (classification, non-overlapping patches)...")
    train_model(model, train_dl, val_dl, class_w=class_w, patience=cfg.EARLY_STOP_PATIENCE)

    best_path = cfg.PROC_DIR / "ct_patchtst_har_best.pth"
    if best_path.exists(): model.load_state_dict(torch.load(best_path, map_location=device))

    # HMM
    pi, A = estimate_hmm_from_dataset(train_ds, num_classes, laplace=cfg.HMM_LAPLACE)
    print("\n📈 HMM priors (π):", np.round(pi, 4))
    print("📊 HMM transitions (A):"); 
    with np.printoptions(precision=3, suppress=True): print(A)

    # Test
    print("\n🎯 Testing...")
    proba_raw, pred_raw, true_raw, pid_raw, first_raw = get_prob_predictions(model, test_dl, temperature=cfg.SOFTMAX_T)

    metrics_raw = compute_all_metrics(true_raw, pred_raw, num_classes, proba=proba_raw)
    cm_raw = confusion_matrix(true_raw, pred_raw, labels=np.arange(num_classes))
    save_confusion(cm_raw, classes, f"Confusion Matrix (Raw) — MacroF1 {metrics_raw['f1_macro']:.3f}", cfg.PROC_DIR / "cm_ct_patchtst_raw.png")

    pred_hmm = apply_hmm_viterbi_over_subjects(proba_raw, pid_raw, first_raw, pi, A)
    metrics_hmm = compute_all_metrics(true_raw, pred_hmm, num_classes, proba=proba_raw)
    cm_hmm = confusion_matrix(true_raw, pred_hmm, labels=np.arange(num_classes))
    save_confusion(cm_hmm, classes, f"Confusion Matrix (HMM) — MacroF1 {metrics_hmm['f1_macro']:.3f}", cfg.PROC_DIR / "cm_ct_patchtst_hmm.png")

    def fmt(x): 
        return "nan" if (x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.3f}"
    print("\n📋 Test Metrics (Raw vs HMM-smoothed)")
    keys = ["f1_macro","f1_weighted","f2_macro","precision_macro","recall_macro",
            "balanced_acc","kappa","mcc","phi_yule_macro","yules_q_macro","auroc_macro_ovr"]
    header = "{:>18} | {:>8} | {:>8}".format("Metric","RAW","HMM")
    print(header); print("-"*len(header))
    for k in keys:
        print("{:>18} | {:>8} | {:>8}".format(k, fmt(metrics_raw[k]), fmt(metrics_hmm[k])))

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "metrics_raw": metrics_raw,
        "metrics_hmm": metrics_hmm,
        "hmm": {"pi": pi, "A": A},
        "config": {k:getattr(cfg, k) for k in dir(cfg) if k.isupper()},
    }, cfg.PROC_DIR / "ct_patchtst_hmm_artifacts.pth")
    print(f"\n💾 Saved: {cfg.PROC_DIR / 'ct_patchtst_hmm_artifacts.pth'}")

if __name__ == "__main__":
    main()
