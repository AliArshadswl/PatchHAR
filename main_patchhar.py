#!/usr/bin/env python3
"""
PatchTST-HAR on CAPTURE-24 (Simplified Features)
- Loads preprocessed windows from disk (processed_minimal/*.npz)
- Subject-wise split: first 80 → train, next 20 → val, rest → test
- Time features: 5-D from FIRST timestamp per window
- Statistical features: 22-D (from figure)
- Topological features: 15-D (H0 full + H1 counts only)
- Encoder: Transformer with RoPE attention (no absolute pos embeddings)
- Validation metrics: Macro-F1, Cohen's κ, MCC
- Post-processing: HMM transitions estimated from TRAIN; Viterbi decode TEST
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict
import math, random, json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
class Config:
    PROC_DIR   = Path("E:/HAR_datasets_codes/processed_minimal")
    TRAIN_N    = 80
    VAL_N      = 20

    SIGNAL_RATE = 100
    WINDOW_SIZE = 1000
    PATCH_LEN   = 100
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN   # 10
    CHANNELS    = 3

    # RoPE-friendly dims
    D_MODEL   = 56     # per-head must be even
    N_HEADS   = 2      # 56/2 = 28 (even) -> OK
    N_LAYERS  = 2
    DROPOUT   = 0.3

    # Feature dims (SIMPLIFIED)
    N_STAT_FEATURES = 22   # From figure
    N_TOPO_FEATURES = 15   # H0 full (12) + H1 counts only (3)

    # Takens embedding for topology
    TAKENS_M = 3                  # embedding dimension
    TAKENS_TAU = 5                # samples (at 100 Hz → 50 ms)
    TOPO_MAX_POINTS = 150         # optional downsample for speed

    BATCH_SIZE    = 32
    EPOCHS        = 30
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 8

    SEED = 42

    # HMM
    HMM_SMOOTH   = 1.0   # add-k smoothing
    HMM_MIN_PROB = 1e-6

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

n_train = min(cfg.TRAIN_N, len(pids_all))
n_val   = min(cfg.VAL_N,  max(0, len(pids_all) - n_train))
train_pids = pids_all[:n_train]
val_pids   = pids_all[n_train:n_train+n_val]
test_pids  = pids_all[n_train+n_val:]
print(f"Subject split: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")

# =========================
# Time features (5-D) from FIRST timestamp
# =========================
def time_features_from_first_ns(first_ns: int) -> np.ndarray:
    ts = pd.to_datetime(int(first_ns), unit="ns", utc=True).tz_convert(None)
    hour, minute, weekday = ts.hour, ts.minute, ts.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    time_of_day = hour // 6
    out = np.zeros(5, dtype=np.float32)
    out[0] = hour / 24.0
    out[1] = minute / 60.0
    out[2] = weekday / 7.0
    out[3] = float(is_weekend)
    out[4] = float(time_of_day)
    return out

# =========================
# Statistical features (22-D) from figure
# =========================
try:
    from scipy.signal import find_peaks, peak_prominences
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-8 or sb < 1e-8: return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(c) else c

def _spectral_entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return 0.0
    H = -float(np.sum(p * np.log(p)))
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0

def _lowpass_gravity(a: np.ndarray, sr: int, fc: float = 0.5) -> np.ndarray:
    alpha = math.exp(-2.0 * math.pi * fc / float(sr))
    g = np.zeros_like(a, dtype=np.float32)
    g[0] = a[0]
    for t in range(1, a.shape[0]):
        g[t] = alpha * g[t-1] + (1.0 - alpha) * a[t]
    return g

def _angles_from_vec(vx, vy, vz):
    roll = np.arctan2(vy, vz)
    pitch = np.arctan2(-vx, np.sqrt(vy*vy + vz*vz) + 1e-8)
    yaw_p = np.arctan2(vx, vy)
    return roll, pitch, yaw_p

def compute_statistical_features(window_norm: np.ndarray, sr: int = 100) -> np.ndarray:
    """
    Compute 22 statistical features exactly as shown in figure.
    """
    x, y, z = window_norm[:,0], window_norm[:,1], window_norm[:,2]
    mag = np.linalg.norm(window_norm, axis=1)
    feats = []

    # 1. autocorr_lag1
    L = min(sr, mag.shape[0] - 1)
    if L <= 1:
        feats.append(0.0)
    else:
        a, b = mag[:-L], mag[L:]
        feats.append(_safe_corr(a, b))

    # 2. peak_prominence
    if _HAVE_SCIPY:
        peaks, _ = find_peaks(mag)
        if peaks.size:
            prom = peak_prominences(mag, peaks)[0]
            feats.append(float(np.median(prom)))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # 3. mag_min
    feats.append(float(np.min(mag)))

    # 4. x_std
    feats.append(float(np.std(x)))

    # 5. x_range
    feats.append(float(np.max(x) - np.min(x)))

    # 6. z_range
    feats.append(float(np.max(z) - np.min(z)))

    # 7. y_std
    feats.append(float(np.std(y)))

    # 8. z_std
    feats.append(float(np.std(z)))

    # 9. y_range
    feats.append(float(np.max(y) - np.min(y)))

    # 10. mag_max
    feats.append(float(np.max(mag)))

    # 11. mag_std
    feats.append(float(np.std(mag)))

    # 12. freq_peak1, 13. freq_peak2, 18. freq_energy
    X = np.fft.rfft(mag)
    ps = (np.abs(X) ** 2).astype(np.float64)
    s = ps.sum()
    if s > 0: ps = ps / s
    freqs_hz = np.fft.rfftfreq(mag.shape[0], d=1.0/sr)
    
    if ps.size <= 2:
        feats.extend([0.0, 0.0, 0.0])
    else:
        p = ps.copy(); f = freqs_hz.copy()
        p[0] = 0.0  # drop DC
        i1 = int(np.argmax(p))
        p1 = float(p[i1]); f1 = float(f[i1])
        p[i1] = -1.0
        i2 = int(np.argmax(p))
        p2 = max(float(p[i2]), 0.0); f2 = float(f[i2])
        feats.extend([f1, f2, p1])  # freq_peak1, freq_peak2, freq_energy

    # 14. pitch_mean, 15. roll_mean
    g = _lowpass_gravity(window_norm, sr=sr, fc=0.5)
    dr, dp, _ = _angles_from_vec(g[:,0], g[:,1], g[:,2])
    feats.append(float(np.mean(dp)))  # pitch_mean
    feats.append(float(np.mean(dr)))  # roll_mean

    # 16. mag_mean
    feats.append(float(np.mean(mag)))

    # 17. spectral_entropy
    feats.append(_spectral_entropy(ps))

    # 19. x_mean, 20. y_mean, 21. z_mean
    feats.append(float(np.mean(x)))
    feats.append(float(np.mean(y)))
    feats.append(float(np.mean(z)))

    # 22. magnitude_vector (norm of mean acceleration vector)
    mean_vec = np.array([np.mean(x), np.mean(y), np.mean(z)])
    feats.append(float(np.linalg.norm(mean_vec)))

    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[0] == cfg.N_STAT_FEATURES, f"Expected {cfg.N_STAT_FEATURES} stats, got {arr.shape[0]}"
    return arr

# =========================
# Topological features (15-D: H0 full + H1 counts only)
# =========================
try:
    from ripser import ripser
    _HAVE_RIPSER = True
except Exception:
    _HAVE_RIPSER = False

def _takens_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    T = x.shape[0]
    L = T - (m - 1) * tau
    if L <= 5:
        return np.zeros((5, m), dtype=np.float32)
    out = np.stack([x[i:i+L] for i in range(0, m*tau, tau)], axis=1)
    return out.astype(np.float32)

def _persistence_entropy(diag: np.ndarray) -> float:
    if diag.size == 0:
        return 0.0
    births = diag[:, 0]; deaths = diag[:, 1]
    finite = np.isfinite(deaths)
    births, deaths = births[finite], deaths[finite]
    pers = np.maximum(deaths - births, 0.0)
    s = pers.sum()
    if s <= 0:
        return 0.0
    p = pers / s
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = np.log(len(p))
    return float(H / (Hmax + 1e-12))

def _topk_lifetimes(diag: np.ndarray, k: int = 3) -> list[float]:
    if diag.size == 0:
        return [0.0]*k
    births = diag[:,0]; deaths = diag[:,1]
    finite = np.isfinite(deaths)
    lifetimes = np.maximum(deaths[finite] - births[finite], 0.0)
    lifetimes = np.sort(lifetimes)[::-1]
    pad = np.zeros(k, dtype=np.float32)
    pad[:min(k, lifetimes.size)] = lifetimes[:k]
    return pad.tolist()

def compute_topological_features(window_norm: np.ndarray, sr: int = 100,
                                 m: int = cfg.TAKENS_M, tau: int = cfg.TAKENS_TAU) -> np.ndarray:
    """
    15-D: H0 full persistence (12) + H1 count features only (3).
    """
    mag = np.linalg.norm(window_norm, axis=1).astype(np.float32)
    if mag.shape[0] > cfg.TOPO_MAX_POINTS:
        step = int(np.ceil(mag.shape[0] / cfg.TOPO_MAX_POINTS))
        mag = mag[::step]
    X = _takens_embed(mag, m=m, tau=tau)

    if _HAVE_RIPSER and X.shape[0] >= 8:
        try:
            res = ripser(X, maxdim=1)
            D0, D1 = res['dgms'][0], res['dgms'][1]
        except:
            D0, D1 = np.empty((0,2)), np.empty((0,2))
    else:
        D0, D1 = np.empty((0,2)), np.empty((0,2))

    feats = []
    
    # H0: All 12 features (Persistence + Count)
    if D0.size == 0:
        feats.extend([0.0] * 12)
    else:
        births, deaths = D0[:,0], D0[:,1]
        finite = np.isfinite(deaths)
        births, deaths = births[finite], deaths[finite]
        pers = np.maximum(deaths - births, 0.0)
        
        feats.append(float(pers.max() if pers.size else 0.0))
        feats.append(float(pers.mean() if pers.size else 0.0))
        feats.append(float(pers.sum() if pers.size else 0.0))
        feats.append(_persistence_entropy(D0))
        feats.extend(_topk_lifetimes(D0, k=3))
        feats.append(float(births.max() if births.size else 0.0))
        feats.append(float(deaths.max() if deaths.size else 0.0))
        
        if pers.size >= 5:
            qs = np.quantile(pers, [0.5, 0.75, 0.9])
            feats.extend([float((pers > qs[0]).sum()),
                         float((pers > qs[1]).sum()),
                         float((pers > qs[2]).sum())])
        else:
            feats.extend([0.0, 0.0, 0.0])
    
    # H1: Count features only (3)
    if D1.size == 0:
        feats.extend([0.0] * 3)
    else:
        births, deaths = D1[:,0], D1[:,1]
        finite = np.isfinite(deaths)
        births, deaths = births[finite], deaths[finite]
        pers = np.maximum(deaths - births, 0.0)
        
        if pers.size >= 5:
            qs = np.quantile(pers, [0.5, 0.75, 0.9])
            feats.extend([float((pers > qs[0]).sum()),
                         float((pers > qs[1]).sum()),
                         float((pers > qs[2]).sum())])
        else:
            feats.extend([0.0, 0.0, 0.0])

    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[0] == cfg.N_TOPO_FEATURES, f"Expected {cfg.N_TOPO_FEATURES}, got {arr.shape[0]}"
    return arr

# =========================
# Dataset
# =========================
class ProcessedDataset(Dataset):
    """
    Returns:
      patches: (3, 10, 100) float32
      tfeat:   (5,)         float32
      sfeat:   (22,)        float32  <-- Reduced stats
      tafeat:  (15,)        float32  <-- Reduced topology
      label:   ()           long
      pid:     str
      first_ns:()           long
    """
    def __init__(self, pid_list, proc_dir: Path, class_to_idx: dict):
        self.proc_dir = proc_dir
        self.class_to_idx = class_to_idx
        self.entries = []
        for pid in pid_list:
            npz_path = proc_dir / f"{pid}.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path, allow_pickle=True)
            W = npz["windows"].astype(np.float32)
            L = npz["labels_str"].astype(str)
            F = npz["first_ts_epoch_ns"].astype(np.int64)

            order = np.argsort(F)
            W, L, F = W[order], L[order], F[order]
            for w, lab, f in zip(W, L, F):
                if lab in class_to_idx:
                    self.entries.append((pid, w, int(class_to_idx[lab]), int(f)))
        self.len = len(self.entries)

    def __len__(self): return self.len

    def __getitem__(self, idx):
        pid, window, label, first_ns = self.entries[idx]
        assert window.shape == (cfg.WINDOW_SIZE, cfg.CHANNELS)

        # Instance norm per channel
        normed = np.zeros_like(window, dtype=np.float32)
        for c in range(cfg.CHANNELS):
            ch = window[:, c]
            mu, sd = float(ch.mean()), float(ch.std())
            normed[:, c] = (ch - mu) / (sd + 1e-8)
        normed = np.clip(normed, -10, 10)

        # Patches: (3, 10, 100)
        patches = normed.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS).transpose(2, 0, 1).astype(np.float32)

        # Time features from FIRST timestamp (5)
        tfeat = time_features_from_first_ns(first_ns)

        # Statistical features from normalized window (22)
        sfeat = compute_statistical_features(normed, sr=cfg.SIGNAL_RATE)

        # Topological features from normalized window (15)
        tafeat = compute_topological_features(normed, sr=cfg.SIGNAL_RATE)

        return (
            torch.from_numpy(patches),
            torch.from_numpy(tfeat),
            torch.from_numpy(sfeat),
            torch.from_numpy(tafeat),
            torch.tensor(label, dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
        )

def make_loader(pid_list, batch_size=32, shuffle=False):
    ds = ProcessedDataset(pid_list, cfg.PROC_DIR, class_to_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=GPU)
    return ds, dl

# =========================
# RoPE helpers
# =========================
def precompute_freqs_cis(dim: int, n_tokens: int, theta: float = 10000.0):
    assert dim % 2 == 0, "RoPE requires even per-head dimension."
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(n_tokens)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # complex64

def apply_rotary_pos_emb(q, k, freqs_cis):
    B, H, N, D = q.shape
    assert D % 2 == 0
    d2 = D // 2
    freqs = freqs_cis[:N].to(q.device).view(1, 1, N, d2)
    q_ = q.float().contiguous().view(B, H, N, d2, 2)
    k_ = k.float().contiguous().view(B, H, N, d2, 2)
    q_c = torch.view_as_complex(q_)
    k_c = torch.view_as_complex(k_)
    q_out = torch.view_as_real(q_c * freqs).view(B, H, N, D)
    k_out = torch.view_as_real(k_c * freqs).view(B, H, N, D)
    return q_out.type_as(q), k_out.type_as(k)

# =========================
# Model (Patch tokens + Stats token + Topo token with time; RoPE encoder)
# =========================
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model=128, n_features=5, p=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return self.drop(self.act(self.proj(x)))

class PatchEmbedding(nn.Module):
    def __init__(self, patch_length=100, channels=3, d_model=128,
                 n_patches=10, n_time_features=5,
                 n_stat_features=22, n_topo_features=15):
        super().__init__()
        self.n_patches = n_patches
        self.patch_proj = nn.Linear(channels * patch_length, d_model)
        self.stat_proj  = nn.Linear(n_stat_features, d_model)
        self.topo_proj  = nn.Linear(n_topo_features, d_model)
        self.time_emb   = TemporalEmbedding(d_model, n_time_features, p=0.1)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, patches, times, stats, topo):
        B, C, NP, PL = patches.shape
        assert NP == self.n_patches
        x = patches.permute(0, 2, 1, 3).reshape(B, NP, C * PL)
        x = self.patch_proj(x)
        t = self.time_emb(times)
        t_patch = t.unsqueeze(1).expand(-1, NP, -1)
        x = x + t_patch
        s = self.stat_proj(stats) + t
        s = s.unsqueeze(1)
        g = self.topo_proj(topo) + t
        g = g.unsqueeze(1)
        x = torch.cat([x, s, g], dim=1)
        return self.norm(x)

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=2, dropout=0.25):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "per-head dim must be even for RoPE"

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, d).permute(0, 2, 1, 3, 4)
        q = qkv[:, 0].transpose(1, 2)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, freqs_cis)
        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, D)
        x = self.norm1(x + self.drop(self.proj(out)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x

class RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=2, n_layers=2, dropout=0.25, n_tokens=12):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        head_dim = d_model // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, n_tokens)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.embed    = PatchEmbedding(cfg.PATCH_LEN, cfg.CHANNELS, cfg.D_MODEL,
                                       cfg.N_PATCHES, n_time_features=5,
                                       n_stat_features=cfg.N_STAT_FEATURES,
                                       n_topo_features=cfg.N_TOPO_FEATURES)
        self.backbone = RoPETransformerEncoder(cfg.D_MODEL, cfg.N_HEADS, cfg.N_LAYERS,
                                               cfg.DROPOUT, n_tokens=cfg.N_PATCHES+2)
        self.cls      = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.D_MODEL//2, num_classes)
        )
    def forward(self, patches, times, stats, topo):
        x = self.embed(patches, times, stats, topo)
        x = self.backbone(x)
        x = x.mean(dim=1)
        return self.cls(x)

# =========================
# Metrics (val + test)
# =========================
def cohen_kappa_standard(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
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

# =========================
# Training utils
# =========================
def compute_class_weights(train_ds, K):
    counts = np.zeros(K, dtype=np.int64)
    for _, w, lab, _ in train_ds.entries:
        counts[lab] += 1
    weights = counts.max() / np.clip(counts, 1, None)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * K

def train_model(model, train_loader, val_loader, class_w: torch.Tensor | None = None, patience: int = 8):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)
    scaler = GradScaler(enabled=GPU)

    best_score = -1e9
    best_path = cfg.PROC_DIR / "patchtst_simplified_best.pth"
    best_epoch, patience_ctr = -1, 0

    for epoch in range(cfg.EPOCHS):
        # Train
        model.train()
        total = 0.0
        for patches, times, stats, topo, labels, pids, first_ns in train_loader:
            patches = patches.to(device); times = times.to(device)
            stats   = stats.to(device);   topo  = topo.to(device)
            labels  = labels.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                logits = model(patches, times, stats, topo)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for patches, times, stats, topo, labels, pids, first_ns in val_loader:
                patches = patches.to(device); times = times.to(device)
                stats   = stats.to(device);   topo  = topo.to(device)
                labels  = labels.to(device).view(-1)
                pred    = model(patches, times, stats, topo).argmax(1)
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
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model (F1={f1:.4f}, Kappa={kappa:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}; best epoch {best_epoch+1}")
                break

@torch.no_grad()
def get_raw_predictions(model, loader):
    model.eval()
    preds, truths, pids, first_ns = [], [], [], []
    for patches, times, stats, topo, labels, sub_ids, firsts in loader:
        patches = patches.to(device); times = times.to(device)
        stats   = stats.to(device);   topo  = topo.to(device)
        logits  = model(patches, times, stats, topo)
        pred    = logits.argmax(1).cpu().numpy()
        preds.extend(pred.tolist())
        truths.extend(labels.numpy().tolist())
        pids.extend(list(sub_ids))
        first_ns.extend(firsts.numpy().tolist())
    return np.array(preds), np.array(truths), np.array(pids, dtype=object), np.array(first_ns, dtype=np.int64)

# =========================
# HMM (estimate from train, decode test)
# =========================
def estimate_hmm_from_train(train_ds, K, smooth=cfg.HMM_SMOOTH, min_prob=cfg.HMM_MIN_PROB):
    by_pid = defaultdict(list)
    for pid, w, lab, first_ns in train_ds.entries:
        by_pid[pid].append((first_ns, lab))

    A = np.full((K, K), smooth, dtype=np.float64)
    pi = np.full(K, smooth, dtype=np.float64)
    for pid, seq in by_pid.items():
        seq.sort(key=lambda x: x[0])
        if not seq: continue
        pi[seq[0][1]] += 1
        for (_, a), (_, b) in zip(seq[:-1], seq[1:]):
            A[a, b] += 1

    A = A / A.sum(axis=1, keepdims=True)
    pi = pi / pi.sum()
    A = np.clip(A, min_prob, 1.0); A = A / A.sum(axis=1, keepdims=True)
    pi = np.clip(pi, min_prob, 1.0); pi = pi / pi.sum()
    return pi, A

def viterbi_decode(emission_log_probs, log_pi, log_A):
    T, K = emission_log_probs.shape
    dp = np.full((T,K), -np.inf, dtype=np.float64)
    bp = np.full((T,K), -1, dtype=np.int32)
    dp[0] = log_pi + emission_log_probs[0]
    for t in range(1,T):
        prev = dp[t-1][:,None] + log_A
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = prev[bp[t], np.arange(K)] + emission_log_probs[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1, path[t+1]]
    return path

# =========================
# Run
# =========================
def main():
    cfg.PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Loaders
    train_ds, train_dl = make_loader(train_pids, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ds,   val_dl   = make_loader(val_pids,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_ds,  test_dl  = make_loader(test_pids,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("\n🧩 Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

    # Model
    model = PatchTSTClassifier(cfg, num_classes).to(device)

    # Class weights
    class_w = compute_class_weights(train_ds, num_classes).to(device)

    # Train
    print("\n🎯 Training (RoPE encoder + 22 Stats + 15 Topo token)...")
    train_model(model, train_dl, val_dl, class_w=class_w, patience=cfg.EARLY_STOP_PATIENCE)

    # Load best weights
    best_path = cfg.PROC_DIR / "patchtst_simplified_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Raw predictions
    print("\n🎯 Testing (raw)...")
    raw_pred, raw_true, raw_pid, raw_first = get_raw_predictions(model, test_dl)

    # Raw metrics
    f1_raw = f1_score(raw_true, raw_pred, average="macro")
    k_raw  = cohen_kappa_standard(raw_true, raw_pred)
    m_raw  = multiclass_mcc_gorodkin(raw_true, raw_pred)
    print(f"\n✅ Raw Test — MacroF1={f1_raw:.3f} | Kappa={k_raw:.3f} | MCC={m_raw:.3f}")

    cm_raw = confusion_matrix(raw_true, raw_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, square=True)
    plt.title(f"Confusion Matrix (Raw) — MacroF1 {f1_raw:.3f}")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "cm_raw_simplified.png", dpi=200); plt.close()
    print(f"🖼️ Saved: {cfg.PROC_DIR / 'cm_raw_simplified.png'}")

    # HMM post-process
    print("\n🤖 HMM: estimate transitions from TRAIN and decode TEST...")
    pi, A = estimate_hmm_from_train(train_ds, num_classes)
    log_pi = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.0))
    log_A  = np.log(np.clip(A,  cfg.HMM_MIN_PROB, 1.0))

    by_pid_probs, by_pid_truth, by_pid_time = defaultdict(list), defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for patches, times, stats, topo, labels, pids, first_ns in test_dl:
            patches = patches.to(device); times = times.to(device)
            stats   = stats.to(device);   topo  = topo.to(device)
            probs   = torch.softmax(model(patches, times, stats, topo), dim=1).cpu().numpy()
            labs    = labels.numpy(); tns = first_ns.numpy()
            for pr, lb, pid, tn in zip(probs, labs, pids, tns):
                by_pid_probs[pid].append(pr)
                by_pid_truth[pid].append(int(lb))
                by_pid_time[pid].append(int(tn))

    hmm_preds, hmm_truth = [], []
    for pid in by_pid_probs.keys():
        order = np.argsort(by_pid_time[pid])
        E = np.log(np.clip(np.vstack([by_pid_probs[pid][i] for i in order]), cfg.HMM_MIN_PROB, 1.0))
        dec = viterbi_decode(E, log_pi, log_A)
        hmm_preds.extend(dec.tolist())
        hmm_truth.extend([by_pid_truth[pid][i] for i in order])

    hmm_preds = np.asarray(hmm_preds, dtype=np.int64)
    hmm_truth = np.asarray(hmm_truth, dtype=np.int64)

    f1_hmm = f1_score(hmm_truth, hmm_preds, average="macro")
    k_hmm  = cohen_kappa_standard(hmm_truth, hmm_preds)
    m_hmm  = multiclass_mcc_gorodkin(hmm_truth, hmm_preds)
    print(f"\n✅ +HMM Test — MacroF1={f1_hmm:.3f} | Kappa={k_hmm:.3f} | MCC={m_hmm:.3f}")

    cm_hmm = confusion_matrix(hmm_truth, hmm_preds, labels=np.arange(num_classes))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_hmm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, square=True)
    plt.title(f"Confusion Matrix (+HMM) — MacroF1 {f1_hmm:.3f}")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "cm_hmm_simplified.png", dpi=200); plt.close()
    print(f"🖼️ Saved: {cfg.PROC_DIR / 'cm_hmm_simplified.png'}")

    # Save bundle
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "pi": pi, "A": A,
        "metrics_raw": {"macro_f1": f1_raw, "kappa": k_raw, "mcc": m_raw},
        "metrics_hmm": {"macro_f1": f1_hmm, "kappa": k_hmm, "mcc": m_hmm},
        "config": vars(cfg),
    }, cfg.PROC_DIR / "patchtst_simplified_artifacts.pth")
    print(f"💾 Saved: {cfg.PROC_DIR / 'patchtst_simplified_artifacts.pth'}")

if __name__ == "__main__":
    main()
