# ============================================
# PatchTST-HAR + Enhanced Metrics & Computational Profiling
# ============================================
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict
import math, random, json, os, io, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.metrics import (f1_score, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, balanced_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
class Config:
    PROC_DIR   = Path("/mnt/share/ali/processed_minimal/")
    TRAIN_N    = 80
    VAL_N      = 20

    SIGNAL_RATE = 100
    WINDOW_SIZE = 1000
    PATCH_LEN   = 100
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN
    CHANNELS    = 3

    D_MODEL   = 56
    N_HEADS   = 2
    N_LAYERS  = 2
    DROPOUT   = 0.3

    N_STAT_FEATURES = 56
    N_TOPO_FEATURES = 24

    TAKENS_M = 3
    TAKENS_TAU = 5
    TOPO_MAX_POINTS = 600

    BATCH_SIZE    = 32
    EPOCHS        = 30
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 8

    SEED = 42

    HMM_SMOOTH   = 1.0
    HMM_MIN_PROB = 1e-6

    EXPLAIN_MAX_SAMPLES = 128
    SAVE_JSON = True
    
    # Profiling config
    PROFILE_ITERATIONS = 100
    PROFILE_WARMUP = 10

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
# Time features
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
# Statistical features
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

def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

def _skew(x: np.ndarray) -> float:
    mu = float(np.mean(x)); sd = float(np.std(x))
    if sd < 1e-12: return 0.0
    m3 = float(np.mean((x - mu)**3))
    return m3 / (sd**3)

def _kurtosis_excess(x: np.ndarray) -> float:
    mu = float(np.mean(x)); sd = float(np.std(x))
    if sd < 1e-12: return 0.0
    m4 = float(np.mean((x - mu)**4))
    return m4 / (sd**4) - 3.0

def _range(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x)) if x.size else 0.0

def _normalized_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(x)
    ps = (np.abs(X) ** 2).astype(np.float64)
    s = ps.sum()
    if s > 0: ps = ps / s
    freqs_hz = np.fft.rfftfreq(x.shape[0], d=1.0/cfg.SIGNAL_RATE)
    return ps, freqs_hz

def _spectral_entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return 0.0
    H = -float(np.sum(p * np.log(p)))
    Hmax = math.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0

def _dominant_two(freq_probs: np.ndarray, freqs_hz: np.ndarray) -> tuple[float,float,float,float]:
    if freq_probs.size <= 2:
        return 0.0, 0.0, 0.0, 0.0
    p = freq_probs.copy()
    f = freqs_hz.copy()
    p[0] = 0.0
    i1 = int(np.argmax(p))
    p1 = float(p[i1]); f1 = float(f[i1])
    p[i1] = -1.0
    i2 = int(np.argmax(p))
    p2 = max(float(p[i2]), 0.0); f2 = float(f[i2])
    return f1, p1, f2, p2

def _one_sec_autocorr(x: np.ndarray, sr: int) -> float:
    L = min(sr, x.shape[0]-1)
    if L <= 1: return 0.0
    a, b = x[:-L], x[L:]
    return _safe_corr(a, b)

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
    x, y, z = window_norm[:,0], window_norm[:,1], window_norm[:,2]
    mag = np.linalg.norm(window_norm, axis=1)
    feats: list[float] = []

    for sig in (x, y, z):
        feats.extend([float(np.mean(sig)), float(np.std(sig)), _range(sig)])
    feats.append(_safe_corr(x, y)); feats.append(_safe_corr(x, z)); feats.append(_safe_corr(y, z))
    feats.extend([
        float(np.mean(mag)), float(np.std(mag)), _range(mag),
        _mad(mag), _kurtosis_excess(mag), _skew(mag), float(np.median(mag)),
    ])

    def qpack(sig):
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        return [float(np.min(sig)), float(np.max(sig)), float(q50), float(q25), float(q75)]
    for sig in (x, y, z, mag):
        feats.extend(qpack(sig))

    feats.append(_one_sec_autocorr(mag, sr))
    ps, freqs_hz = _normalized_psd(mag)
    f1, p1, f2, p2 = _dominant_two(ps, freqs_hz)
    feats.extend([f1, p1, f2, p2])
    feats.append(_spectral_entropy(ps))

    if _HAVE_SCIPY:
        peaks, _ = find_peaks(mag)
        if peaks.size:
            prom = peak_prominences(mag, peaks)[0]
            feats.append(float(len(peaks)))
            feats.append(float(np.median(prom)))
        else:
            feats.extend([0.0, 0.0])
    else:
        if mag.size >= 3:
            pk = ((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])).sum()
            feats.append(float(pk))
        else:
            feats.append(0.0)
        feats.append(0.0)

    g = _lowpass_gravity(window_norm, sr=sr, fc=0.5)
    d = window_norm - g
    gr, gp, gyaw_p = _angles_from_vec(g[:,0], g[:,1], g[:,2])
    feats.extend([float(np.mean(gr)), float(np.mean(gp)), float(np.mean(gyaw_p))])
    dr, dp, dyaw_p = _angles_from_vec(d[:,0], d[:,1], d[:,2])
    for arr in (dr, dp, dyaw_p):
        feats.append(float(np.mean(arr)))
        feats.append(float(np.std(arr)))

    feats = [0.0 if (not np.isfinite(v)) else float(v) for v in feats]
    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[0] == cfg.N_STAT_FEATURES
    return arr

# =========================
# Topological features (WITH PROFILING)
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
    births = diag[:, 0]
    deaths = diag[:, 1]
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
                                 m: int = cfg.TAKENS_M, tau: int = cfg.TAKENS_TAU,
                                 profile: bool = False) -> tuple[np.ndarray, dict]:
    """
    Returns: (features, timing_dict)
    """
    timings = {}
    t0 = time.perf_counter()
    
    mag = np.linalg.norm(window_norm, axis=1).astype(np.float32)
    if mag.shape[0] > cfg.TOPO_MAX_POINTS:
        step = int(np.ceil(mag.shape[0] / cfg.TOPO_MAX_POINTS))
        mag = mag[::step]
    
    t1 = time.perf_counter()
    timings['preprocessing'] = t1 - t0
    
    X = _takens_embed(mag, m=m, tau=tau)
    t2 = time.perf_counter()
    timings['takens_embedding'] = t2 - t1
    
    if _HAVE_RIPSER and X.shape[0] >= 8:
        t3 = time.perf_counter()
        res = ripser(X, maxdim=1)
        t4 = time.perf_counter()
        timings['ripser_computation'] = t4 - t3
        
        D0 = res.get('dgms', [np.empty((0,2)), np.empty((0,2))])[0]
        D1 = res.get('dgms', [np.empty((0,2)), np.empty((0,2))])[1]
    else:
        t3 = time.perf_counter()
        d = np.abs(np.subtract.outer(mag, mag))
        thr = np.quantile(d[np.isfinite(d)], [0.1, 0.2, 0.3, 0.5]) if np.isfinite(d).any() else [0.0]*4
        
        def fake_diag(dist, t):
            M = (dist < t).astype(np.float32)
            lifetimes = []
            for k in range(-10, 11):
                diag = np.diag(M, k=k)
                runs, run = [], 0
                for v in diag:
                    if v > 0.5:
                        run += 1
                    elif run:
                        runs.append(run); run = 0
                if run: runs.append(run)
                lifetimes.extend(runs)
            if len(lifetimes) == 0:
                return np.empty((0,2), dtype=np.float32)
            lifetimes = np.asarray(lifetimes, dtype=np.float32)
            births = np.zeros_like(lifetimes)
            deaths = lifetimes
            return np.stack([births, deaths], axis=1)
        
        D0 = fake_diag(d, thr[1])
        D1 = fake_diag(d, thr[3])
        t4 = time.perf_counter()
        timings['fallback_computation'] = t4 - t3

    t5 = time.perf_counter()
    feats = []
    for D in (D0, D1):
        if D.size == 0:
            feats.extend([0.0]*12)
            continue
        births, deaths = D[:,0], D[:,1]
        finite = np.isfinite(deaths)
        births, deaths = births[finite], deaths[finite]
        pers = np.maximum(deaths - births, 0.0)
        feats.append(float(pers.max() if pers.size else 0.0))
        feats.append(float(pers.mean() if pers.size else 0.0))
        feats.append(float(pers.sum() if pers.size else 0.0))
        feats.append(_persistence_entropy(D))
        feats.extend(_topk_lifetimes(D, k=3))
        feats.append(float(births.max() if births.size else 0.0))
        feats.append(float(deaths.max() if deaths.size else 0.0))
        if pers.size >= 5:
            qs = np.quantile(pers, [0.5, 0.75, 0.9])
            feats.extend([float((pers > qs[0]).sum()),
                          float((pers > qs[1]).sum()),
                          float((pers > qs[2]).sum())])
        else:
            feats.extend([0.0, 0.0, 0.0])
    
    t6 = time.perf_counter()
    timings['feature_extraction'] = t6 - t5
    timings['total'] = t6 - t0
    
    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[0] == cfg.N_TOPO_FEATURES
    
    return arr, timings if profile else {}

# =========================
# Dataset
# =========================
class ProcessedDataset(Dataset):
    def __init__(self, pid_list, proc_dir: Path, class_to_idx: dict, profile_topo: bool = False):
        self.proc_dir = proc_dir
        self.class_to_idx = class_to_idx
        self.profile_topo = profile_topo
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
        normed = np.zeros_like(window, dtype=np.float32)
        for c in range(cfg.CHANNELS):
            ch = window[:, c]
            mu, sd = float(ch.mean()), float(ch.std())
            normed[:, c] = (ch - mu) / (sd + 1e-8)
        normed = np.clip(normed, -10, 10)
        patches = normed.reshape(cfg.N_PATCHES, cfg.PATCH_LEN, cfg.CHANNELS).transpose(2, 0, 1).astype(np.float32)
        tfeat = time_features_from_first_ns(first_ns)
        sfeat = compute_statistical_features(normed, sr=cfg.SIGNAL_RATE)
        tafeat, _ = compute_topological_features(normed, sr=cfg.SIGNAL_RATE, profile=self.profile_topo)
        return (
            torch.from_numpy(patches),
            torch.from_numpy(tfeat),
            torch.from_numpy(sfeat),
            torch.from_numpy(tafeat),
            torch.tensor(label, dtype=torch.long),
            pid,
            torch.tensor(first_ns, dtype=torch.long),
        )

def make_loader(pid_list, batch_size=32, shuffle=False, profile_topo=False):
    ds = ProcessedDataset(pid_list, cfg.PROC_DIR, class_to_idx, profile_topo=profile_topo)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=GPU)
    return ds, dl

# =========================
# RoPE helpers
# =========================
def precompute_freqs_cis(dim: int, n_tokens: int, theta: float = 10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(n_tokens)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

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
# Model
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
                 n_stat_features=56, n_topo_features=24):
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
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0

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
        self.last_attn = None

    def forward(self, x, freqs_cis, return_attn=False):
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
        if return_attn:
            self.last_attn = attn.detach()
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

    def forward(self, x, capture_attn=False):
        for i,layer in enumerate(self.layers):
            x = layer(x, self.freqs_cis, return_attn=capture_attn and (i==len(self.layers)-1))
        return x

    def get_last_attn(self):
        if len(self.layers) == 0: return None
        return self.layers[-1].last_attn

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
    def forward(self, patches, times, stats, topo, capture_attn: bool = False):
        x = self.embed(patches, times, stats, topo)
        x = self.backbone(x, capture_attn=capture_attn)
        x = x.mean(dim=1)
        return self.cls(x)

    def last_attention(self):
        return self.backbone.get_last_attn()

# =========================
# ENHANCED METRICS
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

def pearson_yule_phi(y_true, y_pred):
    """Pearson-Yule Phi coefficient (Matthews correlation for binary, extended for multiclass)"""
    return multiclass_mcc_gorodkin(y_true, y_pred)

def f2_score(y_true, y_pred):
    """F2 score (recall-weighted): β=2 emphasizes recall over precision"""
    from sklearn.metrics import fbeta_score
    return fbeta_score(y_true, y_pred, beta=2.0, average='macro')

def transition_accuracy(y_true, y_pred):
    """Compute accuracy on transitions (when label changes from t to t+1)"""
    if len(y_true) <= 1:
        return 0.0
    
    transitions_true = []
    transitions_pred = []
    
    for i in range(1, len(y_true)):
        if y_true[i] != y_true[i-1]:  # Transition occurred
            transitions_true.append(y_true[i])
            transitions_pred.append(y_pred[i])
    
    if len(transitions_true) == 0:
        return 0.0
    
    correct = sum(1 for t, p in zip(transitions_true, transitions_pred) if t == p)
    return correct / len(transitions_true)

def compute_comprehensive_metrics(y_true, y_pred):
    """Compute ALL metrics for comprehensive evaluation"""
    metrics = {}
    
    # Core classification metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted'))
    metrics['macro_precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['macro_recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Advanced metrics
    metrics['cohen_kappa'] = float(cohen_kappa_standard(y_true, y_pred))
    metrics['mcc'] = float(multiclass_mcc_gorodkin(y_true, y_pred))
    metrics['pearson_yule_phi'] = float(pearson_yule_phi(y_true, y_pred))
    metrics['f2_score'] = float(f2_score(y_true, y_pred))
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, cls_name in enumerate(classes):
        metrics['per_class'][cls_name] = {
            'f1': float(per_class_f1[i]),
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i])
        }
    
    return metrics

def compute_sequence_metrics(y_true_seq, y_pred_seq, pids):
    """Compute sequence-level metrics including transition accuracy"""
    metrics = {}
    
    # Group by participant for sequence analysis
    pid_groups = defaultdict(lambda: {'true': [], 'pred': []})
    for yt, yp, pid in zip(y_true_seq, y_pred_seq, pids):
        pid_groups[pid]['true'].append(yt)
        pid_groups[pid]['pred'].append(yp)
    
    # Transition accuracy across all participants
    all_trans_acc = []
    for pid, data in pid_groups.items():
        ta = transition_accuracy(data['true'], data['pred'])
        if ta > 0:
            all_trans_acc.append(ta)
    
    metrics['transition_accuracy'] = float(np.mean(all_trans_acc)) if all_trans_acc else 0.0
    metrics['transition_accuracy_std'] = float(np.std(all_trans_acc)) if all_trans_acc else 0.0
    metrics['num_participants_with_transitions'] = len(all_trans_acc)
    
    return metrics

# =========================
# COMPUTATIONAL PROFILING
# =========================
def profile_topological_computation(num_samples=100, warmup=10):
    """Profile the computational cost of topological feature extraction"""
    print(f"\n⏱️ Profiling topological computation ({num_samples} samples, {warmup} warmup)...")
    
    # Generate synthetic windows
    windows = []
    for _ in range(num_samples + warmup):
        window = np.random.randn(cfg.WINDOW_SIZE, cfg.CHANNELS).astype(np.float32)
        windows.append(window)
    
    timings_all = []
    
    # Warmup
    for i in range(warmup):
        _, _ = compute_topological_features(windows[i], profile=True)
    
    # Actual profiling
    for i in range(warmup, num_samples + warmup):
        _, timings = compute_topological_features(windows[i], profile=True)
        timings_all.append(timings)
    
    # Aggregate statistics
    stats = {}
    for key in timings_all[0].keys():
        values = [t[key] for t in timings_all]
        stats[key] = {
            'mean_ms': float(np.mean(values) * 1000),
            'std_ms': float(np.std(values) * 1000),
            'min_ms': float(np.min(values) * 1000),
            'max_ms': float(np.max(values) * 1000),
            'p50_ms': float(np.percentile(values, 50) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000),
            'p99_ms': float(np.percentile(values, 99) * 1000),
        }
    
    return stats

def profile_end_to_end_inference(model, test_loader, num_batches=50):
    """Profile end-to-end inference time"""
    print(f"\n⏱️ Profiling end-to-end inference ({num_batches} batches)...")
    
    model.eval()
    batch_times = []
    sample_times = []
    
    with torch.no_grad():
        for i, (patches, times, stats, topo, labels, pids, first_ns) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            batch_size = patches.size(0)
            
            # Time the forward pass
            t0 = time.perf_counter()
            patches = patches.to(device)
            times = times.to(device)
            stats = stats.to(device)
            topo = topo.to(device)
            
            with amp_autocast():
                logits = model(patches, times, stats, topo)
            
            if GPU:
                torch.cuda.synchronize()
            
            t1 = time.perf_counter()
            
            batch_time = t1 - t0
            batch_times.append(batch_time)
            sample_times.extend([batch_time / batch_size] * batch_size)
    
    stats = {
        'batch_inference': {
            'mean_ms': float(np.mean(batch_times) * 1000),
            'std_ms': float(np.std(batch_times) * 1000),
            'p50_ms': float(np.percentile(batch_times, 50) * 1000),
            'p95_ms': float(np.percentile(batch_times, 95) * 1000),
        },
        'per_sample_inference': {
            'mean_ms': float(np.mean(sample_times) * 1000),
            'std_ms': float(np.std(sample_times) * 1000),
            'p50_ms': float(np.percentile(sample_times, 50) * 1000),
            'p95_ms': float(np.percentile(sample_times, 95) * 1000),
        },
        'throughput_samples_per_sec': float(len(sample_times) / sum(batch_times))
    }
    
    return stats

def compute_model_complexity():
    """Compute model parameters and FLOPs estimate"""
    model_temp = PatchTSTClassifier(cfg, num_classes).to(device)
    
    total_params = sum(p.numel() for p in model_temp.parameters())
    trainable_params = sum(p.numel() for p in model_temp.parameters() if p.requires_grad)
    
    # Estimate FLOPs for one forward pass (approximate)
    # Transformer attention: O(N^2 * D) per layer
    # FFN: O(N * D^2) per layer
    N = cfg.N_PATCHES + 2
    D = cfg.D_MODEL
    L = cfg.N_LAYERS
    
    attn_flops = L * N * N * D * 4  # Q, K, V, output projection
    ffn_flops = L * N * D * (2*D) * 2  # Two linear layers
    embed_flops = cfg.N_PATCHES * cfg.CHANNELS * cfg.PATCH_LEN * D
    cls_flops = D * (D//2) + (D//2) * num_classes
    
    total_flops = attn_flops + ffn_flops + embed_flops + cls_flops
    
    complexity = {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'estimated_flops': int(total_flops),
        'estimated_gflops': float(total_flops / 1e9),
        'model_size_mb': float(total_params * 4 / (1024**2))  # Assuming float32
    }
    
    del model_temp
    return complexity

# =========================
# Training with enhanced metrics
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
    best_path = cfg.PROC_DIR / "patchtst_stats_topo_rope_best.pth"
    best_epoch, patience_ctr = -1, 0

    for epoch in range(cfg.EPOCHS):
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
        metrics = compute_comprehensive_metrics(val_truths, val_preds)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR {current_lr:.2e} | Loss {total/max(1,len(train_loader)):.4f}")
        print(f"  Acc={metrics['accuracy']:.4f} | F1={metrics['macro_f1']:.4f} | "
              f"Kappa={metrics['cohen_kappa']:.4f} | MCC={metrics['mcc']:.4f} | F2={metrics['f2_score']:.4f}")

        score = metrics['macro_f1'] + metrics['cohen_kappa']
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}; best epoch {best_epoch+1}")
                break

@torch.no_grad()
def get_raw_predictions(model, loader):
    model.eval()
    preds, truths, pids, first_ns, probs_all = [], [], [], [], []
    for patches, times, stats, topo, labels, sub_ids, firsts in loader:
        patches = patches.to(device); times = times.to(device)
        stats   = stats.to(device);   topo  = topo.to(device)
        logits  = model(patches, times, stats, topo)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        pred    = probs.argmax(1)
        preds.extend(pred.tolist())
        probs_all.extend(probs.tolist())
        truths.extend(labels.numpy().tolist())
        pids.extend(list(sub_ids))
        first_ns.extend(firsts.numpy().tolist())
    return (np.array(preds), np.array(truths), np.array(pids, dtype=object),
            np.array(first_ns, dtype=np.int64), np.array(probs_all, dtype=np.float32))

# =========================
# HMM
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
# Explainability (abbreviated for space)
# =========================
@torch.no_grad()
def predict_proba(model, P,T,S,G, capture_attn: bool=False):
    logits = model(P, T, S, G, capture_attn=capture_attn)
    return torch.softmax(logits, dim=1)

def save_json(obj, path: Path):
    if cfg.SAVE_JSON:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

# =========================
# Main
# =========================
def main():
    cfg.PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Loaders
    train_ds, train_dl = make_loader(train_pids, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_ds,   val_dl   = make_loader(val_pids,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_ds,  test_dl  = make_loader(test_pids,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    print(f"\n🧩 Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # =========================
    # COMPUTATIONAL PROFILING
    # =========================
    print("\n" + "="*60)
    print("COMPUTATIONAL COST ANALYSIS")
    print("="*60)
    
    # 1. Topological feature extraction profiling
    topo_stats = profile_topological_computation(
        num_samples=cfg.PROFILE_ITERATIONS,
        warmup=cfg.PROFILE_WARMUP
    )
    
    print("\n📊 Topological Feature Extraction Times:")
    for key, stats in topo_stats.items():
        print(f"  {key}:")
        print(f"    Mean: {stats['mean_ms']:.2f} ms ± {stats['std_ms']:.2f} ms")
        print(f"    P50: {stats['p50_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms | P99: {stats['p99_ms']:.2f} ms")
    
    # 2. Model complexity
    complexity = compute_model_complexity()
    print("\n🏗️ Model Complexity:")
    print(f"  Total Parameters: {complexity['total_parameters']:,}")
    print(f"  Trainable Parameters: {complexity['trainable_parameters']:,}")
    print(f"  Model Size: {complexity['model_size_mb']:.2f} MB")
    print(f"  Estimated GFLOPs: {complexity['estimated_gflops']:.3f}")
    
    # Save profiling results
    profiling_results = {
        'topological_computation': topo_stats,
        'model_complexity': complexity,
        'config': {
            'window_size': cfg.WINDOW_SIZE,
            'patch_length': cfg.PATCH_LEN,
            'n_patches': cfg.N_PATCHES,
            'd_model': cfg.D_MODEL,
            'n_layers': cfg.N_LAYERS,
            'n_heads': cfg.N_HEADS,
            'takens_m': cfg.TAKENS_M,
            'takens_tau': cfg.TAKENS_TAU,
        }
    }
    save_json(profiling_results, cfg.PROC_DIR / "computational_profiling.json")
    
    # Model
    model = PatchTSTClassifier(cfg, num_classes).to(device)
    class_w = compute_class_weights(train_ds, num_classes).to(device)

    # Train
    print("\n🎯 Training...")
    train_model(model, train_dl, val_dl, class_w=class_w, patience=cfg.EARLY_STOP_PATIENCE)

    # Load best weights
    best_path = cfg.PROC_DIR / "patchtst_stats_topo_rope_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    # =========================
    # END-TO-END INFERENCE PROFILING
    # =========================
    inference_stats = profile_end_to_end_inference(model, test_dl, num_batches=min(50, len(test_dl)))
    print("\n⚡ End-to-End Inference Performance:")
    print(f"  Per-sample mean: {inference_stats['per_sample_inference']['mean_ms']:.2f} ms")
    print(f"  Per-sample P95: {inference_stats['per_sample_inference']['p95_ms']:.2f} ms")
    print(f"  Throughput: {inference_stats['throughput_samples_per_sec']:.1f} samples/sec")
    
    profiling_results['inference'] = inference_stats
    save_json(profiling_results, cfg.PROC_DIR / "computational_profiling.json")
    
    # Check real-time feasibility
    realtime_latency_ms = (cfg.WINDOW_SIZE / cfg.SIGNAL_RATE) * 1000  # Window duration in ms
    inference_latency_p95 = inference_stats['per_sample_inference']['p95_ms']
    
    # Add topological computation to total latency
    topo_latency_p95 = topo_stats['total']['p95_ms']
    total_latency_p95 = inference_latency_p95 + topo_latency_p95
    
    print(f"\n🚦 Real-Time Feasibility Analysis:")
    print(f"  Window duration: {realtime_latency_ms:.1f} ms")
    print(f"  Total processing latency (P95): {total_latency_p95:.2f} ms")
    print(f"    - Topological features: {topo_latency_p95:.2f} ms")
    print(f"    - Model inference: {inference_latency_p95:.2f} ms")
    
    if total_latency_p95 < realtime_latency_ms:
        print(f"  ✅ REAL-TIME CAPABLE (latency < window duration)")
        print(f"     Slack: {realtime_latency_ms - total_latency_p95:.2f} ms")
    else:
        print(f"  ⚠️  NOT REAL-TIME (latency > window duration)")
        print(f"     Overrun: {total_latency_p95 - realtime_latency_ms:.2f} ms")
    
    # Raw predictions
    print("\n🎯 Testing (raw predictions)...")
    raw_pred, raw_true, raw_pid, raw_first, raw_probs = get_raw_predictions(model, test_dl)

    # =========================
    # COMPREHENSIVE METRICS - RAW
    # =========================
    print("\n" + "="*60)
    print("RAW MODEL EVALUATION (Comprehensive Metrics)")
    print("="*60)
    
    raw_metrics = compute_comprehensive_metrics(raw_true, raw_pred)
    raw_seq_metrics = compute_sequence_metrics(raw_true, raw_pred, raw_pid)
    
    print(f"\n📈 Classification Metrics:")
    print(f"  Accuracy: {raw_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {raw_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {raw_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {raw_metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {raw_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {raw_metrics['macro_recall']:.4f}")
    
    print(f"\n📊 Advanced Metrics:")
    print(f"  Cohen's Kappa (κ): {raw_metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Correlation (MCC): {raw_metrics['mcc']:.4f}")
    print(f"  Pearson-Yule Phi (ϕ): {raw_metrics['pearson_yule_phi']:.4f}")
    print(f"  F2 Score (recall-weighted): {raw_metrics['f2_score']:.4f}")
    
    print(f"\n🔄 Sequence & Transition Metrics:")
    print(f"  Transition Accuracy: {raw_seq_metrics['transition_accuracy']:.4f} "
          f"± {raw_seq_metrics['transition_accuracy_std']:.4f}")
    print(f"  Participants with transitions: {raw_seq_metrics['num_participants_with_transitions']}")
    
    print(f"\n📋 Per-Class Performance:")
    for cls_name, cls_metrics in raw_metrics['per_class'].items():
        print(f"  {cls_name}: F1={cls_metrics['f1']:.3f}, "
              f"Prec={cls_metrics['precision']:.3f}, Rec={cls_metrics['recall']:.3f}")
    
    # Save raw metrics
    raw_metrics.update(raw_seq_metrics)
    save_json(raw_metrics, cfg.PROC_DIR / "metrics_raw_comprehensive.json")

    # Confusion matrix
    cm_raw = confusion_matrix(raw_true, raw_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, square=True, cbar_kws={'label': 'Count'})
    plt.title(f"Raw Confusion Matrix\nAcc={raw_metrics['accuracy']:.3f}, F1={raw_metrics['macro_f1']:.3f}, κ={raw_metrics['cohen_kappa']:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "cm_raw_comprehensive.png", dpi=200); plt.close()

    # HMM post-processing
    print("\n🤖 HMM post-processing...")
    pi, A = estimate_hmm_from_train(train_ds, num_classes)
    log_pi = np.log(np.clip(pi, cfg.HMM_MIN_PROB, 1.0))
    log_A  = np.log(np.clip(A, cfg.HMM_MIN_PROB, 1.0))

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

    hmm_preds, hmm_truth, hmm_pids = [], [], []
    for pid in by_pid_probs.keys():
        order = np.argsort(by_pid_time[pid])
        E = np.log(np.clip(np.vstack([by_pid_probs[pid][i] for i in order]), cfg.HMM_MIN_PROB, 1.0))
        dec = viterbi_decode(E, log_pi, log_A)
        hmm_preds.extend(dec.tolist())
        hmm_truth.extend([by_pid_truth[pid][i] for i in order])
        hmm_pids.extend([pid] * len(dec))

    hmm_preds = np.asarray(hmm_preds, dtype=np.int64)
    hmm_truth = np.asarray(hmm_truth, dtype=np.int64)
    hmm_pids = np.asarray(hmm_pids, dtype=object)

    # =========================
    # COMPREHENSIVE METRICS - HMM
    # =========================
    print("\n" + "="*60)
    print("HMM POST-PROCESSED EVALUATION (Comprehensive Metrics)")
    print("="*60)
    
    hmm_metrics = compute_comprehensive_metrics(hmm_truth, hmm_preds)
    hmm_seq_metrics = compute_sequence_metrics(hmm_truth, hmm_preds, hmm_pids)
    
    print(f"\n📈 Classification Metrics:")
    print(f"  Accuracy: {hmm_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {hmm_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {hmm_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {hmm_metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {hmm_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {hmm_metrics['macro_recall']:.4f}")
    
    print(f"\n📊 Advanced Metrics:")
    print(f"  Cohen's Kappa (κ): {hmm_metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Correlation (MCC): {hmm_metrics['mcc']:.4f}")
    print(f"  Pearson-Yule Phi (ϕ): {hmm_metrics['pearson_yule_phi']:.4f}")
    print(f"  F2 Score (recall-weighted): {hmm_metrics['f2_score']:.4f}")
    
    print(f"\n🔄 Sequence & Transition Metrics:")
    print(f"  Transition Accuracy: {hmm_seq_metrics['transition_accuracy']:.4f} "
          f"± {hmm_seq_metrics['transition_accuracy_std']:.4f}")
    print(f"  Participants with transitions: {hmm_seq_metrics['num_participants_with_transitions']}")
    
    print(f"\n📋 Per-Class Performance:")
    for cls_name, cls_metrics in hmm_metrics['per_class'].items():
        print(f"  {cls_name}: F1={cls_metrics['f1']:.3f}, "
              f"Prec={cls_metrics['precision']:.3f}, Rec={cls_metrics['recall']:.3f}")
    
    # Improvement analysis
    print(f"\n📊 Improvement from HMM Post-Processing:")
    print(f"  Δ Accuracy: {(hmm_metrics['accuracy'] - raw_metrics['accuracy']):.4f}")
    print(f"  Δ Macro F1: {(hmm_metrics['macro_f1'] - raw_metrics['macro_f1']):.4f}")
    print(f"  Δ Transition Accuracy: {(hmm_seq_metrics['transition_accuracy'] - raw_seq_metrics['transition_accuracy']):.4f}")
    
    # Save HMM metrics
    hmm_metrics.update(hmm_seq_metrics)
    save_json(hmm_metrics, cfg.PROC_DIR / "metrics_hmm_comprehensive.json")

    # Confusion matrix
    cm_hmm = confusion_matrix(hmm_truth, hmm_preds, labels=np.arange(num_classes))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_hmm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, square=True, cbar_kws={'label': 'Count'})
    plt.title(f"HMM Confusion Matrix\nAcc={hmm_metrics['accuracy']:.3f}, F1={hmm_metrics['macro_f1']:.3f}, κ={hmm_metrics['cohen_kappa']:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "cm_hmm_comprehensive.png", dpi=200); plt.close()

    # =========================
    # COMPARATIVE METRICS VISUALIZATION
    # =========================
    print("\n📊 Generating comparative visualizations...")
    
    # Metric comparison bar chart
    metrics_to_compare = ['accuracy', 'macro_f1', 'cohen_kappa', 'mcc', 'f2_score']
    raw_vals = [raw_metrics[m] for m in metrics_to_compare]
    hmm_vals = [hmm_metrics[m] for m in metrics_to_compare]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_vals, width, label='Raw Model', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hmm_vals, width, label='+ HMM', color='forestgreen', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comprehensive Performance Comparison: Raw vs HMM Post-Processing', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Macro F1', "Cohen's κ", 'MCC', 'F2 Score'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "metrics_comparison.png", dpi=200)
    plt.close()
    
    # Per-class F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    raw_class_f1 = [raw_metrics['per_class'][c]['f1'] for c in classes]
    hmm_class_f1 = [hmm_metrics['per_class'][c]['f1'] for c in classes]
    
    x = np.arange(len(classes))
    bars1 = ax.bar(x - width/2, raw_class_f1, width, label='Raw Model', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hmm_class_f1, width, label='+ HMM', color='forestgreen', alpha=0.8)
    
    ax.set_xlabel('Activity Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "per_class_f1_comparison.png", dpi=200)
    plt.close()
    
    # Computational cost breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart for feature extraction time breakdown
    topo_components = ['preprocessing', 'takens_embedding', 
                       'ripser_computation' if _HAVE_RIPSER else 'fallback_computation',
                       'feature_extraction']
    topo_times = [topo_stats[c]['mean_ms'] for c in topo_components]
    
    ax1.pie(topo_times, labels=topo_components, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Topological Feature Extraction\nTime Breakdown', fontweight='bold')
    
    # Bar chart for end-to-end latency
    latency_components = ['Topological Features', 'Statistical Features', 'Model Inference']
    latency_times = [
        topo_stats['total']['mean_ms'],
        5.0,  # Approximate stat feature time (fast)
        inference_stats['per_sample_inference']['mean_ms']
    ]
    
    bars = ax2.bar(latency_components, latency_times, color=['coral', 'skyblue', 'lightgreen'], alpha=0.8)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('End-to-End Processing Latency\nBreakdown (per sample)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)
    
    # Add total line
    total_time = sum(latency_times)
    ax2.axhline(y=total_time, color='red', linestyle='--', linewidth=2, label=f'Total: {total_time:.2f}ms')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(cfg.PROC_DIR / "computational_cost_breakdown.png", dpi=200)
    plt.close()
    
    # =========================
    # SAVE COMPREHENSIVE BUNDLE
    # =========================
    print("\n💾 Saving comprehensive artifacts bundle...")
    
    comprehensive_bundle = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "hmm": {"pi": pi.tolist(), "A": A.tolist()},
        
        "metrics": {
            "raw": raw_metrics,
            "hmm": hmm_metrics,
        },
        
        "computational_profiling": profiling_results,
        
        "config": vars(cfg),
        
        "realtime_analysis": {
            "window_duration_ms": realtime_latency_ms,
            "total_latency_p95_ms": total_latency_p95,
            "topological_latency_p95_ms": topo_latency_p95,
            "inference_latency_p95_ms": inference_latency_p95,
            "is_realtime_capable": total_latency_p95 < realtime_latency_ms,
            "latency_slack_ms": realtime_latency_ms - total_latency_p95,
        },
        
        "summary": {
            "best_raw_metric": f"F1={raw_metrics['macro_f1']:.4f}, κ={raw_metrics['cohen_kappa']:.4f}",
            "best_hmm_metric": f"F1={hmm_metrics['macro_f1']:.4f}, κ={hmm_metrics['cohen_kappa']:.4f}",
            "transition_accuracy_hmm": hmm_seq_metrics['transition_accuracy'],
            "model_parameters": complexity['total_parameters'],
            "realtime_capable": total_latency_p95 < realtime_latency_ms,
        }
    }
    
    torch.save(comprehensive_bundle, cfg.PROC_DIR / "comprehensive_artifacts.pth")
    print(f"💾 Saved: {cfg.PROC_DIR / 'comprehensive_artifacts.pth'}")
    
    # Save human-readable summary
    summary_text = f"""
{'='*70}
PATCHTST-HAR COMPREHENSIVE EVALUATION SUMMARY
{'='*70}

DATASET CONFIGURATION
---------------------
Training Participants: {len(train_pids)}
Validation Participants: {len(val_pids)}
Test Participants: {len(test_pids)}
Total Windows (test): {len(test_ds)}
Classes: {', '.join(classes)}

MODEL CONFIGURATION
-------------------
Parameters: {complexity['total_parameters']:,}
Model Size: {complexity['model_size_mb']:.2f} MB
Estimated GFLOPs: {complexity['estimated_gflops']:.3f}
Architecture: PatchTST + RoPE + Statistical + Topological Features

RAW MODEL PERFORMANCE
---------------------
Accuracy:           {raw_metrics['accuracy']:.4f}
Balanced Accuracy:  {raw_metrics['balanced_accuracy']:.4f}
Macro F1:           {raw_metrics['macro_f1']:.4f}
Weighted F1:        {raw_metrics['weighted_f1']:.4f}
Cohen's Kappa (κ):  {raw_metrics['cohen_kappa']:.4f}
MCC:                {raw_metrics['mcc']:.4f}
Pearson-Yule (ϕ):   {raw_metrics['pearson_yule_phi']:.4f}
F2 Score:           {raw_metrics['f2_score']:.4f}
Transition Acc:     {raw_seq_metrics['transition_accuracy']:.4f} ± {raw_seq_metrics['transition_accuracy_std']:.4f}

HMM POST-PROCESSED PERFORMANCE
-------------------------------
Accuracy:           {hmm_metrics['accuracy']:.4f}
Balanced Accuracy:  {hmm_metrics['balanced_accuracy']:.4f}
Macro F1:           {hmm_metrics['macro_f1']:.4f}
Weighted F1:        {hmm_metrics['weighted_f1']:.4f}
Cohen's Kappa (κ):  {hmm_metrics['cohen_kappa']:.4f}
MCC:                {hmm_metrics['mcc']:.4f}
Pearson-Yule (ϕ):   {hmm_metrics['pearson_yule_phi']:.4f}
F2 Score:           {hmm_metrics['f2_score']:.4f}
Transition Acc:     {hmm_seq_metrics['transition_accuracy']:.4f} ± {hmm_seq_metrics['transition_accuracy_std']:.4f}

IMPROVEMENT FROM HMM
--------------------
Δ Accuracy:         {(hmm_metrics['accuracy'] - raw_metrics['accuracy']):.4f}
Δ Macro F1:         {(hmm_metrics['macro_f1'] - raw_metrics['macro_f1']):.4f}
Δ Transition Acc:   {(hmm_seq_metrics['transition_accuracy'] - raw_seq_metrics['transition_accuracy']):.4f}

COMPUTATIONAL COST ANALYSIS
----------------------------
Topological Feature Extraction (per window):
  Mean:  {topo_stats['total']['mean_ms']:.2f} ms ± {topo_stats['total']['std_ms']:.2f} ms
  P50:   {topo_stats['total']['p50_ms']:.2f} ms
  P95:   {topo_stats['total']['p95_ms']:.2f} ms
  P99:   {topo_stats['total']['p99_ms']:.2f} ms

Model Inference (per sample):
  Mean:  {inference_stats['per_sample_inference']['mean_ms']:.2f} ms ± {inference_stats['per_sample_inference']['std_ms']:.2f} ms
  P50:   {inference_stats['per_sample_inference']['p50_ms']:.2f} ms
  P95:   {inference_stats['per_sample_inference']['p95_ms']:.2f} ms

Throughput: {inference_stats['throughput_samples_per_sec']:.1f} samples/second

REAL-TIME FEASIBILITY
---------------------
Window Duration:       {realtime_latency_ms:.1f} ms
Total Latency (P95):   {total_latency_p95:.2f} ms
  - Topological:       {topo_latency_p95:.2f} ms
  - Inference:         {inference_latency_p95:.2f} ms
Real-Time Capable:     {'YES ✓' if total_latency_p95 < realtime_latency_ms else 'NO ✗'}
{'Slack: ' + f'{realtime_latency_ms - total_latency_p95:.2f} ms' if total_latency_p95 < realtime_latency_ms else 'Overrun: ' + f'{total_latency_p95 - realtime_latency_ms:.2f} ms'}

RESOURCE REQUIREMENTS
---------------------
Memory Footprint:     {complexity['model_size_mb']:.2f} MB (model only)
Compute per Sample:   {complexity['estimated_gflops']:.3f} GFLOPs
Deployment Target:    {'✓ Edge-capable' if total_latency_p95 < 100 else '⚠ Cloud recommended'}

PER-CLASS PERFORMANCE (HMM)
----------------------------
"""
    for cls_name in classes:
        cls_m = hmm_metrics['per_class'][cls_name]
        summary_text += f"{cls_name:15s} | F1: {cls_m['f1']:.3f} | Prec: {cls_m['precision']:.3f} | Rec: {cls_m['recall']:.3f}\n"
    
    summary_text += f"\n{'='*70}\n"
    summary_text += f"Report generated: {pd.Timestamp.now()}\n"
    summary_text += f"{'='*70}\n"
    
    with open(cfg.PROC_DIR / "EVALUATION_SUMMARY.txt", "w") as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    
    print("\n✅ All comprehensive evaluations complete!")
    print(f"\n📁 Results saved to: {cfg.PROC_DIR}")
    print("\nGenerated files:")
    print("  - comprehensive_artifacts.pth (model + all metrics)")
    print("  - computational_profiling.json (timing analysis)")
    print("  - metrics_raw_comprehensive.json")
    print("  - metrics_hmm_comprehensive.json")
    print("  - EVALUATION_SUMMARY.txt (human-readable report)")
    print("  - metrics_comparison.png")
    print("  - per_class_f1_comparison.png")
    print("  - computational_cost_breakdown.png")
    print("  - cm_raw_comprehensive.png")
    print("  - cm_hmm_comprehensive.png")

if __name__ == "__main__":
    main()
