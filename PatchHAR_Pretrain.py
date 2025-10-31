# -*- coding: utf-8 -*-
# =========================================================
# PHASE A ONLY — SELF-SUPERVISED PRETRAINING (MPR) FOR 1D SENSORS
# ---------------------------------------------------------
# - Dataset: CAPTURE-24-like NPZ files with accelerometer windows.
# - Task: Masked Patch Reconstruction (MPR) — reconstruct RAW PATCHES (C×PL).
# - Backbone: RoPE Transformer Encoder.
# - Tokens: Patch tokens + Stats token + Topology token (NO time embeddings).
# - Regularization: dropout on patch tokens and tiny dropout (p=0.05) on stats/topo tokens.
# - Training: AdamW + optional Warmup-Cosine; AMP; early stopping; deterministic loaders.
# =========================================================

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# [A] CORE LIBRARIES & SETTINGS
# ------------------------------
from pathlib import Path
from contextlib import nullcontext
import math, random, json, copy, os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

# (Optional) plotting backend; never blocks training
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================
# [B] CONFIGURATION (DATA, MODEL, TRAINING)
# ============================================
class PretrainCfg:
    # ---- DATA INPUT ----
    PROC_DIR   = Path("/home/ali/capture24/processed_minimal")
    VAL_SUBJECTS = 40  # maximum; actual val = min(VAL_SUBJECTS, ~20% heuristic)

    # ---- SIGNAL & PATCHING ----
    SIGNAL_RATE = 100
    WINDOW_SIZE = 1000       # samples per window
    PATCH_LEN   = 100        # samples per patch
    CHANNELS    = 3
    N_PATCHES   = WINDOW_SIZE // PATCH_LEN  # 10 patches of length 100

    # ---- BACKBONE SHAPES ----
    D_MODEL   = 768
    N_HEADS   = 12
    N_LAYERS  = 12
    DROPOUT   = 0.10

    # ---- FEATURE HEAD DIMS (stat/topo) ----
    N_STAT_FEATURES = 56
    N_TOPO_FEATURES = 24

    # ---- TOPOLOGY OPTIONS ----
    TAKENS_M = 3
    TAKENS_TAU = 5
    TOPO_MAX_POINTS = 600

    # ---- SSL TRAINING (MPR) ----
    BATCH_SIZE    = 32
    EPOCHS        = 150
    LR            = 1e-4          # lower LR for stability
    WEIGHT_DECAY  = 2e-4          # slightly higher WD
    MAX_GRAD_NORM = 0.5           # tighter clip
    EARLY_STOP_PATIENCE = 50
    WARMUP_FRAC   = 0.05

    # Stronger masking (better signal for MPR)
    MASK_RATIO    = 0.40

    # ---- MISC ----
    SEED = 42

    # ---- OUTPUTS ----
    OUT_DIR   = Path("/home/ali/capture24/pretrain_ssl_outputs_no_time")
    CKPT_PATH = OUT_DIR / "pretrained_backbone_ssl.pth"

PT = PretrainCfg()
PT.OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# [C] REPRODUCIBILITY & DEVICE SELECTION
# ============================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(PT.SEED)

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Device: {device}")
try:
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
except Exception:
    pass

def amp_autocast():
    """
    Prefer BF16 if available; else FP16; CPU -> no-op.
    """
    if device.type == "cuda":
        try:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()
    return nullcontext()

def _amp_uses_fp16() -> bool:
    """
    Determine whether we'll be using FP16 autocast (vs BF16).
    """
    if device.type != "cuda":
        return False
    # If BF16 autocast doesn't raise, we'll use BF16 (thus no scaler).
    try:
        _ = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return False
    except Exception:
        return True

# ===================================================
# [D] FEATURE ENGINEERING (STATS & TOPOLOGY ONLY)
#      -> NO TIME FEATURES; timestamps ignored
# ===================================================
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

def _normalized_psd(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(x)
    ps = (np.abs(X) ** 2).astype(np.float64)
    s = ps.sum()
    if s > 0: ps = ps / s
    freqs_hz = np.fft.rfftfreq(x.shape[0], d=1.0/sr)
    return ps, freqs_hz

def _spectral_entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return 0.0
    H = -float(np.sum(p * np.log(p)))
    Hmax = math.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0

def _dominant_two(freq_probs: np.ndarray, freqs_hz: np.ndarray) -> tuple[float,float,float,float]:
    if freq_probs.size <= 2: return 0.0, 0.0, 0.0, 0.0
    p = freq_probs.copy(); f = freqs_hz.copy()
    p[0] = 0.0
    i1 = int(np.argmax(p)); p1 = float(p[i1]); f1 = float(f[i1])
    p[i1] = -1.0
    i2 = int(np.argmax(p)); p2 = max(float(p[i2]), 0.0); f2 = float(f[i2])
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
    yaw_p = np.arctan2(vx, vy)  # proxy
    return roll, pitch, yaw_p

def compute_statistical_features(window_norm: np.ndarray,
                                 sr: int = 100,
                                 expect_dim: int = 56) -> np.ndarray:
    x, y, z = window_norm[:,0], window_norm[:,1], window_norm[:,2]
    mag = np.linalg.norm(window_norm, axis=1)
    feats: list[float] = []

    # channel-wise moments/range
    for sig in (x, y, z):
        feats.extend([float(np.mean(sig)), float(np.std(sig)), float(np.max(sig)-np.min(sig))])

    # cross-channel corr
    feats.append(_safe_corr(x, y)); feats.append(_safe_corr(x, z)); feats.append(_safe_corr(y, z))

    # magnitude summary
    feats.extend([
        float(np.mean(mag)), float(np.std(mag)),
        float(np.max(mag)-np.min(mag)),
        _mad(mag), _kurtosis_excess(mag), _skew(mag),
        float(np.median(mag)),
    ])

    # quantiles and min/max per axis and magnitude
    def qpack(sig):
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        return [float(np.min(sig)), float(np.max(sig)), float(q50), float(q25), float(q75)]
    for sig in (x, y, z, mag):
        feats.extend(qpack(sig))

    # short-lag autocorr + spectrum
    feats.append(_one_sec_autocorr(mag, sr))
    ps, freqs_hz = _normalized_psd(mag, sr)
    f1, p1, f2, p2 = _dominant_two(ps, freqs_hz)
    feats.extend([f1, p1, f2, p2])
    feats.append(_spectral_entropy(ps))

    # simple peak stats
    if _HAVE_SCIPY:
        peaks, _ = find_peaks(mag)
        if peaks.size:
            prom = peak_prominences(mag, peaks)[0]
            feats.extend([float(len(peaks)), float(np.median(prom))])
        else:
            feats.extend([0.0, 0.0])
    else:
        if mag.size >= 3:
            pk = ((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])).sum()
            feats.append(float(pk))
        else:
            feats.append(0.0)
        feats.append(0.0)

    # gravity/dynamic components orientation
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
    assert arr.shape[0] == expect_dim, f"Expected {expect_dim} stats, got {arr.shape[0]}"
    return arr

# ----- Topology features (Takens + persistent homology) -----
try:
    from ripser import ripser
    _HAVE_RIPSER = True
except Exception:
    _HAVE_RIPSER = False

def _takens_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    T = x.shape[0]; L = T - (m - 1) * tau
    if L <= 5: return np.zeros((5, m), dtype=np.float32)
    out = np.stack([x[i:i+L] for i in range(0, m*tau, tau)], axis=1)
    return out.astype(np.float32)

def _persistence_entropy(diag: np.ndarray) -> float:
    if diag.size == 0: return 0.0
    births = diag[:,0]; deaths = diag[:,1]
    finite = np.isfinite(deaths)
    births, deaths = births[finite], deaths[finite]
    pers = np.maximum(deaths - births, 0.0)
    s = pers.sum()
    if s <= 0: return 0.0
    p = pers / s
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = np.log(len(p))
    return float(H / (Hmax + 1e-12))

def _topk_lifetimes(diag: np.ndarray, k: int = 3) -> list[float]:
    if diag.size == 0: return [0.0]*k
    births = diag[:,0]; deaths = diag[:,1]
    finite = np.isfinite(deaths)
    lifetimes = np.maximum(deaths[finite] - births[finite], 0.0)
    lifetimes = np.sort(lifetimes)[::-1]
    pad = np.zeros(k, dtype=np.float32)
    pad[:min(k, lifetimes.size)] = lifetimes[:k]
    return pad.tolist()

def compute_topological_features(window_norm: np.ndarray,
                                 sr: int = 100,
                                 m: int = 3,
                                 tau: int = 5,
                                 topo_max_points: int = 600,
                                 expect_dim: int = 24) -> np.ndarray:
    mag = np.linalg.norm(window_norm, axis=1).astype(np.float32)
    if mag.shape[0] > topo_max_points:
        step = int(np.ceil(mag.shape[0] / topo_max_points))
        mag = mag[::step]

    X = _takens_embed(mag, m=m, tau=tau)
    if _HAVE_RIPSER and X.shape[0] >= 8:
        res = ripser(X, maxdim=1)
        D0 = res.get('dgms', [np.empty((0,2)), np.empty((0,2))])[0]
        D1 = res.get('dgms', [np.empty((0,2)), np.empty((0,2))])[1]
    else:
        d = np.abs(np.subtract.outer(mag, mag))
        thr = np.quantile(d[np.isfinite(d)], [0.1, 0.2, 0.3, 0.5]) if np.isfinite(d).any() else [0.0,0.0,0.0,0.0]
        def fake_diag(dist, t):
            M = (dist < t).astype(np.float32)
            lifetimes = []
            for k in range(-10, 11):
                diag = np.diag(M, k=k)
                runs, run = [], 0
                for v in diag:
                    if v > 0.5: run += 1
                    elif run: runs.append(run); run = 0
                if run: runs.append(run)
                lifetimes.extend(runs)
            if len(lifetimes) == 0:
                return np.empty((0,2), dtype=np.float32)
            lifetimes = np.asarray(lifetimes, dtype=np.float32)
            births = np.zeros_like(lifetimes); deaths = lifetimes
            return np.stack([births, deaths], axis=1)
        D0 = fake_diag(d, thr[1]); D1 = fake_diag(d, thr[3])

    feats = []
    for D in (D0, D1):
        if D.size == 0:
            feats.extend([0.0,0.0,0.0]); feats.append(0.0)
            feats.extend([0.0,0.0,0.0]); feats.extend([0.0,0.0])
            feats.extend([0.0,0.0,0.0]); continue
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
            feats.extend([0.0,0.0,0.0])

    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[0] == expect_dim, f"Expected {expect_dim}, got {arr.shape[0]}"
    return arr

# ==============================================
# [E] POSITIONAL ENCODING — RoPE (NO time tok)
# ==============================================
def precompute_freqs_cis(dim: int, n_tokens: int, theta: float = 10000.0):
    """
    Rotary frequencies as complex unit-circle values.
    Returns shape: (n_tokens, dim//2) complex.
    """
    assert dim % 2 == 0, "RoPE needs even per-head dim"
    freq_seq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(n_tokens, dtype=torch.float32)
    freqs = torch.outer(t, freq_seq)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_pos_emb(q, k, freqs_cis):
    """
    q,k: (B,H,N,D) real. freqs_cis: (N, D/2) complex.
    """
    B, H, N, D = q.shape
    d2 = D // 2
    def _to_complex(x):
        x_2 = x.float().contiguous().view(B, H, N, d2, 2)
        return torch.view_as_complex(x_2)
    freqs = freqs_cis[:N].to(q.device)
    q_c = _to_complex(q); k_c = _to_complex(k)
    q_rot = q_c * freqs.view(1,1,N,d2)
    k_rot = k_c * freqs.view(1,1,N,d2)
    q_out = torch.view_as_real(q_rot).view(B,H,N,D).type_as(q)
    k_out = torch.view_as_real(k_rot).view(B,H,N,D).type_as(k)
    return q_out, k_out

# ============================================
# [F] DATASET & DATALOADER (NPZ -> Tokens)
# ============================================
class PT_SSL_Dataset(Dataset):
    """
    .npz per subject:
      - 'windows' : float32 array (N, WINDOW_SIZE, CHANNELS)
      - optional timestamps ignored
    Returns:
      patches : (C, NP, PL)     # C=channels, NP=patches, PL=patch_len
      sfeat   : (N_STAT_FEATURES,)
      tafeat  : (N_TOPO_FEATURES,)
    All windows are per-channel standardized and clipped to [-10, 10].
    """
    def __init__(self, pid_list, proc_dir: Path):
        self.win = []   # (window_normed, sfeat, tafeat)
        for pid in pid_list:
            npz_path = proc_dir / f"{pid}.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path, allow_pickle=True)
            if "windows" not in npz:
                continue
            W   = npz["windows"].astype(np.float32)  # (N, T, C)
            if W.ndim != 3:
                continue
            if W.shape[1] != PT.WINDOW_SIZE or W.shape[2] != PT.CHANNELS:
                # allow auto-trim if longer; skip otherwise
                if W.shape[1] > PT.WINDOW_SIZE and W.shape[2] == PT.CHANNELS:
                    W = W[:, :PT.WINDOW_SIZE, :]
                else:
                    continue

            # sort by time-like key if present
            order = None
            if "times_epoch_ns" in npz:
                mids = np.median(npz["times_epoch_ns"], axis=1).astype(np.int64)
                order = np.argsort(mids)
            elif "first_ts_epoch_ns" in npz:
                order = np.argsort(npz["first_ts_epoch_ns"].astype(np.int64))
            if order is not None:
                W = W[order]

            for i in range(W.shape[0]):
                w = W[i]  # (T, C)
                wn = np.zeros_like(w, dtype=np.float32)
                for c in range(PT.CHANNELS):
                    ch = w[:, c]; mu, sd = float(ch.mean()), float(ch.std())
                    wn[:, c] = (ch - mu) / (sd + 1e-8)
                wn = np.clip(wn, -10, 10)

                # handcrafted features (no time)
                sfeat  = compute_statistical_features(wn, sr=PT.SIGNAL_RATE,
                                                      expect_dim=PT.N_STAT_FEATURES)
                tafeat = compute_topological_features(wn, sr=PT.SIGNAL_RATE,
                                                      m=PT.TAKENS_M, tau=PT.TAKENS_TAU,
                                                      topo_max_points=PT.TOPO_MAX_POINTS,
                                                      expect_dim=PT.N_TOPO_FEATURES)
                self.win.append((wn.astype(np.float32),
                                 sfeat.astype(np.float32),
                                 tafeat.astype(np.float32)))
        self.len = len(self.win)
        assert PT.WINDOW_SIZE % PT.PATCH_LEN == 0, "WINDOW_SIZE must be divisible by PATCH_LEN"
        if self.len == 0:
            raise RuntimeError("No valid windows found. Check PROC_DIR and npz contents.")

    def __len__(self): return self.len

    def __getitem__(self, idx):
        wn, sfeat, tafeat = self.win[idx]
        # (T,C) -> (C, NP, PL)
        NP = PT.N_PATCHES; PL = PT.PATCH_LEN; C = PT.CHANNELS
        patches = wn.reshape(NP, PL, C).transpose(2, 0, 1).astype(np.float32)  # (C,NP,PL)
        return (
            torch.from_numpy(patches),       # (C, NP, PL)  -- RAW PATCHES
            torch.from_numpy(sfeat),         # (Sf,)
            torch.from_numpy(tafeat),        # (Gf,)
        )

def _discover_pids(proc_dir: Path) -> list[str]:
    manifest = proc_dir / "manifest.csv"
    if manifest.exists():
        man = pd.read_csv(manifest)
        cand_cols = [c for c in man.columns if c.lower() in ("pid","subject","id","participant")]
        if cand_cols:
            pids = man[cand_cols[0]].astype(str).str.strip().tolist()
        elif "outfile" in man.columns:
            pids = man["outfile"].astype(str).apply(lambda s: Path(s).stem).tolist()
        else:
            file_cols = [c for c in man.columns if ("file" in c.lower()) or ("path" in c.lower())]
            if not file_cols:
                raise ValueError(f"manifest.csv has no subject or path-like columns: {list(man.columns)}")
            pids = man[file_cols[0]].astype(str).apply(lambda s: Path(s).stem).tolist()
        pids = sorted({pid for pid in pids if pid})
    else:
        pids = sorted([p.stem for p in proc_dir.glob("*.npz")])
    return pids

def _seed_worker(worker_id):
    seed = PT.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def pt_make_loader(pid_list, batch_size=32, shuffle=False):
    ds = PT_SSL_Dataset(pid_list, PT.PROC_DIR)
    g = torch.Generator()
    g.manual_seed(PT.SEED)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=2,
                    pin_memory=(device.type=="cuda"),
                    worker_init_fn=_seed_worker,
                    generator=g)
    return ds, dl

# ============================================
# [G] MODEL (RoPE Encoder + SSL MPR Head)
# ============================================
class RoPETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with RoPE in MHSA (Pre-LN).
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "per-head dim must be even for RoPE"

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2*d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model)
        )

    def forward(self, x, freqs_cis):
        B, N, D = x.shape; H = self.n_heads; d = self.head_dim

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, H, d).permute(0, 2, 1, 3, 4)  # (B,3,N,H,d)
        q = qkv[:, 0].transpose(1, 2)  # (B,H,N,d)
        k = qkv[:, 1].transpose(1, 2)
        v = qkv[:, 2].transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, freqs_cis[:N])

        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.drop(self.proj(out))
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x

class RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=8, dropout=0.3,
                 n_tokens=12, max_tokens=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        head_dim = d_model // n_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, max_tokens))

    def forward(self, x):
        freqs = self.freqs_cis[: x.size(1)]
        for layer in self.layers:
            x = layer(x, freqs)
        return x

class PT_SSL_PatchEmbedding(nn.Module):
    """
    Build input token sequence:
      • patch tokens: projection of RAW patches (C×PL) -> D (+dropout)
      • +1 stats token  (tiny dropout p=0.05)
      • +1 topo token   (tiny dropout p=0.05)
    Returns tokens and RAW patch targets (for reconstruction).
    """
    def __init__(self, cfg: PretrainCfg, d_model=256, stats_topo_dropout=0.05, patch_dropout=0.10):
        super().__init__()
        self.n_patches  = cfg.N_PATCHES
        self.mask_ratio = cfg.MASK_RATIO
        self.C = cfg.CHANNELS
        self.PL = cfg.PATCH_LEN

        # projections
        self.patch_proj = nn.Sequential(
            nn.Linear(cfg.CHANNELS * cfg.PATCH_LEN, d_model),
            nn.Dropout(patch_dropout)
        )
        self.stat_proj  = nn.Linear(cfg.N_STAT_FEATURES, d_model)
        self.topo_proj  = nn.Linear(cfg.N_TOPO_FEATURES, d_model)

        self.stat_drop  = nn.Dropout(stats_topo_dropout)
        self.topo_drop  = nn.Dropout(stats_topo_dropout)

        self.norm       = nn.LayerNorm(d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))

    def forward_tokens_and_targets(self, patches, stats, topo, mask_idx=None):
        """
        patches: (B, C, NP, PL)  [RAW]
        stats:   (B, Sf)
        topo:    (B, Gf)
        Returns:
          x_tokens     : (B, NP+2, D)
          patch_targets: (B, NP, C*PL)   (RAW targets; stopgrad)
          mask_idx     : (B, NP) bool
        """
        B, C, NP, PL = patches.shape
        assert C == self.C and PL == self.PL and NP == self.n_patches, "Patch shape mismatch"
        flat_raw = patches.permute(0,2,1,3).reshape(B, NP, C*PL)      # (B,NP,C*PL)
        patch_targets = flat_raw.detach()                             # RAW targets

        patch_vec = self.patch_proj(flat_raw)                         # (B,NP,D)

        # stats & topo tokens with tiny dropout
        s = self.stat_drop(self.stat_proj(stats)).unsqueeze(1)        # (B,1,D)
        g = self.topo_drop(self.topo_proj(topo)).unsqueeze(1)         # (B,1,D)

        # patch masking
        if mask_idx is None:
            num_mask = max(1, int(self.mask_ratio * NP))
            mask_idx = torch.zeros(B, NP, dtype=torch.bool, device=patch_vec.device)
            for b in range(B):
                mpos = torch.randperm(NP, device=patch_vec.device)[:num_mask]
                mask_idx[b, mpos] = True

        masked_patch = patch_vec.clone()
        num_masked_total = int(mask_idx.sum().item())
        if num_masked_total > 0:
            mask_tok = self.mask_token.to(dtype=masked_patch.dtype, device=masked_patch.device)
            mask_tok = mask_tok.expand(num_masked_total, -1)
            masked_patch[mask_idx] = mask_tok

        x_tokens = torch.cat([masked_patch, s, g], dim=1)             # (B, NP+2, D)
        x_tokens = self.norm(x_tokens)
        return x_tokens, patch_targets, mask_idx

class PT_SSL_Model(nn.Module):
    """
    SSL model: embedding -> RoPE Transformer -> MPR decoder to RAW patches.
    """
    def __init__(self, cfg: PretrainCfg):
        super().__init__()
        D = cfg.D_MODEL
        self.C = cfg.CHANNELS
        self.PL = cfg.PATCH_LEN
        self.embed    = PT_SSL_PatchEmbedding(cfg, d_model=D, stats_topo_dropout=0.05, patch_dropout=0.10)
        self.backbone = RoPETransformerEncoder(D, cfg.N_HEADS, cfg.N_LAYERS, cfg.DROPOUT,
                                               n_tokens=cfg.N_PATCHES+2, max_tokens=2048)
        # MPR head predicts RAW patch (C*PL)
        self.mpr_head = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, self.C * self.PL)
        )

    def forward_mpr(self, base_patches, sfeat, tafeat):
        """
        Compute masked-patch reconstruction loss on masked positions only (RAW targets).
        """
        x_tokens, patch_targets, mask_idx = self.embed.forward_tokens_and_targets(
            base_patches, sfeat, tafeat, mask_idx=None
        )
        x = self.backbone(x_tokens)                  # (B, NP+2, D)
        B, NP = mask_idx.shape
        token_out = x[:, :NP, :]                     # (B,NP,D)
        pred_raw  = self.mpr_head(token_out)         # (B,NP,C*PL)

        mask_flat = mask_idx.view(B, NP, 1).expand_as(pred_raw)
        mpr_loss = F.mse_loss(pred_raw[mask_flat], patch_targets[mask_flat]) if mask_flat.any() \
                   else torch.tensor(0.0, device=x.device)
        return {"mpr": mpr_loss, "total": mpr_loss, "mask_cov": float(mask_idx.float().mean().item())}

# ============================================
# [H] TRAINING UTILITIES (LR SCHEDULE, LOOP)
# ============================================
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(1, int(total_steps))
        self.min_lr = min_lr
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]['lr']
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            t = (self.step_num - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

def pt_ssl_train(model: PT_SSL_Model, train_loader, val_loader,
                 epochs=PT.EPOCHS, patience=PT.EARLY_STOP_PATIENCE):
    """
    Training loop with AMP, grad clipping, warmup-cosine (optional),
    early stopping on validation total loss, plus LR & mask coverage logs.
    """
    optimizer = optim.AdamW(model.parameters(), lr=PT.LR, weight_decay=PT.WEIGHT_DECAY)
    use_fp16 = _amp_uses_fp16()
    scaler = GradScaler(enabled=(device.type=="cuda" and use_fp16))

    use_warmup = PT.WARMUP_FRAC > 0.0
    if use_warmup:
        total_steps = epochs * max(1, len(train_loader))
        warmup_steps = int(PT.WARMUP_FRAC * total_steps)
        sched = WarmupCosine(optimizer, warmup_steps, total_steps, min_lr=PT.LR*0.05, base_lr=PT.LR)
    else:
        torch_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None
    patience_ctr = 0

    for ep in range(1, epochs+1):
        model.train()
        loss_meter = 0.0
        mpr_meter = 0.0
        mask_cov_meter = 0.0
        steps = 0

        for base_p, sfeat, tafeat in train_loader:
            base_p = base_p.to(device, non_blocking=True)
            sfeat  = sfeat.to(device, non_blocking=True)
            tafeat = tafeat.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                out = model.forward_mpr(base_p, sfeat, tafeat)
                loss = out["total"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), PT.MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), PT.MAX_GRAD_NORM)
                optimizer.step()

            cur_lr = sched.step() if use_warmup else optimizer.param_groups[0]['lr']

            loss_meter  += float(loss.item())
            mpr_meter   += float(out["mpr"].item())
            mask_cov_meter += float(out["mask_cov"])
            steps += 1

        if not use_warmup:
            torch_sched.step()

        n_batches = max(1, len(train_loader))
        print(f"[SSL-MPR][Train] Ep{ep:03d}/{epochs} | total {loss_meter/n_batches:.6f} | mpr {mpr_meter/n_batches:.6f} | "
              f"mask {mask_cov_meter/n_batches:.2%} | lr {cur_lr:.2e}")

        # ---- Validation ----
        model.eval()
        val_tot = 0.0
        with torch.no_grad():
            for base_p, sfeat, tafeat in val_loader:
                base_p = base_p.to(device, non_blocking=True)
                sfeat  = sfeat.to(device, non_blocking=True)
                tafeat = tafeat.to(device, non_blocking=True)
                with amp_autocast():
                    losses = model.forward_mpr(base_p, sfeat, tafeat)
                    val_tot += float(loses := losses["total"].item())

        val_avg = val_tot / max(1, len(val_loader))
        print(f"[SSL-MPR][Val]   Ep{ep:03d}/{epochs} | total {val_avg:.6f}")

        # early stopping
        if val_avg < best_loss - 1e-6:
            best_loss = val_avg
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("  ⏹️ SSL-MPR Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ============================================
# [I] DRIVER: DISCOVER SUBJECTS, TRAIN, SAVE
# ============================================
def phase_a_pretrain_ssl():
    pids = _discover_pids(PT.PROC_DIR)
    if len(pids) == 0:
        raise FileNotFoundError(f"No npz files found in {PT.PROC_DIR}")

    # subject-wise split: ~20% or VAL_SUBJECTS (whichever smaller)
    val_n_heur = max(1, len(pids)//5)
    val_n = min(PT.VAL_SUBJECTS, val_n_heur)
    train_pids = pids[:-val_n] if len(pids) > val_n else pids[:1]
    val_pids   = pids[-val_n:] if len(pids) > 1 else pids[:1]
    print(f"🩺 [SSL] Subjects: {len(pids)} | train={len(train_pids)} | val={len(val_pids)}")
    print(f"      train_pids[:5]: {train_pids[:5]}")
    print(f"      val_pids: {val_pids}")

    # DataLoaders
    _, train_loader = pt_make_loader(train_pids, batch_size=PT.BATCH_SIZE, shuffle=True)
    _, val_loader   = pt_make_loader(val_pids,   batch_size=PT.BATCH_SIZE, shuffle=False)

    # Model + Train
    model = PT_SSL_Model(PT).to(device)
    model = pt_ssl_train(model, train_loader, val_loader,
                         epochs=PT.EPOCHS, patience=PT.EARLY_STOP_PATIENCE)

    # Save best weights
    PT.OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(model.state_dict(), PT.CKPT_PATH)
        with open(PT.OUT_DIR / "pretrain_meta.json", "w") as f:
            json.dump({"best_val_total": "see logs", "cfg": PT.__dict__}, f, indent=2, default=str)
        print(f"✅ [SSL-MPR] Saved pretrained backbone to: {PT.CKPT_PATH}")
    except Exception as e:
        print(f"⚠️ [SSL-MPR] Could not save to disk ({e}); proceeding with in-memory weights only.")

    return copy.deepcopy(model.state_dict())

# ============================================
# [J] ENTRY POINT
# ============================================
if __name__ == "__main__":
    _ = phase_a_pretrain_ssl()
