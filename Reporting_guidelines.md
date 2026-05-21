# PatchHAR Reporting & Visualization Style Specification

This document defines the exact reporting style used in the PatchHAR v7 codebase.
Any LLM or developer reproducing baselines should follow this spec to ensure visual
consistency across all datasets for publication.

---

## 1. Global Figure Settings

### Output Format
- **DPI:** 300
- **Formats:** PDF + PNG (both saved for every figure)
- **Background:** White (`facecolor="white"`)
- **Bounding box:** `bbox_inches="tight"` on all saves

### Font
| Element | Font Family | Size |
|---|---|---|
| Base / body | `serif` → Times New Roman → DejaVu Serif (fallback) | 9 pt |
| Axis labels | serif | 9 pt |
| Axis titles | serif | 10 pt |
| Tick labels | serif | 8 pt |
| Legend | serif | 8 pt |

### Line & Spine Styling
| Property | Value |
|---|---|
| Axis line width | 0.8 |
| Grid line width | 0.4 |
| Plot line width | 1.4 (default) |
| Top spine | **Hidden** |
| Right spine | **Hidden** |
| PDF/PS fonttype | 42 (embed fonts for camera-ready) |

---

## 2. Color Palette (Wong Color-Blind Safe)

Use **only** these 8 colors. Never use default matplotlib colors.

| Name | Hex | Primary use |
|---|---|---|
| `black` | `#000000` | |
| `orange` | `#E69F00` | val κ curves, bar accents |
| `skyblue` | `#56B4E9` | class color |
| `green` | `#009E73` | val F1 curves |
| `yellow` | `#F0E442` | class color |
| `blue` | `#0072B2` | train loss, precision bars, main bars |
| `red` | `#D55E00` | val loss, best-epoch lines, annotation |
| `pink` | `#CC79A7` | class color |

### Class Color Assignment (fixed order, cycle if >8 classes)
```
[blue, red, green, orange, pink, skyblue, yellow, black]
[#0072B2, #D55E00, #009E73, #E69F00, #CC79A7, #56B4E9, #F0E442, #000000]
```

---

## 3. Figures to Produce (Full Suite)

Every dataset / experiment must produce **all** of the following figures.
Per-fold versions are saved inside `outputs/<dataset>/subj_<ID>/`.
Pooled versions are saved in `outputs/<dataset>/`.

---

### fig01 — Architecture Diagram
- **Size:** `(14, 4.2)` inches
- **Style:** `FancyBboxPatch` with `round,pad=0.02,rounding_size=0.8`, `lw=1.0`, edge `#333`
- **Arrows:** `arrowstyle="->"`, `lw=0.9`, color `#333`
- **Title font:** 8.5 pt, pad=8
- **Content to show:**
  - Input block (dataset name, T, C, instance norm + windowing)
  - Feature extraction block (sensor locations, feature dimensionality)
  - Patch embedding block (conv streams, stat gate, token count)
  - Encoder block (attention type, heads Q/KV, FFN type, MoE config, norm, early exit)
  - Classification head (pooling → dropout → linear → GELU → linear)
  - Post-processing (temperature scaling, Viterbi HMM, predictions)

---

### fig02 — Training Curves (3-panel)
- **Size:** `(9.5, 2.8)` inches, 1 row × 3 columns, `sharex=True`
- **Panel (a): Loss**
  - Raw train loss: `blue`, `lw=0.8`, `alpha=0.3`
  - EMA-smoothed train loss (α=0.15): `blue`, `lw=1.6`
  - Val loss: `red`, `lw=1.4`, marker `"o"`, markersize 2.5
  - Best epoch vertical line: `red`, `lw=1.0`, `ls="--"`, `alpha=0.7`
- **Panel (b): Val Macro-F1**
  - Color: `green`, marker `"o"`, markersize 3
  - Peak horizontal dotted line: `red`, `lw=0.6`, `ls=":"`
  - Peak annotated as `max=X.XXX` at right edge, fontsize 7.5
- **Panel (c): Val Cohen's κ**
  - Color: `orange`, marker `"s"`, markersize 3
  - Same peak annotation style as (b)
- **Grid:** `alpha=0.3` on all panels
- **Legend:** `frameon=False`, `loc="upper right"` (a), `loc="lower right"` (b, c)
- **Saved per fold** with fold label in filename

---

### fig03 — Confusion Matrix
- **Size:** `(max(4.5, K×0.95), max(4.0, K×0.85))` — scales with number of classes K
- **Colormap:** `"Blues"`, `vmin=0`, `vmax=100` (percentage)
- **Values displayed:** raw count on top line + `(X.X%)` on second line, fontsize 7.5
- **Text color:** white if cell value > 55%, else `#222`
- **Cell lines:** `linewidths=0.6`, `linecolor="white"`, `square=True`
- **Colorbar:** `shrink=0.7`, label `"%"`
- **X-axis:** "Predicted", rotated 30°; **Y-axis:** "True"
- **Title:** includes macro-F1 and weighted-F1
- **Saved:** per fold + pooled

---

### fig04 — Per-Class Metrics (Precision / Recall / F1) with Bootstrap CI
- **Size:** `(max(6, K×1.0), 3.6)` inches
- **Bar width:** 0.26 (3 grouped bars per class)
- **Colors:** Precision=`blue`, Recall=`orange`, F1=`green`, all `alpha=0.9`, `edgecolor="white"`, `lw=0.8`
- **Error bars (F1 only):** 95% bootstrap CI, n=1000 resamples, `ecolor="#333"`, `elinewidth=0.9`, `capsize=2.5`
- **Support annotation:** shown as `n=XXX` inside each group at y=0.02, fontsize 6.8, white text on dark `#333` background box (`alpha=0.75`, `boxstyle="round,pad=0.2"`)
- **Mean macro-F1 line:** `#333`, `ls="--"`, `lw=0.8`
- **Y-axis:** 0 to 1.12; `x-tick` labels rotated 20°, `ha="right"`
- **Legend:** 4 columns, `loc="upper center"`, `bbox_to_anchor=(0.5, 1.08)`, `frameon=False`
- **Subplot adjust:** `bottom=0.22`, `top=0.80`
- **Saved:** per fold + pooled

---

### fig06 — CGA Attention Heatmap (per class, last layer)
- **Layout:** up to 6 columns, rows as needed; `(3.2×ncols, 3.0×nrows)` inches
- **Source:** Last CGA encoder layer, mean over heads → patch×patch block only (stat token row/col excluded)
- **Colormap:** `"viridis"`, `aspect="equal"`, `vmin=0`, `vmax=mean_attn.max()`
- **Colorbar:** `shrink=0.75`
- **Subplot title:** `"{class} (n=XX)"`, fontsize 8
- **Super title:** `"Avg last-layer CGA attention (patch block, head-averaged)"`, fontsize 10
- **Axis labels:** "Key patch" (x), "Query patch" (y, leftmost column only)
- **Saved:** per fold

---

### fig08 — Per-Subject F1 Bar Chart
- **Size:** `(max(5, N_subjects×0.7), 4.0)` inches
- **Bars:** `blue`, `alpha=0.85`, `edgecolor="white"`, `lw=0.8`
- **Value labels:** F1 to 3 decimal places, placed above each bar
- **Mean line:** `red`, `lw=1.4`, `ls="--"`, labeled `Mean=X.XXX`
- **Median line:** `orange`, `lw=1.0`, `ls=":"`, labeled `Median=X.XXX`
- **Shaded band:** ±1 std around mean, `red`, `alpha=0.10`
- **Title:** includes IQR `[Q25, Q75]`
- **Y-axis:** 0 to 1.12
- **X-axis labels:** rotated 30°, `ha="right"`
- **Grid:** `axis="y"`, `alpha=0.3`
- **Also exports:** `.csv` file with per-subject F1 values
- **Saved:** pooled only

---

### fig09 — Reliability Diagram + Confidence Histogram (2-panel)
- **Size:** `(8, 3.4)` inches, 1×2 panels
- **Bins:** 15 equal-width bins over [0, 1]
- **Panel (a): Reliability diagram**
  - Perfect calibration diagonal: `#555`, `ls="--"`, `lw=0.9`
  - Observed accuracy bars: `blue`, `alpha=0.85`, `edgecolor="white"`, `lw=0.6`
  - Gap bars (over/under confidence): `red`, `alpha=0.35`, `edgecolor=red`, `lw=0.6`
  - Title includes ECE value: `"Reliability diagram  (ECE=X.XXX)"`
  - Legend: `loc="upper left"`, `frameon=False`
- **Panel (b): Confidence histogram**
  - Bars: `green`, `alpha=0.85`, `edgecolor="white"`, `lw=0.6`
  - Title: `"Confidence histogram"`
- **Grid:** `alpha=0.3` both panels
- **Returns ECE value** (used in compute summary)
- **Saved:** per fold + pooled

---

### fig10 — Embedding UMAP / t-SNE
- **Size:** `(5.5, 5.0)` inches
- **Max points:** 3000 (random subsample if larger)
- **Method priority:** UMAP if installed → fallback to t-SNE
  - UMAP: `n_neighbors=30`, `min_dist=0.1`, `metric="cosine"`, `random_state=42`
  - t-SNE: `perplexity=30`, `init="pca"`, `learning_rate="auto"`, `random_state=42`
- **Scatter:** `s=12`, `alpha=0.6`, `edgecolors="none"`, colors from class color list
- **Legend:** label format `"{class} (n=XX)"`, `frameon=False`, `markerscale=1.4`, fontsize 7.5
- **Axes:** ticks hidden (`set_xticks([])`, `set_yticks([])`)
- **Title:** includes method name and dataset/eval info
- **Source:** mean-pool of final encoder output (all tokens averaged), NOT logits
- **Saved:** per fold + pooled

---

### fig11 — Compute Summary Table
- **Size:** `(9.0, max(5.0, N_rows×0.28 + 1.0))` inches
- **Style:** `matplotlib.table`, `cellLoc="left"`, `colWidths=[0.22, 0.52, 0.26]`
- **Font size:** 8.5 pt, `scale(1, 1.25)` row height
- **Header row:** `blue` background, white bold text
- **Group header cells:** `#E5EDF5` background, bold text
- **Even data rows:** `#F7F9FB` background
- **Cell edges:** `#CCCCCC`
- **Columns:** Group | Metric | Value
- **Groups to include (in order):**
  1. Architecture (model name, params, channels, patches, patch_len, d_model, heads, layers, dropout, drop_path, MoE config, window, stride, stat features)
  2. LOGO Setup (evaluation type, n_subjects, val_strategy, maj_threshold)
  3. Training (epochs_trained_mean, best_epoch_mean, best_val_f1_mean, total_training_time_s, warmup_epochs, focal_gamma, use_ema, use_temp_scaling, use_viterbi_hmm)
  4. LOGO Results – Raw (F1 mean±std, κ mean±std, MCC mean±std)
  5. LOGO Results – HMM (F1 mean±std, κ mean±std, MCC mean±std, ECE mean)
  6. Hardware (device, GPU name, batch_size, peak_memory_MB, reserved_MB)

---

### fig_logo_summary — Per-Fold LOGO Bar Chart (F1 + κ side by side)
- **Size:** `(max(9, N_subjects×0.9), 3.6)` inches, 1×2 panels
- **Bars:** `blue` (F1), `orange` (κ), `alpha=0.85`, `edgecolor="white"`, `lw=0.8`
- **Mean line:** `red`, `lw=1.4`, `ls="--"`, labeled `Mean=X.XXX ± X.XXX`
- **Shaded band:** ±1 std, `red`, `alpha=0.10`
- **Value labels:** 3 decimal places above each bar
- **Y-axis:** 0 to `min(1.08, max_val×1.12 + 0.05)`
- **X-axis:** subject IDs, rotated 30°, `ha="right"`
- **Grid:** `axis="y"`, `alpha=0.3`
- **Also exports:** `logo_fold_metrics.csv` with per-fold and mean/std rows
- **Saved:** pooled only

---

### LaTeX Table Export
- Standard `\begin{table}` with `\toprule`, `\midrule`, `\bottomrule` (booktabs style)
- Caption format: `"PatchHAR v7 <DATASET> LOGO-CV results."`
- Label format: `"tab:patchhar_<dataset>_logo"`
- Two-column format: Metric | Value
- Saved as: `latex_compute_table.tex`

---

## 4. Metrics to Report (in every experiment)

### Per-fold (saved to `fold_metrics.json`)
| Metric | Description |
|---|---|
| `test_macro_f1` | Macro-averaged F1 (HMM predictions) |
| `test_weighted_f1` | Weighted F1 (HMM predictions) |
| `test_kappa` | Cohen's κ (HMM predictions) |
| `test_mcc` | Multiclass MCC (HMM predictions) |
| `raw_macro_f1` | Macro-F1 before HMM smoothing |
| `ece` | Expected Calibration Error |
| `best_epoch` | Epoch of best val F1 |
| `best_val_f1` | Best validation macro-F1 |
| `epochs_trained` | Total epochs run (may be < max due to early stop) |
| `training_time_s` | Wall-clock training time for this fold |
| `hmm_best_lam` | Best Viterbi change-penalty λ |
| `n_train/val/test_windows` | Window counts per split |
| `n_trainable_parameters` | Model parameter count |

### Summary (reported in paper / compute profile)
| Metric | Format |
|---|---|
| LOGO macro-F1 (HMM) | mean ± std across folds |
| LOGO weighted-F1 (HMM) | mean ± std |
| LOGO Cohen's κ (HMM) | mean ± std |
| LOGO MCC (HMM) | mean ± std |
| LOGO macro-F1 (raw, no HMM) | mean ± std |
| Pooled macro-F1 | single value over all test windows |
| Pooled Cohen's κ | single value |
| ECE (pooled) | single value |
| GPU forward latency | mean ± std ms, P50, P95 |
| Batch throughput | samples/sec |
| GPU peak memory | MB |
| Total training wall-clock | seconds |

### Printed to console (per fold)
- Epoch log: `Ep XXX/XXX | loss X.XXXX | val_loss X.XXXX | F1 X.XXXX | κ X.XXXX | lr X.XXe-XX`
- Raw vs HMM results: `Raw: F1=X.XXXX κ=X.XXXX | HMM: F1=X.XXXX κ=X.XXXX`
- Full `sklearn classification_report` (precision, recall, F1, support per class)

---

## 5. Directory Structure

```
outputs/
└── <dataset_name>/
    ├── fig01_architecture.{pdf,png}
    ├── fig03_confusion_matrix_pooled.{pdf,png}
    ├── fig04_per_class_metrics_pooled.{pdf,png}
    ├── fig08_per_subject_f1_pooled.{pdf,png}
    ├── fig09_reliability_pooled.{pdf,png}
    ├── fig10_embedding_umap_pooled.{pdf,png}
    ├── fig11_compute_summary.{pdf,png}
    ├── fig_logo_summary.{pdf,png}
    ├── logo_summary.json
    ├── logo_fold_metrics.csv
    ├── compute_profile.json
    ├── latex_compute_table.tex
    └── subj_<ID>/                        ← one folder per fold
        ├── fig02_training_curves_subj_<ID>.{pdf,png}
        ├── fig03_confusion_matrix_subj_<ID>.{pdf,png}
        ├── fig04_per_class_metrics_subj_<ID>.{pdf,png}
        ├── fig06_attention_heatmap_subj_<ID>.{pdf,png}
        ├── fig09_reliability_subj_<ID>.{pdf,png}
        ├── fig10_embedding_umap_subj_<ID>.{pdf,png}
        └── fold_metrics.json
```

---

## 6. Quick-Reference LLM Prompt Snippet

When asking an LLM to implement reporting for a new baseline, include this:

> Reproduce the following exact reporting style:
> - Font: Times New Roman serif, 9pt body, 10pt title, 8pt ticks/legend
> - Top and right spines hidden on all axes
> - Colors: Wong palette only — blue=#0072B2, red=#D55E00, green=#009E73, orange=#E69F00, pink=#CC79A7, skyblue=#56B4E9, yellow=#F0E442, black=#000000
> - Save every figure as both PDF and PNG at 300 DPI with bbox_inches="tight"
> - Produce figures: fig01 architecture, fig02 training curves (3-panel), fig03 confusion matrix (% + count), fig04 per-class metrics with 95% bootstrap CI, fig06 attention heatmap, fig08 per-subject F1, fig09 reliability diagram + confidence histogram, fig10 UMAP/t-SNE embeddings, fig11 compute summary table, fig_logo_summary
> - Report metrics: macro-F1, weighted-F1, Cohen's κ, MCC, ECE, raw vs HMM, per-fold mean±std, pooled, GPU latency
> - Use LOGO cross-validation: one subject test, next subject (cyclically) val, rest train
> - Export latex_compute_table.tex and logo_fold_metrics.csv alongside all figures
