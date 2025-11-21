# 🧠 PatchHAR: Patch-Transformer with Statistical & Topological Embeddings for Wearable Human Activity Recognition

PatchHAR is a compact, interpretable, and high-performance Transformer framework for **sensor-based Human Activity Recognition (HAR)**.  
It extends the idea of patching from PatchTST but introduces a completely new modeling pipeline tailored for wearable accelerometer data.

This repository contains the full implementation of PatchHAR, including:
- Multi-axis patch embedding
- Rotary positional embeddings (RoPE)
- Statistical token (56-D)
- Topological token (24-D via persistent homology)
- Self-supervised masked patch reconstruction
- Transformer encoder
- HMM temporal smoothing
- Comprehensive evaluation and computational profiling

---

# 🌟 Why PatchHAR?

Wearable HAR requires models that can:
1. Capture **local temporal dynamics**  
2. Preserve **cross-axis motion coupling** (x–y–z accelerometer)
3. Recognize **global structure** of motion (periodicity, fragmentation, posture)
4. Adapt across subjects and datasets  
5. Remain compact and efficient  

PatchHAR addresses all of these needs by combining:
- **Patch-level temporal modeling**
- **Statistical summarization of each 10-s window**
- **Topological descriptors of the motion trajectory**
- **RoPE attention** for improved temporal reasoning
- **Self-supervised pretraining** for generalization

---

# 📌 What’s New in PatchHAR?

PatchHAR introduces two global tokens that capture structure beyond raw time series:

### **1. 56-D Statistical Token**
Includes:
- Per-axis and magnitude statistics  
- Skewness, kurtosis  
- Cross-axis correlations  
- Quartiles  
- Spectral entropy and dominant frequencies  
- Peak morphology  
- Gravity & dynamic orientation angles  

### **2. 24-D Topological Token**
Built from:
- Takens embeddings of motion magnitude  
- Vietoris–Rips persistent homology  
- Persistence entropy  
- Lifetimes (birth–death)  
- Topological complexity signatures  

These two tokens summarize:
- **Intensity**
- **Periodicity**
- **Geometric structure**
- **Fragmentation**
- **Postural characteristics**

and complement the patch embeddings beautifully.

---

# 🔍 How PatchHAR Differs from PatchTST and CT-PatchTST

PatchHAR only inherits the *idea of patching*.  
Everything else is different.

## **PatchTST**
- Processes **each channel independently** (x, y, z)
- Channel-wise patches  
- Additive positional embeddings  
- No early cross-channel interaction  
- Designed for forecasting, not HAR  

## **CT-PatchTST**
- Also begins with **channel-wise patching**
- Adds channel attention + time attention
- Still relies on separated channel token streams

---

# 🚀 PatchHAR (Ours)

**Our pipeline is fundamentally different:**

### ✔ Joint Multi-Axis Windowing  
We take the full window:  
→ **Cross-axis coupling is preserved from the very beginning.**

### ✔ Linear Projection Before Attention  
Each patch is:
1. Flattened  
2. Linearly projected to a 56-D embedding  

No attention is applied yet.

This is **NOT** how PatchTST or CT-PatchTST works.

### ✔ Attention Across Patches (Not Channels)  
Self-attention operates across:
- 10 patch tokens  
- 1 statistical token  
- 1 topological token  

### ✔ Mean pooling → Classifier head → Activity prediction

---

# 🖼 Model Overview

Below is the core PatchHAR architecture:

![Graphical Abstract](sandbox:/mnt/data/framework_har%20%282%29.pdf)

---

# 📚 Model Architecture

### **Patch Embedding**
- Converts each 100×3 patch into a 56-D vector  
- Preserves multivariate structure

### **RoPE Transformer Encoder**
- 2 layers  
- 2 heads  
- 56-D hidden size  
- Rotary positional embeddings for better temporal consistency  

### **Multi-View Token Fusion**
Tokens fed into Transformer:

| Token Type | Description |
|------------|-------------|
| Patch tokens | 10 tokens representing local temporal patterns |
| Statistical token | 56-D global statistical summary |
| Topological token | 24-D geometric summary from persistent homology |

### **Self-Supervised Pretraining**
- Masked Patch Reconstruction (MPR)
- Improves representation learning in unlabeled data

### **Fine-Tuning**
- Classification head  
- Optional HMM smoothing for sequence consistency  

---

# 📊 Evaluation Metrics

PatchHAR supports a complete evaluation suite:

### **Standard Metrics**
- Accuracy  
- Balanced accuracy  
- Macro/Weighted F1  

### **Advanced Metrics**
- Cohen’s κ  
- Matthews Correlation Coefficient  
- Pearson–Yule φ  
- F2-score  

### **Sequence Metrics**
- Transition accuracy  
- Per-participant evaluation  

---

# ⚙ Computational Profiling

We provide full profiling scripts covering:

### **Topological Feature Costs**
- Embedding latency  
- PH computation time  
- Percentile statistics  

### **Model Complexity**
- Parameter counts  
- FLOPs  
- Model size  

### **Inference Profiling**
- Per-batch latency  
- Per-sample latency  
- Throughput (samples/s)  

### **Real-Time Feasibility**
- p95 latency analysis  
- Window-duration comparison  

---

# 📦 Dataset Format

Folder structure:
processed_minimal/
├── classes.json
├── label_encoder.json
├── manifest.csv
├── participant_001.npz
└── participant_002.npz

Each `.npz` contains:

- `windows` → (N, 1000, 3)  
- `labels_str`  
- `first_ts_epoch_ns`  

---

# ▶ Usage

### **Train + Evaluate**
```bash
python main.py
What the script does:

Load and split participant data

Train PatchHAR with early stopping

Compute all metrics

Apply HMM smoothing

Generate visualizations

Save results + reports

Run computational profiling

📁 Output Files

computational_profiling.json

metrics_raw_comprehensive.json

metrics_hmm_comprehensive.json

comprehensive_artifacts.pth

EVALUATION_SUMMARY.txt

Confusion matrices (raw + HMM)

F1 comparison plots

Cost breakdown visualizations

🔧 Dependencies

PyTorch

NumPy

Pandas

Scikit-Learn

SciPy

Ripser (optional)

Matplotlib / Seaborn

🔒 Reproducibility

The implementation fixes seeds for:

Python

NumPy

PyTorch

CUDA
with deterministic settings enabled.

❤️ Citation

If you use PatchHAR, please cite our paper (coming soon).








