# PatchTST-HAR + Enhanced Metrics & Computational Profiling
Human Activity Recognition implementation combining PatchTST with RoPE, statistical features, topological features, and comprehensive evaluation metrics with computational profiling.


## Overview
This implementation provides:
- PatchTST architecture with Rotary Positional Embeddings (RoPE)
- Statistical feature extraction (56 features)
- Topological feature extraction using persistent homology (24 features)
- Comprehensive evaluation metrics including Cohen's Kappa, MCC, F2-score
- HMM post-processing for temporal smoothing
- Computational profiling and real-time feasibility analysis


## Configuration
Key parameters defined in the Config class:
```python
# Data Configuration
WINDOW_SIZE = 1000      # Window size in samples
PATCH_LEN = 100         # Patch length
N_PATCHES = 10          # Number of patches per window
CHANNELS = 3            # Number of sensor channels
# Model Configuration  
D_MODEL = 56            # Model embedding dimension
N_HEADS = 2             # Number of attention heads
N_LAYERS = 2            # Number of transformer layers
DROPOUT = 0.3           # Dropout rate
# Feature Configuration
N_STAT_FEATURES = 56    # Number of statistical features
N_TOPO_FEATURES = 24    # Number of topological features
TAKENS_M = 3            # Takens embedding dimension
TAKENS_TAU = 5          # Takens embedding delay
# Training Configuration
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4               # Learning rate
WEIGHT_DECAY = 1e-4     # Weight decay
```


## Architecture
### Model Components
1. **TemporalEmbedding**: Projects time features to embeddings
2. **PatchEmbedding**: Processes time series patches and fuses with features
3. **RoPETransformerEncoderLayer**: Transformer layer with rotary positional embeddings
4. **RoPETransformerEncoder**: Multi-layer transformer encoder
5. **PatchTSTClassifier**: Complete classification model
### Feature Extraction


#### Statistical Features (56 features)
- Basic statistics (mean, std, range) for each channel and magnitude
- Cross-correlations between channels
- Quartiles and percentiles
- Spectral analysis (dominant frequencies, spectral entropy)
- Peak detection and prominences
- Gravity vector angles
- Autocorrelation


#### Topological Features (24 features)
- Persistent homology using Takens embedding
- Persistence entropy
- Top-k lifetimes
- Birth/death statistics
- Quantile-based features
- Fallback computation when ripser unavailable


#### Time Features (5 features)
- Hour of day (normalized)
- Minute of hour (normalized)  
- Weekday (normalized)
- Weekend indicator
- Time of day category
## Dataset Requirements
Expected data structure:
```
processed_minimal/
├── classes.json              # Class labels
├── label_encoder.json        # Label encoding mapping
├── manifest.csv             # Data manifest
├── participant_001.npz      # Individual participant data
├── participant_002.npz      # Individual participant data
└── ...
```
Each `.npz` file contains:
- `windows`: numpy array of shape (N_windows, 1000, 3)
- `labels_str`: numpy array of activity labels
- `first_ts_epoch_ns`: numpy array of timestamps
## Evaluation Metrics


### Standard Metrics
- Accuracy
- Balanced Accuracy  
- F1-score (macro and weighted)
- Precision (macro)
- Recall (macro)


### Advanced Metrics
- Cohen's Kappa (κ)
- Matthews Correlation (MCC)
- Pearson-Yule Phi coefficient
- F2 Score (recall-weighted)


### Sequence-Level Metrics
- Transition accuracy (accuracy on activity changes)
- Per-participant analysis


## Computational Profiling
The implementation includes comprehensive profiling:


### Topological Feature Profiling
- Timing breakdown for preprocessing, embedding, computation, and feature extraction
- Statistical analysis (mean, std, min, max, percentiles)
- Configurable number of samples and warmup iterations


### Model Complexity Analysis
- Total parameters count
- Trainable parameters count
- FLOPs estimation
- Model size calculation


### End-to-End Inference Profiling
- Batch inference timing
- Per-sample inference timing
- Throughput calculation (samples/second)


### Real-Time Feasibility Analysis
- Comparison of total latency vs. window duration
- P95 latency analysis
- Real-time capability assessment


## HMM Post-Processing
Hidden Markov Model implementation for temporal smoothing:
- Estimates transition matrix and initial probabilities from training data
- Viterbi decoding for sequence prediction
- Configurable smoothing parameters
- Per-participant sequence processing


## Usage
Run the main implementation:
```python
python patchtst_har.py
```
The script will:
1. Load and split participant data
2. Profile computational costs
3. Train the model with early stopping
4. Evaluate with comprehensive metrics
5. Apply HMM post-processing
6. Generate visualizations and reports
7. Save all results and artifacts


## Output Files
The implementation generates:
- `computational_profiling.json` - Profiling results
- `metrics_raw_comprehensive.json` - Raw model metrics
- `metrics_hmm_comprehensive.json` - HMM post-processed metrics  
- `comprehensive_artifacts.pth` - Complete model and results bundle
- `EVALUATION_SUMMARY.txt` - Human-readable summary report
- `cm_raw_comprehensive.png` - Raw model confusion matrix
- `cm_hmm_comprehensive.png` - HMM confusion matrix
- `metrics_comparison.png` - Metrics comparison visualization
- `per_class_f1_comparison.png` - Per-class F1 comparison
- `computational_cost_breakdown.png` - Cost analysis visualization


## Dependencies
Required packages:
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- ripser (optional, for topological features)
- scipy (optional, for advanced signal processing)


## Reproducibility
The implementation sets random seeds for:
- Python random
- NumPy
- PyTorch (CPU and GPU)
- CUDA deterministic mode
- Disabled benchmark mode


## Key Features
- AMP (Automatic Mixed Precision) support for GPU training
- Gradient clipping for training stability
- Class weight computation for imbalanced datasets
- Early stopping with configurable patience
- Comprehensive timing and memory profiling
- Multiple visualization outputs
- Structured JSON output for all metrics
- Human-readable summary reports
