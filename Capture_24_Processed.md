# Capture-24 Dataset — Accelerometer Data Processing Pipeline

This document describes the full preprocessing pipeline for the [Capture-24 dataset](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001), converting raw `.csv.gz` accelerometer files from 151 participants into windowed, labeled samples ready for machine learning.

---

## Dataset Overview

| Property | Detail |
|---|---|
| Participants | 151 (P001 – P151) |
| Sampling rate | 100 Hz |
| File format | `.csv.gz` per participant |
| Columns | `time`, `x`, `y`, `z`, `annotation` |
| Label scheme used | `label:Walmsley2020` |

The `annotation` column contains raw strings such as `"7030 sleeping;MET 0.95"`. These are mapped to clean activity categories (e.g., `sleep`, `sedentary`, `light`, `moderate`) using a separate dictionary file.

**Activity classes (Walmsley2020):**
- `sleep`
- `sedentary`
- `light`
- `moderate`

---

## File Structure

```
capture24/
├── P001.csv.gz
├── P002.csv.gz
│   ...
├── P151.csv.gz
├── annotation-label-dictionary.csv
└── processed/          ← output folder (created by pipeline)
    ├── P001.npz
    ├── P002.npz
    │   ...
    └── P151.npz
```

---

## Processing Pipeline

### Step 1 — Load and Inspect

Each file is loaded with `pandas.read_csv(..., low_memory=False)` to handle mixed-type columns.

```python
df = pd.read_csv(file_path, low_memory=False)
```

---

### Step 2 — Label Extraction and Mapping

The numeric activity code is extracted from the raw annotation string using a regex, then mapped to the `Walmsley2020` label scheme via a dictionary lookup.

```python
# Extract numeric code from annotation string (e.g., "7030 sleeping;MET 0.95" → 7030)
df["ann_code"] = pd.to_numeric(
    df["annotation"].str.extract(r"(\d+)")[0],
    errors="coerce"
)
df = df.dropna(subset=["ann_code"])
df["ann_code"] = df["ann_code"].astype(int)

# Load and clean mapping file
mapping = pd.read_csv(mapping_path)
mapping["ann_code"] = pd.to_numeric(
    mapping["annotation"].str.extract(r"(\d+)")[0],
    errors="coerce"
)
mapping = mapping.dropna(subset=["ann_code"])
mapping["ann_code"] = mapping["ann_code"].astype(int)

# Build dictionary and apply
map_dict = dict(zip(mapping["ann_code"], mapping["label:Walmsley2020"]))
df["label"] = df["ann_code"].map(map_dict)
```

> **Why dictionary over merge?** Dictionary mapping avoids `dtype` mismatch errors (e.g., `float64` vs `object`) that occur with `pd.merge`.

---

### Step 3 — Drop Unlabeled Rows

Across the 151 participants, missing annotations range from ~17% to ~69% of rows. These are **intentionally unlabeled periods** (transitions, uncertain activities) — not random noise.

```python
df = df.dropna(subset=["annotation", "label"])
```

| Statistic | Value |
|---|---|
| Min missing | ~17% |
| Max missing | ~69% |
| Typical missing | ~25–40% |

> Unlabeled rows are dropped before windowing. They must not be included in supervised training.

---

### Step 4 — Segment by Label Continuity

After dropping unlabeled rows, **time gaps appear** in the data. Sliding windows must never cross these gaps or label transitions, as this would mix activities within a single window.

Continuous, single-label segments are identified by detecting label change boundaries:

```python
df["label_change"] = df["label"] != df["label"].shift()
segment_ids = df["label_change"].cumsum()
```

Each `segment_id` group is a contiguous block of identical label with no gaps.

**Segment length statistics (P001, in seconds):**

| Stat | Value |
|---|---|
| Min | ~48 s |
| 25th percentile | ~162 s |
| Median | ~568 s (~9.5 min) |
| 75th percentile | ~1185 s (~20 min) |
| Max | ~14820 s (~4.1 hrs) |

---

### Step 5 — Sliding Window

Sliding windows are applied **inside each segment** independently.

| Parameter | Value |
|---|---|
| Window size | 30 seconds = **3000 samples** |
| Stride | 15 seconds = **1500 samples** (50% overlap) |
| Min segment length | 3000 samples (shorter segments are discarded) |
| Label assignment | Segment label (uniform within segment) |

```python
WINDOW_SIZE = 3000   # 30s × 100Hz
STRIDE = 1500        # 15s × 100Hz

for start in range(0, len(seg) - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window = seg[["x", "y", "z"]].values[start:end]
    X_all.append(window)
    y_all.append(label)
```

> **Why 30 seconds?** The median segment duration is ~568 s, so 30 s windows fit comfortably within segments. This duration is also consistent with Human Activity Recognition literature (e.g., UK Biobank accelerometer studies).

---

### Step 6 — Save Output

Each participant's windows are saved as a compressed NumPy archive:

```python
np.savez_compressed(save_path, X=X_all, y=y_all)
```

**Output format per file:**

| Array | Shape | Description |
|---|---|---|
| `X` | `(num_windows, 3000, 3)` | Raw accelerometer windows (x, y, z) |
| `y` | `(num_windows,)` | Activity label strings |

---

## Full Pipeline Code

```python
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
data_folder   = r"E:\HAR_datasets_codes\capture24"
mapping_path  = r"E:\HAR_datasets_codes\capture24\annotation-label-dictionary.csv"
output_folder = r"E:\HAR_datasets_codes\capture24\processed"

os.makedirs(output_folder, exist_ok=True)

# ==============================
# PARAMETERS
# ==============================
FS          = 100
WINDOW_SEC  = 30
STRIDE_SEC  = 15
WINDOW_SIZE = WINDOW_SEC * FS   # 3000
STRIDE      = STRIDE_SEC * FS   # 1500

# ==============================
# LOAD & CLEAN MAPPING
# ==============================
mapping = pd.read_csv(mapping_path)
mapping["ann_code"] = pd.to_numeric(
    mapping["annotation"].str.extract(r"(\d+)")[0],
    errors="coerce"
)
mapping = mapping.dropna(subset=["ann_code"])
mapping["ann_code"] = mapping["ann_code"].astype(int)
mapping = mapping[["ann_code", "label:Walmsley2020"]]
map_dict = dict(zip(mapping["ann_code"], mapping["label:Walmsley2020"]))

# ==============================
# PROCESS EACH FILE
# ==============================
files = sorted(glob.glob(os.path.join(data_folder, "P*.csv.gz")))
print(f"Total files: {len(files)}")

for file in tqdm(files):
    try:
        df = pd.read_csv(file, low_memory=False)

        # Drop missing annotations
        df = df.dropna(subset=["annotation"])
        if len(df) == 0:
            continue

        # Extract annotation code
        df["ann_code"] = pd.to_numeric(
            df["annotation"].str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        df = df.dropna(subset=["ann_code"])
        df["ann_code"] = df["ann_code"].astype(int)

        # Map labels
        df["label"] = df["ann_code"].map(map_dict)
        df = df.dropna(subset=["label"])
        if len(df) == 0:
            continue

        # Segment by label continuity
        df["label_change"] = df["label"] != df["label"].shift()
        segment_ids = df["label_change"].cumsum()

        X_all, y_all = [], []

        for _, seg in df.groupby(segment_ids):
            if len(seg) < WINDOW_SIZE:
                continue

            x     = seg[["x", "y", "z"]].values
            label = seg["label"].iloc[0]

            for start in range(0, len(seg) - WINDOW_SIZE + 1, STRIDE):
                X_all.append(x[start:start + WINDOW_SIZE])
                y_all.append(label)

        if len(X_all) == 0:
            continue

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all)

        subject_id = os.path.basename(file).replace(".csv.gz", "")
        np.savez_compressed(
            os.path.join(output_folder, f"{subject_id}.npz"),
            X=X_all,
            y=y_all
        )

    except Exception as e:
        print(f"Error in {file}: {e}")

print("Processing complete.")
```

---

## Loading Processed Data

```python
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder

data_folder = r"E:\HAR_datasets_codes\capture24\processed"
files = sorted(glob.glob(f"{data_folder}/*.npz"))

X_list, y_list = [], []

for file in files:
    data = np.load(file, allow_pickle=True)
    X_list.append(data["X"])
    y_list.append(data["y"])

X = np.concatenate(X_list, axis=0)   # (total_windows, 3000, 3)
y = np.concatenate(y_list, axis=0)   # (total_windows,)

le = LabelEncoder()
y_enc = le.fit_transform(y)

print("X shape:", X.shape)
print("Classes:", le.classes_)
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Drop unlabeled rows | Intentionally unlabeled periods should not supervise training |
| Segment before windowing | Prevents windows from crossing label transitions or time gaps |
| 30 s window size | Matches median segment duration (~568 s); standard in HAR literature |
| 50% overlap | Increases number of training samples; smooths predictions |
| Dictionary mapping over merge | Avoids `dtype` mismatch errors between annotation codes |
| 100 Hz assumed and verified | Confirmed from raw timestamps (`Δt ≈ 0.01 s`) |

---

## References

- Walmsley et al. (2020). *Reallocation of time between device-measured movement behaviours and risk of incident cardiovascular disease.* British Journal of Sports Medicine.
- Capture-24 Dataset: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
