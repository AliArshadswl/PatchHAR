# Create a Markdown file summarizing the CAPTURE-24 P001 analysis output

from textwrap import dedent
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

md_path = "/mnt/data/capture24_p001_useful_info.md"

# Arrays and shapes
arrays = [
    ("windows", "(7469, 1000, 3)", "float32"),
    ("labels_str", "(7469,)", "object"),
    ("times_epoch_ns", "(7469, 1000)", "int64"),
    ("first_ts_epoch_ns", "(7469,)", "int64"),
    ("pid", "()", "<U4"),
    ("window_size", "()", "int32"),
    ("signal_rate", "()", "int32"),
]

arrays_df = pd.DataFrame(arrays, columns=["Array", "Shape", "Dtype"])

# Accelerometer stats
accel_stats = {
    "Shape": "(7469, 1000, 3)",
    "Data type": "float32",
    "Value range": "[-8.010, 7.255]",
    "Mean": "-0.158",
    "Std": "0.557",
    "Chosen key": "windows",
}

# Label distribution
labels = [
    ("light", 2102, "28.1%"),
    ("moderate-vigorous", 14, "0.2%"),
    ("sedentary", 2612, "35.0%"),
    ("sleep", 2741, "36.7%"),
]
labels_df = pd.DataFrame(labels, columns=["Label", "Count", "Percent"])

md_content = dedent(f"""
# CAPTURE-24 — Subject P001: Useful Information

This file captures the key, actionable details parsed from the analysis output you provided.

## Overview
- Dataset: **CAPTURE-24**, Subject **P001**
- Source artifacts: NPZ with multiple arrays; downstream script also produced image and report files.

---

## Available arrays (from NPZ)
Below is a concise inventory of arrays discovered in the NPZ file.
""")

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# Append arrays table
arrays_md = arrays_df.to_markdown(index=False)
with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n" + arrays_md + "\n")

# Accelerometer section
accel_md = dedent(f"""
---

## Accelerometer data (primary key: `windows`)
- **Shape:** {accel_stats["Shape"]}
- **Dtype:** {accel_stats["Data type"]}
- **Value range:** {accel_stats["Value range"]}
- **Mean:** {accel_stats["Mean"]}
- **Std:** {accel_stats["Std"]}

Interpretation:
- 3 axes per sample (likely x, y, z) at *{100} Hz* (from `signal_rate`), each window spanning **10 s** if windows are 1000 samples.
- The mean close to zero and range within ~±8g suggests raw/near-raw acceleration values.
- Using the largest array (`windows`) as accel source is reasonable when an explicit accelerometer key isn't present.
""")
with open(md_path, "a", encoding="utf-8") as f:
    f.write(accel_md)

# Labels section
with open(md_path, "a", encoding="utf-8") as f:
    f.write(dedent("""
---

## Activity labels
Discrete activity classes and their distribution across windows.
"""))

labels_md = labels_df.to_markdown(index=False)
with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n" + labels_md + "\n")

# Temporal data section
temporal_md = dedent("""
---

## Temporal data
- **times_epoch_ns**: shape (7469, 1000), dtype int64 — per-sample nanosecond UNIX timestamps for each window.
- **first_ts_epoch_ns**: shape (7469,), dtype int64 — first timestamp per window.
- **signal_rate**: **100 Hz** (int32).

Notes:
- The samples in each window increase by **1e6 ns = 1 ms**, consistent with 1000 samples per 10 s at 100 Hz.
- Use `pd.to_datetime(ts, unit='ns', utc=True)` to convert epoch ns to timestamps.
""")
with open(md_path, "a", encoding="utf-8") as f:
    f.write(temporal_md)

# Outputs produced by the original script
outputs_md = dedent("""
---

## Artifacts reported by the original analysis script
- **Visualization:** `/workspace/charts/capture24_p001_analysis.png`
- **Summary report:** `/workspace/docs/capture24_p001_report.md`

> If you want, I can regenerate these under a different path as needed.
""")
with open(md_path, "a", encoding="utf-8") as f:
    f.write(outputs_md)

# Practical tips
tips_md = dedent("""
---

## Practical tips
- When training models, consider class imbalance (only **0.2%** 'moderate-vigorous'). Use class weights or resampling.
- Validate window contiguity via `times_epoch_ns`; drop windows with irregular spacing or missing samples.
- Standardize per-axis using the provided mean/std or recomputed stats from training data only.
- Store label mapping (`labels_str` order) alongside models to avoid mismatch across subjects.
- Confirm PID value (dtype `<U4`) is `P001`; propagate to derived outputs for provenance.
""")
with open(md_path, "a", encoding="utf-8") as f:
    f.write(tips_md)

print(f"Wrote: {md_path}")
