# PAMAP2 Dataset Specification

## File Location

```
/mnt/share/ali/PaMP2_dataset/Protocol/_clean/pamap2_protocol_combined_common_labels.parquet
```

**Format:** Parquet

---

## Columns

| Column | Description |
|---|---|
| `subject_id` | Subject identifier |
| `timestamp` | Sensor-relative timestamp |
| `activity_id` | Numeric activity ID |
| `activity_label` | String activity label **(target)** |
| `heart_rate` | Heart rate (present in file, not used as model input) |
| `hand_temp` | Temperature, hand |
| `chest_temp` | Temperature, chest |
| `ankle_temp` | Temperature, ankle |
| `hand_acc16_x/y/z` | 16g accelerometer, hand |
| `hand_gyro_x/y/z` | Gyroscope, hand |
| `hand_mag_x/y/z` | Magnetometer, hand |
| `chest_acc16_x/y/z` | 16g accelerometer, chest |
| `chest_gyro_x/y/z` | Gyroscope, chest |
| `chest_mag_x/y/z` | Magnetometer, chest |
| `ankle_acc16_x/y/z` | 16g accelerometer, ankle |
| `ankle_gyro_x/y/z` | Gyroscope, ankle |
| `ankle_mag_x/y/z` | Magnetometer, ankle |

---

## Sensor Locations

3 locations: **hand**, **chest**, **ankle**

---

## Sampling Rate

**100 Hz**

---

## Channels Used for Modeling

**30 channels total**, in this fixed order:

1. All acc16 axes — 9 channels
   - `hand_acc16_x`, `hand_acc16_y`, `hand_acc16_z`
   - `chest_acc16_x`, `chest_acc16_y`, `chest_acc16_z`
   - `ankle_acc16_x`, `ankle_acc16_y`, `ankle_acc16_z`

2. All gyro axes — 9 channels
   - `hand_gyro_x`, `hand_gyro_y`, `hand_gyro_z`
   - `chest_gyro_x`, `chest_gyro_y`, `chest_gyro_z`
   - `ankle_gyro_x`, `ankle_gyro_y`, `ankle_gyro_z`

3. All magnetometer axes — 9 channels
   - `hand_mag_x`, `hand_mag_y`, `hand_mag_z`
   - `chest_mag_x`, `chest_mag_y`, `chest_mag_z`
   - `ankle_mag_x`, `ankle_mag_y`, `ankle_mag_z`

4. Temperature scalars — 3 channels
   - `hand_temp`, `chest_temp`, `ankle_temp`

---

## Windowing

| Parameter | Value |
|---|---|
| Window size | 1000 samples (10 seconds @ 100 Hz) |
| Stride | 250 samples (75% overlap) |
| Majority label threshold | ≥ 60% of samples in window must share the same label |

---

## Target

- **Column:** `activity_label` (string)
- Windows where the majority label is below the 60% threshold are discarded.

---

## Preprocessing

Per-window, per-channel **instance normalization**:
- Zero mean, unit standard deviation
- Clipped to ±10
- NaN / Inf handled via: forward-fill → backward-fill → fill with zero

---

## Validation Strategy

**Leave-One-Subject-Out (LOSO) cross-validation**

For each fold:
- **Test set:** one subject held out
- **Validation set:** the next subject (cyclically)
- **Train set:** all remaining subjects

---

## Output Directory

```
/mnt/share/ali/pamap2_patchhar_Final_results/
```
