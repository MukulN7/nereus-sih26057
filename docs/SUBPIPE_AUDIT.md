# SubPipe Dataset Audit

## 1. Audit Scope

This document provides the authoritative Phase 0 dataset audit for project **AquaGuard** (SIH Problem Statement **PS26057**). The audit evaluates the properties of the local **SubPipe** dataset (`D:\SIH26057\SubPipe`) based on the baseline audit results in `outputs/dataset_audit.json` and official project specifications.

- **Status:** Phase 0 Complete — Baseline Read-Only Audit.
- **Data Policy:** The original SubPipe dataset remains strictly read-only. No files inside `SubPipe` were modified, moved, renamed, or deleted.
- **Classification Standard:** Findings are strictly classified as `[VERIFIED]`, `[INFERRED]`, or `[UNKNOWN]`.

---

## 2. Dataset Structure

### 2.1 Top-Level Layout `[VERIFIED]`
- **Root Files:** `config.yaml`, `structure.txt`.
- **Primary Data Directory:** `DATA/` containing 5 sequential survey chunks: `Chunk0`, `Chunk1`, `Chunk2`, `Chunk3`, and `Chunk4`.

### 2.2 Per-Chunk Organization `[VERIFIED]`
Each chunk contains:
- 10 Telemetry CSV files (`Acceleration.csv`, `Altitude.csv`, `AngularVelocity.csv`, `Depth.csv`, `EstimatedState.csv`, `ForwardDistance.csv`, `Pressure.csv`, `Rpm.csv`, `Temperature.csv`, `WaterVelocity.csv`).
- Optical camera directory: `Cam0_images/` (`.jpg` format).
- Side-Scan Sonar (SSS) directories:
  - `SSS_LF_images/` (Low Frequency)
  - `SSS_HF_images/` (High Frequency)
- Each SSS directory contains:
  - `Image/`: Netpbm image files.
  - `YOLO_Annotation/`: YOLO format text files (`<timestamp>.txt`) and `classes.txt`.
  - `COCO_Annotation/`: `coco_format.json`.

---

## 3. Sonar Image Statistics

### 3.1 Image Counts and Dimensions `[VERIFIED]`
Based on `outputs/dataset_audit.json` and the official SubPipe Zenodo record:

- **Low-Frequency (LF) SSS Images:**
  - Total Images: **5,000** (Chunk0: 1,055; Chunk1: 541; Chunk2: 539; Chunk3: 189; Chunk4: 2,676).
  - Resolution: **`2500 × 500` pixels**.
  - Frequency: **455 kHz** (from `config.yaml`).
- **High-Frequency (HF) SSS Images:**
  - Total Images: **5,030** (Chunk0: 1,011; Chunk1: 541; Chunk2: 539; Chunk3: 190; Chunk4: 2,749).
  - Resolution: **`5000 × 500` pixels**.
  - Frequency: **900 kHz** (from `config.yaml`).
- **Combined Total SSS Images:** **10,030**.
- **Image Encoding `[VERIFIED]`:** All images use the binary Netpbm PPM (`P6`) header format representing 3-channel 8-bit RGB imagery.

---

## 4. Annotation Statistics

### 4.1 Annotation Counts `[VERIFIED]`
- **Total YOLO Annotation Files:** **6,335** (LF: 3,163 files; HF: 3,172 files).
- **Total Bounding Box Rows:** **6,458** (LF: 3,226 bounding boxes; HF: 3,232 bounding boxes).
- **Object Distribution per Image `[VERIFIED]`:**
  - Single Object (1 bounding box): **6,212 files** (~98.1%).
  - Multiple Objects (2 bounding boxes): **123 files** (~1.9%, observed in Chunk0 and Chunk4).
  - Empty Annotation Files: **0**.

---

## 5. Class Mapping

### 5.1 Verification of Class Labels `[VERIFIED]`
- Every `classes.txt` in the dataset contains exactly one line:
  ```text
  Pipeline
  ```
- **YOLO Class IDs `[VERIFIED]`:** All 6,458 bounding box annotations use class ID `0`.
- **COCO Categories `[VERIFIED]`:** All `coco_format.json` files specify category ID `1`, name `"Pipeline"`.
- **Mapping Consistency `[VERIFIED]`:** LF and HF use the exact same single-class mapping:
  $$\text{Class ID } 0 \iff \text{"Pipeline"}$$

---

## 6. Image/Annotation Matching

### 6.1 Filename Matching `[VERIFIED]`
- Sonar images and YOLO annotations share matching timestamp stems (e.g., `1693569378.780.pbm` $\leftrightarrow$ `1693569378.780.txt`).
- **Exact Matched Image-Label Pairs:** **6,296**.
- **Unannotated Images (Background Frames):** **3,734** (LF: 1,869; HF: 1,865). These represent negative background seabed samples where no pipeline is present.
- **Orphan Annotation Files:** **39 files** (32 in Chunk4 LF, 6 in Chunk4 HF, 1 corrupted filename `d.860.txt` in Chunk0 HF) caused by acoustic ping dropouts during acquisition.

---

## 7. Annotation Validity

### 7.1 Coordinate Syntax and Bounds `[VERIFIED]`
- All YOLO annotations follow the standard normalized format:
  $$\langle\text{class\_id}\rangle\quad\langle x_{\text{center}}\rangle\quad\langle y_{\text{center}}\rangle\quad\langle\text{width}\rangle\quad\langle\text{height}\rangle$$
- All coordinates strictly lie within the normalized range $[0.0, 1.0]$.
- There are no malformed, out-of-bounds, or negative dimension values.

---

## 8. Telemetry Structure

### 8.1 CSV Telemetry Headers and Units `[VERIFIED]`
All 10 CSV files across all chunks share identical column headers:

1. **`Acceleration.csv`**: `image,timestamp, x (m/s/s), y (m/s/s), z (m/s/s)`
2. **`Altitude.csv`**: `image,timestamp, DVL - Beam 0 (m), DVL - Beam 1 (m), DVL - Beam 2 (m), DVL - Beam 3 (m), DVL Filtered`
3. **`AngularVelocity.csv`**: `image,timestamp, x (rad/s), y (rad/s), z (rad/s)`
4. **`Depth.csv`**: `image,timestamp, value (m)`
5. **`EstimatedState.csv`**: `image,timestamp, x (m), y (m), z (m), phi (rad), theta (rad), psi (rad), u (m/s), v (m/s), w (m/s), vx (m/s), vy (m/s), vz (m/s), p (rad/s), q (rad/s), r (rad/s), depth (m), alt (m)`
6. **`ForwardDistance.csv`**: `image,timestamp, Echo Sounder (m)`
7. **`Pressure.csv`**: `image,timestamp, value (hpa)`
8. **`Rpm.csv`**: `image,timestamp, value (rpm)`
9. **`Temperature.csv`**: `image,timestamp, value (°c)`
10. **`WaterVelocity.csv`**: `image,timestamp, x (m/s), y (m/s), z (m/s)`

### 8.2 Sampling Rate `[VERIFIED]`
- Telemetry rows match the optical camera frames (`Cam0_images`) sampled at 30 Hz (77,799 total rows across all chunks).

---

## 9. Timestamp Synchronization

### 9.1 Format and Alignment `[VERIFIED]`
- Timestamps use standard Unix epoch decimal seconds (e.g., `1693572852.904`).
- Telemetry and optical video are sampled at 30 Hz, while sonar images are produced at ~1 Hz (every 20 pings).
- **Telemetry Coverage `[VERIFIED]`:**
  - Chunks 1, 2, and 3: 100% of sonar images fall within the telemetry recording window.
  - Chunk 0 and Chunk 4: Sonar recording spans a longer duration than optical/telemetry logging; hence, a subset of sonar frames has telemetry coverage.
- **Synchronization Method `[INFERRED]`:** Nearest-neighbor lookup or 1D linear interpolation on `timestamp` provides alignment with $< 0.017\text{ s}$ error.

---

## 10. Geolocation Availability

### 10.1 Absence of Global GPS Coordinates `[VERIFIED]`
- There are **no global geodetic coordinates** (latitude, longitude, GPS, WGS84, or UTM) anywhere in the SubPipe dataset.
- Spatial position is provided exclusively in **local Cartesian coordinates** $(x, y, z)$ in meters from the AUV's dead-reckoning navigation system (`EstimatedState.csv`).

### 10.2 Geolocation Policy Compliance `[VERIFIED]`
- AquaGuard must never fabricate geographic coordinates.
- Spatial reporting must use local vehicle pose $(x, y, z)$, depth, and altitude when telemetry is available, and designate global coordinates as `null` / `UNAVAILABLE` unless an external mission geodetic origin is provided.

---

## 11. Dataset Splitting Considerations

### 11.1 Temporal Correlation `[VERIFIED]`
- Sonar frames are acquired at ~1 Hz along continuous survey tracks.
- Consecutive frames exhibit significant spatial overlap of the seafloor.

### 11.2 Splitting Strategy `[INFERRED]`
- **Random Frame Splitting (UNSAFE):** Causes severe data leakage between adjacent frames.
- **Recommended Splitting (Chunk / Block Splitting):** Splits must be made across whole chunks (e.g., Train: Chunks 0, 1, 4; Val: Chunk 2; Test: Chunk 3) or contiguous time blocks to ensure valid generalization testing.

---

## 12. Important Dataset Issues

1. **File Extension Typos (`.bpm` vs `.pbm`) `[VERIFIED]`:** 218 images (108 in Chunk0 HF, 110 in Chunk4 LF) use `.bpm` extension due to a typo in the original dataset creation. Dataset loaders must handle both extensions.
2. **Netpbm P6 Header with `.pbm` Extension `[VERIFIED]`:** Images are 3-channel binary PPM (`P6`) rather than 1-bit monochrome PBM.
3. **High Aspect Ratios (`5:1` and `10:1`) `[VERIFIED]`:** LF is $2500 \times 500$; HF is $5000 \times 500$. Naive resizing to standard square YOLO input ($640 \times 640$) will distort acoustic textures; tiling or patch extraction is recommended.
4. **Single Class Dataset vs. SIH Multi-Debris Scope `[VERIFIED]`:** The dataset provides annotations strictly for `"Pipeline"`.
5. **Partial Telemetry Coverage `[VERIFIED]`:** Sonar frames outside telemetry windows must be handled gracefully without crashing metadata association pipelines.

---

## 13. Verified Facts

1. `[VERIFIED]` 5 chunks exist (`Chunk0`–`Chunk4`).
2. `[VERIFIED]` Total SSS images: 10,030 (5,000 LF, 5,030 HF).
3. `[VERIFIED]` LF dimensions: $2500 \times 500$ px; HF dimensions: $5000 \times 500$ px.
4. `[VERIFIED]` Image format: 3-channel 8-bit Netpbm `P6` RGB.
5. `[VERIFIED]` Total YOLO annotations: 6,335 files, 6,458 bounding box instances.
6. `[VERIFIED]` Single class: `0: "Pipeline"`.
7. `[VERIFIED]` 3,734 unannotated background images.
8. `[VERIFIED]` All 10 CSV telemetry files exist across all chunks with standardized headers.
9. `[VERIFIED]` No GPS latitude/longitude fields exist; only local Cartesian $(x, y, z)$ in meters.

---

## 14. Unknowns

1. `[UNKNOWN]` **Mission Geodetic Origin:** Global latitude/longitude of $(0, 0, 0)$ is not specified in local files.
2. `[UNKNOWN]` **DVL Sensor Frame Rotation Matrix:** Mounting matrix from DVL beams to vehicle body frame is not explicitly defined in `config.yaml`.

---

## 15. Recommendations for Phase 1

1. **Read-Only Dataset Preparation:** Build a converter that ingests both `.pbm` and `.bpm` files from `D:\SIH26057\SubPipe` and writes converted data to `AquaGuard/data/`.
2. **Tiling / Patch Extraction:** Implement a sliding window tiling approach to handle $5:1$ and $10:1$ aspect ratios cleanly.
3. **Background Frame Inclusion:** Include a controlled proportion of unannotated background frames to minimize false positives.
4. **Chunk-Level Partitioning:** Generate train/val/test splits partitioned by chunk to eliminate temporal leakage.
5. **Telemetry Synchronization Utility:** Implement binary search timestamp matching for `EstimatedState.csv`.

---

### Audit Summary

| Parameter | Final Verified Value |
| :--- | :--- |
| **Audit Status** | Phase 0 Complete — Authoritative Baseline Established |
| **Total SSS Images** | 10,030 (5,000 LF, 5,030 HF) |
| **Total Bounding Boxes** | 6,458 across 6,335 YOLO label files |
| **Classes** | Exactly 1 class: `0: "Pipeline"` |
| **Coordinates** | Local Cartesian INS $(x, y, z)$ in meters; No GPS Latitude/Longitude |
| **Telemetry Coverage** | Full in Chunks 1–3; partial in Chunks 0 & 4 |

### Recommended Next Step
- Stop Phase 0 and review audit findings before proceeding to Phase 1 (Dataset Preparation).

### Files Inspected
- `D:\SIH26057\SubPipe\config.yaml`
- `D:\SIH26057\SubPipe\structure.txt`
- `d:\SIH26057\AquaGuard\AQUAGUARD_SPEC.md`
- `d:\SIH26057\AquaGuard\docs\SIH_PS26057.md`
- `d:\SIH26057\AquaGuard\docs\SUBPIPE_DATASET.md`
- `d:\SIH26057\AquaGuard\outputs\dataset_audit.json`

### Files Created
- [`docs/SUBPIPE_AUDIT.md`](file:///d:/SIH26057/AquaGuard/docs/SUBPIPE_AUDIT.md)

### Validation Performed
- Verified statistics directly against `dataset_audit.json` and project reference documentation.
- Ensured absolute read-only integrity of `D:\SIH26057\SubPipe`.
- Removed temporary scripts to maintain a clean codebase.
