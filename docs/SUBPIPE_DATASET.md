# SubPipe Dataset

## 1. Dataset Identity

**Dataset:** SubPipe — A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization

**Zenodo DOI:** 10.5281/zenodo.10808161

**Zenodo record:** https://zenodo.org/records/10808161

The SubPipe dataset is an underwater dataset designed for SLAM, object detection, and image segmentation. It was recorded using a Lightweight Autonomous Underwater Vehicle (LAUV) operated by OceanScan Marine Systems & Technology.

The sensor suite includes:
- Two cameras
- Side-scan sonar
- Inertial navigation system
- Additional vehicle/sensor telemetry

The AUV was deployed in a submarine pipeline inspection environment in which a submarine pipe is partially covered by sand.

The dataset provides pose ground truth estimated from navigation sensors.

The side-scan sonar data contains object-detection annotations, while the RGB camera data contains segmentation annotations.

---

## 2. Dataset Versions

The official Zenodo record provides three versions:

- `SubPipe.zip` — full dataset
- `SubPipeMini.zip` — subsample focused on semantic segmentation and camera data
- `SubPipeMini2.zip` — subsample mainly focused on side-scan sonar images of the seabed and ground-truth object-detection bounding boxes of the pipeline

The AquaGuard project uses the downloaded SubPipe dataset available in the external `SubPipe` directory.

The original dataset must remain READ-ONLY.

---

## 3. Local Dataset Structure

The attached dataset structure shows:

```text
SubPipe/
├── config.yaml
├── structure.txt
└── DATA/
```

The `DATA` directory contains five sequential chunks:

```text
DATA/
├── Chunk0/
├── Chunk1/
├── Chunk2/
├── Chunk3/
└── Chunk4/
```

The five chunks are part of the original dataset and must not be renamed, moved, deleted, or modified.

---

## 4. Chunk Structure

Each chunk contains vehicle and sensor telemetry files including:

```text
Acceleration.csv
Altitude.csv
AngularVelocity.csv
Depth.csv
EstimatedState.csv
ForwardDistance.csv
Pressure.csv
Rpm.csv
Temperature.csv
WaterVelocity.csv
```

Each chunk also contains camera imagery and side-scan sonar data.

---

## 5. Camera Data

The dataset contains a `Cam0_images` directory within each chunk.

Example filenames use timestamp-like names such as:

```text
1693572852.904.jpg
1693572852.938.jpg
1693572852.971.jpg
1693572853.004.jpg
```

The filenames therefore contain timestamp information that may be useful when associating camera frames with telemetry.

The official Zenodo record identifies Cam0 as a GoPro Hero 10.

### Cam0 Parameters

According to the official Zenodo record:

- Resolution: `1520 × 2704`
- `fx = 1612.36`
- `fy = 1622.56`
- `cx = 1365.43`
- `cy = 741.27`
- Distortion coefficients:
  - `k1 = -0.247`
  - `k2 = 0.0869`
  - `p1 = -0.006`
  - `p2 = 0.001`

These camera parameters are background information and are not automatically required for the initial AquaGuard SSS detection pipeline.

---

## 6. Side-Scan Sonar Data

Each chunk contains two side-scan sonar directories:

```text
SSS_HF_images/
SSS_LF_images/
```

These represent the high-frequency and low-frequency side-scan sonar data.

The local dataset structure confirms both directories contain:

```text
SSS_HF_images/
├── COCO_Annotation/
│   └── coco_format.json
├── Image/
│   └── <timestamp>.pbm
└── YOLO_Annotation/
    └── <timestamp>.txt
```

and:

```text
SSS_LF_images/
├── COCO_Annotation/
│   └── coco_format.json
├── Image/
│   └── <timestamp>.pbm
└── YOLO_Annotation/
    └── <timestamp>.txt
```

The same organization is present across the dataset chunks.

---

## 7. Sonar Frequencies

The dataset documentation identifies two side-scan sonar frequency groups:

- Low Frequency (LF)
- High Frequency (HF)

The project documentation previously identifies the frequencies as:

- LF SSS: `455 kHz`
- HF SSS: `900 kHz`

These values should still be verified against the actual dataset configuration during the Phase-0 audit before they are used as hard-coded assumptions in software.

---

## 8. Sonar Image Statistics

According to the official Zenodo dataset description:

### Low Frequency

- Number of images: `5,000`
- Image size: `2500 × 500`

### High Frequency

- Number of images: `5,030`
- Image size: `5000 × 500`

### Total

- Total SSS images: `10,030`
- LF annotations: `3,163`
- HF annotations: `3,172`
- Total annotations: `6,335`

These are dataset-level statistics reported by the official Zenodo record.

The Phase-0 audit must independently verify the dimensions and counts in the locally downloaded dataset before the project treats them as validated local statistics.

---

## 9. Sonar Image Generation

According to the official Zenodo documentation:

Each sonar image is created after 20 sonar pings, corresponding to approximately one image per second.

This means the sonar imagery is temporally associated with the vehicle/sensor data and should not automatically be treated as an arbitrary collection of independent images.

Timestamp relationships must be explicitly investigated during the dataset audit.

---

## 10. Object Detection Annotations

The official dataset provides object-detection annotations in both:

- COCO format
- YOLO format

The local dataset structure confirms:

```text
COCO_Annotation/
└── coco_format.json
```

and:

```text
YOLO_Annotation/
└── <timestamp>.txt
```

The YOLO annotations are provided per SSS image.

The COCO annotation file is provided for each chunk and frequency.

---

## 11. Class Information

The local dataset contains `classes.txt` files associated with the SSS annotation data.

The exact class IDs and class names MUST be obtained by inspecting the actual local `classes.txt` files.

Do not assume class names from:

- the SIH problem statement
- the Zenodo description
- external articles
- previous project documentation

until the actual local annotation files have been inspected.

The Phase-0 audit must determine:

1. Exact class names
2. Exact class IDs
3. Whether the mapping is identical across chunks
4. Whether the mapping is identical between LF and HF
5. Whether any annotation files use classes not present in `classes.txt`

---

## 12. Annotation/Image Matching

The SSS image filenames and YOLO annotation filenames use timestamp-like identifiers.

Example:

```text
Image/
└── 1693569378.780.pbm

YOLO_Annotation/
└── 1693569378.780.txt
```

This indicates a likely direct filename-based image-to-label relationship.

However, this relationship must be verified programmatically during the dataset audit.

The audit must check:

- Images without labels
- Labels without corresponding images
- Filename mismatches
- Duplicate identifiers
- Malformed annotation files
- Invalid class IDs

---

## 13. Vehicle and Sensor Telemetry

Each chunk contains multiple telemetry CSV files:

```text
Acceleration.csv
Altitude.csv
AngularVelocity.csv
Depth.csv
EstimatedState.csv
ForwardDistance.csv
Pressure.csv
Rpm.csv
Temperature.csv
WaterVelocity.csv
```

These files may provide information useful for associating sonar detections with vehicle state.

In particular, `EstimatedState.csv` is potentially important for localization and geolocation analysis.

However, the actual columns, timestamp representation, units, coordinate frame, and geographic fields must be verified directly from the CSV files.

No metadata semantics should be assumed before inspection.

---

## 14. Geolocation and Localization

The official dataset description states that the AUV pose ground truth is estimated from navigation sensors.

The presence of navigation and vehicle-state data makes localization analysis possible.

However:

**The presence of vehicle pose data does not automatically prove that every detected sonar object has an exact geographic latitude/longitude.**

AquaGuard must distinguish between:

- AUV/vehicle position
- sonar acquisition position
- estimated target position
- exact target latitude/longitude

The project must never fabricate target coordinates.

The Phase-0 audit must determine whether the available metadata is sufficient to associate each SSS image with:

- timestamp
- vehicle pose
- depth
- altitude
- motion state
- geographic coordinates
- any sonar-specific geometry required for target localization

---

## 15. Temporal Association

SSS image filenames contain timestamp-like values.

Camera image filenames also contain timestamp-like values.

Vehicle telemetry is stored separately in CSV files.

Therefore, the Phase-0 audit must determine the exact timestamp conventions and synchronization relationships between:

```text
SSS image
    ↓
SSS annotation
    ↓
vehicle/sensor telemetry
    ↓
camera data
```

The association method must be based on the actual timestamp formats and sampling rates found in the dataset.

Do not assume that filenames from different sensor streams have identical timestamps or sampling frequencies.

---

## 16. Dataset Safety

The original SubPipe dataset is READ-ONLY.

AquaGuard must NEVER:

- Modify files inside `SubPipe`
- Rename files inside `SubPipe`
- Delete files inside `SubPipe`
- Move files inside `SubPipe`
- Overwrite files inside `SubPipe`
- Generate processed files inside `SubPipe`
- Add project code inside `SubPipe`

Any prepared dataset, converted data, cached data, generated labels, or other derived artifacts must be stored inside the AquaGuard project.

---

## 17. Dataset Preparation Policy

The original dataset is the source of truth.

AquaGuard should create a separate prepared dataset containing only the data required for model development.

Dataset preparation must:

- Preserve the original annotations
- Preserve bounding-box correctness
- Preserve class mappings
- Maintain traceability to the original image
- Avoid modifying source files
- Produce reproducible train/validation/test splits

Because the dataset contains temporally related sequential frames, random frame-level splitting must not be assumed to be appropriate.

The correct splitting strategy must be determined during the Phase-0 audit after inspecting the chunk structure and temporal correlation.

---

## 18. Initial AquaGuard Usage

For the AquaGuard MVP, the primary dataset modality is:

```text
Side-Scan Sonar
        ↓
Object Detection
        ↓
Anomaly Detection
        ↓
Confidence / Noise Filtering
        ↓
Metadata Association
        ↓
Reporting
```

Camera imagery and additional telemetry may be used where technically justified by the verified dataset structure.

They should not be introduced into the initial model pipeline merely because they are available.

---

## 19. Verified vs. Unverified Information

### Verified from official Zenodo documentation

- SubPipe is an underwater dataset for SLAM, object detection, and image segmentation.
- It was collected using a LAUV.
- The sensor suite includes cameras, side-scan sonar, and an inertial navigation system.
- SSS data has LF and HF imagery.
- SSS images have object-detection annotations.
- Both COCO and YOLO annotation formats are provided.
- LF contains 5,000 images.
- HF contains 5,030 images.
- LF image size is 2500 × 500.
- HF image size is 5000 × 500.
- Total SSS images are 10,030.
- Total annotations are 6,335.
- Sonar images are generated after 20 pings, approximately one image per second.

### Verified from the locally provided dataset structure

- Five chunks exist: Chunk0 through Chunk4.
- Each chunk contains multiple telemetry CSV files.
- `Cam0_images` exists.
- `SSS_HF_images` exists.
- `SSS_LF_images` exists.
- SSS data contains `Image` directories.
- SSS data contains `YOLO_Annotation` directories.
- SSS data contains `COCO_Annotation` directories.
- COCO annotations use `coco_format.json`.
- YOLO annotations use timestamp-based `.txt` files.
- `classes.txt` files exist.
- Telemetry includes `EstimatedState.csv`, `Depth.csv`, `Altitude.csv`, `ForwardDistance.csv`, and other sensor-state files.

### Must be verified during Phase 0

- Exact class names
- Exact class IDs
- Exact YOLO annotation syntax
- Exact COCO category mapping
- Exact image dimensions in the local copy
- Exact number of local images
- Exact number of local annotations
- Exact timestamp formats
- Exact telemetry column names
- Telemetry units
- Geographic coordinate availability
- Coordinate reference/frame conventions
- Image-to-telemetry synchronization
- Image-to-annotation matching
- Appropriate train/validation/test splitting strategy
- Whether LF and HF annotations use identical class mappings
- Whether all chunks use identical data conventions

If any property cannot be verified from the actual files, mark it as `UNKNOWN` rather than guessing.

---

## 20. Official Dataset Reference

SubPipe official Zenodo record:

https://zenodo.org/records/10808161

DOI:

10.5281/zenodo.10808161
