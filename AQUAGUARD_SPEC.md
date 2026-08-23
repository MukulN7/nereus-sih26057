# AquaGuard — Project Specification

## 1. Project Identity

**Project:** AquaGuard

**SIH Problem Statement:** PS26057

**Problem Statement:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery

AquaGuard is the proposed software prototype for PS26057.

The system is intended to process Side-Scan Sonar (SSS) imagery, identify man-made underwater objects/anomalies, reduce false positives caused by natural seabed structures and acoustic artifacts, associate detections with available metadata, and provide actionable anomaly reports through a user-facing dashboard.

---

## 2. Problem Requirements

The SIH problem statement requires an end-to-end automated computer vision pipeline capable of:

1. Ingesting side-scan sonar imagery.
2. Identifying man-made debris/anomalies against a complex natural background.
3. Generating actionable localized data.
4. Handling acoustic-image challenges including:
   - High speckle noise
   - Varying pixel resolutions
   - Acoustic shadows
   - Data dropouts caused by underwater vehicle motion
5. Separating natural seafloor topology from artificial anomalies.
6. Running efficiently enough to potentially support edge or onboard deployment without heavy cloud dependencies.

---

## 3. Required Functional Components

The final AquaGuard prototype must contain the following major components.

### 3.1 Object Detection / Semantic Segmentation

The system must use an AI/ML model capable of detecting man-made objects in SSS imagery.

Possible architectures mentioned by the SIH problem statement include:

- YOLO
- Faster R-CNN
- U-Net

The actual model architecture will be selected after dataset audit and experimentation.

The model must produce bounding boxes or masks as appropriate for the selected task.

---

### 3.2 Confidence Scoring and Noise Filtering

The system must include a confidence/noise-filtering stage capable of reducing false positives caused by:

- Natural acoustic shadows
- Rock clusters
- Natural seabed structures
- Other non-man-made sonar patterns

Every retained detection must have an associated confidence score.

The final confidence representation must be normalized to a 0–100% user-facing scale.

The exact filtering strategy must be determined through experimentation rather than assumed in advance.

---

### 3.3 Anomaly Reporting and Geotagging

The system must produce structured anomaly information.

The expected report format is:

- JSON
- CSV

The report should contain, where supported by the available dataset metadata:

- Detection/classification
- Confidence
- Location
- Bounding dimensions
- Relevant image/frame identifier
- Relevant timestamp
- Other useful metadata

### Important Geolocation Rule

AquaGuard must never fabricate geographic coordinates.

If the dataset provides only vehicle position rather than exact target position, the system must distinguish between:

- Vehicle position
- Sonar acquisition position
- Estimated target position
- Exact target coordinates

The exact localization method must be established from the actual SubPipe metadata during the dataset audit.

---

## 4. User Interface

AquaGuard must ultimately provide a visual dashboard through which a user can:

1. Upload raw sonar imagery/log data.
2. Run the detection pipeline.
3. View detections overlaid on the sonar imagery.
4. View available localization/geographic information.
5. Inspect confidence scores and classifications.
6. Download generated anomaly reports.

The final UI technology should be selected after the core inference pipeline is functional.

---

## 5. Primary Data Source

The primary dataset for model development is:

**SubPipe**

Official source:

https://zenodo.org/records/10808161

The original dataset is stored outside the AquaGuard source tree:

```text
D:\SIH26057├── SubPipe└── AquaGuard```

The SubPipe dataset is READ-ONLY.

AquaGuard must never modify the original dataset.

---

## 6. Dataset Modality Priority

The primary modality for the AquaGuard MVP is:

```text
Side-Scan Sonar
        ↓
Preprocessing
        ↓
Object Detection
        ↓
Confidence / Noise Filtering
        ↓
Metadata Association
        ↓
Anomaly Reporting
        ↓
Dashboard
```

Camera imagery and additional telemetry may be incorporated only where they provide a technically justified improvement to the required solution.

Their availability alone is not sufficient reason to add them to the initial model pipeline.

---

## 7. Dataset Audit Before Model Development

No model training should begin before the SubPipe dataset has been audited.

The audit must establish the actual properties of the local dataset.

At minimum, verify:

### Dataset Structure

- Number of chunks
- SSS LF directory structure
- SSS HF directory structure
- Camera directory structure
- Telemetry structure

### Images

- Number of LF images
- Number of HF images
- Image dimensions
- Image formats
- Image readability
- Corrupted files
- Duplicate files

### Annotations

- Number of annotations
- Annotation formats
- Class IDs
- Class names
- Bounding-box validity
- Empty annotations
- Malformed annotations
- Images without labels
- Labels without images
- Duplicate identifiers

### Metadata

- Timestamp format
- Telemetry columns
- Units
- Vehicle state
- Depth
- Altitude
- Position
- Geographic coordinates
- Coordinate frames
- Synchronization relationships

### Dataset Splitting

The audit must determine whether sequential frames are strongly correlated.

Random frame-level splitting must not be assumed to be valid.

The final train/validation/test strategy must prevent temporal leakage where possible.

---

## 8. Unknowns Policy

AquaGuard development must distinguish between:

### VERIFIED

Information directly established from:

- Actual local files
- Official SubPipe documentation
- Official SIH PS26057 documentation

### INFERRED

Information derived logically from verified observations but not explicitly documented.

### UNKNOWN

Information that has not yet been verified.

Unknown information must be written as:

`UNKNOWN`

rather than being guessed.

This rule applies particularly to:

- Class mappings
- Metadata semantics
- Geographic coordinates
- Coordinate systems
- Sensor synchronization
- Annotation conventions
- Units

---

## 9. Model Development Strategy

The initial model should be selected based on:

- Actual dataset annotation structure
- Number of classes
- Class imbalance
- Image resolution
- Available compute
- Detection accuracy
- Inference speed
- Ease of deployment
- Edge-device feasibility

The project should prefer a practical model that can produce a reliable demonstration rather than an unnecessarily complex architecture.

Model selection must be evidence-driven.

---

## 10. Preprocessing Requirements

The preprocessing pipeline must account for the characteristics of SSS imagery.

Potential issues include:

- Speckle noise
- Variable intensity
- Acoustic shadows
- Resolution differences
- Seabed texture
- Motion-related artifacts
- Data dropouts

The exact preprocessing operations must be evaluated experimentally.

Potential transformations must not destroy the visual structures required for object detection.

All preprocessing applied during training must be reproducible during inference.

---

## 11. False Positive Reduction

Natural seabed structures can resemble artificial objects in sonar imagery.

AquaGuard must therefore evaluate false-positive reduction using:

- Model confidence
- Spatial characteristics
- Object size
- Shape
- Acoustic-shadow characteristics
- Temporal consistency where available
- Other validated features

The final filtering pipeline must be based on measurable validation results.

---

## 12. Output Schema

AquaGuard detections should ultimately be representable in a structured format similar to:

```json
{
  "image_id": "example.pbm",
  "timestamp": null,
  "detections": [
    {
      "class_id": 0,
      "class_name": "UNKNOWN",
      "confidence": 0.0,
      "confidence_percent": 0.0,
      "bbox": {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "width": 0,
        "height": 0
      },
      "location": {
        "latitude": null,
        "longitude": null
      }
    }
  ]
}
```

This is a proposed internal/output structure.

The actual fields must be finalized after the dataset audit establishes what metadata can legitimately be provided.

---

## 13. Traceability

Every model prediction should remain traceable to its source data.

At minimum, the system should retain:

- Original image identifier
- Source chunk
- Timestamp where available
- Model version
- Detection class
- Confidence
- Bounding box
- Localization metadata where available

This is required for reproducibility and debugging.

---

## 14. Reproducibility

AquaGuard should use configuration-driven execution wherever practical.

Important settings should not be scattered throughout source code.

Configurations should eventually control:

- Dataset paths
- Model paths
- Image preprocessing
- Confidence thresholds
- IoU thresholds
- Training parameters
- Output paths
- Logging
- Device selection

The original dataset path must remain configurable rather than hard-coded throughout the codebase.

---

## 15. Source Code Organization

The permanent project structure is:

```text
AquaGuard/
├── .agents/
│   └── rules/
├── docs/
├── src/
├── scripts/
├── tests/
├── configs/
├── models/
├── outputs/
├── notebooks/
├── AQUAGUARD_SPEC.md
├── requirements.txt
└── README.md
```

The project should evolve within this structure rather than creating arbitrary files in the project root.

---

## 16. Development Principles

AquaGuard development must follow these principles:

1. Audit before training.
2. Verify before assuming.
3. Keep the source dataset read-only.
4. Separate raw data from derived data.
5. Keep experiments reproducible.
6. Keep model inference modular.
7. Keep reporting modular.
8. Never fabricate metadata.
9. Prefer measurable validation over subjective visual judgment.
10. Build the MVP incrementally.
11. Avoid unnecessary complexity.
12. Preserve traceability from output back to source data.

---

## 17. Current Development Phase

The project is currently in:

**PHASE 0 — Dataset Audit**

The immediate objective is NOT model training.

The immediate objective is to understand and verify the SubPipe dataset sufficiently to design the correct preparation and training pipeline.

No major model architecture decision should be treated as final until the audit is complete.

---

## 18. Phase Sequence

The project should progress through the following broad sequence:

```text
Phase 0
Dataset Audit
    ↓
Dataset Review
    ↓
Dataset Preparation
    ↓
Model Training
    ↓
Inference
    ↓
Noise / Confidence Filtering
    ↓
Metadata / Geolocation
    ↓
Reporting
    ↓
Dashboard
    ↓
Integration
    ↓
Testing / Validation
    ↓
Final Demonstration
```

Each phase should be completed and verified before moving to the next major phase.

---

## 19. Antigravity Development Rule

Antigravity is being used as the primary development environment.

The AI coding agent must not be given the entire project implementation as one uncontrolled task.

Development should proceed incrementally.

For each step:

1. Define the exact objective.
2. Identify the required files.
3. Implement only that objective.
4. Run the relevant verification.
5. Inspect the result.
6. Fix issues before proceeding.

The agent must not silently invent dataset properties.

When a dataset fact is required, the agent should inspect the actual files.

---

## 20. Final Objective

AquaGuard should ultimately demonstrate an end-to-end workflow:

```text
User uploads SSS imagery/log
            ↓
Input validation
            ↓
Sonar preprocessing
            ↓
AI detection
            ↓
Confidence scoring
            ↓
False-positive filtering
            ↓
Metadata association
            ↓
Localization where supported
            ↓
Anomaly visualization
            ↓
JSON / CSV report
            ↓
Dashboard
```

The final prototype must remain aligned with the requirements of SIH PS26057 and the verified capabilities of the SubPipe dataset.

---

## 21. Authoritative References

### SIH Problem Statement

PS26057:

**AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery**

The authoritative project requirements are stored in:

```text
docs/SIH_PS.md
```

### SubPipe Dataset

The dataset reference and audit notes are stored in:

```text
docs/SUBPIPE_DATASET.md
```

Official SubPipe Zenodo record:

https://zenodo.org/records/10808161

### Project Specification

This document:

```text
AQUAGUARD_SPEC.md
```

defines the current project-level engineering requirements and development principles.
