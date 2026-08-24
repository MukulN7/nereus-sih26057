# NEREUS — Underwater Sonar Anomaly Detection

**NEREUS** is an AI-powered Side-Scan Sonar (SSS) inspection system developed for **Smart India Hackathon 2026 — Problem Statement PS26057**:

> **AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery**

NEREUS is a working MVP that ingests Side-Scan Sonar imagery, detects annotated underwater pipeline instances using a lightweight **YOLOv8n** model, assigns confidence scores, visualizes detections in a Streamlit dashboard, and exports structured **JSON/CSV** inspection reports.

> **Current MVP scope:** the SubPipe training data contains a single annotated class, **Pipeline**. The broader PS covers underwater marine debris/anomalies; expanding NEREUS to additional anomaly classes is future work.

---

## Live Demo

**Web dashboard:**  
https://sarvagya-nereus-sih26057.streamlit.app/

**GitHub:**  
https://github.com/MukulN7/nereus-sih26057

---

## Why NEREUS?

Underwater sonar inspection produces large volumes of acoustic imagery that can be difficult to review manually. Side-Scan Sonar also differs significantly from conventional RGB imagery: the SubPipe dataset contains very wide sonar frames, acoustic textures, shadows, and sequential survey data.

NEREUS focuses on turning this difficult input into a simple operational workflow:

**Sonar Input → AI Detection → Confidence → Bounding Geometry → Structured Report → Dashboard**

The result is an AI-assisted first-pass inspection tool that helps an operator identify likely pipeline instances and review them without inspecting every frame manually.

---

## Key Highlights

- **10,030** Side-Scan Sonar images in the SubPipe dataset
- **6,458** annotated pipeline bounding-box instances
- **3,734** unannotated background sonar frames
- Both **Low-Frequency (LF)** and **High-Frequency (HF)** sonar imagery
- LF resolution: **2500 × 500 px**
- HF resolution: **5000 × 500 px**
- Lightweight **YOLOv8n** detector
- **CPU inference** in the deployed dashboard
- Adjustable confidence threshold
- Original vs. detection-result visualization
- Structured detection tables with confidence and bounding-box geometry
- **JSON and CSV** report export
- Live Streamlit deployment

---

## Model Performance

The evaluated YOLOv8n run used a **640 px input size**, **batch size 16**, and **30 training epochs**.

| Metric | Result |
|---|---:|
| Precision | **96.93%** |
| Recall | **85.73%** |
| mAP@50 | **93.34%** |
| mAP@50–95 | **54.87%** |

These values correspond to the evaluated model run used for the project presentation. Metric names are reported explicitly rather than being presented as a generic “accuracy” figure.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   SubPipe SSS Data  │
                    │  LF + HF Sonar      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Input Preparation   │
                    │ Image decoding      │
                    │ Size / format handling│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      YOLOv8n        │
                    │ Pipeline Detection  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Confidence Control  │
                    │ User-set threshold  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     Structured Detection      │
              │ class + confidence + bbox    │
              └───────────────┬────────────────┘
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
       ┌────────────────────┐     ┌────────────────────┐
       │ NEREUS Dashboard   │     │ JSON / CSV Export  │
       │ Visual inspection  │     │ Inspection reports │
       └────────────────────┘     └────────────────────┘
```

---

## Dataset — SubPipe

NEREUS uses the **SubPipe** underwater pipeline inspection dataset.

**Dataset reference:**  
https://zenodo.org/records/10808161  
**DOI:** `10.5281/zenodo.10808161`

The dataset contains:

- 5 sequential survey chunks (`Chunk0`–`Chunk4`)
- 5,000 LF SSS images
- 5,030 HF SSS images
- 6,335 YOLO annotation files
- 6,458 annotated bounding boxes
- One verified class: `Pipeline`
- Background images without annotations
- AUV and sensor telemetry in CSV form

### Important dataset characteristics

- LF images: **2500 × 500 px**
- HF images: **5000 × 500 px**
- Images are wide acoustic frames rather than conventional square photographs.
- The dataset contains temporally correlated survey frames, so random frame-level splitting can lead to leakage if used carelessly.
- The original SubPipe dataset should be treated as **read-only**.

### Geolocation note

The verified local audit found **no global latitude/longitude GPS fields** in the dataset. Spatial information is available as local Cartesian AUV/navigation coordinates. NEREUS therefore does **not fabricate GPS coordinates** in its current MVP.

---

## Dashboard

The deployed Streamlit application provides an operator-facing workflow:

1. Select a sample sonar image or upload a custom sonar image.
2. Set the confidence threshold.
3. Run NEREUS inference.
4. Compare the original sonar image with the annotated detection result.
5. Review detected class, confidence, and bounding-box dimensions.
6. Download the inspection result as JSON or CSV.

The deployment runs inference on **CPU**, making the current MVP lightweight and easy to demonstrate without requiring a dedicated inference GPU.

---

## Technology Stack

### AI / Computer Vision
- **Python**
- **Ultralytics YOLOv8n**
- **OpenCV**
- **NumPy**

### Data / Reporting
- **Pandas**
- **JSON**
- **CSV**

### Application
- **Streamlit**

---

## Example Detection Flow

```text
Raw Side-Scan Sonar
        │
        ▼
   NEREUS / YOLOv8n
        │
        ▼
 Pipeline + Confidence
        │
        ▼
Bounding Box + Dimensions
        │
        ▼
 Dashboard Visualization
        │
        ▼
   JSON / CSV Report
```

---

## Repository Structure

A typical NEREUS project layout is:

```text
NEREUS/
├── app.py
├── models/
│   └── best.pt
├── data/
├── sample_data/
├── src/
├── scripts/
├── notebooks/
├── configs/
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

The exact contents may evolve as the project is developed.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/MukulN7/nereus-sih26057.git
cd nereus-sih26057
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The application should open in the local browser.

---

## Inference Output

For each detected object, NEREUS can expose structured fields including:

```json
{
  "class_name": "Man-Made Anomaly",
  "class_id": 0,
  "confidence": 0.81,
  "confidence_percent": 81.0,
  "bbox": {
    "x1": 0,
    "y1": 0,
    "x2": 0,
    "y2": 0,
    "width": 0,
    "height": 0
  }
}
```

The deployment currently presents the learned class as a user-facing **Man-Made Anomaly** label, while the underlying SubPipe training annotation is the verified `Pipeline` class.

---

## Design & Engineering Principles

NEREUS was developed around a few practical principles:

- **Use real underwater data.**
- **Keep the model lightweight.**
- **Make predictions measurable.**
- **Keep outputs structured and traceable.**
- **Avoid unsupported geolocation claims.**
- **Prefer an operational MVP over unnecessary architectural complexity.**
- **Keep the source dataset separate and read-only.**

---

## Limitations

The current MVP is intentionally focused.

- The trained model currently covers **one class: Pipeline**.
- Global GPS latitude/longitude are not available in the verified SubPipe metadata.
- The current deployment is an AI-assisted inspection tool rather than a fully autonomous underwater decision system.
- Generalization to other sonar sensors, environments, and anomaly classes requires additional data and evaluation.
- Advanced temporal/sequence-aware detection and broader anomaly taxonomies are future extensions.

---

## Future Scope

Potential extensions include:

- Additional Side-Scan Sonar datasets
- Multi-class underwater anomaly detection
- Advanced sonar preprocessing and noise handling
- Temporal / sequence-aware detection across adjacent sonar frames
- Stronger false-positive analysis and filtering
- AUV mission-log integration for richer localization
- Edge deployment optimization for onboard marine systems
- Broader marine debris and infrastructure classes

---

## Smart India Hackathon Context

**Competition:** Smart India Hackathon 2026  
**Problem Statement:** PS26057  
**Theme:** Renewable / Sustainable Energy  
**Category:** Software  
**Team:** Sarvagya

NEREUS was developed as an end-to-end prototype aligned with the PS requirements around Side-Scan Sonar ingestion, AI-based object detection, confidence scoring, structured anomaly reporting, and a user-facing dashboard.

---

## References

1. **SubPipe — A Submarine Pipeline Inspection Dataset for Segmentation and Visual-Inertial Localization**  
   Zenodo: https://zenodo.org/records/10808161  
   DOI: `10.5281/zenodo.10808161`

2. **Ultralytics YOLO documentation**  
   https://docs.ultralytics.com/

3. **Smart India Hackathon 2026 — PS26057**  
   AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery

---

## Project Status

**Working MVP — trained, evaluated, and deployed.**

Live deployment: https://sarvagya-nereus-sih26057.streamlit.app/

---

## Team

**Sarvagya**  
Smart India Hackathon 2026 — PS26057
