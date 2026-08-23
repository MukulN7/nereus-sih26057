"""
Nereus - Underwater Sonar Anomaly Detection
Operational MVP Dashboard for SIH PS26057

This dashboard provides an operational interface for ingesting Side-Scan Sonar (SSS)
imagery, performing real-time inference using a trained detector on CPU, visualizing
identified man-made anomalies, and generating structured JSON and CSV inspection reports.
"""

import io
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Project Paths & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
FALLBACK_SAMPLE_DIR = PROJECT_ROOT / "data" / "images" / "test"

# Page setup with professional sans-serif styling
st.set_page_config(
    page_title="Nereus - Sonar Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for clean typography and restrained UI elements
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    .stDataFrame {
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_active_sample_dir() -> Path:
    """Returns the sample data directory, preferring sample_data/ for deployment."""
    if SAMPLE_DIR.exists() and any(SAMPLE_DIR.iterdir()):
        return SAMPLE_DIR
    if FALLBACK_SAMPLE_DIR.exists() and any(FALLBACK_SAMPLE_DIR.iterdir()):
        return FALLBACK_SAMPLE_DIR
    return SAMPLE_DIR


# ---------------------------------------------------------------------------
# Model Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detection_model(model_path_str: str) -> Optional[YOLO]:
    """Loads and caches the detector on CPU to avoid reloading on user interactions."""
    path = Path(model_path_str)
    if not path.exists():
        return None
    try:
        model = YOLO(str(path))
        # Map class display label to generic product term
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = {0: "Man-Made Anomaly"}
        return model
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image Decoding (Netpbm P6 / PBM / Standard formats)
# ---------------------------------------------------------------------------
def decode_sonar_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes into a BGR numpy array using OpenCV.
    Supports Netpbm (.pbm / .bpm / .ppm) as well as PNG, JPG, and BMP formats.
    """
    try:
        np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core Inference Function
# ---------------------------------------------------------------------------
def run_nereus_inference(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Executes model inference, annotates detections, and extracts structured data.
    """
    # Ensure display names map to generic product term
    if hasattr(model, "model") and hasattr(model.model, "names"):
        model.model.names = {0: "Man-Made Anomaly"}

    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    result.names = {0: "Man-Made Anomaly"}

    # Generate annotated image in RGB
    plotted_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

            detections.append(
                {
                    "class_name": "Man-Made Anomaly",
                    "class_id": cls_id,
                    "confidence": round(confidence, 4),
                    "confidence_percent": round(confidence * 100, 2),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2),
                    },
                }
            )

    return annotated_rgb, detections


# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">Nereus</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Underwater Sonar Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    # Validate model availability
    if not MODEL_PATH.exists():
        st.error(
            f"Trained model not found at {MODEL_PATH}. Please verify the model file path."
        )
        return

    model = load_detection_model(str(MODEL_PATH))
    if model is None:
        st.error("Failed to initialize the detection model. Please check the model file integrity.")
        return

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### Controls")

    input_mode = st.sidebar.radio(
        "Input Source",
        options=["Sample Test Image", "Upload Custom Image"],
        index=0,
    )

    # Collect available sample test images from active sample directory
    active_sample_dir = get_active_sample_dir()
    sample_options = []
    if active_sample_dir.exists():
        sample_options = sorted(
            [p.name for p in active_sample_dir.glob("*.jpg")] +
            [p.name for p in active_sample_dir.glob("*.pbm")] +
            [p.name for p in active_sample_dir.glob("*.bpm")] +
            [p.name for p in active_sample_dir.glob("*.png")] +
            [p.name for p in active_sample_dir.glob("*.jpeg")]
        )

    active_filename = ""
    raw_bytes = None

    if input_mode == "Sample Test Image":
        if not sample_options:
            st.sidebar.warning("No sample images found in sample directory.")
            return

        selected_sample = st.sidebar.selectbox(
            "Select Sample Image",
            options=sample_options,
            index=0,
        )
        active_filename = selected_sample
        sample_file_path = active_sample_dir / selected_sample
        if sample_file_path.exists():
            raw_bytes = sample_file_path.read_bytes()

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Sonar Image",
            type=["pbm", "bpm", "ppm", "png", "jpg", "jpeg", "bmp"],
            help="Supports Netpbm (.pbm, .ppm) and standard image formats.",
        )
        if uploaded_file is not None:
            active_filename = uploaded_file.name
            raw_bytes = uploaded_file.getvalue()

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
    )

    # -----------------------------------------------------------------------
    # Input Validation & Image Decoding
    # -----------------------------------------------------------------------
    if raw_bytes is None:
        if input_mode == "Upload Custom Image":
            st.info("Upload a Side-Scan Sonar image using the sidebar to begin analysis.")
        else:
            st.info("Select a sample image from the sidebar to begin analysis.")
        return

    img_bgr = decode_sonar_bytes(raw_bytes)
    if img_bgr is None:
        st.error(f"Unable to decode image file '{active_filename}'. Please verify the file format.")
        return

    h, w, _ = img_bgr.shape
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------------------------
    # Run Inference
    # -----------------------------------------------------------------------
    try:
        annotated_rgb, detections = run_nereus_inference(
            model=model,
            img_bgr=img_bgr,
            conf_threshold=conf_threshold,
        )
    except Exception as exc:
        st.error(f"Inference error during detection: {exc}")
        return

    # -----------------------------------------------------------------------
    # Visual Output Section (Two Columns)
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">Sonar Imagery</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Original Sonar Image** ({w} x {h} px)")
        st.image(orig_rgb, use_container_width=True)

    with col_right:
        st.markdown(f"**Nereus Detection Result** (Threshold: {int(conf_threshold * 100)}%)")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Detection Summary
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)

    num_detections = len(detections)

    if num_detections == 0:
        st.info("No man-made anomaly detected above the selected confidence threshold.")
    else:
        table_rows = []
        for idx, det in enumerate(detections, start=1):
            table_rows.append(
                {
                    "Anomaly": det["class_name"],
                    "Confidence": f"{det['confidence_percent']:.2f}%",
                    "Bounding Box [x1, y1, x2, y2]": f"[{det['bbox']['x1']}, {det['bbox']['y1']}, {det['bbox']['x2']}, {det['bbox']['y2']}]",
                    "Width (px)": det["bbox"]["width"],
                    "Height (px)": det["bbox"]["height"],
                }
            )

        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Report Export Section
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">Inspection Reports</div>', unsafe_allow_html=True)

    # Structured JSON report
    report_json = {
        "image_filename": active_filename,
        "image_dimensions": {"width": w, "height": h},
        "confidence_threshold": conf_threshold,
        "detection_count": num_detections,
        "detections": detections,
    }
    json_bytes = json.dumps(report_json, indent=2).encode("utf-8")

    # Structured CSV report
    csv_rows = []
    if num_detections > 0:
        for idx, det in enumerate(detections, start=1):
            csv_rows.append(
                {
                    "image_filename": active_filename,
                    "detection_index": idx,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "confidence_percent": det["confidence_percent"],
                    "confidence_threshold": conf_threshold,
                    "bbox_x1": det["bbox"]["x1"],
                    "bbox_y1": det["bbox"]["y1"],
                    "bbox_x2": det["bbox"]["x2"],
                    "bbox_y2": det["bbox"]["y2"],
                    "bbox_width": det["bbox"]["width"],
                    "bbox_height": det["bbox"]["height"],
                }
            )
    else:
        csv_rows.append(
            {
                "image_filename": active_filename,
                "detection_index": 0,
                "class_name": "No Detection",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "confidence_threshold": conf_threshold,
                "bbox_x1": 0.0,
                "bbox_y1": 0.0,
                "bbox_x2": 0.0,
                "bbox_y2": 0.0,
                "bbox_width": 0.0,
                "bbox_height": 0.0,
            }
        )

    df_csv = pd.DataFrame(csv_rows)
    csv_bytes = df_csv.to_csv(index=False).encode("utf-8")

    col_btn_json, col_btn_csv = st.columns(2)
    with col_btn_json:
        st.download_button(
            label="Download JSON Report",
            data=json_bytes,
            file_name=f"{Path(active_filename).stem}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_btn_csv:
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"{Path(active_filename).stem}_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
