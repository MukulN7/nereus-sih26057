"""
Nereus - Underwater Sonar Anomaly Detection
Operational Screening & Inspection Dashboard

This dashboard provides an operational interface for Side-Scan Sonar (SSS) imagery:
1. Single Image Mode: Inspect and analyze individual sonar frames with dual-view visualization.
2. Survey Folder Mode: Automated batch screening of entire sonar survey directories, anomaly ranking,
   targeted flagged-frame review, and consolidated JSON/CSV survey report generation.
"""

import io
import json
import time
import zipfile
from datetime import datetime
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

SUPPORTED_EXTENSIONS = {".pbm", ".bpm", ".ppm", ".jpg", ".jpeg", ".png", ".bmp"}

# Page setup with professional styling
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
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 4px;
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
        # Map class display label to standardized operational term
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
    Supports Netpbm (.pbm / .bpm / .ppm) binary P6 format as well as PNG, JPG, BMP.
    """
    try:
        np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core Inference Helpers
# ---------------------------------------------------------------------------
def extract_detections(result, conf_threshold: float) -> List[Dict[str, Any]]:
    """Extracts structured bounding box and confidence records from YOLO prediction."""
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i].item())
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            detections.append(
                {
                    "class_name": "Man-Made Anomaly",
                    "confidence": round(conf, 4),
                    "confidence_percent": round(conf * 100, 2),
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
    return detections


def run_nereus_inference(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Executes model inference, generates annotated RGB image, and returns structured detections.
    """
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

    plotted_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    detections = extract_detections(result, conf_threshold)

    return annotated_rgb, detections


# ---------------------------------------------------------------------------
# Consolidated Report Generators
# ---------------------------------------------------------------------------
def build_consolidated_csv(
    survey_id: str,
    images_data: List[Dict[str, Any]],
) -> bytes:
    """Generates consolidated CSV report with one row per detected anomaly across the survey."""
    rows = []
    for img_info in images_data:
        filename = img_info.get("image_filename", "")
        detections = img_info.get("detections", [])
        for idx, det in enumerate(detections, start=1):
            bbox = det.get("bbox", {})
            rows.append(
                {
                    "survey_id": survey_id,
                    "image_filename": filename,
                    "detection_index": idx,
                    "class_name": det.get("class_name", "Man-Made Anomaly"),
                    "confidence": det.get("confidence", 0.0),
                    "confidence_percent": det.get("confidence_percent", 0.0),
                    "bbox_x1": bbox.get("x1", 0.0),
                    "bbox_y1": bbox.get("y1", 0.0),
                    "bbox_x2": bbox.get("x2", 0.0),
                    "bbox_y2": bbox.get("y2", 0.0),
                    "bbox_width": bbox.get("width", 0.0),
                    "bbox_height": bbox.get("height", 0.0),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "survey_id",
                "image_filename",
                "detection_index",
                "class_name",
                "confidence",
                "confidence_percent",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_width",
                "bbox_height",
            ]
        )

    return df.to_csv(index=False).encode("utf-8")


def build_consolidated_json(
    survey_summary: Dict[str, Any],
    images_data: List[Dict[str, Any]],
    errors_data: List[Dict[str, Any]],
) -> bytes:
    """Generates structured consolidated JSON report for the complete survey."""
    # Clean images list to strictly match the standardized JSON schema
    schema_images = []
    for item in images_data:
        schema_images.append(
            {
                "image_filename": item["image_filename"],
                "detection_count": item["detection_count"],
                "detections": item["detections"],
            }
        )

    report = {
        "survey_summary": survey_summary,
        "images": schema_images,
        "errors": errors_data,
    }
    return json.dumps(report, indent=2).encode("utf-8")


def build_flagged_images_zip(
    model: YOLO,
    flagged_images: List[Dict[str, Any]],
    file_bytes_map: Dict[str, bytes],
    conf_threshold: float,
) -> bytes:
    """Generates an in-memory ZIP archive containing annotated images of all flagged anomalies."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in flagged_images:
            filename = item["image_filename"]
            raw_bytes = file_bytes_map.get(filename)
            if raw_bytes is None:
                continue

            img_bgr = decode_sonar_bytes(raw_bytes)
            if img_bgr is None:
                continue

            annotated_rgb, _ = run_nereus_inference(
                model=model,
                img_bgr=img_bgr,
                conf_threshold=conf_threshold,
            )
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            success, enc_bytes = cv2.imencode(".jpg", annotated_bgr)
            if success:
                out_name = f"flagged_{Path(filename).stem}.jpg"
                zip_file.writestr(out_name, enc_bytes.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ---------------------------------------------------------------------------
# Workflow: Single Image Mode
# ---------------------------------------------------------------------------
def render_single_image_mode(model: YOLO):
    """Renders the original single-image operational inspection workflow."""
    st.sidebar.markdown("### Controls")

    input_source = st.sidebar.radio(
        "Input Source",
        options=["Sample Test Image", "Upload Custom Image"],
        index=0,
    )

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

    if input_source == "Sample Test Image":
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

    if raw_bytes is None:
        if input_source == "Upload Custom Image":
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

    try:
        annotated_rgb, detections = run_nereus_inference(
            model=model,
            img_bgr=img_bgr,
            conf_threshold=conf_threshold,
        )
    except Exception as exc:
        st.error(f"Inference error during detection: {exc}")
        return

    st.markdown('<div class="section-title">Sonar Imagery</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Original Sonar Image** ({w} x {h} px)")
        st.image(orig_rgb, use_container_width=True)

    with col_right:
        st.markdown(f"**Nereus Detection Result** (Threshold: {int(conf_threshold * 100)}%)")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)
    num_detections = len(detections)

    if num_detections == 0:
        st.info("No man-made anomaly detected above the selected confidence threshold.")
    else:
        table_rows = []
        for det in detections:
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

    st.markdown('<div class="section-title">Inspection Reports</div>', unsafe_allow_html=True)

    report_json = {
        "image_filename": active_filename,
        "image_dimensions": {"width": w, "height": h},
        "confidence_threshold": conf_threshold,
        "detection_count": num_detections,
        "detections": detections,
    }
    json_bytes = json.dumps(report_json, indent=2).encode("utf-8")

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


# ---------------------------------------------------------------------------
# Workflow: Survey Folder Mode (Batch Screening)
# ---------------------------------------------------------------------------
def render_survey_folder_mode(model: YOLO):
    """Renders the automated survey directory batch screening workflow."""
    st.sidebar.markdown("### Controls")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
    )

    st.markdown('<div class="section-title">Survey Ingestion</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a folder of Side-Scan Sonar images. Nereus will automatically screen all images, "
        "rank detected anomalies, and prepare consolidated inspection reports."
    )

    uploaded_files = st.file_uploader(
        "Select Survey Folder",
        accept_multiple_files="directory",
        type=["pbm", "bpm", "ppm", "jpg", "jpeg", "png", "bmp"],
        help="Select a directory containing sonar images to screen.",
    )

    if not uploaded_files:
        st.info("Select a sonar survey directory using the upload area above to begin batch screening.")
        return

    total_files = len(uploaded_files)
    st.write(f"Discovered **{total_files}** file(s) for survey screening.")

    # Survey run button
    btn_analyze = st.button("Analyze Survey", type="primary")

    if btn_analyze:
        survey_id = f"SURVEY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()

        processed_images = []
        errors = []
        flagged_images = []
        file_bytes_map = {}
        total_detections_count = 0

        # Sequential low-memory batch screening
        for idx, file_obj in enumerate(uploaded_files):
            filename = file_obj.name
            ext = Path(filename).suffix.lower()

            status_placeholder.text(
                f"Processing: {idx + 1} / {total_files} | Anomalies Found: {total_detections_count} | Current: {filename}"
            )

            # Validate extension
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append(
                    {
                        "image_filename": filename,
                        "error": f"Unsupported file extension '{ext}'",
                    }
                )
                progress_bar.progress((idx + 1) / total_files)
                continue

            try:
                file_bytes = file_obj.getvalue()
                img_bgr = decode_sonar_bytes(file_bytes)
            except Exception as read_err:
                errors.append(
                    {
                        "image_filename": filename,
                        "error": f"Read/decode error: {read_err}",
                    }
                )
                progress_bar.progress((idx + 1) / total_files)
                continue

            if img_bgr is None:
                errors.append(
                    {
                        "image_filename": filename,
                        "error": "Failed to decode image bytes into valid sonar array.",
                    }
                )
                progress_bar.progress((idx + 1) / total_files)
                continue

            try:
                results = model.predict(
                    source=img_bgr,
                    conf=conf_threshold,
                    device="cpu",
                    verbose=False,
                )
                result = results[0]
                detections = extract_detections(result, conf_threshold)
            except Exception as inf_err:
                errors.append(
                    {
                        "image_filename": filename,
                        "error": f"Inference execution failure: {inf_err}",
                    }
                )
                progress_bar.progress((idx + 1) / total_files)
                continue

            num_dets = len(detections)
            total_detections_count += num_dets
            highest_conf = max([d["confidence"] for d in detections]) if num_dets > 0 else 0.0

            img_record = {
                "image_filename": filename,
                "detection_count": num_dets,
                "detections": detections,
                "highest_confidence": highest_conf,
            }
            processed_images.append(img_record)

            if num_dets > 0:
                flagged_images.append(img_record)
                file_bytes_map[filename] = file_bytes

            progress_bar.progress((idx + 1) / total_files)

        status_placeholder.text(
            f"Screening complete: {len(processed_images)} processed, {len(flagged_images)} flagged with anomalies, {len(errors)} skipped/errors."
        )

        num_processed = len(processed_images)
        num_with_anomalies = len(flagged_images)
        num_without_anomalies = num_processed - num_with_anomalies
        num_skipped = len(errors)

        all_confs = [
            d["confidence"]
            for img in processed_images
            for d in img["detections"]
        ]
        highest_survey_conf = max(all_confs) if all_confs else 0.0
        avg_survey_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

        survey_summary = {
            "total_images": total_files,
            "processed_images": num_processed,
            "images_with_anomalies": num_with_anomalies,
            "images_without_anomalies": num_without_anomalies,
            "skipped_images": num_skipped,
            "total_detections": total_detections_count,
            "confidence_threshold": conf_threshold,
            "highest_confidence": round(highest_survey_conf, 4),
            "average_confidence": round(avg_survey_conf, 4),
        }

        # Store survey state
        st.session_state["survey_results"] = {
            "survey_id": survey_id,
            "survey_summary": survey_summary,
            "processed_images": processed_images,
            "flagged_images": flagged_images,
            "errors": errors,
            "file_bytes_map": file_bytes_map,
            "conf_threshold": conf_threshold,
        }

    # Render survey results from session state if available
    survey_data = st.session_state.get("survey_results")
    if survey_data is None:
        return

    summary = survey_data["survey_summary"]
    flagged_images = survey_data["flagged_images"]
    processed_images = survey_data["processed_images"]
    errors = survey_data["errors"]
    file_bytes_map = survey_data["file_bytes_map"]
    active_conf = survey_data["conf_threshold"]
    survey_id = survey_data["survey_id"]

    st.markdown("---")
    st.markdown('<div class="section-title">Survey Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{summary["processed_images"]}</div><div class="metric-label">Images Processed</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{summary["images_with_anomalies"]}</div><div class="metric-label">Anomalous Images</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{summary["total_detections"]}</div><div class="metric-label">Total Detections</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{summary["images_without_anomalies"]}</div><div class="metric-label">Clean Images</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{summary["skipped_images"]}</div><div class="metric-label">Skipped / Errors</div></div>',
            unsafe_allow_html=True,
        )

    if summary["total_detections"] > 0:
        st.caption(
            f"Highest Detection Confidence: **{summary['highest_confidence'] * 100:.2f}%** | "
            f"Average Detection Confidence: **{summary['average_confidence'] * 100:.2f}%**"
        )

    # Flagged Anomalies Table
    st.markdown('<div class="section-title">Flagged Anomalies</div>', unsafe_allow_html=True)

    if not flagged_images:
        st.info("No man-made anomalies detected above the confidence threshold in this survey.")
    else:
        flagged_table_rows = []
        for img in flagged_images:
            flagged_table_rows.append(
                {
                    "Image": img["image_filename"],
                    "Anomalies": img["detection_count"],
                    "Highest Confidence": f"{img['highest_confidence'] * 100:.2f}%",
                    "Total Detections": img["detection_count"],
                    "_conf_sort": img["highest_confidence"],
                }
            )

        df_flagged = pd.DataFrame(flagged_table_rows)
        df_flagged = df_flagged.sort_values(by="_conf_sort", ascending=False).drop(columns=["_conf_sort"])
        st.dataframe(df_flagged, use_container_width=True, hide_index=True)

    # Flagged Image Inspection
    if flagged_images:
        st.markdown('<div class="section-title">Flagged Image Review</div>', unsafe_allow_html=True)
        flagged_names = [row["image_filename"] for row in sorted(flagged_images, key=lambda x: x["highest_confidence"], reverse=True)]

        selected_image_name = st.selectbox(
            "Select Flagged Image to Inspect",
            options=flagged_names,
            index=0,
        )

        selected_bytes = file_bytes_map.get(selected_image_name)
        if selected_bytes:
            img_bgr = decode_sonar_bytes(selected_bytes)
            if img_bgr is not None:
                h, w, _ = img_bgr.shape
                orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                annotated_rgb, target_dets = run_nereus_inference(
                    model=model,
                    img_bgr=img_bgr,
                    conf_threshold=active_conf,
                )

                col_rev_left, col_rev_right = st.columns(2)
                with col_rev_left:
                    st.markdown(f"**Original Sonar Image** ({w} x {h} px)")
                    st.image(orig_rgb, use_container_width=True)

                with col_rev_right:
                    st.markdown(f"**Nereus Detection Result** (Threshold: {int(active_conf * 100)}%)")
                    st.image(annotated_rgb, use_container_width=True)

                if target_dets:
                    st.markdown("**Image Detections**")
                    det_rows = []
                    for det in target_dets:
                        det_rows.append(
                            {
                                "Anomaly": det["class_name"],
                                "Confidence": f"{det['confidence_percent']:.2f}%",
                                "Bounding Box [x1, y1, x2, y2]": f"[{det['bbox']['x1']}, {det['bbox']['y1']}, {det['bbox']['x2']}, {det['bbox']['y2']}]",
                                "Width (px)": det["bbox"]["width"],
                                "Height (px)": det["bbox"]["height"],
                            }
                        )
                    st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

    # Consolidated Survey Reports
    st.markdown('<div class="section-title">Consolidated Survey Reports</div>', unsafe_allow_html=True)

    csv_bytes = build_consolidated_csv(survey_id, processed_images)
    json_bytes = build_consolidated_json(summary, processed_images, errors)

    col_down_json, col_down_csv, col_down_zip = st.columns(3)

    with col_down_json:
        st.download_button(
            label="Download Survey JSON",
            data=json_bytes,
            file_name=f"{survey_id.lower()}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_down_csv:
        st.download_button(
            label="Download Survey CSV",
            data=csv_bytes,
            file_name=f"{survey_id.lower()}_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_down_zip:
        if flagged_images:
            zip_bytes = build_flagged_images_zip(model, flagged_images, file_bytes_map, active_conf)
            st.download_button(
                label="Download Flagged Images (ZIP)",
                data=zip_bytes,
                file_name=f"{survey_id.lower()}_flagged_images.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button("Download Flagged Images (ZIP)", disabled=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Main Application Entrypoint
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

    # Top-level input mode selection
    st.sidebar.markdown("### Operational Mode")
    input_mode = st.sidebar.radio(
        "Input Mode",
        options=["Survey Folder", "Single Image"],
        index=0,
    )

    if input_mode == "Single Image":
        render_single_image_mode(model)
    else:
        render_survey_folder_mode(model)


if __name__ == "__main__":
    main()
