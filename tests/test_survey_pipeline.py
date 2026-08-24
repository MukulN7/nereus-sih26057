"""
Automated unit and integration test suite for Nereus Survey Screening & Single Image pipeline.
"""

import io
import json
import zipfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from app import (
    MODEL_PATH,
    SUPPORTED_EXTENSIONS,
    load_detection_model,
    decode_sonar_bytes,
    run_nereus_inference,
    extract_detections,
    build_consolidated_csv,
    build_consolidated_json,
    build_flagged_images_zip,
)


def test_model_loading():
    """TEST 8: Verify model loads successfully and sets class name to 'Man-Made Anomaly'."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    model = load_detection_model(str(MODEL_PATH))
    assert model is not None, "Failed to load YOLO model"
    if hasattr(model, "model") and hasattr(model.model, "names"):
        assert model.model.names[0] == "Man-Made Anomaly"


def test_image_decoding():
    """TEST 4 / SUPPORTED FILES: Verify decoding of PBM and JPG files, and graceful error on bad bytes."""
    # Test valid JPG
    sample_jpg = next(Path("sample_data").glob("*.jpg"))
    raw_jpg = sample_jpg.read_bytes()
    decoded_jpg = decode_sonar_bytes(raw_jpg)
    assert decoded_jpg is not None
    assert decoded_jpg.ndim == 3

    # Test valid PBM (Netpbm P6)
    pbm_files = list(Path("data/images/test").glob("*.pbm"))
    if pbm_files:
        raw_pbm = pbm_files[0].read_bytes()
        decoded_pbm = decode_sonar_bytes(raw_pbm)
        assert decoded_pbm is not None
        assert decoded_pbm.shape[0] == 500
        assert decoded_pbm.shape[1] in (2500, 5000)

    # Test invalid / corrupted bytes
    corrupted_bytes = b"NOT_AN_IMAGE_DATA_CORRUPTED_BYTES"
    decoded_bad = decode_sonar_bytes(corrupted_bytes)
    assert decoded_bad is None


def test_single_image_inference_and_anomaly_naming():
    """TEST 1 & TEST 6: Verify single image inference produces 'Man-Made Anomaly' labels."""
    model = load_detection_model(str(MODEL_PATH))
    sample_jpg = next(Path("sample_data").glob("*.jpg"))
    raw_bytes = sample_jpg.read_bytes()
    img_bgr = decode_sonar_bytes(raw_bytes)

    annotated_rgb, detections = run_nereus_inference(model, img_bgr, conf_threshold=0.25)
    assert isinstance(annotated_rgb, np.ndarray)
    assert annotated_rgb.shape == img_bgr.shape

    for det in detections:
        assert det["class_name"] == "Man-Made Anomaly"
        assert "confidence" in det
        assert "confidence_percent" in det
        assert "bbox" in det
        assert all(k in det["bbox"] for k in ("x1", "y1", "x2", "y2", "width", "height"))


def test_threshold_filtering():
    """TEST 4: Verify that high confidence threshold filters out lower confidence detections."""
    model = load_detection_model(str(MODEL_PATH))
    # Chunk3_LF_1693574479.03.jpg has detections around 0.43, 0.33, 0.32
    target_img_path = Path("sample_data") / "Chunk3_LF_1693574479.03.jpg"
    img_bgr = cv2.imread(str(target_img_path))

    # Low threshold: 0.25 -> 3 detections
    _, dets_low = run_nereus_inference(model, img_bgr, conf_threshold=0.25)
    # High threshold: 0.50 -> 0 detections
    _, dets_high = run_nereus_inference(model, img_bgr, conf_threshold=0.50)

    assert len(dets_low) >= 2
    assert len(dets_high) < len(dets_low)


def test_batch_survey_simulation_and_aggregation():
    """TEST 2, TEST 3, TEST 7: Simulate survey batch screening with multiple images and corrupted file."""
    model = load_detection_model(str(MODEL_PATH))
    conf_threshold = 0.25
    survey_id = "SURVEY_TEST_001"

    # Gather test images (at least 3) + 1 invalid mock file
    image_paths = list(Path("sample_data").glob("*.*"))[:3]
    assert len(image_paths) >= 3

    mock_files = [
        {"name": p.name, "bytes": p.read_bytes()} for p in image_paths
    ]
    # Add an invalid corrupted file
    mock_files.append({"name": "corrupted_sonar.jpg", "bytes": b"BAD_BYTES"})
    # Add an unsupported extension file
    mock_files.append({"name": "notes.txt", "bytes": b"Some text"})

    processed_images = []
    errors = []
    flagged_images = []
    file_bytes_map = {}
    total_detections_count = 0

    for file_obj in mock_files:
        filename = file_obj["name"]
        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            errors.append({"image_filename": filename, "error": f"Unsupported extension {ext}"})
            continue

        img_bgr = decode_sonar_bytes(file_obj["bytes"])
        if img_bgr is None:
            errors.append({"image_filename": filename, "error": "Failed to decode image"})
            continue

        annotated_rgb, detections = run_nereus_inference(model, img_bgr, conf_threshold=conf_threshold)
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
            file_bytes_map[filename] = file_obj["bytes"]

    # Verify counts
    assert len(processed_images) == 3
    assert len(errors) == 2  # 1 corrupted, 1 unsupported
    assert len(flagged_images) >= 2
    assert total_detections_count >= 3

    # Survey Summary
    survey_summary = {
        "total_images": len(mock_files),
        "processed_images": len(processed_images),
        "images_with_anomalies": len(flagged_images),
        "images_without_anomalies": len(processed_images) - len(flagged_images),
        "skipped_images": len(errors),
        "total_detections": total_detections_count,
        "confidence_threshold": conf_threshold,
    }

    # TEST 5: Verify Consolidated CSV
    csv_bytes = build_consolidated_csv(survey_id, processed_images)
    csv_str = csv_bytes.decode("utf-8")
    df_csv = pd.read_csv(io.StringIO(csv_str))

    expected_cols = [
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
    for col in expected_cols:
        assert col in df_csv.columns, f"Missing CSV column: {col}"

    assert len(df_csv) == total_detections_count
    assert (df_csv["class_name"] == "Man-Made Anomaly").all()
    assert (df_csv["survey_id"] == survey_id).all()

    # TEST 6: Verify Consolidated JSON
    json_bytes = build_consolidated_json(survey_summary, processed_images, errors)
    parsed_json = json.loads(json_bytes.decode("utf-8"))

    assert "survey_summary" in parsed_json
    assert "images" in parsed_json
    assert "errors" in parsed_json
    assert parsed_json["survey_summary"]["total_detections"] == total_detections_count
    assert len(parsed_json["images"]) == len(processed_images)
    assert len(parsed_json["errors"]) == 2

    for img_entry in parsed_json["images"]:
        assert "image_filename" in img_entry
        assert "detection_count" in img_entry
        assert "detections" in img_entry
        for det in img_entry["detections"]:
            assert det["class_name"] == "Man-Made Anomaly"
            assert "bbox" in det

    # Test ZIP export
    zip_bytes = build_flagged_images_zip(model, flagged_images, file_bytes_map, conf_threshold)
    assert len(zip_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        file_list = zf.namelist()
        assert len(file_list) == len(flagged_images)
        for name in file_list:
            assert name.startswith("flagged_")
            assert name.endswith(".jpg")


if __name__ == "__main__":
    test_model_loading()
    test_image_decoding()
    test_single_image_inference_and_anomaly_naming()
    test_threshold_filtering()
    test_batch_survey_simulation_and_aggregation()
    print("All tests passed successfully!")
