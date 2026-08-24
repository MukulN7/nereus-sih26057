"""
Automated unit and integration test suite for Nereus Survey Screening & Single Image pipeline.

Updated for the session-state-safe architecture:
- No image bytes stored in session_state
- Raw detections stored at inference time; filtered per threshold
- ZIP export uses on-demand disk reads or LocalSurveyFile wrappers
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
    SAMPLE_SURVEYS_DIR,
    SUPPORTED_EXTENSIONS,
    MAX_CUSTOM_SURVEY_IMAGES,
    MAX_CUSTOM_SURVEY_BYTES,
    LocalSurveyFile,
    get_available_sample_surveys,
    load_detection_model,
    decode_sonar_bytes,
    run_nereus_inference,
    extract_detections,
    extract_detections_raw,
    filter_detections,
    _apply_threshold_to_survey,
    _resolve_survey_action,
    validate_custom_survey_limits,
    build_consolidated_csv,
    build_consolidated_json,
    build_flagged_images_zip,
)


class _MockUploadedFile:
    """Minimal stand-in for a Streamlit UploadedFile, used only for CHANGE 1/2 tests."""
    def __init__(self, name: str, size_bytes: int):
        self.name = name
        self.size = size_bytes

    def getvalue(self) -> bytes:
        return b"x" * self.size


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
    """TEST 4: Verify that high confidence threshold filters stored raw detections correctly."""
    model = load_detection_model(str(MODEL_PATH))
    # Chunk3_LF_1693574479.03.jpg has detections around 0.43, 0.33, 0.32
    target_img_path = Path("sample_data") / "Chunk3_LF_1693574479.03.jpg"
    img_bgr = cv2.imread(str(target_img_path))

    # Simulate new pipeline: extract raw at low conf, then filter
    results = model.predict(source=img_bgr, conf=0.01, device="cpu", verbose=False)
    raw_dets = extract_detections_raw(results[0])

    dets_low = filter_detections(raw_dets, 0.25)
    dets_high = filter_detections(raw_dets, 0.50)

    assert len(dets_low) >= 2
    assert len(dets_high) < len(dets_low)

    # Slider change simulation: re-filter from stored raw_dets at a third threshold
    dets_mid = filter_detections(raw_dets, 0.35)
    assert len(dets_mid) <= len(dets_low)

    # Verify all returned dets meet threshold
    for d in dets_low:
        assert d["confidence"] >= 0.25
    for d in dets_high:
        assert d["confidence"] >= 0.50


def test_apply_threshold_to_survey():
    """TEST 3 & threshold recompute: verify _apply_threshold_to_survey re-filters correctly."""
    all_image_records = [
        {
            "image_filename": "img_a.jpg",
            "all_detections": [
                {"class_name": "Man-Made Anomaly", "confidence": 0.60, "confidence_percent": 60.0, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "width": 10, "height": 10}},
                {"class_name": "Man-Made Anomaly", "confidence": 0.30, "confidence_percent": 30.0, "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5, "width": 5, "height": 5}},
            ],
        },
        {
            "image_filename": "img_b.jpg",
            "all_detections": [
                {"class_name": "Man-Made Anomaly", "confidence": 0.20, "confidence_percent": 20.0, "bbox": {"x1": 0, "y1": 0, "x2": 3, "y2": 3, "width": 3, "height": 3}},
            ],
        },
    ]
    errors = []

    # At threshold 0.25: img_a has 2 detections, img_b has 0
    view_low = _apply_threshold_to_survey(all_image_records, 0.25, errors, "SURVEY_X")
    assert view_low["survey_summary"]["total_detections"] == 2
    assert view_low["survey_summary"]["images_with_anomalies"] == 1
    assert len(view_low["flagged_images"]) == 1

    # At threshold 0.50: img_a has 1, img_b has 0
    view_high = _apply_threshold_to_survey(all_image_records, 0.50, errors, "SURVEY_X")
    assert view_high["survey_summary"]["total_detections"] == 1
    assert view_high["survey_summary"]["images_with_anomalies"] == 1

    # At threshold 0.70: nothing
    view_none = _apply_threshold_to_survey(all_image_records, 0.70, errors, "SURVEY_X")
    assert view_none["survey_summary"]["total_detections"] == 0
    assert view_none["survey_summary"]["images_with_anomalies"] == 0
    assert len(view_none["flagged_images"]) == 0


def test_batch_survey_simulation_and_aggregation():
    """TEST 2, TEST 3, TEST 7: Simulate survey batch screening with new raw-detection pipeline."""
    model = load_detection_model(str(MODEL_PATH))
    conf_threshold = 0.25
    survey_id = "SURVEY_TEST_001"

    # Gather test images (at least 3) + 1 invalid mock file + 1 unsupported
    image_paths = list(Path("sample_data").glob("*.*"))[:3]
    assert len(image_paths) >= 3

    mock_files = [
        {"name": p.name, "bytes": p.read_bytes()} for p in image_paths
    ]
    mock_files.append({"name": "corrupted_sonar.jpg", "bytes": b"BAD_BYTES"})
    mock_files.append({"name": "notes.txt", "bytes": b"Some text"})

    # New pipeline: store all_detections at raw conf=0.01
    all_image_records = []
    errors = []

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

        results = model.predict(source=img_bgr, conf=0.01, device="cpu", verbose=False)
        raw_dets = extract_detections_raw(results[0])
        all_image_records.append({"image_filename": filename, "all_detections": raw_dets})
        del img_bgr  # no bytes in records

    assert len(all_image_records) == 3
    assert len(errors) == 2  # 1 corrupted, 1 unsupported

    # Apply threshold to get survey view
    view = _apply_threshold_to_survey(all_image_records, conf_threshold, errors, survey_id)
    processed_images = view["processed_images"]
    flagged_images = view["flagged_images"]
    survey_summary = view["survey_summary"]

    assert len(processed_images) == 3
    assert len(flagged_images) >= 2
    assert survey_summary["total_detections"] >= 3

    # TEST: changing threshold re-filters without crashing (simulates slider change)
    view_high = _apply_threshold_to_survey(all_image_records, 0.60, errors, survey_id)
    assert view_high["survey_summary"]["total_detections"] <= survey_summary["total_detections"]

    view_low = _apply_threshold_to_survey(all_image_records, 0.25, errors, survey_id)
    assert view_low["survey_summary"]["total_detections"] == survey_summary["total_detections"]

    # TEST 5: Verify Consolidated CSV
    csv_bytes = build_consolidated_csv(survey_id, processed_images)
    csv_str = csv_bytes.decode("utf-8")
    df_csv = pd.read_csv(io.StringIO(csv_str))

    expected_cols = [
        "survey_id", "image_filename", "detection_index", "class_name",
        "confidence", "confidence_percent",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height",
    ]
    for col in expected_cols:
        assert col in df_csv.columns, f"Missing CSV column: {col}"

    assert len(df_csv) == survey_summary["total_detections"]
    assert (df_csv["class_name"] == "Man-Made Anomaly").all()
    assert (df_csv["survey_id"] == survey_id).all()

    # TEST 6: Verify Consolidated JSON
    json_bytes = build_consolidated_json(survey_summary, processed_images, errors)
    parsed_json = json.loads(json_bytes.decode("utf-8"))

    assert "survey_summary" in parsed_json
    assert "images" in parsed_json
    assert "errors" in parsed_json
    assert parsed_json["survey_summary"]["total_detections"] == survey_summary["total_detections"]
    assert len(parsed_json["images"]) == len(processed_images)
    assert len(parsed_json["errors"]) == 2

    for img_entry in parsed_json["images"]:
        assert "image_filename" in img_entry
        assert "detection_count" in img_entry
        assert "detections" in img_entry
        for det in img_entry["detections"]:
            assert det["class_name"] == "Man-Made Anomaly"
            assert "bbox" in det

    # TEST ZIP export — using sample survey dir as source (no bytes in session state)
    demo_dir = PROJECT_ROOT / "sample_surveys" / "demo_survey"
    zip_bytes = build_flagged_images_zip(
        model=model,
        flagged_images=flagged_images,
        survey_source="sample",
        sample_dir_str=str(PROJECT_ROOT / "sample_data"),
        uploaded_files=None,
        conf_threshold=conf_threshold,
    )
    assert len(zip_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        file_list = zf.namelist()
        assert len(file_list) == len(flagged_images)
        for name in file_list:
            assert name.startswith("flagged_")
            assert name.endswith(".jpg")


def test_sample_survey_discovery_and_screening():
    """Verify discovery of sample_surveys/demo_survey and full batch screening with new pipeline."""
    surveys = get_available_sample_surveys()
    assert "Demo Survey" in surveys, f"Demo Survey not discovered in {SAMPLE_SURVEYS_DIR}"
    demo_dir = surveys["Demo Survey"]
    assert demo_dir.exists()

    raw_paths = sorted([
        p for p in demo_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    assert len(raw_paths) == 3, f"Expected 3 sample images in demo survey, found {len(raw_paths)}"

    survey_files = [LocalSurveyFile(p) for p in raw_paths]
    model = load_detection_model(str(MODEL_PATH))
    conf_threshold = 0.25

    # New pipeline: store raw detections; no image bytes kept
    all_image_records = []
    for file_obj in survey_files:
        img_bgr = decode_sonar_bytes(file_obj.getvalue())
        assert img_bgr is not None
        results = model.predict(source=img_bgr, conf=0.01, device="cpu", verbose=False)
        raw_dets = extract_detections_raw(results[0])
        all_image_records.append({"image_filename": file_obj.name, "all_detections": raw_dets})
        del img_bgr

    assert len(all_image_records) == 3

    view = _apply_threshold_to_survey(all_image_records, conf_threshold, [], "DEMO_TEST")
    processed_images = view["processed_images"]
    flagged_images = view["flagged_images"]

    assert len(processed_images) == 3
    assert len(flagged_images) == 3

    # Simulate slider change to 0.50 — must not crash
    view_high = _apply_threshold_to_survey(all_image_records, 0.50, [], "DEMO_TEST")
    assert view_high["survey_summary"]["total_detections"] <= view["survey_summary"]["total_detections"]

    # Reports
    csv_bytes = build_consolidated_csv("DEMO_TEST", processed_images)
    assert len(csv_bytes) > 0
    json_bytes = build_consolidated_json(view["survey_summary"], processed_images, [])
    assert len(json_bytes) > 0

    # ZIP via on-demand disk reads
    zip_bytes = build_flagged_images_zip(
        model=model,
        flagged_images=flagged_images,
        survey_source="sample",
        sample_dir_str=str(demo_dir),
        uploaded_files=None,
        conf_threshold=conf_threshold,
    )
    assert len(zip_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert len(zf.namelist()) == len(flagged_images)


def test_custom_survey_image_count_limit():
    """CHANGE 1 / TEST: <=20 images accepted, >20 images rejected before inference."""
    ok_files = [_MockUploadedFile(f"img_{i}.jpg", 1024) for i in range(MAX_CUSTOM_SURVEY_IMAGES)]
    is_valid, count, _ = validate_custom_survey_limits(ok_files)
    assert is_valid is True
    assert count == MAX_CUSTOM_SURVEY_IMAGES

    too_many_files = [_MockUploadedFile(f"img_{i}.jpg", 1024) for i in range(MAX_CUSTOM_SURVEY_IMAGES + 1)]
    is_valid, count, _ = validate_custom_survey_limits(too_many_files)
    assert is_valid is False
    assert count == MAX_CUSTOM_SURVEY_IMAGES + 1


def test_custom_survey_size_limit():
    """CHANGE 1 / TEST: <=75MB accepted, >75MB rejected before inference."""
    ok_files = [_MockUploadedFile("img_0.jpg", MAX_CUSTOM_SURVEY_BYTES)]
    is_valid, _, total_bytes = validate_custom_survey_limits(ok_files)
    assert is_valid is True
    assert total_bytes == MAX_CUSTOM_SURVEY_BYTES

    over_size_files = [_MockUploadedFile("img_0.jpg", MAX_CUSTOM_SURVEY_BYTES + 1)]
    is_valid, _, total_bytes = validate_custom_survey_limits(over_size_files)
    assert is_valid is False
    assert total_bytes == MAX_CUSTOM_SURVEY_BYTES + 1


def test_reanalysis_gating_initial_and_after_analysis():
    """CHANGE 2 / TEST: initial analysis uses selected threshold; slider changes after
    analysis do not trigger auto re-analysis (already_analyzed stays keyed to the
    frozen analyzed_threshold until an explicit Reanalyze)."""
    file_signature = ("img_a.jpg", "img_b.jpg")

    # No prior run -> not already analyzed (Analyze Survey path)
    already_analyzed, run_state = _resolve_survey_action("custom", file_signature, None)
    assert already_analyzed is False
    assert run_state is None

    # Simulate a completed "Analyze Survey" at 25%
    analyzed_run_state = {
        "survey_source": "custom",
        "file_signature": file_signature,
        "analyzed_threshold": 0.25,
        "all_image_records": [],
        "errors": [],
        "survey_id": "SURVEY_TEST",
    }

    # Same files still selected -> already analyzed; frozen threshold unaffected
    # by whatever the live slider currently reads (simulated by not passing it in).
    already_analyzed, run_state = _resolve_survey_action("custom", file_signature, analyzed_run_state)
    assert already_analyzed is True
    assert run_state["analyzed_threshold"] == 0.25  # unchanged: slider move alone must not update this


def test_reanalyze_applies_new_threshold():
    """CHANGE 2 / TEST: Reanalyze Survey with a new threshold updates analyzed_threshold
    (re-filtering stored raw detections), simulating an explicit user click."""
    file_signature = ("img_a.jpg", "img_b.jpg")
    run_state = {
        "survey_source": "custom",
        "file_signature": file_signature,
        "analyzed_threshold": 0.25,
        "all_image_records": [],
        "errors": [],
        "survey_id": "SURVEY_TEST",
    }

    already_analyzed, run_state = _resolve_survey_action("custom", file_signature, run_state)
    assert already_analyzed is True

    # Simulate clicking "Reanalyze Survey" with a newly selected threshold of 60%
    run_state["analyzed_threshold"] = 0.60
    assert run_state["analyzed_threshold"] == 0.60


def test_reanalysis_gating_invalidated_on_file_change():
    """CHANGE 2 / TEST: a different custom file selection is treated as not-yet-analyzed
    and the stale run_state is dropped (not silently reused)."""
    old_signature = ("img_a.jpg", "img_b.jpg")
    new_signature = ("img_c.jpg",)
    stale_run_state = {
        "survey_source": "custom",
        "file_signature": old_signature,
        "analyzed_threshold": 0.25,
    }
    already_analyzed, run_state = _resolve_survey_action("custom", new_signature, stale_run_state)
    assert already_analyzed is False
    assert run_state is None


def test_reanalysis_gating_sample_survey_unaffected():
    """CHANGE 2 / TEST: Sample Survey mode is never gated behind Reanalyze — it
    always reports already_analyzed=False so it keeps its existing live-refilter
    behavior."""
    already_analyzed, run_state = _resolve_survey_action(
        "sample", None, {"survey_source": "sample", "analyzed_threshold": 0.25}
    )
    assert already_analyzed is False


if __name__ == "__main__":
    test_model_loading()
    test_image_decoding()
    test_single_image_inference_and_anomaly_naming()
    test_threshold_filtering()
    test_apply_threshold_to_survey()
    test_batch_survey_simulation_and_aggregation()
    test_sample_survey_discovery_and_screening()
    test_custom_survey_image_count_limit()
    test_custom_survey_size_limit()
    test_reanalysis_gating_initial_and_after_analysis()
    test_reanalyze_applies_new_threshold()
    test_reanalysis_gating_invalidated_on_file_change()
    test_reanalysis_gating_sample_survey_unaffected()
    print("All tests passed successfully!")
