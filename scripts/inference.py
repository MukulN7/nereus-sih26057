"""
AquaGuard MVP inference script.

Usage:
    python scripts/inference.py --source path\to\image.pbm
    python scripts/inference.py --source path\to\folder --conf 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# SubPipe images use P6 Netpbm data with a .pbm extension.
try:
    import ultralytics.data.base as ub
    import ultralytics.data.utils as uu
    ub.IMG_FORMATS.add("pbm")
    uu.IMG_FORMATS.add("pbm")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AquaGuard YOLO inference")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def get_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]

    if source.is_dir():
        extensions = {".pbm", ".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    raise FileNotFoundError(f"Source not found: {source}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    source = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    images = get_images(source)
    if not images:
        raise RuntimeError(f"No supported images found in: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            device=args.device,
            verbose=False,
            save=False,
        )

        result = results[0]

        annotated = result.plot()
        annotated_path = output_dir / f"{image_path.stem}.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        detections = []

        if result.boxes is not None:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                confidence = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                class_name = (
                    result.names.get(cls_id, str(cls_id))
                    if hasattr(result.names, "get")
                    else str(cls_id)
                )

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
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

        report = {
            "image": image_path.name,
            "model": str(model_path),
            "confidence_threshold": args.conf,
            "detections": detections,
        }

        report_path = output_dir / f"{image_path.stem}.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(
            f"{image_path.name}: "
            f"{len(detections)} detection(s) -> {annotated_path.name}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
