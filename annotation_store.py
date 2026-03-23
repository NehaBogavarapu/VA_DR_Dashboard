"""
annotation_store.py — Stores ophthalmologist annotations for retraining
=======================================================================
Each annotation contains:
  - image_id, correct DR grade, drawn shapes, brush colour type
  - These feed into the retraining pipeline as label corrections
    and attention masks.
"""

import json
import os
from datetime import datetime


class AnnotationStore:
    """In-memory annotation store with JSON persistence."""

    def __init__(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "va_export", "annotations.json"
            )
        self.save_path = save_path
        self._annotations = []
        self._load()

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self._annotations = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self._annotations, f, indent=2, default=str)

    def add(self, image_id, correct_grade, shapes, brush_color):
        annotation = {
            "image_id": image_id,
            "correct_grade": int(correct_grade),
            "shapes": shapes,
            "brush_color": brush_color,
            "annotation_type": self._color_to_type(brush_color),
            "timestamp": datetime.now().isoformat(),
        }

        existing_idx = next(
            (i for i, a in enumerate(self._annotations) if a["image_id"] == image_id),
            None,
        )
        if existing_idx is not None:
            self._annotations[existing_idx] = annotation
        else:
            self._annotations.append(annotation)

        self._save()
        return annotation

    def get_all(self):
        return self._annotations

    def get_by_image(self, image_id):
        for ann in self._annotations:
            if ann["image_id"] == image_id:
                return ann
        return None

    def get_corrected_labels(self):
        return {ann["image_id"]: ann["correct_grade"] for ann in self._annotations}

    def count(self):
        return len(self._annotations)

    def clear(self):
        self._annotations = []
        self._save()

    @staticmethod
    def _color_to_type(color):
        if "255,0,0" in color or "255, 0, 0" in color:
            return "lesion"
        elif "0,0,255" in color or "0, 0, 255" in color:
            return "artefact"
        elif "0,255,0" in color or "0, 255, 0" in color:
            return "normal"
        return "unknown"


def shapes_to_binary_mask(shapes, width, height):
    """Convert Plotly drawn shapes to binary masks for retraining.
    Returns dict with lesion_mask, artefact_mask, normal_mask."""
    import numpy as np

    masks = {
        "lesion_mask": np.zeros((height, width), dtype=bool),
        "artefact_mask": np.zeros((height, width), dtype=bool),
        "normal_mask": np.zeros((height, width), dtype=bool),
    }

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        fill_color = shape.get("fillcolor", "")
        if "255,0,0" in fill_color:
            target = "lesion_mask"
        elif "0,0,255" in fill_color:
            target = "artefact_mask"
        elif "0,255,0" in fill_color:
            target = "normal_mask"
        else:
            continue

        shape_type = shape.get("type", "")

        if shape_type == "rect":
            x0, y0 = int(shape["x0"]), int(shape["y0"])
            x1, y1 = int(shape["x1"]), int(shape["y1"])
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            masks[target][y0:y1, x0:x1] = True

        elif shape_type == "circle":
            cx = (shape["x0"] + shape["x1"]) / 2
            cy = (shape["y0"] + shape["y1"]) / 2
            rx = abs(shape["x1"] - shape["x0"]) / 2
            ry = abs(shape["y1"] - shape["y0"]) / 2
            yy, xx = np.ogrid[:height, :width]
            ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
            masks[target][ellipse <= 1] = True

        elif shape_type == "path":
            _rasterise_path(shape.get("path", ""), masks[target], width, height)

    return masks


def _rasterise_path(svg_path, mask, width, height):
    import re
    coords = re.findall(r"[ML]\s*([\d.]+)[,\s]+([\d.]+)", svg_path)
    if len(coords) < 3:
        return
    try:
        from PIL import Image, ImageDraw
        polygon = [(float(x), float(y)) for x, y in coords]
        temp = Image.new("L", (width, height), 0)
        ImageDraw.Draw(temp).polygon(polygon, fill=255)
        import numpy as np
        mask |= np.array(temp) > 0
    except ImportError:
        import numpy as np
        xs = [float(x) for x, y in coords]
        ys = [float(y) for x, y in coords]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        mask[y0:y1, x0:x1] = True
