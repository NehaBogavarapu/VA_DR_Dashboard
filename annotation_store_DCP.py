"""
annotation_store_DCP.py — Stores user annotations for retraining
=================================================================
Each annotation contains:
  - image_id, correct class, drawn shapes, brush colour type
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

    def add(self, image_id, correct_class, shapes, brush_color):
        annotation = {
            "image_id": image_id,
            "correct_class": int(correct_class),
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
        return {ann["image_id"]: ann["correct_class"] for ann in self._annotations}

    def count(self):
        return len(self._annotations)

    def clear(self):
        self._annotations = []
        self._save()

    @staticmethod
    def _color_to_type(color):
        if "255,0,0" in color or "255, 0, 0" in color:
            return "important_feature"
        elif "0,0,255" in color or "0, 0, 255" in color:
            return "artefact"
        elif "0,255,0" in color or "0, 255, 0" in color:
            return "background"
        return "unknown"
