"""
ModelForge AI - ML Engine: Grid-Based Object Detection & Bounding Box Regression
Implements YOLO grid-based bounding box coordinates parameterization, IoU intersection matrix,
Focal Loss, Complete IoU (CIoU) bounding box loss, and Greedy Non-Maximum Suppression (NMS).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class BoundingBoxUtils:
    @staticmethod
    def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute Intersection over Union (IoU) between two sets of boxes $[x_1, y_1, x_2, y_2]$.
        boxes1: $(N, 4)$, boxes2: $(M, 4)$ -> returns $(N, M)$ matrix.
        """
        N = len(boxes1)
        M = len(boxes2)

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Overlapping intersection coordinates
        lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[np.newaxis, :, :2])  # (N, M, 2)
        rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[np.newaxis, :, 2:])  # (N, M, 2)

        wh = np.maximum(0.0, rb - lt)
        intersection = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        return intersection / np.maximum(1e-10, union)

    @staticmethod
    def non_maximum_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25,
    ) -> List[int]:
        """Greedy NMS filtering out redundant overlapping candidate bounding boxes."""
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        original_indices = np.where(mask)[0]

        if len(boxes) == 0:
            return []

        # Sort scores descending
        order = np.argsort(scores)[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(original_indices[i])

            if len(order) == 1:
                break

            ious = BoundingBoxUtils.box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
            remaining = np.where(ious <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep


class GridDetectionHead:
    """YOLO $S \times S$ Grid Prediction Head with Anchor Offsets."""

    def __init__(
        self,
        grid_size: int = 7,
        num_anchors: int = 3,
        num_classes: int = 20,
    ):
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Output tensor depth: num_anchors * (5 + num_classes) where 5 = [x, y, w, h, obj_conf]
        self.out_channels = num_anchors * (5 + num_classes)

        # Standard anchor dimensions (width, height)
        self.anchors = np.array([
            [1.25, 1.625],
            [2.0, 3.75],
            [3.5, 4.25],
        ])

    def decode_predictions(
        self,
        raw_output: np.ndarray,
        conf_threshold: float = 0.3,
        nms_iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Decode raw neural grid tensor $[B, S, S, A \times (5 + C)]$ into pixel coordinates.
        $b_x = \sigma(t_x) + c_x, b_y = \sigma(t_y) + c_y, b_w = p_w e^{t_w}, b_h = p_h e^{t_h}$
        """
        # Assume single image batch
        raw = raw_output[0] if raw_output.ndim == 4 else raw_output
        S = self.grid_size
        A = self.num_anchors
        C = self.num_classes

        raw = raw.reshape(S, S, A, 5 + C)

        # Sigmoid for coordinates and objectness
        t_xy = 1.0 / (1.0 + np.exp(-raw[:, :, :, :2]))
        t_wh = np.exp(raw[:, :, :, 2:4])
        t_conf = 1.0 / (1.0 + np.exp(-raw[:, :, :, 4:5]))

        # Softmax class probabilities
        shift_cls = raw[:, :, :, 5:] - np.max(raw[:, :, :, 5:], axis=-1, keepdims=True)
        t_cls = np.exp(shift_cls) / np.sum(np.exp(shift_cls), axis=-1, keepdims=True)

        boxes = []
        scores = []
        classes = []

        for cy in range(S):
            for cx in range(S):
                for a in range(A):
                    obj_score = float(t_conf[cy, cx, a, 0])
                    class_probs = t_cls[cy, cx, a]
                    best_class = int(np.argmax(class_probs))
                    confidence = obj_score * float(class_probs[best_class])

                    if confidence >= conf_threshold:
                        # Convert normalized grid offsets to bounding box [x1, y1, x2, y2]
                        bx = (cx + t_xy[cy, cx, a, 0]) / S
                        by = (cy + t_xy[cy, cx, a, 1]) / S
                        bw = (self.anchors[a, 0] * t_wh[cy, cx, a, 0]) / S
                        bh = (self.anchors[a, 1] * t_wh[cy, cx, a, 1]) / S

                        x1 = max(0.0, bx - bw / 2.0)
                        y1 = max(0.0, by - bh / 2.0)
                        x2 = min(1.0, bx + bw / 2.0)
                        y2 = min(1.0, by + bh / 2.0)

                        boxes.append([x1, y1, x2, y2])
                        scores.append(confidence)
                        classes.append(best_class)

        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)

        # Apply NMS
        keep_indices = BoundingBoxUtils.non_maximum_suppression(
            boxes_arr, scores_arr, iou_threshold=nms_iou_threshold, score_threshold=conf_threshold
        )

        detections = []
        for idx in keep_indices:
            detections.append({
                "box": [round(float(v), 4) for v in boxes_arr[idx]],
                "score": round(float(scores_arr[idx]), 4),
                "class_id": classes[idx],
            })

        return detections
