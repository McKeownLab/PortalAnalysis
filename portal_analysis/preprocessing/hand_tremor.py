"""
Hand tremor metrics from MediaPipe pose CSVs (both-still task).

Ported from BoothReports ``hand_tremor_processor.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class HandTremorProcessor:
    """
    Compute fingertip distances from a fixed centre and mean fingertip distance.

    Used for MDS-UPDRS hand tremor (both still); inference column:
    ``mean_fingertip_distance_from_center``.
    """

    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    FINGERTIP_LANDMARKS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    def __init__(self, center_coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.center_x, self.center_y, self.center_z = center_coordinates

    def _extract_landmark_point(self, row: pd.Series, idx: int) -> np.ndarray:
        return np.array([row[f"x_{idx}"], row[f"y_{idx}"], row[f"z_{idx}"]], dtype=float)

    def _calculate_distance_from_center(self, point: np.ndarray) -> float:
        center = np.array([self.center_x, self.center_y, self.center_z])
        return float(np.linalg.norm(point - center))

    def compute_tremor_metrics_from_row(self, row: pd.Series) -> Dict:
        metrics: Dict = {}
        fingertip_distances = []

        try:
            wrist_point = self._extract_landmark_point(row, self.WRIST)
            metrics["wrist_distance_from_center"] = self._calculate_distance_from_center(
                wrist_point
            )

            for landmark_idx, finger_name in zip(self.FINGERTIP_LANDMARKS, self.FINGERTIP_NAMES):
                try:
                    fingertip_point = self._extract_landmark_point(row, landmark_idx)
                    distance = self._calculate_distance_from_center(fingertip_point)
                    metrics[f"{finger_name}_tip_distance_from_center"] = distance
                    fingertip_distances.append(distance)
                except KeyError:
                    metrics[f"{finger_name}_tip_distance_from_center"] = np.nan

            if fingertip_distances:
                valid = [d for d in fingertip_distances if not np.isnan(d)]
                metrics["mean_fingertip_distance_from_center"] = (
                    float(np.mean(valid)) if valid else np.nan
                )
            else:
                metrics["mean_fingertip_distance_from_center"] = np.nan

        except Exception as exc:
            print(f"Error computing distance metrics: {exc}")
            for finger_name in self.FINGERTIP_NAMES:
                metrics[f"{finger_name}_tip_distance_from_center"] = np.nan
            metrics["wrist_distance_from_center"] = np.nan
            metrics["mean_fingertip_distance_from_center"] = np.nan

        return metrics

    def process_csv_row(
        self,
        row: pd.Series,
        frame_col: str = "frame_number",
        hand_id_col: str = "hand_id",
        handedness_col: str = "hand_label",
    ) -> Dict:
        result = {
            frame_col: row.get(frame_col, 0),
            hand_id_col: row.get(hand_id_col, ""),
            handedness_col: row.get(handedness_col, ""),
        }
        result.update(self.compute_tremor_metrics_from_row(row))
        return result

    def process_csv_file(
        self,
        input_csv_path: str | Path,
        output_csv_path: str | Path | None = None,
        frame_col: str = "frame_number",
        hand_id_col: str = "hand_id",
        handedness_col: str = "hand_label",
        separate_hands: bool = True,
    ) -> pd.DataFrame:
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

        df = pd.read_csv(input_path)

        results = []
        for i, row in df.iterrows():
            landmarks = [
                [row[f"x_{j}"], row[f"y_{j}"], row[f"z_{j}"]] for j in range(21)
            ]
            if all(x == 0 and y == 0 and z == 0 for x, y, z in landmarks):
                continue
            try:
                results.append(
                    self.process_csv_row(row, frame_col, hand_id_col, handedness_col)
                )
            except Exception as exc:
                print(f"Skipping row {i}: {exc}")

        output_df = pd.DataFrame(results)

        if output_csv_path:
            output_path = Path(output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if separate_hands and handedness_col in output_df.columns:
                left_hand_df = output_df[
                    output_df[handedness_col].str.lower().str.contains("left", na=False)
                ]
                right_hand_df = output_df[
                    output_df[handedness_col].str.lower().str.contains("right", na=False)
                ]

                base_name = output_path.stem
                extension = output_path.suffix
                parent_dir = output_path.parent

                left_output_path = parent_dir / f"{base_name}_left_hand{extension}"
                right_output_path = parent_dir / f"{base_name}_right_hand{extension}"

                if len(left_hand_df) > 0:
                    left_hand_df.to_csv(left_output_path, index=False)
                if len(right_hand_df) > 0:
                    right_hand_df.to_csv(right_output_path, index=False)

                output_df.to_csv(output_path, index=False)
            else:
                output_df.to_csv(output_path, index=False)

        return output_df
