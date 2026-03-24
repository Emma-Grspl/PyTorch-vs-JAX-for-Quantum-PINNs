from __future__ import annotations

import numpy as np


def align_sign(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.dot(prediction, reference) < 0.0:
        return -prediction
    return prediction


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    aligned = align_sign(prediction, reference)
    numerator = np.linalg.norm(aligned - reference)
    denominator = np.linalg.norm(reference)
    return float(numerator / denominator)


def absolute_energy_error(predicted_energy: float, reference_energy: float) -> float:
    return float(abs(predicted_energy - reference_energy))

