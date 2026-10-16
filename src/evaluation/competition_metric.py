"""Competition metric for PhysioNet - Digitization of ECG Images (scoring helpers).

Copied from the public Kaggle notebook https://www.kaggle.com/code/sacuscreed/metric, which reproduces the
competition's scoring code. Each lead is shifted in time by up to 0.2 s to best match the label and has one constant
vertical offset removed; signal and noise power are summed over the 12 leads of an image, the image SNR is capped at
384, and the final score is 10 * log10 of the mean image SNR.
"""
from typing import Dict, Tuple

import numpy as np
import scipy.optimize
import scipy.signal

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 384


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    prediction = prediction.copy()
    prediction[~np.isfinite(prediction)] = 0
    noise = label - prediction
    return np.sum(label ** 2), np.sum(noise ** 2)


def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        return PERFECT_SCORE
    if signal == 0:
        return 0
    return min(signal / noise, PERFECT_SCORE)


def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: float = float('inf')) -> np.ndarray:
    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    correlation = scipy.signal.correlate(label_arr - label_arr.mean(), pred_arr - pred_arr.mean(), mode='full')
    n_label, n_pred = np.size(label_arr), np.size(pred_arr)
    lags = scipy.signal.correlation_lags(n_label, n_pred, mode='full')
    valid = (lags >= -max_shift) & (lags <= max_shift)
    max_correlation = np.nanmax(correlation[valid])
    best_idx = min(np.flatnonzero(correlation == max_correlation), key=lambda i: abs(lags[i]))
    time_shift = lags[best_idx]
    aligned = np.concatenate((
        np.full(max(time_shift, 0), np.nan),
        pred_arr[max(-time_shift, 0):min(n_label - time_shift, n_pred)],
        np.full(max(n_label - n_pred - time_shift, 0), np.nan),
    ))

    def objective(v_shift):
        return np.nansum((label_arr - (aligned - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned)):
        aligned -= scipy.optimize.minimize_scalar(objective, method='Brent').x
    return aligned


def image_snr(truth: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], fs: int) -> float:
    """Capped linear SNR of one image (12 leads)."""
    sum_signal = sum_noise = 0.0
    for lead in LEADS:
        aligned = align_signals(truth[lead], pred[lead], int(fs * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(truth[lead], aligned)
        sum_signal += p_signal
        sum_noise += p_noise
    return compute_snr(sum_signal, sum_noise)


def score_db(image_snrs) -> float:
    """Final score: 10 * log10 of the mean capped image SNR."""
    return max(float(10 * np.log10(np.mean(image_snrs))), -PERFECT_SCORE)
