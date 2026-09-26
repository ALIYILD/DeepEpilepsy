"""Computation of spectral and connectivity metrics for the QEEG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import mne
import numpy as np
import pandas as pd
from mne.connectivity import spectral_connectivity
from mne.time_frequency import psd_welch


@dataclass(frozen=True)
class Band:
    label: str
    fmin: float
    fmax: float


DEFAULT_BANDS: Tuple[Band, ...] = (
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 12.0),
    Band("beta", 12.0, 20.0),
    Band("high_beta", 20.0, 30.0),
    Band("gamma", 30.0, 40.0),
)


def _band_mean(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"Band ({fmin}, {fmax}) outside computed PSD range")
    return psd[..., mask].mean(axis=-1)


def compute_power_features(
    epochs: mne.Epochs,
    bands: Iterable[Band] = DEFAULT_BANDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute absolute and relative band power features."""

    psd, freqs = psd_welch(epochs, fmin=1.0, fmax=40.0, n_fft=1024, verbose="ERROR")
    # psd shape: (n_epochs, n_channels, n_freqs)
    abs_power = {}
    rel_power = {}
    total = psd.mean(axis=-1)
    for band in bands:
        band_values = _band_mean(psd, freqs, band.fmin, band.fmax)
        abs_power[band.label] = band_values.mean(axis=0)
        rel_power[band.label] = abs_power[band.label] / total.mean(axis=0)

    abs_df = pd.DataFrame(abs_power, index=epochs.ch_names)
    rel_df = pd.DataFrame(rel_power, index=epochs.ch_names)
    return abs_df, rel_df, psd, freqs


def theta_beta_ratio(epochs: mne.Epochs, channel: str = "Cz") -> float:
    psd, freqs = psd_welch(epochs, fmin=1.0, fmax=40.0, verbose="ERROR")
    ch_idx = epochs.ch_names.index(channel)
    theta = _band_mean(psd[:, ch_idx : ch_idx + 1, :], freqs, 4.0, 8.0).mean()
    beta = _band_mean(psd[:, ch_idx : ch_idx + 1, :], freqs, 12.0, 20.0).mean()
    return float(theta / beta)


POSTERIOR_CHANNELS = ["O1", "O2", "P3", "Pz", "P4"]


def posterior_alpha_beta_ratio(epochs: mne.Epochs, channels: Iterable[str] = POSTERIOR_CHANNELS) -> float:
    psd, freqs = psd_welch(epochs, fmin=1.0, fmax=40.0, verbose="ERROR")
    idx = [epochs.ch_names.index(ch) for ch in channels if ch in epochs.ch_names]
    if not idx:
        raise ValueError("None of the posterior channels were found in epochs")
    alpha = _band_mean(psd[:, idx, :], freqs, 8.0, 12.0).mean()
    beta = _band_mean(psd[:, idx, :], freqs, 12.0, 20.0).mean()
    return float(alpha / beta)


def posterior_dominant_rhythm(
    epochs: mne.Epochs,
    channels: Iterable[str] = POSTERIOR_CHANNELS,
) -> Tuple[float, Dict[str, float]]:
    psd, freqs = psd_welch(epochs, fmin=6.0, fmax=14.0, n_fft=2048, verbose="ERROR")
    idx = [epochs.ch_names.index(ch) for ch in channels if ch in epochs.ch_names]
    if not idx:
        raise ValueError("No channels available for PDR computation")

    channel_peaks: Dict[str, float] = {}
    for ch_idx in idx:
        mean_psd = psd[:, ch_idx, :].mean(axis=0)
        peak_freq = float(freqs[np.argmax(mean_psd)])
        channel_peaks[epochs.ch_names[ch_idx]] = peak_freq

    return float(np.mean(list(channel_peaks.values()))), channel_peaks


ASYMMETRY_PAIRS = [("F3", "F4"), ("F7", "F8"), ("P3", "P4")]


def alpha_log_power(epochs: mne.Epochs) -> pd.Series:
    psd, freqs = psd_welch(epochs, fmin=8.0, fmax=12.0, verbose="ERROR")
    band = _band_mean(psd, freqs, 8.0, 12.0).mean(axis=0)
    return pd.Series(np.log(band + 1e-12), index=epochs.ch_names)


def alpha_asymmetry(epochs: mne.Epochs, pairs: Iterable[Tuple[str, str]] = ASYMMETRY_PAIRS) -> pd.DataFrame:
    log_power = alpha_log_power(epochs)
    rows = []
    for left, right in pairs:
        if left not in log_power.index or right not in log_power.index:
            continue
        rows.append(
            {
                "pair": f"{left}-{right}",
                "left": float(log_power[left]),
                "right": float(log_power[right]),
                "bias": "Left" if log_power[left] > log_power[right] else "Right",
            }
        )
    return pd.DataFrame(rows)


CONNECTIVITY_BANDS: Mapping[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 40.0),
}


def _lower_to_matrix(values: np.ndarray, n_channels: int, diagonal: float) -> np.ndarray:
    matrix = np.zeros((n_channels, n_channels), dtype=float)
    triu_no_diag = np.triu_indices(n_channels, k=1)
    triu_with_diag = np.triu_indices(n_channels, k=0)

    if values.shape[0] == len(triu_with_diag[0]):
        matrix[triu_with_diag] = values
    elif values.shape[0] == len(triu_no_diag[0]):
        matrix[triu_no_diag] = values
    else:
        raise ValueError("Unexpected connectivity vector length")

    matrix[(triu_no_diag[1], triu_no_diag[0])] = matrix[triu_no_diag]
    np.fill_diagonal(matrix, diagonal)
    return matrix


def connectivity_matrices(epochs: mne.Epochs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    n_channels = data.shape[1]
    coh: Dict[str, np.ndarray] = {}
    pli: Dict[str, np.ndarray] = {}
    for label, (fmin, fmax) in CONNECTIVITY_BANDS.items():
        coh_values, _, _, _, _ = spectral_connectivity(
            data,
            method="coh",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose="ERROR",
        )
        pli_values, _, _, _, _ = spectral_connectivity(
            data,
            method="pli",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose="ERROR",
        )
        coh[label] = _lower_to_matrix(coh_values.squeeze(), n_channels, diagonal=1.0)
        pli[label] = _lower_to_matrix(pli_values.squeeze(), n_channels, diagonal=0.0)
    return coh, pli
