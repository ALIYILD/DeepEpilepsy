"""Visualization utilities for the QEEG pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_welch


def _prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def topomap_band(epochs: mne.Epochs, fmin: float, fmax: float, output: Path, title: str | None = None) -> Path:
    """Save a topographic map of band power."""

    psd, freqs = psd_welch(epochs, fmin=fmin, fmax=fmax, verbose="ERROR")
    band_power = psd.mean(axis=0).mean(axis=-1)
    _prepare_output_path(output)
    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(band_power, epochs.info, axes=ax, show=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_channel_psd(epochs: mne.Epochs, channels: Iterable[str], output: Path, title: str) -> Path:
    psd, freqs = psd_welch(epochs, fmin=1.0, fmax=40.0, verbose="ERROR")
    _prepare_output_path(output)
    fig, ax = plt.subplots(figsize=(6, 4))
    for channel in channels:
        if channel not in epochs.ch_names:
            continue
        idx = epochs.ch_names.index(channel)
        ax.plot(freqs, psd[:, idx, :].mean(axis=0), label=channel)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_connectivity_matrix(matrix: np.ndarray, ch_names: Iterable[str], output: Path, title: str) -> Path:
    _prepare_output_path(output)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Connectivity")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def summarize_top_edges(matrix: np.ndarray, ch_names: Iterable[str], percentile: float = 90.0) -> Dict[str, Tuple[str, str, float]]:
    names = list(ch_names)
    upper_indices = np.triu_indices_from(matrix, k=1)
    weights = matrix[upper_indices]
    threshold = np.percentile(weights, percentile)
    strongest_idx = np.argmax(weights)
    weakest_idx = np.argmin(weights)
    strong_edge = (names[upper_indices[0][strongest_idx]], names[upper_indices[1][strongest_idx]], float(weights[strongest_idx]))
    weak_edge = (names[upper_indices[0][weakest_idx]], names[upper_indices[1][weakest_idx]], float(weights[weakest_idx]))
    return {"strongest": strong_edge, "weakest": weak_edge, "threshold": float(threshold)}
