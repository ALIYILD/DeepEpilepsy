"""Preprocessing utilities for the QEEG pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mne
import numpy as np
import pandas as pd

from .config import ConditionWindow, PatientMetadata, PipelineConfig


def ensure_directories(config: PipelineConfig) -> None:
    """Create expected output directories if they do not already exist."""

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)


def _parse_condition_windows(entries: Iterable[Dict[str, float]]) -> List[ConditionWindow]:
    windows: List[ConditionWindow] = []
    for item in entries:
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            raise ValueError(f"Invalid condition window: end {end} <= start {start}")
        windows.append(ConditionWindow(start, end))
    return windows


def load_metadata(meta_path: Path, client_id: str) -> PatientMetadata:
    """Load patient metadata from a JSON or CSV file."""

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    if meta_path.suffix.lower() == ".json":
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif meta_path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(meta_path)
        payload = df.iloc[0].dropna().to_dict()
    else:
        raise ValueError(f"Unsupported metadata format: {meta_path.suffix}")

    conditions_payload = payload.get("conditions", {})
    conditions: Dict[str, List[ConditionWindow]] = {}
    for label, entries in conditions_payload.items():
        conditions[label] = _parse_condition_windows(entries)

    medications = payload.get("medications")
    if isinstance(medications, str):
        medications_list = [med.strip() for med in medications.split(",") if med.strip()]
    else:
        medications_list = medications or []

    return PatientMetadata(
        client_id=client_id,
        name=payload.get("name"),
        date_of_birth=payload.get("date_of_birth") or payload.get("dob"),
        recording_date=payload.get("recording_date") or payload.get("recorded_on"),
        handedness=payload.get("handedness"),
        medications=medications_list,
        notes=payload.get("notes"),
        conditions=conditions,
    )


def load_raw_recording(config: PipelineConfig, client_id: str) -> mne.io.BaseRaw:
    """Load EDF data for the given client."""

    edf_path = config.data_dir / "raw" / f"{client_id}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.set_montage(config.montage, match_case=False)
    raw.set_eeg_reference(config.reference)
    return raw


def apply_filters(raw: mne.io.BaseRaw, config: PipelineConfig) -> mne.io.BaseRaw:
    """Apply band-pass filtering to the raw recording."""

    filt = config.filter
    raw.filter(filt.l_freq, filt.h_freq, fir_design=filt.fir_design, verbose="ERROR")
    return raw


def detect_bad_channels(raw: mne.io.BaseRaw, z_thresh: float = 3.0, flat_threshold: float = 1e-6) -> List[str]:
    """Detect flat or high-variance channels using simple heuristics."""

    data = raw.get_data(reject_by_annotation="omit")
    variances = data.var(axis=1)
    z_scores = (variances - np.mean(variances)) / np.std(variances)
    bads = [raw.ch_names[idx] for idx, score in enumerate(z_scores) if score > z_thresh]

    for idx, channel in enumerate(raw.ch_names):
        if np.all(np.abs(data[idx]) < flat_threshold):
            bads.append(channel)
    return sorted(set(bads))


def mark_and_interpolate_bads(raw: mne.io.BaseRaw, bads: List[str]) -> mne.io.BaseRaw:
    """Mark bad channels and interpolate them."""

    if not bads:
        return raw
    raw.info["bads"] = bads
    raw.interpolate_bads(reset_bads=True)
    return raw


def run_ica(raw: mne.io.BaseRaw, config: PipelineConfig) -> mne.preprocessing.ICA:
    """Fit ICA and flag blink-related components."""

    ica_config = config.ica
    ica = mne.preprocessing.ICA(
        n_components=ica_config.n_components,
        random_state=ica_config.random_state,
        max_iter=ica_config.max_iter,
    )
    filtered = raw.copy().filter(1.0, 40.0, verbose="ERROR")
    ica.fit(filtered, verbose="ERROR")
    eog_indices, _ = mne.preprocessing.ica.find_bads_eog(ica, raw)
    ica.exclude = eog_indices
    return ica


def apply_ica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> mne.io.BaseRaw:
    """Apply ICA to remove artefactual components."""

    return ica.apply(raw.copy())


def _windows_from_annotations(raw: mne.io.BaseRaw, annotation_key: str) -> List[ConditionWindow]:
    windows: List[ConditionWindow] = []
    key = annotation_key.lower()
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if description.lower() == key:
            windows.append(ConditionWindow(float(onset), float(onset + duration)))
    return windows


def split_conditions(raw: mne.io.BaseRaw, metadata: PatientMetadata) -> Dict[str, mne.io.BaseRaw]:
    """Split the cleaned recording into conditions (e.g., eyes open/closed)."""

    segments: Dict[str, mne.io.BaseRaw] = {}
    for label, windows in metadata.conditions.items():
        if not windows:
            continue
        pieces = []
        for window in windows:
            pieces.append(raw.copy().crop(tmin=window.start, tmax=window.end))
        if pieces:
            segments[label] = mne.concatenate_raws(pieces)

    if not segments and raw.annotations:
        for label in {"eo", "eyes-open", "eyes open"}:
            windows = _windows_from_annotations(raw, label)
            if windows:
                segments["EO"] = mne.concatenate_raws([raw.copy().crop(*w.to_tuple()) for w in windows])
        for label in {"ec", "eyes-closed", "eyes closed"}:
            windows = _windows_from_annotations(raw, label)
            if windows:
                segments["EC"] = mne.concatenate_raws([raw.copy().crop(*w.to_tuple()) for w in windows])

    if not segments:
        segments["FULL"] = raw
    return segments


def epoch_condition(raw: mne.io.BaseRaw, config: PipelineConfig) -> mne.Epochs:
    """Epoch a raw segment into fixed-length epochs."""

    epoch_cfg = config.epoch
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_cfg.duration,
        overlap=epoch_cfg.overlap,
        preload=True,
        verbose="ERROR",
    )
    reject = {"eeg": epoch_cfg.reject_threshold_uv * 1e-6}
    epochs.drop_bad(reject=reject)
    return epochs


def epoch_conditions(segments: Dict[str, mne.io.BaseRaw], config: PipelineConfig) -> Dict[str, mne.Epochs]:
    """Epoch all available segments."""

    return {label: epoch_condition(segment, config) for label, segment in segments.items()}
