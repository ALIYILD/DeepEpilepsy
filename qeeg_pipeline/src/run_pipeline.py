"""Command-line entry point for running the QEEG pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .preprocess import (
    apply_filters,
    apply_ica,
    detect_bad_channels,
    ensure_directories,
    epoch_conditions,
    load_metadata,
    load_raw_recording,
    mark_and_interpolate_bads,
    run_ica,
    split_conditions,
)
from .metrics import (
    alpha_asymmetry,
    connectivity_matrices,
    compute_power_features,
    posterior_alpha_beta_ratio,
    posterior_dominant_rhythm,
    theta_beta_ratio,
)
from .visualize import plot_channel_psd, plot_connectivity_matrix, topomap_band
from .report import build_report


def _metadata_path(config: PipelineConfig, client_id: str) -> Path:
    json_path = config.data_dir / "meta" / f"{client_id}.json"
    if json_path.exists():
        return json_path
    csv_path = config.data_dir / "meta" / f"{client_id}.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No metadata file found for client {client_id}")


def _summaries_for_condition(label: str, epochs) -> Dict[str, float]:
    summary: Dict[str, float] = {"condition": label}
    summary["tbr_cz"] = theta_beta_ratio(epochs)
    summary["abr"] = posterior_alpha_beta_ratio(epochs)
    pdr_avg, _ = posterior_dominant_rhythm(epochs)
    summary["pdr_hz"] = pdr_avg
    summary["n_epochs"] = float(len(epochs))
    return summary


def run_pipeline(client_id: str, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    ensure_directories(config)
    metadata = load_metadata(_metadata_path(config, client_id), client_id)

    raw = load_raw_recording(config, client_id)
    raw = apply_filters(raw, config)
    bads = detect_bad_channels(raw)
    raw = mark_and_interpolate_bads(raw, bads)
    ica = run_ica(raw, config)
    clean_raw = apply_ica(raw, ica)

    segments = split_conditions(clean_raw, metadata)
    epochs_map = epoch_conditions(segments, config)

    summaries: List[Dict[str, float]] = []
    figure_paths: List[Path] = []

    tables: Dict[str, pd.DataFrame] = {}
    asym_tables: Dict[str, pd.DataFrame] = {}
    pdr_tables: Dict[str, pd.DataFrame] = {}

    for label, epochs in epochs_map.items():
        abs_power, rel_power, _, _ = compute_power_features(epochs)
        tables[f"abs_{label}"] = abs_power
        tables[f"rel_{label}"] = rel_power

        asym = alpha_asymmetry(epochs)
        asym["condition"] = label
        asym_tables[label] = asym

        summaries.append(_summaries_for_condition(label, epochs))
        _, pdr_detail = posterior_dominant_rhythm(epochs)
        pdr_tables[label] = pd.DataFrame([pdr_detail])

        # Visualizations
        alpha_map_path = config.figures_dir / f"{client_id}_{label}_alpha_topomap.png"
        figure_paths.append(topomap_band(epochs, 8.0, 12.0, alpha_map_path, title=f"{label} Alpha"))

        psd_path = config.figures_dir / f"{client_id}_{label}_psd.png"
        figure_paths.append(
            plot_channel_psd(
                epochs,
                channels=["Cz", "Pz", "O1", "O2"],
                output=psd_path,
                title=f"{label} PSD",
            )
        )

        coh, pli = connectivity_matrices(epochs)
        for method, matrices in {"coh": coh, "pli": pli}.items():
            for band, matrix in matrices.items():
                conn_path = config.figures_dir / f"{client_id}_{label}_{method}_{band}.png"
                figure_paths.append(
                    plot_connectivity_matrix(
                        matrix,
                        epochs.ch_names,
                        conn_path,
                        title=f"{label} {method.upper()} {band}",
                    )
                )

    summary_csv = config.tables_dir / f"{client_id}_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    for table_label, table in tables.items():
        table.to_csv(config.tables_dir / f"{client_id}_{table_label}.csv", index_label="channel")

    if asym_tables:
        pd.concat(asym_tables.values(), ignore_index=True).to_csv(
            config.tables_dir / f"{client_id}_asymmetry.csv", index=False
        )

    if pdr_tables:
        pd.concat(pdr_tables.values(), keys=pdr_tables.keys()).to_csv(
            config.tables_dir / f"{client_id}_pdr.csv"
        )

    report_path = config.reports_dir / f"{client_id}_report.docx"
    asym_combined = pd.concat(asym_tables.values(), ignore_index=True) if asym_tables else pd.DataFrame()
    pdr_combined = pd.concat(
        [df.assign(condition=label) for label, df in pdr_tables.items()], ignore_index=True
    ) if pdr_tables else pd.DataFrame()

    build_report(
        metadata=metadata,
        summary_rows=summaries,
        figure_paths=figure_paths,
        asymmetry_table=asym_combined,
        pdr_detail=pdr_combined,
        output_path=report_path,
    )

    print(f"Pipeline completed for {client_id}. Report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QEEG pipeline for a client ID")
    parser.add_argument("client_id", help="Identifier matching EDF and metadata filenames")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON file overriding default configuration values.",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.config:
        with args.config.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        config = PipelineConfig.from_base_dir(config.base_dir)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    run_pipeline(args.client_id, config)


if __name__ == "__main__":
    main()
