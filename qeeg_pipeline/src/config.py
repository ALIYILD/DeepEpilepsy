"""Configuration objects and helpers for the QEEG processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ConditionWindow:
    """Time window definition for a recording condition."""

    start: float
    end: float

    def to_tuple(self) -> Tuple[float, float]:
        return self.start, self.end


@dataclass
class PatientMetadata:
    """Container describing patient demographics and acquisition details."""

    client_id: str
    name: str | None = None
    date_of_birth: str | None = None
    recording_date: str | None = None
    handedness: str | None = None
    medications: List[str] = field(default_factory=list)
    notes: str | None = None
    conditions: Dict[str, List[ConditionWindow]] = field(default_factory=dict)


@dataclass
class ICAConfig:
    """Settings for ICA decomposition."""

    n_components: int = 20
    random_state: int = 97
    max_iter: str | int = "auto"


@dataclass
class FilterConfig:
    """Filter configuration parameters."""

    l_freq: float = 1.0
    h_freq: float = 40.0
    fir_design: str = "firwin"


@dataclass
class EpochConfig:
    """Epoching configuration."""

    duration: float = 4.0
    overlap: float = 0.0
    reject_threshold_uv: float = 100.0


@dataclass
class PipelineConfig:
    """Aggregated configuration for the QEEG pipeline."""

    base_dir: Path
    data_dir: Path
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    reports_dir: Path
    montage: str = "standard_1020"
    reference: str = "average"
    filter: FilterConfig = field(default_factory=FilterConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "PipelineConfig":
        data_dir = base_dir / "qeeg_pipeline" / "data"
        output_dir = base_dir / "qeeg_pipeline" / "outputs"
        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            figures_dir=output_dir / "figures",
            tables_dir=output_dir / "tables",
            reports_dir=output_dir / "reports",
        )


DEFAULT_CONFIG = PipelineConfig.from_base_dir(Path(__file__).resolve().parents[2])
