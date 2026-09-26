# QEEG Pipeline Starter Kit

This directory contains a modular Python implementation of a quantitative EEG
(QEEG) processing workflow inspired by the step-by-step plan shared in the
project brief. The code is organised into small modules so individual stages of
the analysis can be customised or extended with minimal friction.

## Directory layout

```
qeeg_pipeline/
  data/
    raw/          # Place EDF recordings here (e.g. CLIENT_ID.edf)
    meta/         # Place patient metadata JSON/CSV files here
    montages/     # Optional custom montage definitions
  outputs/
    figures/      # Generated topomaps, PSD plots, connectivity matrices
    tables/       # Exported CSV summaries
    reports/      # DOCX clinical reports
  src/
    config.py     # Configuration dataclasses
    preprocess.py
    metrics.py
    visualize.py
    report.py
    run_pipeline.py
```

## Usage

1. Install the dependencies inside a virtual environment:

   ```bash
   pip install mne yasa neurokit2 numpy scipy pandas matplotlib python-docx jinja2 autoreject
   ```

2. Place `CLIENT_ID.edf` into `qeeg_pipeline/data/raw/` and create the matching
   metadata file `qeeg_pipeline/data/meta/CLIENT_ID.json`. A minimal JSON example
   is shown below:

   ```json
   {
     "name": "Client Example",
     "date_of_birth": "1986-05-01",
     "recording_date": "2024-03-18",
     "handedness": "Right",
     "medications": ["None"],
     "conditions": {
       "EC": [{"start": 60, "end": 180}],
       "EO": [{"start": 200, "end": 320}]
     }
   }
   ```

3. Run the pipeline using the module syntax so relative imports resolve
   correctly:

   ```bash
   python -m qeeg_pipeline.src.run_pipeline CLIENT_ID
   ```

   The script will perform filtering, channel QC, ICA-based artefact removal,
   spectral metric extraction, connectivity estimation, visualisation, and
   report generation. Outputs are written to the `qeeg_pipeline/outputs/`
   subdirectories.

## Configuration overrides

`run_pipeline.py` loads defaults from `config.py`. You can override values by
passing a JSON configuration file via the `--config` option. Only the keys you
set in the JSON payload will replace the defaults.

```bash
python -m qeeg_pipeline.src.run_pipeline CLIENT_ID --config custom_config.json
```

```json
{
  "montage": "standard_1020",
  "reference": "average"
}
```

## Next steps

The modules deliberately expose intermediate data structures (e.g. absolute and
relative power data frames, connectivity matrices, and summary tables) so you
can integrate additional analytics, normative comparisons, or custom reporting
narratives without reworking the core pipeline.
