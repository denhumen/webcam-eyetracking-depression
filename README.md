# Investigating dynamics of eye-tracking attention indicators for depression assessment

> **Bachelor thesis** · Faculty of Applied Sciences, Ukrainian Catholic University · Lviv, 2026
> Author: Denis Humeniuk · Supervisor: PhD Anton Popov

## Research questions

The thesis tests four hypotheses, organised around two analysis tracks.

**Participant-level prediction.**

- **H1.** Higher depression severity is associated with a stronger negative-versus-positive attentional bias on session-aggregated static features, on both PHQ-9 and BDI-II.
- **H2.** At least one dynamic feature family (distributional, temporal, or trial-level bias score) improves prediction headline performance by ∆AUC ≥ 0.02 over the static-aggregate baseline, holding model class fixed.

**Trial-level mixed effects.**

- **H3.** On negative-versus-positive trials, the depression effect on per-valence attention differs between the negative and positive images on both scales, in the theoretically predicted direction.
- **H4.** The depression effect on per-valence attention strengthens across the course of a session, consistent with the maintenance phase of attentional bias described in the vigilance–maintenance framework.

## Repository structure

### Top-level layout

| Path | Purpose |
|---|---|
| `config.py` | Centralised constants: paths, valence keywords, quality-gate thresholds, output directories. |
| `pipelines/` | Databricks PySpark jobs (run on Databricks runtime). |
| `src/` | Reusable Python modules used by pipelines and notebooks. |
| `notebooks/exploration/` | Exploration notebooks used during development. |
| `notebooks/analysis/` | Notebooks producing the results reported in the thesis. |
| `tests/` | Unit tests for the metric-computation utilities. |
| `reports/figures/` | Figures produced by analysis notebooks. |
| `reports/notebook_exports/` | Exported analysis notebooks in PDF / HTML format. |
| `dashboards/` | Databricks dashboard JSON for the session explorer. |

### `pipelines/`

| File | Purpose |
|---|---|
| `01_build_stimuli_table.py` | Parse stimulus-set JSON files into a stimuli table. |
| `02_build_forms_table.py` | Load `forms.csv` and assign severity bands. |
| `03_build_sessions_table.py` | Scan raw session CSVs into a sessions dimension table. |
| `04_compute_scene_metrics.py` | Main job: raw gaze → fixation sequence → scene metrics. |

### `src/`

| File | Purpose |
|---|---|
| `data_loading.py` | Spark / Pandas readers for the produced tables. |
| `preprocessing.py` | Quality gate and per-session filtering. |
| `scene_processor.py` | Per-scene segmentation and fixation extraction. |
| `scene_metrics.py` | All five metric families (general, vigilance, maintenance, bias scores, pair-stratified). |
| `features/session_aggregation.py` | Static, distributional, and temporal aggregations. |
| `features/tlbs.py` | Trial-Level Bias Score time-series parameters. |
| `evaluation/classification.py` | Model definitions, GroupKFold CV, plotting helpers for binary, multiclass, regression tasks. |
| `evaluation/lmm_temporal.py` | Mixed-effects models testing depression × trial-position interactions (predicate for H4). |
| `evaluation/lmm_valence.py` | Mixed-effects models testing depression × valence interactions (predicate for H3). |
| `visualization/scene.py` | Per-scene scanpath and heatmap plots. |
| `visualization/session.py` | Session-level distribution and severity-trend plots. |
| `visualization/io.py` | Centralised figure-saving utility used by every plotting function. |

### `notebooks/analysis/`

| Notebook | Purpose |
|---|---|
| `00_dataset_audit.ipynb` | Cohort sizes, sessions-per-user, severity distributions, sampling-rate descriptives. |
| `01_static_correlations.ipynb` | Per-metric correlations with PHQ-9 and BDI-II (preliminary screening for H1). |
| `02_classification_static.ipynb` | Static aggregation, all four prediction tasks (H1 / H2 baseline). |
| `03_classification_distributional.ipynb` | Distributional aggregation, all four tasks (H2). |
| `04_classification_temporal.ipynb` | Temporal aggregation, all four tasks (H2). |
| `05_classification_tlbs.ipynb` | Trial-Level Bias Score aggregation, all four tasks (H2). |
| `06_lmm_temporal.ipynb` | Within-session depression × trial-position interactions (H4). |
| `07_lmm_valence.ipynb` | Depression × valence interactions (H3). |

## Data flow

The analysis runs in two stages. The first stage is a Databricks pipeline that turns raw per-session CSV recordings into a single scene-level metrics table. The second stage is a set of analysis notebooks that consume that table, aggregate it to the session or trial level, and run the statistical models.

```
Raw session CSVs (one per recorded session)
        │
        ▼
[pipelines/01-04, run on Databricks]
        │
        ▼
Scene-level metrics table (anima.scene_metrics)
        │
        ├──► notebooks/analysis/02-05  (participant-level classification, H1/H2)
        │
        └──► notebooks/analysis/06-07  (trial-level mixed effects, H3/H4)
```

Five metric families are computed per scene: general oculomotor metrics (fixation count, saccade amplitude, scanpath length, transition entropy, blink rate), vigilance metrics (early-window dwell, time to first fixation, first-fixation valence), maintenance metrics (per-valence dwell time, fixation count, fixation proportion, revisit count), bias scores (per-pair attention asymmetry), and pair-stratified versions of the per-valence metrics.

These are then aggregated to the session level in four ways: static (mean of each metric), distributional (quartiles plus IQR), temporal (early third, late third, and their difference), and Trial-Level Bias Score (seven time-series parameters per pair type).

## Setup

All of the project code was developed in Databricks. It is highly recommended to download the repository to Databricks platform and run the notebooks there directly.

### Data access

The dataset is the property of Anima and is **not distributed with this repository**. Replication on the same dataset requires direct request to the platform.

### Pipeline (Databricks)

The four pipeline jobs in `pipelines/` are written for Databricks runtime and assume access to:

- A Databricks workspace with a runtime image including Python 3.10+, PySpark, and the libraries pinned by that runtime.
- A volume containing the raw session CSVs and the four stimulus-set JSON files. The default path is set in `config.py` via `VOLUME_BASE`.

Pipelines are run in order (01, 02, 03, 04). Each writes a Delta table in the workspace catalogue (default names: `anima.stimuli`, `anima.forms`, `anima.sessions`, `anima.scene_metrics`).

### Analysis notebooks

The notebooks in `notebooks/analysis/` consume the tables produced by the pipelines. They run either inside the same Databricks workspace or locally with PySpark configured to read from the workspace catalogue.

Required Python libraries (all available in the Databricks runtime):

- `pyspark` for table access and pre-aggregation.
- `pandas`, `numpy`, `scipy` for in-memory analysis.
- `scikit-learn`, `xgboost` for classification and regression.
- `statsmodels` for linear mixed-effects models.
- `matplotlib`, `seaborn` for figures.

Library versions are pinned by the Databricks runtime used during the original analysis. Outside Databricks, install the latest stable releases of the above and adjust paths in `config.py` accordingly.

### Configuration

`config.py` centralises all constants used across the pipeline and notebooks: volume paths, table names, valence-keyword lists, quality-gate thresholds, and the figure output directory. Any change to these values takes effect across the entire workflow without further edits.
