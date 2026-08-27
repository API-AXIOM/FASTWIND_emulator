# FASTWIND DeepONet emulators

This repository contains the consolidated training and plotting code developed
for the MSc thesis *Exploring the Composition and Evolution of Massive Stars:
Predicting stellar spectra of O-type stars using neural networks* by Vasilis
Kolympiris.

The project trains coordinate-based Deep Operator Networks (DeepONets) to
emulate three kinds of FASTWIND output:

1. continuum-normalised hydrogen and helium line profiles controlled by five
   independently varied physical parameters;
2. ultraviolet and optical line profiles controlled by thirteen stellar,
   abundance, wind, and clumping parameters; and
3. the broad continuum spectral energy distribution stored in the FASTWIND
   `FLUXCONT` output.

The large FASTWIND datasets and trained checkpoints are not stored in Git.
See [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md) for their expected
layout and availability.

## Repository contents

| File | Purpose |
|---|---|
| `five_parameter_training.py` | Convergence selection, training, architecture search, and inference for the five-parameter line emulators |
| `five_parameter_plots.py` | All five-parameter evaluation and thesis plots |
| `export_parallel_plot_metadata.py` | Builds the metadata used by the five-parameter architecture comparison |
| `parallel_plot_from_metadata.py` | Interactive Plotly architecture comparison |
| `parallel_plot_thesis.py` | Static and interactive thesis versions of the architecture comparison |
| `thirteen_parameter_training.py` | Training for the five historical thirteen-parameter configurations, including the adopted deep network |
| `thirteen_parameter_plots.py` | Thirteen-parameter line-emulator evaluation and plotting |
| `fluxcont_training.py` | Consolidated implementation of the FLUXCONT v1-v18 architecture search |
| `fluxcont_plots.py` | FLUXCONT evaluation, architecture summaries, and thesis plots |

The three parallel-coordinate helpers intentionally remain separate. They are
loaded by `five_parameter_plots.py` for the `export-metadata`, `parallel`, and
`parallel-interactive` commands.

## Installation

Python 3.12 was used for the final consolidated repository.

```bash
git clone https://github.com/API-AXIOM/FASTWIND_emulator.git
cd FASTWIND_emulator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `zstd` command-line program is additionally required when the FASTWIND
files are supplied as `.zst` archives. On Ubuntu it can be installed with:

```bash
sudo apt-get install zstd
```

The thirteen-parameter convergence figure optionally uses `pdflatex` and
PGFPlots. All other Matplotlib figures can be produced without a TeX
installation.

## Portable paths

No personal or cluster-specific path is required. The default locations are
resolved relative to the repository:

```text
FASTWIND_emulator/
├── data/                         # external inputs, not tracked by Git
│   ├── five_parameter/
│   ├── thirteen_parameter/
│   ├── fluxcont/
│   └── datasets/                 # optional Sobol/LHC HDF5 comparisons
└── outputs/                      # checkpoints, metrics, and plots; not tracked
    ├── five_parameter/
    ├── thirteen_parameter/
    └── fluxcont_runs/
```

All important locations can be overridden on the command line. The following
environment variables provide convenient global alternatives:

| Variable | Meaning |
|---|---|
| `FASTWIND_EMULATOR_ROOT` | Repository root; defaults to the directory containing the scripts |
| `FASTWIND_DATA_ROOT` | Root of all external datasets; defaults to `<repository>/data` |
| `FASTWIND_OUTPUT_ROOT` | Root of generated results; defaults to `<repository>/outputs` |
| `FASTWIND_5PAR_DATA` | Five-parameter input tree |
| `FASTWIND_5PAR_OUTPUT` | Five-parameter output tree |
| `FASTWIND_5PAR_CHECKPOINTS` | Adopted five-parameter checkpoints |
| `FASTWIND_FLUXCONT_RUNS` | FLUXCONT run tree used by the plotting script |

The historical `ML13_*` environment variables are still accepted by the
thirteen-parameter and FLUXCONT training scripts. Explicit command-line paths
take precedence over environment variables, which in turn take precedence over
the portable defaults.

Example with data stored outside the repository:

```bash
export FASTWIND_DATA_ROOT=/path/to/fastwind_data
export FASTWIND_OUTPUT_ROOT=/path/to/fastwind_results
```

## Minimal reproduction commands

The following commands illustrate one small run or diagnostic for each case.
The architecture searches whose original drivers were preserved are exposed
through the configuration-listing commands and are computationally expensive.

### Five-parameter line emulator

Inspect the available architectures and search stages:

```bash
python five_parameter_training.py --mode list-configs
```

Train the adopted architecture for one line:

```bash
python five_parameter_training.py \
  --mode line \
  --baseline original \
  --line OUT.HGAMMA_VTV010 \
  --models-root data/five_parameter/models_LHC \
  --output-dir outputs/five_parameter/emulators_per_line_hg
```

Generate its loss curve:

```bash
python five_parameter_plots.py \
  --figure loss-curves \
  --emulator-dir outputs/five_parameter/emulators_per_line_hg \
  --output-dir outputs/five_parameter/plots
```

To reproduce the architecture-search figures, first export the metadata and
then draw the parallel-coordinate plot:

```bash
python five_parameter_plots.py \
  --figure export-metadata \
  --search-root outputs/five_parameter/fine_tuning_emulator_exploration \
  --hg-dir outputs/five_parameter/emulators_per_line_hg

python five_parameter_plots.py \
  --figure parallel \
  --metadata outputs/five_parameter/parallel_plot_metadata.json \
  --output-dir outputs/five_parameter/plots
```

### Thirteen-parameter line emulator

List the historical configurations:

```bash
python thirteen_parameter_training.py --list
```

Train the adopted network for one diagnostic window from compressed FASTWIND
files:

```bash
python thirteen_parameter_training.py \
  --config deep_relu_nodrop_nobn \
  --storage-dir data/thirteen_parameter \
  --filter-json data/thirteen_parameter/ml13_training_filter_config_linelevel_absflux15_1p5vinf_37lines.json \
  --line OUT.HALPHAHEII6527_VTV010.zst
```

Generate the available plots from the resulting run folders:

```bash
python thirteen_parameter_plots.py \
  --figure loss-curves \
  --runs-root outputs/thirteen_parameter/runs \
  --output-dir outputs/thirteen_parameter/plots
```

For an HDF5 campaign, use the
`deep_relu_nodrop_nobn_newh5` configuration and supply `--dataset-dir`,
`--filter-json`, and either `--line` or `--line-index`.

### FLUXCONT emulator

List every architecture-search configuration:

```bash
python fluxcont_training.py --list
```

Train the adopted v17a configuration:

```bash
python fluxcont_training.py \
  --config v17a \
  --storage-dir data/fluxcont \
  --output-root outputs/fluxcont_runs
```

Generate the training-history figure for the adopted run:

```bash
python fluxcont_plots.py \
  --figure loss-curves \
  --fluxcont-root outputs/fluxcont_runs
```

Run any training or plotting script with `--help` for the complete set of
options. Long searches should be launched through the scheduler available on
the target system while passing the same command-line arguments.

## Reproducibility notes

- Dataset partitions use the seeds recorded in the scripts.
- The preserved five-parameter drivers define 282 executable search
  configurations. The thesis comparison contains 460 configurations because it
  also includes earlier sweeps, including part of the 64-Fourier-mode
  comparison, whose driver code was not preserved. Their saved outputs can
  still be included in the metadata and parallel-coordinate analysis, but
  those missing training runs cannot honestly be reconstructed from this
  repository alone.
- Some historical architecture-search runs deliberately did not set a global
  PyTorch seed because the original scripts did not do so. Their exact weights
  may therefore vary between executions even though the configuration is the
  same.
- GPU results can differ slightly across CUDA, cuDNN, driver, and PyTorch
  versions. `requirements.txt` records the Python packages used when this
  repository was prepared, while the CUDA runtime is system-specific.
- The scripts preserve historical configuration choices. Consolidation reduces
  duplicated code but does not retune the models.

## Citation

If you use this software, please cite the associated MSc thesis. Publication
metadata or a DOI can be added here when one becomes available.

## License

The code is released under the MIT License. FASTWIND itself and any FASTWIND
input or output datasets remain subject to their own access and licensing
conditions.
