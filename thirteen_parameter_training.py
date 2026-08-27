#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
thirteen_parameter_training.py
==============================

Training of the **thirteen-parameter FASTWIND line-profile emulator** of the
MSc thesis: one DeepONet per spectral line, trained on the 13-dimensional
Latin-hypercube and Sobol model grids.

This single file replaces the five near-duplicate research scripts that lived
in ``13_par/py_codes/``, without changing any numerical behaviour:

    13_emulator_deep_relu_nodrop_nobn.py        -> CONFIGS["deep_relu_nodrop_nobn"]
    13_emulator_deep_relu_nodrop_nobn_newh5.py  -> CONFIGS["deep_relu_nodrop_nobn_newh5"]
    13_emulator_orig_silu_fm64.py               -> CONFIGS["orig_silu_fm64"]
    13_emulator_anja_style.py                   -> CONFIGS["anja_style"]
    13_emulator_backup_orig.py                  -> CONFIGS["backup_orig"]

Nothing was tuned, rounded or re-derived while consolidating.  Every numeric
literal below was copied from the corresponding source file, and every
behavioural difference between two sources is expressed as a configuration key,
never as a code change.  Where two sources genuinely did different things -- for
example the crop/edge-pad profile handling of the adopted script versus the
fixed-cropped-length rejection of ``orig_silu_fm64``, or the per-line quality
filter of the adopted script versus the global one of ``orig_silu_fm64`` versus
no filter at all in ``anja_style`` -- *both* code paths are present and are
selected by configuration.

The ADOPTED emulator of the thesis is ``CONFIGS["deep_relu_nodrop_nobn"]``
(Latin-hypercube campaign, per-line directories of zstd-compressed FASTWIND
output) and its byte-identical HDF5 twin ``CONFIGS["deep_relu_nodrop_nobn_newh5"]``
(Sobol and merged Sobol+LHC campaigns).  Both produce the architecture tag

    deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048_lat128_fm32_relu_nodrop_nobn


--------------------------------------------------------------------------------
1.  What the emulator is
--------------------------------------------------------------------------------

FASTWIND is a non-LTE, spherically symmetric, line-blanketed model-atmosphere
and wind code for hot massive stars.  One converged model costs of order tens of
minutes of CPU time and writes, among other products, a set of
continuum-normalised spectral line profiles, one file per diagnostic line
window.  Fitting an observed spectrum requires many thousands of such
evaluations, hence the surrogate.

The surrogate is a **DeepONet** (deep operator network), which factorises the
map from stellar parameters to a *function of wavelength* into two subnetworks
combined by an inner product:

        F(theta)(lambda)  ~=  sum_k  b_k(theta) * t_k(lambda)

  * the **branch** network ``b`` sees only the 13 stellar parameters;
  * the **trunk** network ``t`` sees only the normalised wavelength coordinate;
  * their dot product gives the flux at that one wavelength.

Wavelength is a genuine *input coordinate*, not an output index, because
FASTWIND does not use a uniform wavelength grid and the grid is not identical
between models.  Treating lambda as a coordinate keeps every flux value paired
with its own physical wavelength and lets the trained emulator be evaluated on
any grid inside the line window.

The trunk input is not the raw scalar but a deterministic Fourier feature
expansion ``[1, sin(2*pi*n*x), cos(2*pi*n*x)]`` for n = 1..M, with
M = ``fourier_modes`` (32 in every configuration except ``orig_silu_fm64``,
which used 64).  The leading feature is the constant 1, not x -- this matches
all five sources.  The trunk input dimension is therefore ``2*M + 1``.

One emulator is trained per line window, as in the five-parameter chapter: the
wavelength normalisation is per line, different lines respond to different
physics, and the error budget stays diagnosable line by line.


--------------------------------------------------------------------------------
2.  The 13 branch inputs
--------------------------------------------------------------------------------

    theta = (teff, logg, radius, mdot, yhe, C, N, O, beta, vinf, fic, fvel, fclump)

    - teff, logg, radius : photospheric structure
    - mdot, beta, vinf   : mass-loss rate, velocity-law exponent, terminal speed
    - yhe, C, N, O       : helium and CNO abundances
    - fic, fvel, fclump  : the three clumping parameters

``mdot`` is transformed to ``log10(mdot)`` **first**, because mass-loss rates
span several decades, and only then are all 13 components min-max scaled to
[0, 1].  A column whose range is exactly zero has its range replaced by 1.0 so
that the normalised column is exactly zero rather than NaN.

The min/max are taken over a *population that differs between the two readers*,
and this difference is preserved because it changes the numbers written into
``parameter_normalization.json`` and into every checkpoint:

  * per-line-directory reader ("zst"):  over the models that survive the green
    convergence flag, the parameter completeness check and the training-filter
    JSON (``FILTER_VALID_MODEL_IDS``);
  * HDF5 reader ("hdf5"):  over **all** models present in ``models.h5``, before
    any filtering (the sources record this explicitly as
    ``"normalization_population": "all_models_in_models_h5"``).


--------------------------------------------------------------------------------
3.  The two input paths
--------------------------------------------------------------------------------

The Latin-hypercube and Sobol campaigns used different readers, and both are
reproducible here through ``cfg["reader"]``.

**reader = "zst"** (Latin-hypercube campaign; the adopted checkpoints)

    <models_root>/<model_id>/OUT.<LINE>_VTV010.zst          (zstd-compressed)
    <storage_dir>/grid_large_log_LHC.json                   parameter grid
    <storage_dir>/grid_quality_large_lhc.json               convergence flags

    A model is usable when its quality flag is ``"green"`` **and** its directory
    exists **and** the grid JSON has all 13 parameters for it.  Each line file
    is decompressed by shelling out to the ``zstd`` binary (``zstd -dc``) and
    parsed with ``np.loadtxt(..., max_rows=161)``; column index 2 is the
    wavelength and the last column is the normalised flux.  Only
    *non-rotational* files are used: a file qualifies when it starts with
    ``OUT.``, ends with ``.zst`` and does not contain ``vrot_``.  The line list
    is discovered from the *lowest-numbered* usable model directory.

**reader = "hdf5"** (Sobol and merged Sobol+LHC campaigns)

    <dataset_dir>/models.h5      model_ids (N,), parameters (N,13), param_cols (13,)
    <dataset_dir>/out_lines.h5   out_lines/<LINE>/{model_ids,n_points,wavelength,flux}
    <dataset_dir>/manifest.json  provenance of the merge
    <dataset_dir>/fluxcont.h5, convergence.h5   (present, not used here)

    No decompression, no per-model directory scan.  Rows with a duplicate model
    id, with no parameter row, or with fewer than 20 points are skipped, and
    non-finite (lambda, flux) pairs are dropped before the crop.  The line list
    is the sorted set of keys of the filter JSON's ``crop_ranges``, and every one
    of them must exist in ``out_lines.h5``.


--------------------------------------------------------------------------------
4.  The line-level quality filter
--------------------------------------------------------------------------------

The filter JSON is produced by a separate quality-control job
(``prepare_ml13_training_filters.py`` for the zstd path,
``prepare_ml13_newds_training_filters_h5.py`` for the HDF5 path).  Training
refuses to start unless the file exists and carries ``"status": "completed"``.

Three consumption modes existed and all three are kept:

  * ``filter_mode = "line_level"`` (adopted).  ``valid_model_ids_by_line`` maps
    each line to the set of model ids that passed QC *for that line*; the global
    ``valid_model_ids`` set is then **overwritten** by the union of the per-line
    sets, and that union is what prunes the parameter table.  During the dataset
    build, a line uses its own list, falling back to the union if the line has
    no entry.  This is the behaviour of ``13_emulator_deep_relu_nodrop_nobn.py``,
    ``13_emulator_backup_orig.py`` and the HDF5 script.
  * ``filter_mode = "global"``.  Only the flat ``valid_model_ids`` list is read;
    the same set is applied to every line.  This is
    ``13_emulator_orig_silu_fm64.py``, which pointed at the older
    ``ml13_training_filter_config.json``.
  * ``filter_mode = "none"``.  No filter JSON is read at all and no crop is
    applied.  This is ``13_emulator_anja_style.py``.

``crop_ranges`` maps each line to a (lambda_min, lambda_max) window in Angstrom.
Line keys are normalised before use by ``normalize_line_name``, which strips a
leading ``emulator_``, a trailing ``_test_outputs.npz`` and a trailing ``.zst``,
so that filter JSONs keyed by any of those spellings work unchanged.  The HDF5
script matches the raw line names instead, because the new datasets are keyed by
the bare ``OUT.*`` name.

Note the QC filenames actually used, taken from the sbatch scripts:

    ml13_training_filter_config_linelevel_absflux15_1p5vinf.json          (34 lines)
    ml13_training_filter_config_linelevel_absflux15_1p5vinf_37lines.json  (37 lines, group 3)
    ml13_training_filter_config.json                                      (global, orig_silu)
    ml13_sobol_training_filter_config_linelevel_absflux15_1p5vinf_nocorrupt.json
    ml13_merged_training_filter_config_linelevel_absflux15_1p5vinf_sobol_lhc_nocorrupt.json


--------------------------------------------------------------------------------
5.  Cropping and padding to a fixed profile length
--------------------------------------------------------------------------------

Cropping a profile to the QC window leaves a variable number of points per
model, but the training tensors are rectangular.  Three strategies existed:

  * ``crop_mode = "crop_pad"`` (adopted).  After the crop, the profile is padded
    back up to ``target_wavelength_points`` (default 161) by inserting
    linearly spaced wavelengths in the empty margins between the crop bound and
    the first/last surviving point, half on each side, with the flux held
    constant at the nearest surviving value.  If one margin is numerically
    unusable the whole remainder goes to the other.  **No original FASTWIND
    point is modified.**  A profile that has *more* than the target number of
    points after cropping is dropped (``skipped_crop_too_long``).
  * ``crop_mode = "crop_fixed_length"``.  The first cropped profile defines the
    expected length and every later profile with a different length is dropped
    (``skipped_crop_length_mismatch``).  This is ``orig_silu_fm64``.
  * ``crop_mode = "none"``.  No crop; profiles are used at their native length.
    This is ``anja_style``.

In every mode a profile with fewer than ``MIN_POINTS_PER_LINE = 20`` points
before or after the crop is dropped.


--------------------------------------------------------------------------------
6.  Target, loss and the amplitude weighting
--------------------------------------------------------------------------------

The network predicts the **residual** ``flux - FLUX_OFFSET`` with
``USE_RESIDUAL = True`` and ``FLUX_OFFSET = 1.0``.  A continuum-normalised
profile is 1.0 almost everywhere, so predicting the residual removes a large
constant offset and spends the network's dynamic range on the line itself.

The loss is an amplitude-weighted mean squared error on that residual:

    w = 1 + ALPHA * |target|          with target = F - 1  and  ALPHA = 5.0
    L = mean[ (pred - target)^2 * w ]

so the line core and wings -- exactly the diagnostic parts -- carry up to six
times the weight of the flat continuum.  ``ALPHA = 5.0`` in all five sources.
The weights are recomputed from the *targets* inside every batch, in both the
training and the validation pass, and the epoch loss is the mean over batches
(``running / max(len(loader), 1)``), not over samples.


--------------------------------------------------------------------------------
7.  Split
--------------------------------------------------------------------------------

``sklearn.model_selection.train_test_split`` is applied twice with
``random_state = SPLIT_SEED = 42``: first ``test_size = 0.30`` on
``np.arange(n_models)``, then ``test_size = 0.50`` on the remainder.  That is
70 % train / 15 % validation / 15 % test.

The split is done **per line, on that line's own surviving model list**, not
once globally.  Because the per-line QC filter removes a different set of models
from each line, the number of models entering the split differs from line to
line, and therefore so do the actual train/val/test members.  The model ids of
each partition are written to ``<stem>_split.json`` so that the partition is
recoverable exactly.  A line with fewer than 3 usable models is skipped.


--------------------------------------------------------------------------------
8.  Architecture, optimiser and early stopping
--------------------------------------------------------------------------------

Branch and trunk are plain MLPs built by the same ``build_mlp``: for each hidden
width, optionally a ``BatchNorm1d`` on the *input* of the layer, then
``Linear``, then the activation for that position in ``activation_sequence``,
then optionally ``Dropout`` if that layer index is listed in
``dropout_after_layer_indices`` and ``dropout > 0``; a final ``Linear`` maps to
``latent_dim``.  The adopted network is

    branch: 13 -> 128 -> 256 -> 512 -> 512 -> 1024 -> 2048 -> 128
    trunk : (2*32+1)=65 -> 128 -> 256 -> 512 -> 512 -> 1024 -> 2048 -> 128
    all ReLU, no dropout, no batch normalisation

Optimiser: ``torch.optim.Adam`` with the config's learning rate and
``weight_decay = 0.0``.  Scheduler: ``ReduceLROnPlateau(mode="min", factor=0.5,
patience=10)`` stepped on the validation loss.  Multi-GPU runs wrap the model in
``nn.DataParallel`` when two or more CUDA devices are visible.

Early stopping: an epoch improves when ``val < best_val`` (strict, no minimum
delta).  After ``early_stop_patience`` consecutive non-improving epochs the loop
stops.  The state dict at the epoch of minimum validation loss is deep-copied
and restored before the test pass, so the saved checkpoint is always the best
one, never the last one.


--------------------------------------------------------------------------------
9.  Artifacts (downstream plotting scripts read these -- do not change them)
--------------------------------------------------------------------------------

Per line, in ``<base_dir>/runs/<run_tag>/``, with ``stem = emulator_<line_file>``:

    <stem>.pth                 torch checkpoint: model_state_dict plus the full
                               architecture description, the parameter
                               normalisation vectors, lambda_min/lambda_max, the
                               loss curves and the best epoch
    <stem>_loss.json           train_losses, val_losses, best_epoch,
                               best_val_loss, epochs_run, stopped_early
    <stem>_split.json          train/val/test model ids
    <stem>_test_outputs.npz    line_file, test_model_ids, wavelengths_phys,
                               wavelengths_norm, target_residual, pred_residual,
                               target_flux, pred_flux   (compressed)
    <stem>_meta.json           counts, wavelength range, crop/pad bookkeeping,
                               training_seconds, best epoch and loss

plus, per run directory: ``parameter_normalization.json``, ``run_config.json``,
``training_summary.json`` and (zstd reader only) ``selected_lines.json``.

Two format differences between sources are preserved rather than harmonised:

  * ``backup_orig`` does **not** write ``wavelengths_phys`` into the npz (its
    dataset builder never returned the physical wavelength array), and its
    checkpoint records ``dropout`` but not ``activation_sequence``,
    ``dropout_after_layer_indices`` or ``use_batch_norm``.
  * the HDF5 configuration records ``line_index``, ``dataset_dir`` and
    ``data_source`` instead of ``line_group``, and its meta carries the
    ``skipped`` counter dictionary instead of the individual crop counters.

**Resume policy (all sources):** only fully completed lines are saved.  A line
counts as complete when all five artifacts exist.  An interrupted line writes
nothing, and the next submission of the same job skips the completed ones.


--------------------------------------------------------------------------------
10.  Walltime guard
--------------------------------------------------------------------------------

The jobs ran on a SLURM queue that kills tasks at the walltime.  Being killed
inside ``torch.save`` corrupts a checkpoint, so the scripts watch the clock and
raise ``DeadlineReached`` at a safe point -- between epochs, between lines, or
while scanning models during a dataset build -- then write the summary and exit
cleanly.  The deadline is the earlier of

    SLURM_TIMELIMIT - ML13_STOP_BEFORE_TIMEOUT_SECONDS
    run start + ML13_MAX_RUNTIME_SECONDS

and both are unset by default, in which case there is no deadline at all.  The
sbatch scripts used ``ML13_MAX_RUNTIME_SECONDS=81000`` and
``ML13_STOP_BEFORE_TIMEOUT_SECONDS=1800`` against a 24 h walltime.

The margin required before a *new line* is started differs between sources and
is a configuration key: 5400 s (env-overridable) for the two line-level-filter
scripts, a hard-coded 180 s for ``orig_silu_fm64`` and ``anja_style``.


--------------------------------------------------------------------------------
11.  How the runs were actually launched, and the matching CLI
--------------------------------------------------------------------------------

The zstd campaigns were submitted as three *line groups* over the sorted pool of
non-rotational line files, truncated to ``ML13_TOTAL_LINES_FOR_GROUPING``
entries: group 1 = lines 1-10, group 2 = lines 11-20, group 3 = the rest.

    run_13par_group{1,2,3}.sbatch                      backup_orig,      34 lines
    run_13par_orig_silu_fm64_group{1,2,3}.sbatch       orig_silu_fm64,   34 lines
    run_13par_deep_relu_nodrop_nobn_group1.sbatch      adopted,          34 lines
    run_13par_deep_relu_nodrop_nobn_group2.sbatch      adopted,          34 lines
    run_13par_deep_relu_nodrop_nobn_group3.sbatch      adopted,          37 lines *

    * group 3 of the adopted run was re-submitted with
      ML13_TOTAL_LINES_FOR_GROUPING=37 and the 37-line filter JSON, to include
      the three Si/P lines (PV1118, SiIII4552, SiIV1400) that the original
      34-line cap silently dropped from the sorted pool.  The first 34 entries
      of the 37-line JSON are identical to the 34-line one.

The HDF5 campaigns were submitted as a SLURM *array*, one array task per line:

    run_13par_newh5_array.sbatch            #SBATCH --array=1-37   (Sobol)
    run_13par_newh5_array_sobol_lhc.sbatch  #SBATCH --array=1-37   (merged Sobol+LHC)

with ``ML13_LINE_INDEX=$SLURM_ARRAY_TASK_ID`` (1-based, into the sorted
filter-JSON line list) and a run tag suffixed ``_line%02d``.

The CLI mirrors all three access patterns::

    # list every configuration
    python thirteen_parameter_training.py --list

    # the adopted Latin-hypercube run, one group at a time (as submitted)
    python thirteen_parameter_training.py --config deep_relu_nodrop_nobn \
        --line-group 1 --total-lines-for-grouping 34 \
        --run-tag deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad_group1

    python thirteen_parameter_training.py --config deep_relu_nodrop_nobn \
        --line-group 3 --total-lines-for-grouping 37 \
        --filter-json <storage>/ml13_training_filter_config_linelevel_absflux15_1p5vinf_37lines.json

    # the adopted Sobol run, one array index at a time (as submitted)
    python thirteen_parameter_training.py --config deep_relu_nodrop_nobn_newh5 \
        --dataset-dir <storage>/.../final_unique/sobol_dataset_hdf5 \
        --line-index 5

    # a single named line, either reader
    python thirteen_parameter_training.py --config deep_relu_nodrop_nobn \
        --line OUT.HGAMMA_VTV010.zst

    # inspect without training
    python thirteen_parameter_training.py --config anja_style --print-config
    python thirteen_parameter_training.py --config backup_orig --line-group 2 --dry-run

No personal or cluster path is required.  Precedence is: command-line flag >
``ML13_*`` environment variable > the portable data/ and outputs/ directories
documented in the repository README.
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# SECTION 1.  FIXED PHYSICS / DATA CONSTANTS
# =============================================================================
# These were identical in all five sources.  They are collected here so that the
# whole data contract is visible in one place.

#: The 13 columns handed to the branch network, in this exact order.  The order
#: is part of the checkpoint contract: inference code indexes into it.
PARAM_COLS = [
    "teff",
    "logg",
    "radius",
    "mdot",
    "yhe",
    "C",
    "N",
    "O",
    "beta",
    "vinf",
    "fic",
    "fvel",
    "fclump",
]

#: Number of branch inputs.  Hard-coded as the literal 13 in every source.
INPUT_DIM = 13

#: Rows read from each ``OUT.*`` line file.  FASTWIND writes 161
#: wavelength/flux pairs followed by a footer that must not be parsed.
MAX_ROWS_PER_LINE_FILE = 161

#: A profile contributing fewer than this many points -- before or after the
#: crop -- is dropped.
MIN_POINTS_PER_LINE = 20

#: Residual-target settings.  The network predicts (flux - FLUX_OFFSET).
USE_RESIDUAL = True
FLUX_OFFSET = 1.0

#: Amplitude-weighting strength of the loss:  w = 1 + ALPHA * |residual|.
ALPHA = 5.0

#: Train / validation / test split.  Two successive calls to
#: sklearn.train_test_split with this seed: 0.30, then 0.50 of the remainder,
#: i.e. 70 % / 15 % / 15 %.  Applied per line, to that line's own model list.
SPLIT_SEED = 42
SPLIT_TEST_SIZE_FIRST = 0.30
SPLIT_TEST_SIZE_SECOND = 0.50

#: A line needs at least this many models before it can be split at all.
MIN_MODELS_FOR_SPLIT = 3

#: Optimiser.  Adam; weight decay was 0.0 in every source.
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0

#: LR scheduler: ReduceLROnPlateau on the validation loss.
SCHEDULER = "reduce_on_plateau"
SCHEDULER_MODE = "min"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10

#: Maximum epochs.  300 in every source.
MAX_EPOCHS = 300

#: Deadline margins, in seconds, required before continuing.  These literals
#: appear inline in the sources and are not configurable there.
MIN_SECONDS_BEFORE_EPOCH = 60
MIN_SECONDS_DURING_DATASET_BUILD = 120

#: Every this many models scanned, the zstd dataset builder re-checks the
#: deadline (and always on the first model).
DATASET_BUILD_CHECK_EVERY = 1000

#: Every this many HDF5 rows scanned, the HDF5 dataset builder re-checks the
#: deadline (row % 50000 == 0, so also on row 0).
H5_BUILD_CHECK_EVERY = 50000

#: nn.DataParallel is used when this many CUDA devices or more are visible.
DATA_PARALLEL_MIN_GPUS = 2

#: Line grouping used by the zstd campaigns: 10, 10, rest.
GROUP_SPLITS = {
    "1": (0, 10),
    "2": (10, 20),
    "3": (20, None),
    "all": (0, None),
}

#: Default size of the pool that grouping is applied to.  The sbatch scripts
#: used 34 everywhere except group 3 of the adopted run, which used 37.
DEFAULT_TOTAL_LINES_FOR_GROUPING = 34

#: Default number of wavelength points a cropped profile is padded back to.
DEFAULT_TARGET_WAVELENGTH_POINTS = MAX_ROWS_PER_LINE_FILE  # 161


# =============================================================================
# SECTION 2.  PORTABLE PATHS AND ENVIRONMENT DEFAULTS
# =============================================================================
# Large datasets and outputs are deliberately kept outside version control.  A
# researcher may point to them with CLI arguments or environment variables;
# otherwise the scripts use the documented data/ and outputs/ tree beside this
# file.  The historical ML13_* variables remain supported for existing jobs.

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(
    os.environ.get("FASTWIND_EMULATOR_ROOT", SCRIPT_DIR)
).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("FASTWIND_DATA_ROOT", REPOSITORY_ROOT / "data")
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("FASTWIND_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs")
).expanduser()
DEFAULT_BASE_DIR = os.environ.get(
    "ML13_BASE_DIR", str(OUTPUT_ROOT / "thirteen_parameter")
)
DEFAULT_STORAGE_DIR = os.environ.get(
    "ML13_STORAGE_DIR", str(DATA_ROOT / "thirteen_parameter")
)

#: Sub-paths of the storage directory, as hard-coded in the zstd sources.
MODELS_ROOT_BASENAME = "all_models_more_info"
GRID_JSON_BASENAME = "grid_large_log_LHC.json"
QUALITY_JSON_BASENAME = "grid_quality_large_lhc.json"

#: The HDF5 dataset directory.  The sources have NO default for this: the
#: script exits if ML13_NEW_DATASET_DIR is unset.  That behaviour is preserved.
DEFAULT_NEW_DATASET_DIR = os.environ.get("ML13_NEW_DATASET_DIR", "")

#: Names of the files inside an HDF5 dataset directory.
MODELS_H5_BASENAME = "models.h5"
OUT_LINES_H5_BASENAME = "out_lines.h5"
MANIFEST_JSON_BASENAME = "manifest.json"

DEFAULT_NUM_WORKERS = int(os.environ.get("ML13_NUM_WORKERS", "4"))
DEFAULT_ZSTD_BIN = os.environ.get("ML13_ZSTD_BIN", "zstd")

DEFAULT_STOP_BEFORE_TIMEOUT_SECONDS = int(
    os.environ.get("ML13_STOP_BEFORE_TIMEOUT_SECONDS", "300")
)
_MAX_RUNTIME_ENV = os.environ.get("ML13_MAX_RUNTIME_SECONDS")
DEFAULT_MAX_RUNTIME_SECONDS = None if _MAX_RUNTIME_ENV is None else int(_MAX_RUNTIME_ENV)

#: Margin before starting a new line, for the configurations that read it from
#: the environment (deep_relu_nodrop_nobn and backup_orig).
DEFAULT_MIN_SECONDS_BEFORE_NEW_LINE = int(
    os.environ.get("ML13_MIN_SECONDS_BEFORE_NEW_LINE", "5400")
)

#: Quality-control filter JSON basenames, as used by the sbatch scripts.  These
#: are documentation as much as defaults; --filter-json overrides them.
FILTER_JSON_LINELEVEL_34 = "ml13_training_filter_config_linelevel_absflux15_1p5vinf.json"
FILTER_JSON_LINELEVEL_37 = (
    "ml13_training_filter_config_linelevel_absflux15_1p5vinf_37lines.json"
)
FILTER_JSON_GLOBAL = "ml13_training_filter_config.json"
FILTER_JSON_H5_DEFAULT = "training_filter_config_linelevel_absflux15_1p5vinf.json"

#: The two HDF5 campaigns, for reference.  Their locations are portable
#: defaults under FASTWIND_DATA_ROOT and can be replaced with --dataset-dir and
#: --filter-json.  No machine-specific cluster path is retained.
H5_CAMPAIGNS = {
    "sobol": {
        "sbatch": "run_13par_newh5_array.sbatch",
        "array": "1-37",
        "dataset_dir": str(DATA_ROOT / "thirteen_parameter" / "sobol_dataset_hdf5"),
        "filter_json": str(
            DATA_ROOT
            / "thirteen_parameter"
            / "ml13_sobol_training_filter_config_linelevel_absflux15_1p5vinf_nocorrupt.json"
        ),
        "run_tag_template": (
            "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
            "_lat128_fm32_relu_nodrop_nobn_sobol_h5_nocorrupt_line{index:02d}"
        ),
        "note": "267079 unique complete models, 37 lines",
    },
    "merged_sobol_lhc": {
        "sbatch": "run_13par_newh5_array_sobol_lhc.sbatch",
        "array": "1-37",
        "dataset_dir": str(
            DATA_ROOT / "thirteen_parameter" / "merged_sobol_lhc_hdf5"
        ),
        "filter_json": str(
            DATA_ROOT
            / "thirteen_parameter"
            / "ml13_merged_training_filter_config_linelevel_absflux15_1p5vinf_sobol_lhc_nocorrupt.json"
        ),
        "run_tag_template": (
            "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
            "_lat128_fm32_relu_nodrop_nobn_merged_sobol_lhc_h5_nocorrupt_line{index:02d}"
        ),
        "note": "merged LHC + Sobol, same training script as the Sobol run",
    },
}

#: The line-group submissions of the zstd campaigns, transcribed from the
#: sbatch files.  ``run_tag`` is exactly the ML13_RUN_TAG that was exported.
ZST_CAMPAIGNS = {
    "deep_relu_nodrop_nobn": [
        {
            "sbatch": "run_13par_deep_relu_nodrop_nobn_group1.sbatch",
            "line_group": "1",
            "total_lines_for_grouping": 34,
            "filter_json_basename": FILTER_JSON_LINELEVEL_34,
            "run_tag": (
                "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
                "_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad_group1"
            ),
        },
        {
            "sbatch": "run_13par_deep_relu_nodrop_nobn_group2.sbatch",
            "line_group": "2",
            "total_lines_for_grouping": 34,
            "filter_json_basename": FILTER_JSON_LINELEVEL_34,
            "run_tag": (
                "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
                "_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad_group2"
            ),
        },
        {
            "sbatch": "run_13par_deep_relu_nodrop_nobn_group3.sbatch",
            "line_group": "3",
            # 37: include the Si/P lines (PV1118, SiIII4552, SiIV1400) that the
            # original 34-line cap silently dropped from the sorted pool.
            "total_lines_for_grouping": 37,
            "filter_json_basename": FILTER_JSON_LINELEVEL_37,
            "run_tag": (
                "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
                "_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad_group3"
            ),
        },
    ],
    "backup_orig": [
        {
            "sbatch": f"run_13par_group{g}.sbatch",
            "line_group": str(g),
            "total_lines_for_grouping": 34,
            "filter_json_basename": FILTER_JSON_LINELEVEL_34,
            "run_tag": (
                "orig_branch128_256_512_trunk256_512_512_lat128_fm32"
                f"_linefilter_cropedgepad_group{g}"
            ),
        }
        for g in (1, 2, 3)
    ],
    "orig_silu_fm64": [
        {
            "sbatch": f"run_13par_orig_silu_fm64_group{g}.sbatch",
            "line_group": str(g),
            "total_lines_for_grouping": 34,
            "filter_json_basename": FILTER_JSON_GLOBAL,
            "run_tag": (
                f"orig_silu_b128_256_512_t256_512_512_lat128_fm64_nodrop_nobn_group{g}"
            ),
        }
        for g in (1, 2, 3)
    ],
    # anja_style: no sbatch file is preserved in 13_par/sbatch_codes/.  See the
    # note on CONFIGS["anja_style"] below.
    "anja_style": [],
}


# =============================================================================
# SECTION 3.  CONFIGS -- one entry per architecture variant
# =============================================================================
#
# Every behavioural difference between the five source files is a key here.  The
# defaults below are the values of the adopted script; each entry states only
# what it changes.  A key that an entry does not list genuinely had the default
# value in that source file.
#
# Keys that disagree between sources and are therefore kept separately:
#   reader, filter_mode, crop_mode, default_filter_json, activation_sequence,
#   dropout, dropout_after_layer_indices, use_batch_norm, fourier_modes,
#   learning_rate, batch_size, early_stop_patience, line_selection,
#   new_line_budget, dataset_build_deadline_checks, paths_from_env,
#   batch_size_from_env, max_epochs_from_env, checkpoint_style,
#   npz_includes_wavelengths_phys and the three *_data_keys lists.

_CONFIG_DEFAULTS = {
    # --- identity ----------------------------------------------------------
    "architecture_name": None,
    "architecture_tag": None,
    "activation_label": "relu",       # the string written as "activation"
    "source": None,                   # source file this entry reproduces
    "adopted": False,
    "description": "",

    # --- data --------------------------------------------------------------
    # "zst"  : per-model directories of zstd-compressed OUT.* files (LHC)
    # "hdf5" : models.h5 + out_lines.h5 (Sobol / merged Sobol+LHC)
    "reader": "zst",
    # "line_level" | "global" | "none"
    "filter_mode": "line_level",
    "default_filter_json": FILTER_JSON_LINELEVEL_34,
    # "crop_pad" | "crop_fixed_length" | "none"
    "crop_mode": "crop_pad",
    "target_wavelength_points": DEFAULT_TARGET_WAVELENGTH_POINTS,
    # Only the adopted script and backup_orig checked the deadline while
    # scanning models inside the dataset build.
    "dataset_build_deadline_checks": True,

    # --- architecture ------------------------------------------------------
    "branch_widths": [128, 256, 512, 512, 1024, 2048],
    "trunk_widths": [128, 256, 512, 512, 1024, 2048],
    "activation_sequence": ["relu", "relu", "relu", "relu", "relu", "relu"],
    "dropout": 0.0,
    "dropout_after_layer_indices": [],
    "use_batch_norm": False,
    "latent_dim": 128,
    "fourier_modes": 32,

    # --- optimisation ------------------------------------------------------
    "learning_rate": 3e-4,
    "batch_size": 1024,
    "batch_size_from_env": True,      # backup_orig hard-coded BATCH_SIZE = 4096
    "max_epochs": MAX_EPOCHS,
    "max_epochs_from_env": False,     # only the HDF5 script read ML13_MAX_EPOCHS
    "early_stop_patience": 30,
    "weight_decay": WEIGHT_DECAY,
    "alpha": ALPHA,
    "use_residual": USE_RESIDUAL,
    "flux_offset": FLUX_OFFSET,
    "split_seed": SPLIT_SEED,

    # --- run layout --------------------------------------------------------
    # "group" : ML13_LINE_GROUP over the sorted non-rotational pool
    # "index" : ML13_LINE_INDEX / ML13_LINE_FILE, one line per job
    "line_selection": "group",
    # seconds of budget required before a new line is started:
    # "env" -> ML13_MIN_SECONDS_BEFORE_NEW_LINE (default 5400), or an int
    "new_line_budget": "env",
    # Whether the historical source read BASE_DIR and STORAGE_DIR from the
    # environment.  The public resolver is portable for both values.
    "paths_from_env": True,

    # --- artifacts ---------------------------------------------------------
    # "deep"          : activation_sequence + dropout + dropout_after + bn
    # "orig_notebook" : dropout only
    # "hdf5"          : line_index / dataset_dir / data_source instead of line_group
    "checkpoint_style": "deep",
    "npz_includes_wavelengths_phys": True,
    "checkpoint_data_keys": [
        "crop_range",
        "skipped_crop_empty",
        "skipped_crop_too_long",
        "padded_left_points",
        "padded_right_points",
        "line_filter_valid_model_count",
    ],
    "meta_data_keys": [
        "crop_range",
        "skipped_crop_empty",
        "skipped_crop_too_long",
        "padded_left_points",
        "padded_right_points",
        "line_filter_valid_model_count",
    ],
    "summary_data_keys": [
        "crop_range",
        "skipped_crop_empty",
        "skipped_crop_too_long",
        "padded_left_points",
        "padded_right_points",
        "line_filter_valid_model_count",
    ],
}


def _config(**overrides):
    """Return a full configuration: the defaults updated by ``overrides``."""
    cfg = copy.deepcopy(_CONFIG_DEFAULTS)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise KeyError(f"Unknown configuration keys: {unknown}")
    cfg.update(copy.deepcopy(overrides))
    return cfg


CONFIGS = {
    # -----------------------------------------------------------------------
    # THE ADOPTED ARCHITECTURE, Latin-hypercube campaign.
    #
    # Deep six-layer branch and trunk, ReLU throughout, no dropout and no batch
    # normalisation, so that train and validation losses are directly comparable
    # between lines apart from normal data-shuffling effects.
    #
    #   Branch: 13 -> 128 -> 256 -> 512 -> 512 -> 1024 -> 2048 -> 128
    #   Trunk : Fourier(32) -> 128 -> 256 -> 512 -> 512 -> 1024 -> 2048 -> 128
    #
    # Source: 13_emulator_deep_relu_nodrop_nobn.py
    # -----------------------------------------------------------------------
    "deep_relu_nodrop_nobn": _config(
        architecture_name="deep_relu_nodrop_nobn_deeponet",
        architecture_tag=(
            "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
            "_lat128_fm32_relu_nodrop_nobn"
        ),
        activation_label="relu",
        source="13_emulator_deep_relu_nodrop_nobn.py",
        adopted=True,
        description=(
            "Adopted 13-parameter emulator: deep ReLU DeepONet, no dropout, no "
            "batch norm, per-line QC filter with crop and edge padding, trained "
            "on the Latin-hypercube grid read from per-model zstd directories."
        ),
    ),

    # -----------------------------------------------------------------------
    # THE ADOPTED ARCHITECTURE, HDF5 datasets (Sobol and merged Sobol+LHC).
    #
    # Identical network, loss, split and artifact formats.  What differs:
    #   - parameters come from models.h5, not grid_large_log_LHC.json;
    #   - spectra come from out_lines.h5, no zstd decompression;
    #   - the parameter min-max is computed over ALL models in models.h5,
    #     not over the filtered subset (recorded in parameter_normalization.json
    #     as normalization_population = "all_models_in_models_h5");
    #   - one line per job, selected by ML13_LINE_INDEX (1-based) or
    #     ML13_LINE_FILE, mirroring #SBATCH --array=1-37;
    #   - the filter JSON is mandatory and its keys are the raw line names;
    #   - ML13_MAX_EPOCHS is honoured (it is not, in the zstd script).
    #
    # NOTE: architecture_name gains the "_newh5" suffix but architecture_tag is
    # byte-for-byte the same string as the entry above, exactly as in the
    # sources -- the two campaigns are meant to be directly comparable.
    #
    # Source: 13_emulator_deep_relu_nodrop_nobn_newh5.py
    # -----------------------------------------------------------------------
    "deep_relu_nodrop_nobn_newh5": _config(
        architecture_name="deep_relu_nodrop_nobn_deeponet_newh5",
        architecture_tag=(
            "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
            "_lat128_fm32_relu_nodrop_nobn"
        ),
        activation_label="relu",
        source="13_emulator_deep_relu_nodrop_nobn_newh5.py",
        adopted=True,
        description=(
            "The adopted architecture on the merged HDF5 datasets; one SLURM "
            "array task per line.  Used for the Sobol and merged Sobol+LHC "
            "campaigns."
        ),
        reader="hdf5",
        filter_mode="line_level",
        default_filter_json=FILTER_JSON_H5_DEFAULT,
        crop_mode="crop_pad",
        line_selection="index",
        max_epochs_from_env=True,
        checkpoint_style="hdf5",
        checkpoint_data_keys=["crop_range"],
        meta_data_keys=[
            "crop_range",
            "skipped",
            "padded_left_points",
            "padded_right_points",
        ],
        summary_data_keys=[],
    ),

    # -----------------------------------------------------------------------
    # Earlier variant: the original three-layer widths, but with SiLU
    # activations and 64 Fourier modes instead of ReLU and 32, to test whether a
    # smoother activation plus a richer trunk basis helps at the original size.
    #
    #   Branch: 13 -> 128 -> 256 -> 512 -> 128
    #   Trunk : Fourier(64) -> 256 -> 512 -> 512 -> 128
    #
    # It also predates the per-line QC filter: it reads the older, GLOBAL
    # ml13_training_filter_config.json, and its crop path rejects any profile
    # whose cropped length differs from the first one seen instead of padding.
    # Larger batch (4096) and a larger learning rate (5e-4) than the adopted
    # run, and patience 40 rather than 30.
    #
    # Source: 13_emulator_orig_silu_fm64.py
    # -----------------------------------------------------------------------
    "orig_silu_fm64": _config(
        architecture_name="original_width_silu_fm64_deeponet",
        architecture_tag="orig_silu_b128_256_512_t256_512_512_lat128_fm64_nodrop_nobn",
        activation_label="silu",
        source="13_emulator_orig_silu_fm64.py",
        description=(
            "Original-width DeepONet with SiLU activations and 64 Fourier "
            "modes; global (not per-line) QC filter; fixed-cropped-length "
            "rejection instead of edge padding."
        ),
        filter_mode="global",
        default_filter_json=FILTER_JSON_GLOBAL,
        crop_mode="crop_fixed_length",
        dataset_build_deadline_checks=False,
        branch_widths=[128, 256, 512],
        trunk_widths=[256, 512, 512],
        activation_sequence=["silu", "silu", "silu"],
        fourier_modes=64,
        learning_rate=5e-4,
        batch_size=4096,
        early_stop_patience=40,
        new_line_budget=180,
        checkpoint_data_keys=[
            "crop_range",
            "skipped_crop_empty",
            "skipped_crop_length_mismatch",
        ],
        meta_data_keys=[
            "crop_range",
            "skipped_crop_empty",
            "skipped_crop_length_mismatch",
        ],
        summary_data_keys=[
            "crop_range",
            "skipped_crop_empty",
            "skipped_crop_length_mismatch",
        ],
    ),

    # -----------------------------------------------------------------------
    # Variant transplanted from the deep, regularised network of Anja's Chapter
    # 3: same six-layer widths as the adopted run, but with BatchNorm before
    # every hidden linear layer, 30 % dropout after the second 512-wide block
    # and after the 2048-wide block (indices 3 and 5), and a mixed activation
    # sequence -- LeakyReLU(0.01) for the first two layers, SiLU for the rest.
    #
    # It is the ONLY configuration that applies no quality filter at all: it
    # reads no filter JSON, does not prune the parameter table, and does not
    # crop.  Its historical source hard-coded BASE_DIR and STORAGE_DIR, but the
    # consolidated public script resolves them portably like every other run.
    #
    # No sbatch file for this variant survives in 13_par/sbatch_codes/, so the
    # run tag it was submitted with could not be recovered; the default run tag
    # is therefore the architecture tag, which is what the script itself falls
    # back to.
    #
    # Source: 13_emulator_anja_style.py
    # -----------------------------------------------------------------------
    "anja_style": _config(
        architecture_name="anja_ch3_deep_dropout_deeponet",
        architecture_tag=(
            "deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
            "_lat128_fm32_drop0p3_bn"
        ),
        activation_label="mixed_leaky_relu_silu",
        source="13_emulator_anja_style.py",
        description=(
            "Deep DeepONet with BatchNorm and 30 % dropout, LeakyReLU->SiLU "
            "activation sequence; NO quality filter and NO wavelength crop."
        ),
        filter_mode="none",
        default_filter_json=None,
        crop_mode="none",
        dataset_build_deadline_checks=False,
        activation_sequence=["leaky_relu", "leaky_relu", "silu", "silu", "silu", "silu"],
        dropout=0.30,
        dropout_after_layer_indices=[3, 5],
        use_batch_norm=True,
        new_line_budget=180,
        paths_from_env=False,
        checkpoint_data_keys=[],
        meta_data_keys=[],
        summary_data_keys=[],
    ),

    # -----------------------------------------------------------------------
    # The original baseline: the DeepONet of the five-parameter notebook
    # (fw_emulator_per_line_comparison_hg.ipynb), re-targeted at 13 inputs.
    #
    #   Branch: 13 -> 128 -> 256 -> 512 -> 128     (ReLU)
    #   Trunk : Fourier(32) -> 256 -> 512 -> 512 -> 128   (ReLU)
    #
    # In the source this network is written out as two explicit nn.Sequential
    # blocks rather than through build_mlp.  With activation_sequence
    # ["relu"]*3, dropout 0.0 and use_batch_norm False, build_mlp emits exactly
    # the same module list in exactly the same order, so the state dicts are
    # interchangeable and the generic builder is used here.
    #
    # It uses the modern per-line filter with crop and edge padding, like the
    # adopted run, but keeps the original optimisation settings (lr 5e-4, batch
    # 4096, patience 40).  Two artifact quirks are preserved: BATCH_SIZE is a
    # hard-coded 4096 (ML13_BATCH_SIZE is NOT read), and the npz has no
    # wavelengths_phys array because the builder never returned X_waves.
    #
    # Source: 13_emulator_backup_orig.py
    # -----------------------------------------------------------------------
    "backup_orig": _config(
        architecture_name="original_notebook_deeponet",
        architecture_tag="orig_branch128_256_512_trunk256_512_512_lat128_fm32",
        activation_label="relu",
        source="13_emulator_backup_orig.py",
        description=(
            "Original notebook DeepONet at 13 inputs; per-line QC filter with "
            "crop and edge padding; batch size fixed at 4096; npz without "
            "wavelengths_phys."
        ),
        branch_widths=[128, 256, 512],
        trunk_widths=[256, 512, 512],
        activation_sequence=["relu", "relu", "relu"],
        learning_rate=5e-4,
        batch_size=4096,
        batch_size_from_env=False,
        early_stop_patience=40,
        checkpoint_style="orig_notebook",
        npz_includes_wavelengths_phys=False,
    ),
}

#: The configuration used for every 13-parameter result quoted in the thesis.
DEFAULT_CONFIG_KEY = "deep_relu_nodrop_nobn"

#: Convenience handle.
ADOPTED_CONFIG = CONFIGS[DEFAULT_CONFIG_KEY]


def resolve_config(key):
    """Return a deep copy of the named configuration."""
    if key not in CONFIGS:
        raise KeyError(f"Unknown configuration key: {key!r}. Known: {sorted(CONFIGS)}")
    return copy.deepcopy(CONFIGS[key])


# =============================================================================
# SECTION 4.  WALLTIME GUARD
# =============================================================================
# Transcribed from the sources.  The deadline is the earlier of the SLURM
# walltime minus a safety margin and an optional hard runtime cap.  When neither
# is set there is no deadline and ensure_time_budget is a no-op.

RUN_START_TIME = time.time()
DEADLINE_TS = None


class DeadlineReached(RuntimeError):
    """Raised at a safe point when the walltime budget is nearly exhausted."""


def _slurm_time_limit_seconds():
    """Parse ``SLURM_TIMELIMIT`` (``[days-]HH[:MM[:SS]]``) into seconds."""
    raw = os.environ.get("SLURM_TIMELIMIT")
    if not raw:
        return None

    parts = raw.split("-")
    days = 0
    hhmmss = raw
    if len(parts) == 2:
        days = int(parts[0])
        hhmmss = parts[1]

    vals = [int(x) for x in hhmmss.split(":")]
    if len(vals) == 3:
        h, m, s = vals
    elif len(vals) == 2:
        h, m = vals
        s = 0
    else:
        h = vals[0]
        m = 0
        s = 0
    return days * 86400 + h * 3600 + m * 60 + s


def configure_deadline(stop_before_timeout_seconds, max_runtime_seconds):
    """Set the module-level deadline; return (time_limit_seconds, deadline_ts)."""
    global DEADLINE_TS

    time_limit_seconds = _slurm_time_limit_seconds()
    slurm_deadline_ts = (
        None
        if time_limit_seconds is None
        else RUN_START_TIME + time_limit_seconds - stop_before_timeout_seconds
    )
    runtime_deadline_ts = (
        None if max_runtime_seconds is None else RUN_START_TIME + max_runtime_seconds
    )
    candidates = [ts for ts in (slurm_deadline_ts, runtime_deadline_ts) if ts is not None]
    DEADLINE_TS = min(candidates) if candidates else None
    return time_limit_seconds, DEADLINE_TS


def time_remaining_seconds():
    if DEADLINE_TS is None:
        return None
    return DEADLINE_TS - time.time()


def ensure_time_budget(context, minimum_seconds=0):
    remaining = time_remaining_seconds()
    if remaining is not None and remaining <= minimum_seconds:
        raise DeadlineReached(
            f"Stopping before walltime during {context}. Remaining budget: {remaining:.1f}s"
        )


# =============================================================================
# SECTION 5.  SUMMARY / RESUME BOOKKEEPING
# =============================================================================
# Resume policy, identical in all five sources: a line counts as done only when
# all five artifacts exist.  An interrupted line writes nothing at all, so a
# resubmission simply retrains it from scratch.


def line_artifact_paths(output_dir, line_file):
    line_stem = f"emulator_{line_file}"
    return {
        "model": os.path.join(output_dir, f"{line_stem}.pth"),
        "loss": os.path.join(output_dir, f"{line_stem}_loss.json"),
        "split": os.path.join(output_dir, f"{line_stem}_split.json"),
        "pred": os.path.join(output_dir, f"{line_stem}_test_outputs.npz"),
        "meta": os.path.join(output_dir, f"{line_stem}_meta.json"),
    }


def is_line_complete(output_dir, line_file):
    paths = line_artifact_paths(output_dir, line_file)
    return all(os.path.exists(p) for p in paths.values())


def load_existing_summary(summary_path):
    if not os.path.exists(summary_path):
        return []
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[summary] Could not read existing summary at {summary_path}: {e}", flush=True)
        return []


def save_summary(summary_path, records):
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)


def upsert_summary_record(summary_records, new_record):
    line_file = new_record.get("line_file")
    for i, rec in enumerate(summary_records):
        if rec.get("line_file") == line_file:
            summary_records[i] = new_record
            return
    summary_records.append(new_record)


def load_completed_line_info(output_dir, line_file):
    """Rebuild a summary record for a line whose artifacts already exist."""
    paths = line_artifact_paths(output_dir, line_file)
    out = {
        "line_file": line_file,
        "status": "completed",
    }

    try:
        if os.path.exists(paths["loss"]):
            with open(paths["loss"], "r") as f:
                loss_data = json.load(f)
            out["best_val_loss"] = loss_data.get("best_val_loss")
            out["best_epoch"] = loss_data.get("best_epoch")
            out["epochs_run"] = loss_data.get("epochs_run")
    except Exception as e:
        print(f"[resume] Could not read loss json for {line_file}: {e}", flush=True)

    try:
        if os.path.exists(paths["meta"]):
            with open(paths["meta"], "r") as f:
                meta = json.load(f)
            out.update({
                "n_total_models": meta.get("n_total_models"),
                "n_train": meta.get("n_train"),
                "n_val": meta.get("n_val"),
                "n_test": meta.get("n_test"),
                "n_wavelength_points": meta.get("n_wavelength_points"),
                "lambda_min": meta.get("lambda_min"),
                "lambda_max": meta.get("lambda_max"),
                "training_seconds": meta.get("training_seconds"),
            })
    except Exception as e:
        print(f"[resume] Could not read meta json for {line_file}: {e}", flush=True)

    return out


# =============================================================================
# SECTION 6.  LINE NAMING AND THE QUALITY-FILTER JSON
# =============================================================================


def normalize_line_name(line):
    """Strip the artifact decorations so that any spelling of a line matches.

    ``emulator_OUT.HGAMMA_VTV010.zst_test_outputs.npz`` and
    ``OUT.HGAMMA_VTV010`` both reduce to ``OUT.HGAMMA_VTV010``.
    """
    name = str(line)
    if name.startswith("emulator_"):
        name = name[len("emulator_"):]
    if name.endswith("_test_outputs.npz"):
        name = name[:-len("_test_outputs.npz")]
    if name.endswith(".zst"):
        name = name[:-4]
    return name


def line_short_name(line):
    """``OUT.HGAMMA_VTV010`` -> ``HGAMMA``.  Used in the HDF5 run tag."""
    name = normalize_line_name(line)
    if name.startswith("OUT."):
        name = name[len("OUT."):]
    for suffix in ("_VTV010", "_VT010"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def load_training_filter_config(path, filter_mode, normalize_keys):
    """Read the QC filter JSON.

    ``filter_mode`` selects how the per-line information is consumed:

      * ``"line_level"``: read ``valid_model_ids_by_line`` and let the union of
        those per-line sets *replace* the flat ``valid_model_ids`` list.  This
        is the behaviour of the adopted script, backup_orig and the HDF5 script.
      * ``"global"``: read only the flat ``valid_model_ids`` list; every line
        then uses the same set.  This is orig_silu_fm64.

    ``normalize_keys`` is True for the zstd reader (keys are normalised with
    ``normalize_line_name``) and False for the HDF5 reader (the datasets are
    already keyed by the bare ``OUT.*`` name).

    Returns ``(config, valid_ids, valid_ids_by_line, crop_ranges)``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training filter JSON not found: {path}. "
            "Run prepare_ml13_training_filters.py before training."
        )

    with open(path, "r") as f:
        config = json.load(f)

    status = config.get("status")
    if status != "completed":
        raise RuntimeError(
            f"Training filter JSON is not completed: {path} has status={status!r}"
        )

    def key_of(line):
        return normalize_line_name(line) if normalize_keys else line

    valid_ids_by_line = {}
    if filter_mode == "line_level":
        raw_valid_by_line = config.get("valid_model_ids_by_line", {})
        for line, model_ids in raw_valid_by_line.items():
            valid_ids_by_line[key_of(line)] = {int(model_id) for model_id in model_ids}

    valid_ids = {int(model_id) for model_id in config.get("valid_model_ids", [])}
    if valid_ids_by_line:
        # The union of the per-line lists REPLACES the flat list, exactly as in
        # the sources.  This is what prunes the parameter table.
        valid_ids = set().union(*valid_ids_by_line.values())
    if not valid_ids and filter_mode == "line_level":
        raise RuntimeError(
            f"Training filter JSON has no valid model IDs: {path}. "
            "Expected valid_model_ids_by_line or valid_model_ids."
        )
    if not valid_ids and filter_mode == "global":
        raise RuntimeError(f"Training filter JSON has no valid_model_ids: {path}")

    crop_ranges = {}
    for line, bounds in config.get("crop_ranges", {}).items():
        if len(bounds) != 2:
            raise RuntimeError(f"Invalid crop range for {line}: {bounds}")
        crop_ranges[key_of(line)] = (float(bounds[0]), float(bounds[1]))

    return config, valid_ids, valid_ids_by_line, crop_ranges


def empty_filter():
    """The ``filter_mode = "none"`` case (anja_style): nothing is filtered."""
    return {}, set(), {}, {}


# =============================================================================
# SECTION 7.  PARAMETER HANDLING
# =============================================================================
# log10 of the mass-loss rate FIRST, then min-max scaling of all 13 columns.
# A column of zero range gets range 1.0 so the normalised column is exactly 0.


def normalize_parameters(params_arr):
    """Return ``(params_norm, param_mins, param_maxs, param_range, mdot_idx)``.

    ``params_arr`` must have the 13 columns of ``PARAM_COLS`` in order.
    """
    params_norm = params_arr.copy()

    mdot_idx = PARAM_COLS.index("mdot")
    params_norm[:, mdot_idx] = np.log10(params_arr[:, mdot_idx])

    param_mins = params_norm.min(axis=0)
    param_maxs = params_norm.max(axis=0)
    param_range = np.where(param_maxs - param_mins == 0.0, 1.0, param_maxs - param_mins)
    params_norm = (params_norm - param_mins) / param_range

    return params_norm, param_mins, param_maxs, param_range, mdot_idx


def write_parameter_normalization(output_dir, param_mins, param_maxs, param_range,
                                  mdot_idx, normalization_population=None):
    """Write ``parameter_normalization.json`` in the source format.

    ``normalization_population`` is present only for the HDF5 reader, where the
    sources record that the min/max were taken over every model in models.h5.
    """
    norm_metadata = {
        "param_cols": PARAM_COLS,
        "param_mins_after_mdot_log": param_mins.tolist(),
        "param_maxs_after_mdot_log": param_maxs.tolist(),
        "param_range_after_mdot_log": param_range.tolist(),
        "mdot_index": mdot_idx,
        "mdot_log10_applied": True,
    }
    if normalization_population is not None:
        norm_metadata["normalization_population"] = normalization_population

    with open(os.path.join(output_dir, "parameter_normalization.json"), "w") as f:
        json.dump(norm_metadata, f, indent=2)
    return norm_metadata


def load_parameter_table_zst(runtime, valid_model_ids, apply_filter):
    """Build the parameter table for the per-model-directory reader.

    A model survives when its quality flag is ``green``, its directory exists,
    the grid JSON has all 13 parameters for it and -- when a filter is active --
    it is in ``valid_model_ids``.
    """
    with open(runtime["grid_json_path"], "r") as f:
        grid_data = json.load(f)

    with open(runtime["quality_json_path"], "r") as f:
        quality_data = json.load(f)

    green_model_ids = {
        int(model_id_str)
        for model_id_str, q in quality_data.items()
        if q.get("flag") == "green"
    }

    available_model_ids = {
        int(os.path.basename(d))
        for d in glob.glob(os.path.join(runtime["models_root"], "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    }

    usable_model_ids = sorted(green_model_ids & available_model_ids)
    print(f"Total model directories found: {len(available_model_ids)}", flush=True)
    print(f"Green models with directories: {len(usable_model_ids)}", flush=True)

    grid_rows = []
    for model_id in usable_model_ids:
        values = grid_data.get(str(model_id))
        if values is None:
            continue

        missing = [col for col in PARAM_COLS if col not in values]
        if missing:
            print(f"[WARN] Model {model_id} missing parameters: {missing}", flush=True)
            continue

        row = {"model": model_id}
        for col in PARAM_COLS:
            row[col] = values[col]
        grid_rows.append(row)

    params_df = pd.DataFrame(grid_rows).sort_values("model").reset_index(drop=True)
    print(f"Models kept after parameter filtering: {len(params_df)}", flush=True)

    if apply_filter:
        pre_training_filter_param_count = len(params_df)
        params_df = params_df[params_df["model"].isin(valid_model_ids)].reset_index(drop=True)
        print(
            f"Models kept after training filter JSON: {len(params_df)} / "
            f"{pre_training_filter_param_count}",
            flush=True,
        )

    if len(params_df) == 0:
        raise RuntimeError("No usable models found after filtering.")

    return params_df


def load_parameter_table_h5(models_h5_path):
    """Read ``models.h5``: model ids, the 13-column parameter matrix, column names."""
    import h5py

    with h5py.File(models_h5_path, "r") as h5:
        ds_model_ids = h5["model_ids"][:].astype(np.int64)
        ds_parameters = h5["parameters"][:].astype(np.float64)
        ds_param_cols = [
            c.decode() if isinstance(c, bytes) else str(c) for c in h5["param_cols"][:]
        ]

    print(
        f"models.h5: {ds_model_ids.shape[0]} models, param_cols={ds_param_cols}",
        flush=True,
    )
    if ds_param_cols != PARAM_COLS:
        raise SystemExit(
            f"param_cols mismatch!\n  models.h5: {ds_param_cols}\n  expected : {PARAM_COLS}"
        )
    if len(np.unique(ds_model_ids)) != ds_model_ids.shape[0]:
        raise SystemExit("models.h5 contains duplicate model_ids; expected unique models.")

    return ds_model_ids, ds_parameters


# =============================================================================
# SECTION 8.  READERS
# =============================================================================


def is_base_line_file(filename):
    """True for a non-rotational, zstd-compressed FASTWIND line file."""
    return (
        filename.startswith("OUT.")
        and filename.endswith(".zst")
        and "vrot_" not in filename
    )


def check_zstd_binary(zstd_bin):
    """Fail early if the zstd binary is neither on PATH nor an existing file."""
    probe = subprocess.run(
        ["which", zstd_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if probe.returncode != 0:
        if not os.path.exists(zstd_bin):
            raise RuntimeError(f"zstd binary not found: {zstd_bin}")


def load_fastwind_line_zst(path, zstd_bin, max_rows=MAX_ROWS_PER_LINE_FILE):
    """Decompress one ``OUT.*.zst`` and return ``(wavelength, flux)`` as float32.

    Column 2 is the wavelength in Angstrom, the last column is the
    continuum-normalised flux.  ``max_rows`` stops the parse before the footer.
    """
    proc = subprocess.run(
        [zstd_bin, "-dc", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    text = proc.stdout.decode("utf-8", errors="replace")
    arr = np.loadtxt(io.StringIO(text), max_rows=max_rows)

    if arr.ndim == 1:
        arr = arr[None, :]

    waves = arr[:, 2].astype(np.float32)
    flux = arr[:, -1].astype(np.float32)
    return waves, flux


# =============================================================================
# SECTION 9.  CROPPING AND EDGE PADDING
# =============================================================================
# Used by crop_mode = "crop_pad".  The padding only ever adds points inside the
# empty margin between a crop bound and the outermost surviving FASTWIND point,
# and holds the flux at the nearest surviving value; no original point moves.


def edge_pad_points(start, stop, n_points, side):
    if n_points <= 0:
        return np.empty((0,), dtype=np.float32)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return np.empty((0,), dtype=np.float32)

    if side == "left":
        points = np.linspace(start, stop, n_points + 1, dtype=np.float32)[:-1]
    elif side == "right":
        points = np.linspace(start, stop, n_points + 1, dtype=np.float32)[1:]
    else:
        raise ValueError(f"Unknown padding side: {side}")
    return points


def crop_pad_profile_to_target_length(waves, flux, crop_range, target_points):
    """Pad a cropped profile back to exactly ``target_points`` samples.

    Returns ``(waves, flux, n_left, n_right)``.  Raises ``ValueError`` when the
    profile is longer than the target, or when the margins cannot supply enough
    room -- the caller counts those as ``skipped_crop_too_long``.
    """
    n_kept = int(waves.shape[0])
    if n_kept > target_points:
        raise ValueError(
            f"Cropped profile has {n_kept} points, more than target {target_points}"
        )
    if n_kept == target_points:
        return waves.astype(np.float32), flux.astype(np.float32), 0, 0

    crop_min, crop_max = crop_range
    missing = target_points - n_kept
    first_wave = float(waves[0])
    last_wave = float(waves[-1])

    left_capacity = max(first_wave - crop_min, 0.0)
    right_capacity = max(crop_max - last_wave, 0.0)
    n_left = missing // 2
    n_right = missing - n_left

    if left_capacity <= 0.0:
        n_right += n_left
        n_left = 0
    if right_capacity <= 0.0:
        n_left += n_right
        n_right = 0

    left_waves = edge_pad_points(crop_min, first_wave, n_left, "left")
    right_waves = edge_pad_points(last_wave, crop_max, n_right, "right")

    # If one edge had a numerically unusable gap, place the remaining padding
    # on the other edge. This keeps all original FASTWIND points unchanged.
    missing_after_edges = missing - int(left_waves.size) - int(right_waves.size)
    if missing_after_edges > 0 and left_capacity > right_capacity and left_capacity > 0.0:
        extra = edge_pad_points(crop_min, first_wave, missing_after_edges, "left")
        left_waves = np.sort(np.concatenate([left_waves, extra]).astype(np.float32))
    elif missing_after_edges > 0 and right_capacity > 0.0:
        extra = edge_pad_points(last_wave, crop_max, missing_after_edges, "right")
        right_waves = np.sort(np.concatenate([right_waves, extra]).astype(np.float32))

    if int(left_waves.size) + n_kept + int(right_waves.size) != target_points:
        raise ValueError(
            f"Could not pad cropped profile from {n_kept} to {target_points} points"
        )

    left_flux = np.full(left_waves.shape, float(flux[0]), dtype=np.float32)
    right_flux = np.full(right_waves.shape, float(flux[-1]), dtype=np.float32)

    padded_waves = np.concatenate([
        left_waves,
        waves.astype(np.float32),
        right_waves,
    ]).astype(np.float32)
    padded_flux = np.concatenate([
        left_flux,
        flux.astype(np.float32),
        right_flux,
    ]).astype(np.float32)

    return padded_waves, padded_flux, int(left_waves.size), int(right_waves.size)


# =============================================================================
# SECTION 10.  DATASET BUILDERS
# =============================================================================


def build_dataset_for_line_zst(cfg, runtime, line_file, model_param_norm, filt):
    """Assemble the training tensors for one line from per-model directories.

    ``filt`` is ``(valid_ids, valid_ids_by_line, crop_ranges)``.
    """
    print(f"[DATA] Building dataset for {line_file}", flush=True)

    valid_ids, valid_ids_by_line, crop_ranges = filt
    crop_mode = cfg["crop_mode"]
    target_points = runtime["target_wavelength_points"]

    all_params = []
    all_waves = []
    all_flux = []
    all_model_ids = []

    line_key = normalize_line_name(line_file)
    crop_range = crop_ranges.get(line_key) if crop_mode != "none" else None
    line_valid_model_ids = valid_ids_by_line.get(line_key, valid_ids)

    expected_cropped_points = None
    skipped_crop_empty = 0
    skipped_crop_too_long = 0
    skipped_crop_length_mismatch = 0
    padded_left_points = 0
    padded_right_points = 0

    if crop_range is not None:
        print(
            f"[DATA] {line_file}: applying wavelength crop "
            f"{crop_range[0]:g}-{crop_range[1]:g} A",
            flush=True,
        )
    if cfg["filter_mode"] == "line_level":
        print(
            f"[DATA] {line_file}: using {len(line_valid_model_ids)} valid model IDs "
            "from training filter",
            flush=True,
        )

    for n_seen, model_id in enumerate(sorted(model_param_norm.keys()), start=1):
        if cfg["dataset_build_deadline_checks"] and (
            n_seen == 1 or n_seen % DATASET_BUILD_CHECK_EVERY == 0
        ):
            ensure_time_budget(
                f"dataset build for line {line_file} after scanning {n_seen} models",
                minimum_seconds=MIN_SECONDS_DURING_DATASET_BUILD,
            )

        # Only the line-level configurations restrict per line here; the global
        # and none modes have already pruned model_param_norm (or not at all).
        if cfg["filter_mode"] == "line_level" and model_id not in line_valid_model_ids:
            continue

        model_dir = os.path.join(runtime["models_root"], str(model_id))
        line_path = os.path.join(model_dir, line_file)

        if not os.path.exists(line_path):
            continue

        try:
            waves, flux = load_fastwind_line_zst(
                line_path,
                runtime["zstd_bin"],
                max_rows=MAX_ROWS_PER_LINE_FILE,
            )
        except Exception as e:
            print(f"[WARN] Failed to read {line_path}: {e}", flush=True)
            continue

        if waves.shape[0] < MIN_POINTS_PER_LINE:
            continue

        if crop_range is not None:
            crop_min, crop_max = crop_range
            crop_mask = np.isfinite(waves) & (waves >= crop_min) & (waves <= crop_max)
            if int(crop_mask.sum()) < MIN_POINTS_PER_LINE:
                skipped_crop_empty += 1
                continue
            waves = waves[crop_mask]
            flux = flux[crop_mask]

            if crop_mode == "crop_pad":
                try:
                    waves, flux, n_left, n_right = crop_pad_profile_to_target_length(
                        waves,
                        flux,
                        crop_range,
                        target_points,
                    )
                    padded_left_points += n_left
                    padded_right_points += n_right
                except ValueError:
                    skipped_crop_too_long += 1
                    continue
            elif crop_mode == "crop_fixed_length":
                # orig_silu_fm64: the first cropped profile sets the expected
                # length; anything else is dropped rather than padded.
                if expected_cropped_points is None:
                    expected_cropped_points = int(waves.shape[0])
                elif int(waves.shape[0]) != expected_cropped_points:
                    skipped_crop_length_mismatch += 1
                    continue
        elif crop_mode == "crop_pad" and int(waves.shape[0]) != target_points:
            # No crop range for this line: the profile must already have the
            # target length, otherwise np.stack would fail later.
            print(
                f"[WARN] {line_file} model {model_id}: expected "
                f"{target_points} rows without crop, got {waves.shape[0]}",
                flush=True,
            )
            continue

        all_params.append(model_param_norm[model_id])
        all_waves.append(waves)
        all_flux.append(flux)
        all_model_ids.append(model_id)

    if not all_params:
        raise RuntimeError(f"No usable data for line {line_file}")

    X_params = np.stack(all_params).astype(np.float32)
    X_waves = np.stack(all_waves).astype(np.float32)
    Y_flux = np.stack(all_flux).astype(np.float32)
    model_ids = np.array(all_model_ids, dtype=int)

    lambda_min = float(X_waves.min())
    lambda_max = float(X_waves.max())

    if lambda_max == lambda_min:
        raise RuntimeError(f"Degenerate wavelength range for line {line_file}")

    X_waves_norm = (X_waves - lambda_min) / (lambda_max - lambda_min)

    print(
        f"[DATA] {line_file}: kept {len(all_model_ids)} models, "
        f"{X_waves.shape[1]} wavelength points",
        flush=True,
    )
    if crop_range is not None and crop_mode == "crop_pad":
        print(
            f"[DATA] {line_file}: crop/pad summary "
            f"empty_or_lt20={skipped_crop_empty}, "
            f"too_long_after_crop={skipped_crop_too_long}, "
            f"padded_left_points={padded_left_points}, "
            f"padded_right_points={padded_right_points}",
            flush=True,
        )
    elif crop_range is not None and crop_mode == "crop_fixed_length":
        print(
            f"[DATA] {line_file}: crop skipped "
            f"empty_or_lt20={skipped_crop_empty}, "
            f"length_mismatch={skipped_crop_length_mismatch}",
            flush=True,
        )

    return {
        "X_params": X_params,
        "X_waves": X_waves,
        "X_waves_norm": X_waves_norm,
        "Y_flux": Y_flux,
        "model_ids": model_ids,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "crop_range": crop_range,
        "skipped_crop_empty": skipped_crop_empty,
        "skipped_crop_too_long": skipped_crop_too_long,
        "skipped_crop_length_mismatch": skipped_crop_length_mismatch,
        "padded_left_points": padded_left_points,
        "padded_right_points": padded_right_points,
        "line_filter_valid_model_count": len(line_valid_model_ids),
    }


def build_dataset_for_line_h5(cfg, runtime, line_file, params_norm, row_of_model, filt):
    """Assemble the training tensors for one line from ``out_lines.h5``."""
    import h5py

    out_lines_h5 = runtime["out_lines_h5"]
    print(f"[DATA] Building dataset for {line_file} from {out_lines_h5}", flush=True)
    t0 = time.time()

    _valid_ids, valid_ids_by_line, crop_ranges = filt
    target_points = runtime["target_wavelength_points"]

    crop_range = crop_ranges.get(line_file)
    valid_ids = valid_ids_by_line.get(line_file)
    if crop_range is None or valid_ids is None:
        raise RuntimeError(
            f"Filter JSON has no crop_range/valid IDs for {line_file!r} - "
            "rerun the QC job for this dataset."
        )
    print(f"[DATA] {line_file}: crop {crop_range[0]:g}-{crop_range[1]:g} A", flush=True)
    print(f"[DATA] {line_file}: {len(valid_ids)} valid IDs from filter JSON", flush=True)

    with h5py.File(out_lines_h5, "r") as h5:
        grp = h5["out_lines"][line_file]
        ensure_time_budget(
            f"before HDF5 read for {line_file}",
            minimum_seconds=MIN_SECONDS_DURING_DATASET_BUILD,
        )
        h5_ids = grp["model_ids"][:]
        h5_npts = grp["n_points"][:]
        h5_w = grp["wavelength"][:]
        h5_f = grp["flux"][:]
    print(
        f"[DATA] {line_file}: read {h5_ids.shape[0]} rows in {time.time() - t0:.1f} s",
        flush=True,
    )

    all_params, all_waves, all_flux, all_ids = [], [], [], []
    seen = set()
    skipped = {"duplicate": 0, "no_params": 0, "not_valid": 0, "short": 0,
               "nonfinite": 0, "wrong_length": 0, "crop_empty": 0, "crop_too_long": 0}
    padded_left = padded_right = 0

    for row in range(h5_ids.shape[0]):
        if row % H5_BUILD_CHECK_EVERY == 0:
            ensure_time_budget(
                f"dataset build after {row} rows",
                minimum_seconds=MIN_SECONDS_DURING_DATASET_BUILD,
            )

        mid = int(h5_ids[row])
        if mid in seen:
            skipped["duplicate"] += 1
            continue
        prow = row_of_model.get(mid)
        if prow is None:
            skipped["no_params"] += 1
            continue
        if valid_ids is not None and mid not in valid_ids:
            skipped["not_valid"] += 1
            continue

        npts = int(h5_npts[row])
        if npts < MIN_POINTS_PER_LINE:
            skipped["short"] += 1
            continue
        waves = h5_w[row, :npts]
        flux = h5_f[row, :npts]
        finite = np.isfinite(waves) & np.isfinite(flux)
        if not finite.all():
            waves, flux = waves[finite], flux[finite]
            if waves.shape[0] < MIN_POINTS_PER_LINE:
                skipped["nonfinite"] += 1
                continue

        if crop_range is not None:
            crop_min, crop_max = crop_range
            mask = (waves >= crop_min) & (waves <= crop_max)
            if int(mask.sum()) < MIN_POINTS_PER_LINE:
                skipped["crop_empty"] += 1
                continue
            waves, flux = waves[mask], flux[mask]
            try:
                waves, flux, nl, nr = crop_pad_profile_to_target_length(
                    waves, flux, crop_range, target_points)
                padded_left += nl
                padded_right += nr
            except ValueError:
                skipped["crop_too_long"] += 1
                continue
        elif int(waves.shape[0]) != target_points:
            skipped["wrong_length"] += 1
            continue

        seen.add(mid)
        all_params.append(params_norm[prow])
        all_waves.append(waves.astype(np.float32))
        all_flux.append(flux.astype(np.float32))
        all_ids.append(mid)

    del h5_w, h5_f
    if not all_params:
        raise RuntimeError(f"No usable data for line {line_file}")

    X_params = np.stack(all_params).astype(np.float32)
    X_waves = np.stack(all_waves).astype(np.float32)
    Y_flux = np.stack(all_flux).astype(np.float32)
    model_ids = np.array(all_ids, dtype=int)

    lambda_min = float(X_waves.min())
    lambda_max = float(X_waves.max())
    if lambda_max == lambda_min:
        raise RuntimeError(f"Degenerate wavelength range for {line_file}")
    X_waves_norm = (X_waves - lambda_min) / (lambda_max - lambda_min)

    print(
        f"[DATA] {line_file}: kept {len(all_ids)} models, "
        f"{X_waves.shape[1]} wavelength points "
        f"(build {time.time() - t0:.1f} s)",
        flush=True,
    )
    print(
        f"[DATA] {line_file}: skip summary {skipped}, "
        f"padded_left={padded_left}, padded_right={padded_right}",
        flush=True,
    )

    return {
        "X_params": X_params,
        "X_waves": X_waves,
        "X_waves_norm": X_waves_norm,
        "Y_flux": Y_flux,
        "model_ids": model_ids,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "crop_range": crop_range,
        "skipped": skipped,
        "padded_left_points": padded_left,
        "padded_right_points": padded_right,
    }


# =============================================================================
# SECTION 11.  THE DEEPONET
# =============================================================================


def activation_layer(name):
    key = str(name).lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"gelu"}:
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def build_mlp(
    input_dim,
    output_dim,
    hidden_widths,
    activation_sequence,
    dropout,
    dropout_after_layer_indices,
    use_batch_norm,
):
    """Stack of [BatchNorm?] -> Linear -> activation -> [Dropout?] plus a head.

    The BatchNorm, when enabled, normalises the *input* of each hidden layer.
    The activation for position ``idx`` is ``activation_sequence[idx]``, with the
    last entry reused if the sequence is shorter than the width list.  Dropout
    is inserted only when ``dropout > 0`` and ``idx`` is listed.
    """
    layers = []
    in_dim = input_dim

    for idx, out_dim in enumerate(hidden_widths):
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_dim))

        layers.append(nn.Linear(in_dim, out_dim))

        act_name = (
            activation_sequence[idx]
            if idx < len(activation_sequence)
            else activation_sequence[-1]
        )
        layers.append(activation_layer(act_name))

        if dropout > 0.0 and idx in dropout_after_layer_indices:
            layers.append(nn.Dropout(dropout))

        in_dim = out_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """Maps the 13 normalised stellar parameters to the latent coefficients."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_widths,
        activation_sequence,
        dropout,
        dropout_after_layer_indices,
        use_batch_norm,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_widths=hidden_widths,
            activation_sequence=activation_sequence,
            dropout=dropout,
            dropout_after_layer_indices=dropout_after_layer_indices,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """Maps the normalised wavelength to the latent basis, via Fourier features.

    The feature vector is ``[1, sin(2*pi*n*x), cos(2*pi*n*x)]`` for n = 1..M, so
    the input dimension is ``2*M + 1``.  The leading feature is the constant 1,
    not x -- this is what all five sources do.
    """

    def __init__(
        self,
        output_dim,
        fourier_modes,
        hidden_widths,
        activation_sequence,
        dropout,
        dropout_after_layer_indices,
        use_batch_norm,
    ):
        super().__init__()
        self.fourier_modes = fourier_modes
        in_dim = 2 * fourier_modes + 1
        self.net = build_mlp(
            input_dim=in_dim,
            output_dim=output_dim,
            hidden_widths=hidden_widths,
            activation_sequence=activation_sequence,
            dropout=dropout,
            dropout_after_layer_indices=dropout_after_layer_indices,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, npts, _ = x.shape
        n = self.fourier_modes

        freqs = 2.0 * math.pi * torch.arange(
            1, n + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, n)

        sin_feats = torch.sin(freqs * x)
        cos_feats = torch.cos(freqs * x)
        ones = torch.ones(batch_size, npts, 1, device=x.device, dtype=x.dtype)

        feats = torch.cat([ones, sin_feats, cos_feats], dim=-1)
        feats = feats.view(-1, feats.shape[-1])

        out = self.net(feats)
        out = out.view(batch_size, npts, -1)
        return out


class DeepONetModel(nn.Module):
    """F(theta)(lambda) = sum_k b_k(theta) * t_k(lambda)."""

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net

    def forward(self, params, coords):
        B = self.branch(params)
        T = self.trunk(coords)
        B_expanded = B.unsqueeze(1)
        pred = (T * B_expanded).sum(-1)
        return pred


def build_model(cfg):
    """Instantiate the DeepONet described by ``cfg``.

    For ``backup_orig`` the source writes the two subnetworks out as explicit
    nn.Sequential blocks.  With its activation sequence, zero dropout and no
    batch norm, ``build_mlp`` emits the identical module list in the identical
    order, so the state dicts are interchangeable.
    """
    branch_net = BranchNet(
        input_dim=INPUT_DIM,
        output_dim=cfg["latent_dim"],
        hidden_widths=cfg["branch_widths"],
        activation_sequence=cfg["activation_sequence"],
        dropout=cfg["dropout"],
        dropout_after_layer_indices=cfg["dropout_after_layer_indices"],
        use_batch_norm=cfg["use_batch_norm"],
    )
    trunk_net = TrunkNet(
        output_dim=cfg["latent_dim"],
        fourier_modes=cfg["fourier_modes"],
        hidden_widths=cfg["trunk_widths"],
        activation_sequence=cfg["activation_sequence"],
        dropout=cfg["dropout"],
        dropout_after_layer_indices=cfg["dropout_after_layer_indices"],
        use_batch_norm=cfg["use_batch_norm"],
    )
    return DeepONetModel(branch_net, trunk_net)


def get_base_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_base_model_state_dict(model, state):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def resolve_device():
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    device = torch.device("cuda:0" if cuda_available else "cpu")
    return device, cuda_available, n_gpus


# =============================================================================
# SECTION 12.  LOSS
# =============================================================================
# The amplitude-weighted MSE on the residual target.  ``targets`` is already
# (flux - FLUX_OFFSET) when cfg["use_residual"] is True, so |targets| is the
# distance from the continuum and the weight w = 1 + alpha*|F - 1| rises to
# 1 + alpha in a saturated line core.  alpha = 5.0 in every source.


def weighted_line_loss(preds, targets, alpha):
    weights = 1.0 + alpha * targets.abs()
    return ((preds - targets) ** 2 * weights).mean()


# =============================================================================
# SECTION 13.  TRAINING ONE LINE
# =============================================================================


def split_indices(n_models, split_seed):
    """The 70/15/15 split, applied per line to that line's own model list."""
    indices = np.arange(n_models)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=SPLIT_TEST_SIZE_FIRST,
        random_state=split_seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=SPLIT_TEST_SIZE_SECOND,
        random_state=split_seed,
    )
    return train_idx, val_idx, test_idx


def _make_loader(X_params, X_waves, Y_flux, idx, batch_size, shuffle, runtime):
    dataset = TensorDataset(
        torch.tensor(X_params[idx], dtype=torch.float32),
        torch.tensor(X_waves[idx], dtype=torch.float32),
        torch.tensor(Y_flux[idx], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        persistent_workers=(runtime["num_workers"] > 0),
    )


def train_one_line(cfg, runtime, line_file, data):
    """Train, early-stop, restore the best checkpoint and run the test pass.

    Returns a dictionary with everything the artifact writers need.  Raises
    ``DeadlineReached`` if the walltime budget runs out inside the epoch loop,
    in which case nothing is saved for this line.
    """
    device = runtime["device"]
    alpha = cfg["alpha"]
    batch_size = runtime["batch_size"]
    max_epochs = runtime["max_epochs"]
    early_stop_patience = cfg["early_stop_patience"]

    X_params_all = data["X_params"]
    X_waves_phys_all = data["X_waves"]
    X_waves_all = data["X_waves_norm"]
    Y_flux_all = data["Y_flux"]
    model_ids_all = data["model_ids"]

    if cfg["use_residual"]:
        Y_flux_all = Y_flux_all - cfg["flux_offset"]

    n_models = X_params_all.shape[0]
    if n_models < MIN_MODELS_FOR_SPLIT:
        raise ValueError(
            f"not enough models ({n_models}) for train/val/test split"
        )

    train_idx, val_idx, test_idx = split_indices(n_models, cfg["split_seed"])

    train_loader = _make_loader(
        X_params_all, X_waves_all, Y_flux_all, train_idx, batch_size, True, runtime
    )
    val_loader = _make_loader(
        X_params_all, X_waves_all, Y_flux_all, val_idx, batch_size, False, runtime
    )
    test_loader = _make_loader(
        X_params_all, X_waves_all, Y_flux_all, test_idx, batch_size, False, runtime
    )

    base_model = build_model(cfg).to(device)

    using_data_parallel = False
    if runtime["cuda_available"] and runtime["n_gpus"] >= DATA_PARALLEL_MIN_GPUS:
        print(f"[GPU] Wrapping model with DataParallel over {runtime['n_gpus']} GPUs",
              flush=True)
        model = nn.DataParallel(base_model)
        using_data_parallel = True
    else:
        print("[GPU] Using single GPU / CPU execution", flush=True)
        model = base_model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        ensure_time_budget(
            f"epoch loop for line {line_file}",
            minimum_seconds=MIN_SECONDS_BEFORE_EPOCH,
        )

        model.train()
        running_train = 0.0

        for batch_params, batch_coords, batch_targets in train_loader:
            batch_params = batch_params.to(device, non_blocking=True)
            batch_coords = batch_coords.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_params, batch_coords)
            loss = weighted_line_loss(preds, batch_targets, alpha)
            loss.backward()
            optimizer.step()
            running_train += loss.item()

        avg_train_loss = running_train / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch_params, batch_coords, batch_targets in val_loader:
                batch_params = batch_params.to(device, non_blocking=True)
                batch_coords = batch_coords.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)

                preds = model(batch_params, batch_coords)
                running_val += weighted_line_loss(preds, batch_targets, alpha).item()

        avg_val_loss = running_val / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Best-checkpoint tracking: strict improvement, no minimum delta.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(get_base_model_state_dict(model))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"[{line_file}] epoch {epoch:03d} "
            f"train={avg_train_loss:.6e} val={avg_val_loss:.6e}",
            flush=True,
        )

        if epochs_no_improve >= early_stop_patience:
            stopped_early = True
            print(f"[{line_file}] Early stop at epoch {epoch}", flush=True)
            break

    # Restore the weights of the minimum-validation-loss epoch before testing,
    # so the saved checkpoint is never the last epoch but always the best one.
    if best_state is not None:
        load_base_model_state_dict(model, best_state)

    test_preds = []
    test_targets = []

    model.eval()
    with torch.no_grad():
        for batch_params, batch_coords, batch_targets in test_loader:
            batch_params = batch_params.to(device, non_blocking=True)
            batch_coords = batch_coords.to(device, non_blocking=True)

            preds = model(batch_params, batch_coords).cpu().numpy()
            test_preds.append(preds)
            test_targets.append(batch_targets.numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    if cfg["use_residual"]:
        test_preds_flux = test_preds + cfg["flux_offset"]
        test_targets_flux = test_targets + cfg["flux_offset"]
    else:
        test_preds_flux = test_preds
        test_targets_flux = test_targets

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "using_data_parallel": using_data_parallel,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "model_ids_all": model_ids_all,
        "X_params_all": X_params_all,
        "X_waves_phys_all": X_waves_phys_all,
        "X_waves_all": X_waves_all,
        "test_preds": test_preds,
        "test_targets": test_targets,
        "test_preds_flux": test_preds_flux,
        "test_targets_flux": test_targets_flux,
    }


# =============================================================================
# SECTION 14.  ARTIFACT WRITERS
# =============================================================================
# The formats below are consumed by 13_plot_after_training.py and by the
# comparison notebooks.  Key names, key order and dtypes are copied from the
# sources; nothing may be renamed.


def _copy_data_keys(target, data, keys):
    for key in keys:
        target[key] = data[key]


def build_checkpoint(cfg, runtime, line_file, data, result, norm, extras):
    """Assemble the ``.pth`` payload in the key order of the source scripts."""
    param_mins, param_maxs, param_range = norm

    ckpt = {
        "model_state_dict": get_base_model_state_dict(result["model"]),
        "architecture_name": cfg["architecture_name"],
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": runtime["run_tag"],
    }

    if cfg["checkpoint_style"] == "hdf5":
        ckpt["line_file"] = line_file
        ckpt["line_index"] = extras["line_index"]
        ckpt["param_cols"] = PARAM_COLS
        ckpt["branch_widths"] = cfg["branch_widths"]
        ckpt["trunk_widths"] = cfg["trunk_widths"]
        ckpt["latent_dim"] = cfg["latent_dim"]
        ckpt["activation"] = cfg["activation_label"]
        ckpt["fourier_modes"] = cfg["fourier_modes"]
        ckpt["learning_rate"] = cfg["learning_rate"]
        ckpt["batch_size"] = runtime["batch_size"]
    else:
        ckpt["line_group"] = extras["line_group"]
        ckpt["param_cols"] = PARAM_COLS
        ckpt["branch_widths"] = cfg["branch_widths"]
        ckpt["trunk_widths"] = cfg["trunk_widths"]
        ckpt["latent_dim"] = cfg["latent_dim"]
        ckpt["activation"] = cfg["activation_label"]
        if cfg["checkpoint_style"] == "deep":
            ckpt["activation_sequence"] = cfg["activation_sequence"]
            ckpt["dropout"] = cfg["dropout"]
            ckpt["dropout_after_layer_indices"] = cfg["dropout_after_layer_indices"]
            ckpt["use_batch_norm"] = cfg["use_batch_norm"]
        else:  # "orig_notebook": the original script recorded dropout only
            ckpt["dropout"] = cfg["dropout"]
        ckpt["fourier_modes"] = cfg["fourier_modes"]
        ckpt["learning_rate"] = cfg["learning_rate"]
        ckpt["batch_size"] = runtime["batch_size"]
        ckpt["line_file"] = line_file

    ckpt["lambda_min"] = data["lambda_min"]
    ckpt["lambda_max"] = data["lambda_max"]
    _copy_data_keys(ckpt, data, cfg["checkpoint_data_keys"])

    ckpt["use_residual"] = cfg["use_residual"]
    ckpt["flux_offset"] = cfg["flux_offset"]
    ckpt["param_mins_after_mdot_log"] = param_mins.tolist()
    ckpt["param_maxs_after_mdot_log"] = param_maxs.tolist()
    ckpt["param_range_after_mdot_log"] = param_range.tolist()
    ckpt["train_losses"] = result["train_losses"]
    ckpt["val_losses"] = result["val_losses"]
    ckpt["best_epoch"] = result["best_epoch"]
    ckpt["best_val_loss"] = result["best_val_loss"]
    ckpt["stopped_early"] = result["stopped_early"]
    ckpt["data_parallel_used"] = result["using_data_parallel"]
    ckpt["n_gpus_visible"] = runtime["n_gpus"]

    if cfg["reader"] == "hdf5":
        ckpt["dataset_dir"] = runtime["dataset_dir"]
        ckpt["data_source"] = "new_merged_hdf5_collection"
    else:
        ckpt["zstd_bin"] = runtime["zstd_bin"]

    return ckpt


def build_meta(cfg, runtime, line_file, data, result, extras, training_seconds):
    meta = {
        "line_file": line_file,
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": runtime["run_tag"],
    }
    if cfg["reader"] == "hdf5":
        meta["line_index"] = extras["line_index"]
    else:
        meta["line_group"] = extras["line_group"]

    meta["n_total_models"] = int(result["X_params_all"].shape[0])
    meta["n_train"] = int(len(result["train_idx"]))
    meta["n_val"] = int(len(result["val_idx"]))
    meta["n_test"] = int(len(result["test_idx"]))
    meta["n_wavelength_points"] = int(result["X_waves_all"].shape[1])
    meta["lambda_min"] = data["lambda_min"]
    meta["lambda_max"] = data["lambda_max"]
    _copy_data_keys(meta, data, cfg["meta_data_keys"])
    meta["training_seconds"] = training_seconds
    meta["best_epoch"] = result["best_epoch"]
    meta["best_val_loss"] = result["best_val_loss"]
    meta["epochs_run"] = len(result["train_losses"])
    meta["stopped_early"] = result["stopped_early"]
    meta["data_parallel_used"] = result["using_data_parallel"]
    meta["n_gpus_visible"] = runtime["n_gpus"]

    if cfg["reader"] == "hdf5":
        meta["dataset_dir"] = runtime["dataset_dir"]
        meta["data_source"] = "new_merged_hdf5_collection"
    else:
        meta["zstd_bin"] = runtime["zstd_bin"]

    return meta


def build_completed_record(cfg, runtime, line_file, data, result, extras):
    """The per-line entry appended to ``training_summary.json``."""
    if cfg["reader"] == "hdf5":
        return {
            "line_file": line_file,
            "status": "completed",
            "run_tag": runtime["run_tag"],
            "line_index": extras["line_index"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "epochs_run": len(result["train_losses"]),
            "stopped_early": result["stopped_early"],
            "n_total_models": int(result["X_params_all"].shape[0]),
            "n_train": int(len(result["train_idx"])),
            "n_val": int(len(result["val_idx"])),
            "n_test": int(len(result["test_idx"])),
        }

    record = {
        "line_file": line_file,
        "status": "completed",
        "architecture_name": cfg["architecture_name"],
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": runtime["run_tag"],
        "line_group": extras["line_group"],
        "branch_widths": cfg["branch_widths"],
        "trunk_widths": cfg["trunk_widths"],
        "latent_dim": cfg["latent_dim"],
        "activation": cfg["activation_label"],
    }
    if cfg["checkpoint_style"] == "deep":
        record["activation_sequence"] = cfg["activation_sequence"]
        record["dropout"] = cfg["dropout"]
        record["dropout_after_layer_indices"] = cfg["dropout_after_layer_indices"]
        record["use_batch_norm"] = cfg["use_batch_norm"]
    else:
        record["dropout"] = cfg["dropout"]
    record["fourier_modes"] = cfg["fourier_modes"]
    record["learning_rate"] = cfg["learning_rate"]
    record["batch_size"] = runtime["batch_size"]
    record["lambda_min"] = data["lambda_min"]
    record["lambda_max"] = data["lambda_max"]
    _copy_data_keys(record, data, cfg["summary_data_keys"])
    record["best_epoch"] = result["best_epoch"]
    record["best_val_loss"] = result["best_val_loss"]
    record["epochs_run"] = len(result["train_losses"])
    record["stopped_early"] = result["stopped_early"]
    record["n_total_models"] = int(result["X_params_all"].shape[0])
    record["n_train"] = int(len(result["train_idx"]))
    record["n_val"] = int(len(result["val_idx"]))
    record["n_test"] = int(len(result["test_idx"]))
    record["data_parallel_used"] = result["using_data_parallel"]
    record["n_gpus_visible"] = runtime["n_gpus"]
    return record


def write_line_artifacts(cfg, runtime, line_file, data, result, norm, extras,
                         training_seconds):
    """Write the five per-line artifacts, in the source formats."""
    output_dir = runtime["output_dir"]
    paths = line_artifact_paths(output_dir, line_file)

    checkpoint = build_checkpoint(cfg, runtime, line_file, data, result, norm, extras)
    torch.save(checkpoint, paths["model"])

    with open(paths["loss"], "w") as f:
        json.dump(
            {
                "line_file": line_file,
                "train_losses": result["train_losses"],
                "val_losses": result["val_losses"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "epochs_run": len(result["train_losses"]),
                "stopped_early": result["stopped_early"],
            },
            f,
            indent=2,
        )

    model_ids_all = result["model_ids_all"]
    with open(paths["split"], "w") as f:
        json.dump(
            {
                "line_file": line_file,
                "train_model_ids": model_ids_all[result["train_idx"]].tolist(),
                "val_model_ids": model_ids_all[result["val_idx"]].tolist(),
                "test_model_ids": model_ids_all[result["test_idx"]].tolist(),
            },
            f,
            indent=2,
        )

    npz_arrays = {
        "line_file": line_file,
        "test_model_ids": model_ids_all[result["test_idx"]],
    }
    if cfg["npz_includes_wavelengths_phys"]:
        npz_arrays["wavelengths_phys"] = result["X_waves_phys_all"][result["test_idx"]]
    npz_arrays["wavelengths_norm"] = result["X_waves_all"][result["test_idx"]]
    npz_arrays["target_residual"] = result["test_targets"]
    npz_arrays["pred_residual"] = result["test_preds"]
    npz_arrays["target_flux"] = result["test_targets_flux"]
    npz_arrays["pred_flux"] = result["test_preds_flux"]
    np.savez_compressed(paths["pred"], **npz_arrays)

    meta = build_meta(cfg, runtime, line_file, data, result, extras, training_seconds)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    return paths


def write_run_config(cfg, runtime, extras, filter_info):
    """Write ``run_config.json``, in the per-reader format of the sources."""
    if cfg["reader"] == "hdf5":
        config = {
            "architecture_name": cfg["architecture_name"],
            "architecture_tag": cfg["architecture_tag"],
            "run_tag": runtime["run_tag"],
            "line_file": extras["line_file"],
            "line_index": extras["line_index"],
            "n_lines_in_dataset": extras["n_lines_in_dataset"],
            "all_line_files": extras["all_line_files"],
            "param_cols": PARAM_COLS,
            "branch_widths": cfg["branch_widths"],
            "trunk_widths": cfg["trunk_widths"],
            "latent_dim": cfg["latent_dim"],
            "activation": cfg["activation_label"],
            "fourier_modes": cfg["fourier_modes"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": runtime["batch_size"],
            "max_epochs": runtime["max_epochs"],
            "early_stop_patience": cfg["early_stop_patience"],
            "weight_decay": cfg["weight_decay"],
            "alpha": cfg["alpha"],
            "use_residual": cfg["use_residual"],
            "flux_offset": cfg["flux_offset"],
            "split_seed": cfg["split_seed"],
            "target_wavelength_points": runtime["target_wavelength_points"],
            "dataset_dir": runtime["dataset_dir"],
            "data_source": "new_merged_hdf5_collection",
            "filter_json_path": runtime["filter_json_path"] or None,
            "n_models_in_dataset": extras["n_models_in_dataset"],
            "cuda_available": runtime["cuda_available"],
            "n_gpus_visible": runtime["n_gpus"],
            "num_workers": runtime["num_workers"],
        }
    else:
        config = {
            "architecture_name": cfg["architecture_name"],
            "architecture_tag": cfg["architecture_tag"],
            "run_tag": runtime["run_tag"],
            "line_group": extras["line_group"],
            "total_lines_for_grouping": runtime["total_lines_for_grouping"],
            "param_cols": PARAM_COLS,
            "branch_widths": cfg["branch_widths"],
            "trunk_widths": cfg["trunk_widths"],
            "latent_dim": cfg["latent_dim"],
            "activation": cfg["activation_label"],
            "activation_sequence": cfg["activation_sequence"],
            "dropout": cfg["dropout"],
            "dropout_after_layer_indices": cfg["dropout_after_layer_indices"],
            "use_batch_norm": cfg["use_batch_norm"],
            "fourier_modes": cfg["fourier_modes"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": runtime["batch_size"],
            "max_epochs": runtime["max_epochs"],
            "early_stop_patience": cfg["early_stop_patience"],
            "weight_decay": cfg["weight_decay"],
            "alpha": cfg["alpha"],
            "use_residual": cfg["use_residual"],
            "flux_offset": cfg["flux_offset"],
            "split_seed": cfg["split_seed"],
            "grid_json_path": runtime["grid_json_path"],
            "quality_json_path": runtime["quality_json_path"],
        }
        if cfg["filter_mode"] != "none":
            filter_config = filter_info["config"]
            config["training_filter_json_path"] = runtime["filter_json_path"]
            config["training_filter_status"] = filter_config.get("status")
            config["training_filter_abs_flux_threshold"] = filter_config.get(
                "abs_flux_threshold"
            )
            config["training_filter_n_valid_model_ids"] = len(filter_info["valid_ids"])
            if cfg["filter_mode"] == "line_level":
                config["training_filter_n_line_valid_model_lists"] = len(
                    filter_info["valid_ids_by_line"]
                )
            config["training_filter_crop_ranges"] = filter_config.get("crop_ranges", {})
        if cfg["crop_mode"] == "crop_pad":
            config["target_wavelength_points"] = runtime["target_wavelength_points"]
        config["n_green_models_used"] = extras["n_green_models_used"]
        config["models_root"] = runtime["models_root"]
        config["all_detected_line_files"] = extras["all_line_files"]
        config["grouping_pool"] = extras["grouping_pool"]
        config["line_files"] = extras["line_files"]
        config["file_mode"] = "non_rotational_zst_only"
        config["cuda_available"] = runtime["cuda_available"]
        config["n_gpus_visible"] = runtime["n_gpus"]
        config["data_parallel_used"] = runtime["n_gpus"] >= DATA_PARALLEL_MIN_GPUS
        config["num_workers"] = runtime["num_workers"]
        config["zstd_bin"] = runtime["zstd_bin"]
        config["stop_before_timeout_seconds"] = runtime["stop_before_timeout_seconds"]
        if cfg["new_line_budget"] == "env":
            config["min_seconds_before_new_line"] = runtime["min_seconds_before_new_line"]
        config["max_runtime_seconds"] = runtime["max_runtime_seconds"]
        config["time_limit_seconds"] = runtime["time_limit_seconds"]

    with open(os.path.join(runtime["output_dir"], "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


# =============================================================================
# SECTION 15.  LINE SELECTION
# =============================================================================


def select_line_group(all_line_files, line_group, total_lines_for_grouping):
    """Apply the 10 / 10 / rest grouping to the sorted non-rotational pool.

    The pool is truncated to ``total_lines_for_grouping`` entries first.  This is
    exactly what made group 3 of the 34-line submissions miss the three Si/P
    lines that the 37-line resubmission recovered.
    """
    selected_pool = sorted(all_line_files)

    if len(selected_pool) > total_lines_for_grouping:
        print(
            f"[grouping] Detected {len(selected_pool)} non-rotational lines; "
            f"restricting to first {total_lines_for_grouping} for 10/10/rest grouping.",
            flush=True,
        )
        selected_pool = selected_pool[:total_lines_for_grouping]

    if line_group not in GROUP_SPLITS:
        raise ValueError(
            f"Invalid line group '{line_group}'. Use one of: {list(GROUP_SPLITS.keys())}"
        )

    start, end = GROUP_SPLITS[line_group]
    chosen = selected_pool[start:end]

    if not chosen:
        raise RuntimeError(
            f"No lines selected for group {line_group}. "
            f"Pool has {len(selected_pool)} lines, split={GROUP_SPLITS[line_group]}"
        )

    return selected_pool, chosen


def discover_zst_line_files(models_root, reference_model_id):
    """List the non-rotational ``.zst`` line files of the reference model dir."""
    ref_dir = os.path.join(models_root, str(reference_model_id))

    if not os.path.isdir(ref_dir):
        raise RuntimeError(f"Reference model directory does not exist: {ref_dir}")

    all_line_files = sorted(f for f in os.listdir(ref_dir) if is_base_line_file(f))

    print(f"Reference model: {reference_model_id}", flush=True)
    print(f"Found {len(all_line_files)} non-rotational compressed line files", flush=True)

    if len(all_line_files) == 0:
        raise RuntimeError(
            f"No non-rotational .zst line files found in reference directory: {ref_dir}"
        )

    return all_line_files


# =============================================================================
# SECTION 16.  CAMPAIGN RUNNERS
# =============================================================================


def run_zst_campaign(cfg, runtime, args):
    """Reproduce one line-group (or single-line) submission of the zstd path."""
    check_zstd_binary(runtime["zstd_bin"])

    # ---- quality filter ----------------------------------------------------
    if cfg["filter_mode"] == "none":
        filter_config, valid_ids, valid_ids_by_line, crop_ranges = empty_filter()
        print("Training filter: disabled for this configuration", flush=True)
    else:
        filter_config, valid_ids, valid_ids_by_line, crop_ranges = (
            load_training_filter_config(
                runtime["filter_json_path"],
                cfg["filter_mode"],
                normalize_keys=True,
            )
        )
        if cfg["filter_mode"] == "line_level":
            print(
                f"Training filter: {len(valid_ids)} valid models across all lines, "
                f"{len(valid_ids_by_line)} line-specific valid-model lists, "
                f"{len(crop_ranges)} crop ranges",
                flush=True,
            )
        else:
            print(
                f"Training filter: {len(valid_ids)} globally valid models, "
                f"{len(crop_ranges)} crop ranges",
                flush=True,
            )

    filter_info = {
        "config": filter_config,
        "valid_ids": valid_ids,
        "valid_ids_by_line": valid_ids_by_line,
        "crop_ranges": crop_ranges,
    }
    filt = (valid_ids, valid_ids_by_line, crop_ranges)

    # ---- parameters --------------------------------------------------------
    params_df = load_parameter_table_zst(
        runtime, valid_ids, apply_filter=(cfg["filter_mode"] != "none")
    )

    params_arr = params_df[PARAM_COLS].values.astype(np.float32)
    params_norm, param_mins, param_maxs, param_range, mdot_idx = normalize_parameters(
        params_arr
    )
    model_param_norm = {
        int(params_df.loc[i, "model"]): params_norm[i] for i in range(len(params_df))
    }
    write_parameter_normalization(
        runtime["output_dir"], param_mins, param_maxs, param_range, mdot_idx
    )
    norm = (param_mins, param_maxs, param_range)

    # ---- line selection ----------------------------------------------------
    reference_model_id = min(model_param_norm.keys())
    all_line_files = discover_zst_line_files(runtime["models_root"], reference_model_id)

    if args.line is not None:
        grouping_pool = sorted(all_line_files)
        if args.line not in grouping_pool:
            raise SystemExit(
                f"--line {args.line!r} is not among the detected line files.\n"
                f"Available: {grouping_pool}"
            )
        line_files = [args.line]
        line_group = runtime["line_group"]
    elif args.line_index is not None:
        grouping_pool, _chosen = select_line_group(
            all_line_files, "all", runtime["total_lines_for_grouping"]
        )
        if not (1 <= args.line_index <= len(grouping_pool)):
            raise SystemExit(
                f"--line-index {args.line_index} out of range 1..{len(grouping_pool)}"
            )
        line_files = [grouping_pool[args.line_index - 1]]
        line_group = runtime["line_group"]
    else:
        line_group = runtime["line_group"]
        grouping_pool, line_files = select_line_group(
            all_line_files, line_group, runtime["total_lines_for_grouping"]
        )

    print(f"[grouping] Pool size used for grouping: {len(grouping_pool)}", flush=True)
    print(
        f"[grouping] Selected {len(line_files)} lines for group {line_group}",
        flush=True,
    )
    for i, lf in enumerate(line_files, start=1):
        print(f"[grouping] {i:02d}: {lf}", flush=True)

    with open(os.path.join(runtime["output_dir"], "selected_lines.json"), "w") as f:
        json.dump(
            {
                "line_group": line_group,
                "total_lines_for_grouping": runtime["total_lines_for_grouping"],
                "grouping_pool": grouping_pool,
                "selected_lines": line_files,
            },
            f,
            indent=2,
        )

    extras = {
        "line_group": line_group,
        "all_line_files": all_line_files,
        "grouping_pool": grouping_pool,
        "line_files": line_files,
        "n_green_models_used": len(model_param_norm),
    }
    write_run_config(cfg, runtime, extras, filter_info)

    if args.dry_run:
        print("\n[dry-run] Nothing was trained.  Lines that would run:", flush=True)
        for lf in line_files:
            print(f"  {lf}", flush=True)
        return 0

    # ---- resume bookkeeping ------------------------------------------------
    summary_path = runtime["summary_path"]
    training_summary = load_existing_summary(summary_path)
    summary_completed = {
        rec.get("line_file")
        for rec in training_summary
        if rec.get("status") == "completed"
    }

    existing_completed_lines = set()
    for line_file in line_files:
        if is_line_complete(runtime["output_dir"], line_file):
            existing_completed_lines.add(line_file)
            if line_file not in summary_completed:
                upsert_summary_record(
                    training_summary,
                    load_completed_line_info(runtime["output_dir"], line_file),
                )

    save_summary(summary_path, training_summary)

    completed_lines = summary_completed | existing_completed_lines
    print(
        f"[resume] Completed lines detected in this group: {len(completed_lines)}",
        flush=True,
    )

    # ---- per-line training -------------------------------------------------
    new_line_budget = (
        runtime["min_seconds_before_new_line"]
        if cfg["new_line_budget"] == "env"
        else cfg["new_line_budget"]
    )

    try:
        for line_file in line_files:
            ensure_time_budget(
                f"before starting line {line_file}",
                minimum_seconds=new_line_budget,
            )

            if line_file in completed_lines:
                print(f"[resume] Skipping already completed line: {line_file}", flush=True)
                continue

            print(f"\n=== Training line: {line_file} ===", flush=True)
            start_time = time.time()

            try:
                data = build_dataset_for_line_zst(
                    cfg, runtime, line_file, model_param_norm, filt
                )
            except DeadlineReached:
                raise
            except Exception as e:
                print(f"[SKIP] {line_file}: {e}", flush=True)
                upsert_summary_record(
                    training_summary,
                    {
                        "line_file": line_file,
                        "status": "skipped_build",
                        "reason": str(e),
                    },
                )
                save_summary(summary_path, training_summary)
                continue

            try:
                result = train_one_line(cfg, runtime, line_file, data)
            except DeadlineReached:
                raise
            except ValueError as e:
                # Too few models to split: recorded, never trained.
                print(f"[SKIP] {line_file}: {e}", flush=True)
                upsert_summary_record(
                    training_summary,
                    {
                        "line_file": line_file,
                        "status": "skipped_split",
                        "reason": str(e),
                    },
                )
                save_summary(summary_path, training_summary)
                continue

            training_seconds = time.time() - start_time
            write_line_artifacts(
                cfg, runtime, line_file, data, result, norm, extras, training_seconds
            )

            upsert_summary_record(
                training_summary,
                build_completed_record(cfg, runtime, line_file, data, result, extras),
            )
            save_summary(summary_path, training_summary)

            completed_lines.add(line_file)

            print(
                f"[DONE] {line_file} finished in {time.time() - start_time:.1f} s",
                flush=True,
            )

    except DeadlineReached as e:
        print(f"[deadline] {e}", flush=True)
        print("[deadline] No partial line checkpoint saved.", flush=True)
        print(f"[deadline] Summary saved to {summary_path}", flush=True)
    else:
        print("\nAll training runs completed.", flush=True)
    finally:
        save_summary(summary_path, training_summary)
        print(f"Final summary written to {summary_path}", flush=True)
        print("\nDone.", flush=True)

    return 0


def run_h5_campaign(cfg, runtime, args):
    """Reproduce one array task of the HDF5 path: exactly one line per process."""
    import h5py

    # ---- quality filter (mandatory, keys are the raw line names) -----------
    filter_config, valid_ids, valid_ids_by_line, crop_ranges = load_training_filter_config(
        runtime["filter_json_path"],
        cfg["filter_mode"],
        normalize_keys=False,
    )
    print(
        f"Filter JSON: {len(valid_ids_by_line)} line entries, "
        f"{len(crop_ranges)} crop ranges",
        flush=True,
    )
    filter_info = {
        "config": filter_config,
        "valid_ids": valid_ids,
        "valid_ids_by_line": valid_ids_by_line,
        "crop_ranges": crop_ranges,
    }
    filt = (valid_ids, valid_ids_by_line, crop_ranges)

    # ---- line selection: the filter JSON's lines, sorted -------------------
    all_line_files = sorted(crop_ranges)
    with h5py.File(runtime["out_lines_h5"], "r") as h5:
        h5_lines = set(h5["out_lines"].keys())
    missing_in_h5 = [line for line in all_line_files if line not in h5_lines]
    if missing_in_h5:
        raise SystemExit(f"Filter JSON lines not present in out_lines.h5: {missing_in_h5}")
    print(
        f"Filter JSON defines {len(all_line_files)} lines "
        f"(out_lines.h5 has {len(h5_lines)})",
        flush=True,
    )

    if args.line is not None:
        if args.line not in all_line_files:
            raise SystemExit(
                f"--line {args.line!r} not in out_lines.h5.\nAvailable: {all_line_files}"
            )
        line_file = args.line
    elif args.line_index is not None:
        if not (1 <= args.line_index <= len(all_line_files)):
            raise SystemExit(
                f"--line-index {args.line_index} out of range 1..{len(all_line_files)}"
            )
        line_file = all_line_files[args.line_index - 1]
    else:
        raise SystemExit(
            "Set --line-index (1-based, mirrors SLURM_ARRAY_TASK_ID) or --line."
        )

    line_index = all_line_files.index(line_file) + 1
    line_short = line_short_name(line_file)
    print(f"Selected line {line_index:02d}/{len(all_line_files)}: {line_file}", flush=True)

    # ---- output directory (depends on the selected line) -------------------
    if runtime["run_tag"] is None:
        runtime["run_tag"] = (
            f"{cfg['architecture_tag']}_newds_h5_line{line_index:02d}_{line_short}"
        )
    finalize_output_dir(runtime)
    print("RUN_TAG:", runtime["run_tag"], flush=True)
    print("OUTPUT_DIR:", runtime["output_dir"], flush=True)

    # ---- parameters, normalised over ALL models in models.h5 ---------------
    ds_model_ids, ds_parameters = load_parameter_table_h5(runtime["models_h5"])
    params_norm, param_mins, param_maxs, param_range, mdot_idx = normalize_parameters(
        ds_parameters
    )
    params_norm = params_norm.astype(np.float32)
    row_of_model = {int(m): i for i, m in enumerate(ds_model_ids)}
    write_parameter_normalization(
        runtime["output_dir"],
        param_mins,
        param_maxs,
        param_range,
        mdot_idx,
        normalization_population="all_models_in_models_h5",
    )
    norm = (param_mins, param_maxs, param_range)

    extras = {
        "line_file": line_file,
        "line_index": line_index,
        "n_lines_in_dataset": len(all_line_files),
        "all_line_files": all_line_files,
        "n_models_in_dataset": int(ds_model_ids.shape[0]),
    }
    write_run_config(cfg, runtime, extras, filter_info)

    if args.dry_run:
        print("\n[dry-run] Nothing was trained.  Line that would run:", flush=True)
        print(f"  {line_index:02d}  {line_file}", flush=True)
        return 0

    # ---- resume check ------------------------------------------------------
    summary_path = runtime["summary_path"]
    training_summary = load_existing_summary(summary_path)
    if is_line_complete(runtime["output_dir"], line_file):
        print(
            f"[resume] Line already completed in {runtime['output_dir']}; nothing to do.",
            flush=True,
        )
        return 0

    # ---- training ----------------------------------------------------------
    try:
        print(f"\n=== Training line: {line_file} ===", flush=True)
        start_time = time.time()

        data = build_dataset_for_line_h5(
            cfg, runtime, line_file, params_norm, row_of_model, filt
        )
        result = train_one_line(cfg, runtime, line_file, data)

        training_seconds = time.time() - start_time
        write_line_artifacts(
            cfg, runtime, line_file, data, result, norm, extras, training_seconds
        )

        upsert_summary_record(
            training_summary,
            build_completed_record(cfg, runtime, line_file, data, result, extras),
        )
        save_summary(summary_path, training_summary)

        print(
            f"[DONE] {line_file} finished in {time.time() - start_time:.1f} s",
            flush=True,
        )

    except DeadlineReached as e:
        print(f"[deadline] {e}", flush=True)
        print("[deadline] No partial line checkpoint saved.", flush=True)
    finally:
        save_summary(summary_path, training_summary)
        print(f"Final summary written to {summary_path}", flush=True)
        print("\nDone.", flush=True)

    return 0


# =============================================================================
# SECTION 17.  COMMAND-LINE INTERFACE
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        prog="thirteen_parameter_training.py",
        description=(
            "Train the 13-parameter FASTWIND line-profile DeepONet emulator. "
            "Every architecture variant of the study is reproducible through "
            "--config; the adopted model is the default, "
            f"{DEFAULT_CONFIG_KEY}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (mirroring the sbatch submissions):\n"
            "  python thirteen_parameter_training.py --list\n"
            "  python thirteen_parameter_training.py --config deep_relu_nodrop_nobn "
            "--line-group 1 --total-lines-for-grouping 34\n"
            "  python thirteen_parameter_training.py --config deep_relu_nodrop_nobn "
            "--line-group 3 --total-lines-for-grouping 37\n"
            "  python thirteen_parameter_training.py --config deep_relu_nodrop_nobn_newh5 "
            "--dataset-dir <dir> --line-index 5\n"
            "  python thirteen_parameter_training.py --config backup_orig "
            "--line OUT.HGAMMA_VTV010.zst\n"
            "  python thirteen_parameter_training.py --config anja_style --print-config\n"
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=f"Configuration key.  Default: {DEFAULT_CONFIG_KEY}.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the configurations and exit."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration as JSON and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything and report what would run, but do not train.",
    )

    lines = parser.add_argument_group("line selection")
    lines.add_argument(
        "--line",
        default=None,
        help=(
            "Train exactly this line.  For the zstd reader give the file name as "
            "it appears on disk (e.g. OUT.HGAMMA_VTV010.zst); for the HDF5 "
            "reader give the out_lines.h5 group name."
        ),
    )
    lines.add_argument(
        "--line-group",
        default=None,
        choices=sorted(GROUP_SPLITS),
        help=(
            "zstd reader only: 10/10/rest grouping of the sorted non-rotational "
            "pool.  Default: the ML13_LINE_GROUP environment variable, or 'all'."
        ),
    )
    lines.add_argument(
        "--line-index",
        type=int,
        default=None,
        help=(
            "1-based index, mirroring SLURM_ARRAY_TASK_ID of the "
            "--array=1-37 submissions.  For the HDF5 reader it indexes the "
            "sorted filter-JSON line list (this is how the Sobol campaign ran); "
            "for the zstd reader it indexes the truncated grouping pool."
        ),
    )
    lines.add_argument(
        "--total-lines-for-grouping",
        type=int,
        default=None,
        help=(
            "zstd reader only: size the sorted pool is truncated to before "
            f"grouping.  Default: {DEFAULT_TOTAL_LINES_FOR_GROUPING} "
            "(the adopted group-3 resubmission used 37)."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--base-dir", default=None, help=f"Default: {DEFAULT_BASE_DIR}")
    paths.add_argument(
        "--storage-dir", default=None, help=f"Default: {DEFAULT_STORAGE_DIR}"
    )
    paths.add_argument(
        "--models-root",
        default=None,
        help=f"Default: <storage-dir>/{MODELS_ROOT_BASENAME}",
    )
    paths.add_argument(
        "--grid-json", default=None, help=f"Default: <storage-dir>/{GRID_JSON_BASENAME}"
    )
    paths.add_argument(
        "--quality-json",
        default=None,
        help=f"Default: <storage-dir>/{QUALITY_JSON_BASENAME}",
    )
    paths.add_argument(
        "--filter-json",
        default=None,
        help=(
            "Quality-control filter JSON.  Default: the config's "
            "default_filter_json under <storage-dir> (zstd) or <dataset-dir> "
            "(HDF5)."
        ),
    )
    paths.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "HDF5 reader only: the final_unique/<TAG> directory holding "
            "models.h5, out_lines.h5 and manifest.json.  There is no default; "
            "the sources exit if it is unset."
        ),
    )
    paths.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Output subdirectory under <base-dir>/runs.  Default: the "
            "architecture tag (zstd) or "
            "<architecture_tag>_newds_h5_line<NN>_<SHORT> (HDF5)."
        ),
    )

    knobs = parser.add_argument_group("run-level overrides")
    knobs.add_argument("--batch-size", type=int, default=None)
    knobs.add_argument("--max-epochs", type=int, default=None)
    knobs.add_argument("--num-workers", type=int, default=None)
    knobs.add_argument("--zstd-bin", default=None)
    knobs.add_argument("--target-wavelength-points", type=int, default=None)
    knobs.add_argument("--stop-before-timeout-seconds", type=int, default=None)
    knobs.add_argument("--max-runtime-seconds", type=int, default=None)
    knobs.add_argument("--min-seconds-before-new-line", type=int, default=None)

    return parser


def _env_int(env, name, fallback):
    raw = env.get(name)
    return fallback if raw in (None, "") else int(raw)


def resolve_runtime(cfg, args, env=None):
    """Build the run-level settings.

    Precedence: explicit command-line flag > ``ML13_*`` environment variable >
    the module constant transcribed from the source file.
    """
    if env is None:
        env = os.environ

    def pick(flag_value, other):
        return other if flag_value is None else flag_value

    # The original anja_style source hard-coded storage paths.  The consolidated
    # public version deliberately uses the same portable path resolution as all
    # other configurations; this changes no numerical training behaviour.
    if cfg["paths_from_env"]:
        base_dir = pick(args.base_dir, env.get("ML13_BASE_DIR", DEFAULT_BASE_DIR))
        storage_dir = pick(
            args.storage_dir, env.get("ML13_STORAGE_DIR", DEFAULT_STORAGE_DIR)
        )
    else:
        base_dir = pick(args.base_dir, DEFAULT_BASE_DIR)
        storage_dir = pick(args.storage_dir, DEFAULT_STORAGE_DIR)

    models_root = args.models_root or os.path.join(storage_dir, MODELS_ROOT_BASENAME)
    grid_json_path = args.grid_json or os.path.join(storage_dir, GRID_JSON_BASENAME)
    quality_json_path = args.quality_json or os.path.join(
        storage_dir, QUALITY_JSON_BASENAME
    )

    dataset_dir = pick(args.dataset_dir, env.get("ML13_NEW_DATASET_DIR", ""))
    models_h5 = os.path.join(dataset_dir, MODELS_H5_BASENAME) if dataset_dir else None
    out_lines_h5 = (
        os.path.join(dataset_dir, OUT_LINES_H5_BASENAME) if dataset_dir else None
    )
    manifest_json = (
        os.path.join(dataset_dir, MANIFEST_JSON_BASENAME) if dataset_dir else None
    )

    # Filter JSON: flag > ML13_FILTER_JSON_PATH > the config's default location.
    if cfg["filter_mode"] == "none":
        filter_json_path = None
    else:
        if cfg["reader"] == "hdf5":
            default_filter = (
                os.path.join(dataset_dir, cfg["default_filter_json"])
                if dataset_dir
                else None
            )
        else:
            default_filter = os.path.join(storage_dir, cfg["default_filter_json"])
        filter_json_path = pick(
            args.filter_json, env.get("ML13_FILTER_JSON_PATH", default_filter)
        )

    batch_size = cfg["batch_size"]
    if cfg["batch_size_from_env"]:
        batch_size = _env_int(env, "ML13_BATCH_SIZE", batch_size)
    batch_size = pick(args.batch_size, batch_size)

    max_epochs = cfg["max_epochs"]
    if cfg["max_epochs_from_env"]:
        max_epochs = _env_int(env, "ML13_MAX_EPOCHS", max_epochs)
    max_epochs = pick(args.max_epochs, max_epochs)

    target_wavelength_points = pick(
        args.target_wavelength_points,
        _env_int(
            env, "ML13_TARGET_WAVELENGTH_POINTS", cfg["target_wavelength_points"]
        ),
    )

    line_group = pick(args.line_group, env.get("ML13_LINE_GROUP", "all").strip().lower())
    total_lines_for_grouping = pick(
        args.total_lines_for_grouping,
        _env_int(
            env, "ML13_TOTAL_LINES_FOR_GROUPING", DEFAULT_TOTAL_LINES_FOR_GROUPING
        ),
    )

    run_tag = pick(args.run_tag, env.get("ML13_RUN_TAG"))
    if run_tag is None and cfg["reader"] != "hdf5":
        run_tag = cfg["architecture_tag"]

    runtime = {
        "config_key": None,
        "base_dir": base_dir,
        "storage_dir": storage_dir,
        "models_root": models_root,
        "grid_json_path": grid_json_path,
        "quality_json_path": quality_json_path,
        "filter_json_path": filter_json_path,
        "dataset_dir": dataset_dir,
        "models_h5": models_h5,
        "out_lines_h5": out_lines_h5,
        "manifest_json": manifest_json,
        "run_tag": run_tag,
        "output_dir": None,
        "summary_path": None,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "target_wavelength_points": target_wavelength_points,
        "line_group": line_group,
        "total_lines_for_grouping": total_lines_for_grouping,
        "num_workers": pick(
            args.num_workers, _env_int(env, "ML13_NUM_WORKERS", DEFAULT_NUM_WORKERS)
        ),
        "zstd_bin": pick(args.zstd_bin, env.get("ML13_ZSTD_BIN", DEFAULT_ZSTD_BIN)),
        "stop_before_timeout_seconds": pick(
            args.stop_before_timeout_seconds,
            _env_int(
                env,
                "ML13_STOP_BEFORE_TIMEOUT_SECONDS",
                DEFAULT_STOP_BEFORE_TIMEOUT_SECONDS,
            ),
        ),
        "min_seconds_before_new_line": pick(
            args.min_seconds_before_new_line,
            _env_int(
                env,
                "ML13_MIN_SECONDS_BEFORE_NEW_LINE",
                DEFAULT_MIN_SECONDS_BEFORE_NEW_LINE,
            ),
        ),
    }

    raw_max_runtime = env.get("ML13_MAX_RUNTIME_SECONDS")
    max_runtime_seconds = None if raw_max_runtime in (None, "") else int(raw_max_runtime)
    runtime["max_runtime_seconds"] = pick(args.max_runtime_seconds, max_runtime_seconds)

    device, cuda_available, n_gpus = resolve_device()
    runtime["device"] = device
    runtime["cuda_available"] = cuda_available
    runtime["n_gpus"] = n_gpus
    runtime["pin_memory"] = cuda_available

    time_limit_seconds, deadline_ts = configure_deadline(
        runtime["stop_before_timeout_seconds"], runtime["max_runtime_seconds"]
    )
    runtime["time_limit_seconds"] = time_limit_seconds
    runtime["deadline_ts"] = deadline_ts

    return runtime


def finalize_output_dir(runtime):
    """Create ``<base_dir>/runs/<run_tag>`` and record the summary path."""
    runs_root = os.path.join(runtime["base_dir"], "runs")
    output_dir = os.path.join(runs_root, runtime["run_tag"])
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    runtime["output_dir"] = output_dir
    runtime["summary_path"] = os.path.join(output_dir, "training_summary.json")
    return output_dir


def print_config_listing():
    print("Available 13-parameter configurations:")
    print()
    for key, cfg in CONFIGS.items():
        marker = "  <-- ADOPTED" if cfg["adopted"] else ""
        print(
            f"  {key:<28} reader={cfg['reader']:<5} "
            f"filter={cfg['filter_mode']:<10} "
            f"crop={cfg['crop_mode']:<18} "
            f"fm={cfg['fourier_modes']:<3} "
            f"drop={cfg['dropout']:<5} "
            f"bn={str(cfg['use_batch_norm']):<5} "
            f"lr={cfg['learning_rate']:<7} "
            f"bs={cfg['batch_size']:<5} "
            f"pat={cfg['early_stop_patience']:<3}"
            f"{marker}"
        )
        print(f"       tag    : {cfg['architecture_tag']}")
        print(f"       source : {cfg['source']}")
    print()
    print("Line-group submissions preserved from sbatch_codes/:")
    for key, campaigns in ZST_CAMPAIGNS.items():
        if not campaigns:
            print(f"  {key:<28} (no sbatch file preserved)")
            continue
        for campaign in campaigns:
            print(
                f"  {key:<28} group={campaign['line_group']} "
                f"pool={campaign['total_lines_for_grouping']} "
                f"{campaign['sbatch']}"
            )
    print()
    print("HDF5 array submissions preserved from sbatch_codes/:")
    for key, campaign in H5_CAMPAIGNS.items():
        print(f"  {key:<18} --array={campaign['array']}  {campaign['sbatch']}")
        print(f"       dataset: {campaign['dataset_dir']}")
        print(f"       filter : {campaign['filter_json']}")


def print_runtime_banner(cfg, runtime, config_key):
    print("Using device:", runtime["device"], flush=True)
    print("CUDA available:", runtime["cuda_available"], flush=True)
    print("Detected GPUs:", runtime["n_gpus"], flush=True)
    print("CONFIG:", config_key, flush=True)
    print("ARCHITECTURE_NAME:", cfg["architecture_name"], flush=True)
    print("ARCHITECTURE_TAG:", cfg["architecture_tag"], flush=True)
    print("BASE_DIR:", runtime["base_dir"], flush=True)
    if cfg["reader"] == "hdf5":
        print("DATASET_DIR:", runtime["dataset_dir"], flush=True)
        print("MODELS_H5:", runtime["models_h5"], flush=True)
        print("OUT_LINES_H5:", runtime["out_lines_h5"], flush=True)
    else:
        print("STORAGE_DIR:", runtime["storage_dir"], flush=True)
        print("MODELS_ROOT:", runtime["models_root"], flush=True)
        print("GRID_JSON_PATH:", runtime["grid_json_path"], flush=True)
        print("QUALITY_JSON_PATH:", runtime["quality_json_path"], flush=True)
        print("ZSTD_BIN:", runtime["zstd_bin"], flush=True)
        print("LINE_GROUP:", runtime["line_group"], flush=True)
        print("TOTAL_LINES_FOR_GROUPING:", runtime["total_lines_for_grouping"], flush=True)
    print(
        "TRAINING_FILTER_JSON_PATH:",
        runtime["filter_json_path"] or "(none: no QC filter, no crop)",
        flush=True,
    )
    print("RUN_TAG:", runtime["run_tag"], flush=True)
    print("OUTPUT_DIR:", runtime["output_dir"], flush=True)
    print("NUM_WORKERS:", runtime["num_workers"], flush=True)
    print("BATCH_SIZE:", runtime["batch_size"], flush=True)
    print("MAX_EPOCHS:", runtime["max_epochs"], flush=True)
    print("EARLY_STOP_PATIENCE:", cfg["early_stop_patience"], flush=True)
    print("TARGET_WAVELENGTH_POINTS:", runtime["target_wavelength_points"], flush=True)
    print("SPLIT_SEED:", cfg["split_seed"], flush=True)
    print("ALPHA:", cfg["alpha"], flush=True)
    print("STOP_BEFORE_TIMEOUT_SECONDS:", runtime["stop_before_timeout_seconds"], flush=True)
    print("MAX_RUNTIME_SECONDS:", runtime["max_runtime_seconds"], flush=True)
    print("TIME_LIMIT_SECONDS:", runtime["time_limit_seconds"], flush=True)
    print("DEADLINE_TS:", runtime["deadline_ts"], flush=True)
    if cfg["new_line_budget"] == "env":
        print(
            "MIN_SECONDS_BEFORE_NEW_LINE:",
            runtime["min_seconds_before_new_line"],
            flush=True,
        )
    else:
        print("MIN_SECONDS_BEFORE_NEW_LINE:", cfg["new_line_budget"], flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_config_listing()
        return 0

    config_key = args.config or DEFAULT_CONFIG_KEY
    if config_key not in CONFIGS:
        parser.error(
            f"Unknown configuration key: {config_key!r}. Try --list."
        )
    cfg = resolve_config(config_key)

    runtime = resolve_runtime(cfg, args)
    runtime["config_key"] = config_key

    if args.print_config:
        print(json.dumps({"config": cfg, "runtime": runtime}, indent=2, default=str))
        return 0

    if cfg["reader"] == "hdf5":
        if not runtime["dataset_dir"]:
            raise SystemExit(
                "Set --dataset-dir (or ML13_NEW_DATASET_DIR) to the "
                "final_unique/<TAG> directory."
            )
        for path, label in (
            (runtime["models_h5"], MODELS_H5_BASENAME),
            (runtime["out_lines_h5"], OUT_LINES_H5_BASENAME),
        ):
            if not os.path.exists(path):
                raise SystemExit(f"Required dataset file not found: {path} ({label})")
        if os.path.exists(runtime["manifest_json"]):
            with open(runtime["manifest_json"], "r") as f:
                manifest = json.load(f)
            print(
                "Manifest:",
                {
                    k: manifest.get(k)
                    for k in ("kind", "unique_model_ids", "n_packages_merged")
                },
                flush=True,
            )
        # The output directory of an HDF5 run depends on the selected line, so
        # it is created inside run_h5_campaign once the line is known.
        print_runtime_banner(cfg, runtime, config_key)
        return run_h5_campaign(cfg, runtime, args)

    finalize_output_dir(runtime)
    print_runtime_banner(cfg, runtime, config_key)
    return run_zst_campaign(cfg, runtime, args)


if __name__ == "__main__":
    sys.exit(main())
