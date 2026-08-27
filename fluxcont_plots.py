#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fluxcont_plots.py
=================

Every figure of Chapter 5 (the 13-parameter FLUXCONT continuum / SED emulator)
and of the FLUXCONT appendix, produced from saved training outputs.

This single file replaces the following research scripts and notebook, without
changing any numerical behaviour:

    fluxcont/FLUXCONT_emulator_performance_plots.ipynb
                                            -> loss-curves        (cell  5)
                                               sed-profiles       (cells 9, 10, 19)
                                               calibration        (cell 12)
                                               binned-error       (cell 14)
                                               per-model-metrics  (cells 16, 17)
                                               parameter-scatter  (cell 21)
    FLUXCONT/FLUXCONT_py_codes/plot_fluxcont_signed_relative_error.py
                                            -> signed-relative-error
    FLUXCONT/FLUXCONT_py_codes/fluxcont_chapter_plots.py
                                            -> worst-corner
    FLUXCONT/FLUXCONT_py_codes/plot_mean_relative_error_vs_parameters.py
                                            -> parameter-trends

and adds one family, ``arch-summary``, which assembles Table
``tab:fluxcont_architecture_metrics`` of the appendix directly from the
``emulator_FLUXCONT_meta.json`` and ``test_metrics_per_model.csv`` files of the
28 completed configurations.  The notebook never produced that table; it was
read off the run folders by hand.

Training and inference live in the companion file ``fluxcont_training.py``.
Nothing here re-runs the network: every figure is a re-reading of the artefacts
that script writes, so a plot can never disagree with the run it describes.


INPUTS
------
All of the printed figures read one *run folder*, the adopted ``v17a``
configuration::

    emulator_FLUXCONT.pth                   checkpoint: weights, the branch
                                            scaling, the wavelength range *and*
                                            train_losses / val_losses /
                                            train_eval_losses / best_epoch /
                                            best_val_loss
    emulator_FLUXCONT_loss.json             the same loss histories, as JSON
    emulator_FLUXCONT_meta.json             run metadata and aggregate test
                                            metrics (n_test, best_epoch,
                                            best_val_loss, test_mse_log_flam,
                                            test_mean_relative_flux_error, ...)
    emulator_FLUXCONT_test_outputs.npz      the flattened native-grid test
                                            predictions: test_model_ids,
                                            model_offsets, model_ids_flat,
                                            wavelengths_phys_flat,
                                            target_log_flam_flat,
                                            pred_log_flam_flat,
                                            target_flux_flat, pred_flux_flat
    test_metrics_per_model.csv              one row per test model

The five ``emulator_FLUXCONT*`` names are exactly the ones
``fluxcont_training.artifact_paths()`` returns; ``--verify-constants`` imports
the training script and asserts that they, and the 13-parameter column order,
still agree.  The import is otherwise deferred, because only
``--loss-source checkpoint`` genuinely needs torch.

Two families need one further input:

    parameter-scatter   the 13-parameter grid file ``grid_large_log_LHC.json``,
                        which it joins against the per-model metrics and writes
                        out as ``performance_plots/test_metrics_with_parameters.csv``
    worst-corner        that joined CSV; it falls back to
                        ``test_metrics_per_model.csv`` + the grid file, which is
                        slower because the grid file is large
    parameter-trends    that joined CSV
    arch-summary        the whole FLUXCONT output tree, one folder per run


THE NUMBERS THE THESIS QUOTES
-----------------------------
Every literal below is copied from the source listed beside it; none of them is
re-derived here, because the thesis text quotes them.

    180 logarithmic wavelength-bin edges       signed-relative-error, binned-error
    179 possible intervals, 130 populated      a *result*, printed by the driver
    percentiles 2.5 / 16 / 84 / 97.5           signed-relative-error
    percentiles 16 / 84 / 95                   binned-error (the notebook's set)
    500 displayed model-wavelength points      signed-relative-error
    seed 42                                    signed-relative-error, worst-corner
    worst fraction 0.01 -> 327 of 32,730       worst-corner
    KS threshold D >= 0.15, minimum 3 params   worst-corner
    25 parameter bins, minimum 20 per bin      parameter-trends
    90th percentile                            parameter-trends
    300,000 hexbin points, gridsize 90         calibration
    80 histogram bins, 12 worst / 12 best      per-model-metrics


FIGURE MAP
----------
    --figure loss-curves
        Training, train-evaluation (dropout off) and validation loss versus
        epoch on a logarithmic axis, with the selected epoch marked.  For the
        adopted run the validation minimum is 1.27e-3 at epoch 84.
        Thesis: fig:fluxcont_loss.

    --figure sed-profiles
        FASTWIND SED against the emulator prediction in log-log space for one
        representative test model, its pointwise absolute relative difference in
        per cent, and the 12 worst test models by mean relative flux error.
        Thesis: fig:fluxcont_representative_model (sub-figures
        fig:fluxcont_profiles and fig:fluxcont_example_relative_error).

    --figure signed-relative-error
        Signed relative error 100 (F_FW - F_emu) / |F_FW| against wavelength,
        binned in log wavelength, with the mean, the median and the 2.5 / 16 /
        84 / 97.5 percentiles, over a reproducible random sample of 500
        individual model-wavelength points.
        Thesis: fig:fluxcont_binned_relative_error.

    --figure binned-error
        The notebook's unsigned pair of the same diagnostic: relative error and
        absolute log-flux error binned by wavelength with median, 16-84 band and
        95th percentile, plus the two CSV tables they are computed from.

    --figure calibration
        Predicted against FASTWIND log10(F_lambda) over the full test set as a
        log-density hexbin, with the diagonal.  The cloud spans roughly nine
        decades, log10(F_lambda) ~ 28 to 37.
        Thesis: fig:fluxcont_pred_true_log_flux.

    --figure per-model-metrics
        Histograms of the per-model log-flux MSE, mean relative error and median
        relative error, and the 12 worst and 12 best models by mean relative
        error.

    --figure worst-corner
        Corner plot locating the worst one per cent of test models (327 of
        32,730) in the input parameters, showing only the parameters whose
        worst-subset distribution differs from the full test set by a two-sample
        Kolmogorov-Smirnov distance D >= 0.15, with a minimum of three
        parameters.  The 13-parameter version is written too, for reference.
        Thesis: fig:fluxcont_parameter_error.

    --figure parameter-trends
        Binned median and 90th percentile of the per-model mean relative flux
        error against each of the 13 parameters separately.
        Thesis: fig:fluxcont_parameter_trends (appendix).

    --figure parameter-scatter
        The notebook's earlier version of the same idea, in which the colour bar
        encodes the same quantity as the y axis.  Not printed in the thesis, but
        it is the step that writes ``test_metrics_with_parameters.csv``, which
        worst-corner and parameter-trends consume.  Run it once first if that
        file is missing.

    --figure arch-summary
        Not printed as a figure in the thesis: the numerical comparison of the
        28 completed FLUXCONT configurations, written as a CSV and as a LaTeX
        longtable body, plus a summary bar chart.
        Thesis: tab:fluxcont_architecture_metrics (appendix).


USAGE
-----
    # every figure that needs nothing but the adopted run folder
    python fluxcont_plots.py --all \
        --fluxcont-root outputs/fluxcont_runs \
        --run-name fluxcont_v17a_binlogflam_balanced_loglambda_weighted_huber_bias_endpoint0p01_no_bn_control_1e3_1e5_fm64_drop0p02_13par

    # one figure
    python fluxcont_plots.py --figure signed-relative-error
    python fluxcont_plots.py --figure worst-corner --ks-threshold 0.15 --min-params 3
    python fluxcont_plots.py --figure parameter-trends --parameter-bins 25

    # the loss curves straight out of the checkpoint instead of the loss JSON
    python fluxcont_plots.py --figure loss-curves --loss-source checkpoint

    # rebuild the metrics/parameter join first, then the two families that need it
    python fluxcont_plots.py --figure parameter-scatter \
        --grid-json data/thirteen_parameter/grid_large_log_LHC.json
    python fluxcont_plots.py --figure worst-corner
    python fluxcont_plots.py --figure parameter-trends

    # the appendix architecture table, over the whole FLUXCONT output tree
    python fluxcont_plots.py --figure arch-summary \
        --search-root outputs/fluxcont_runs

    # check that this file and fluxcont_training.py still agree (needs torch)
    python fluxcont_plots.py --figure loss-curves --verify-constants
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: every figure is written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================================
# SECTION 1.  CONSTANTS
# =====================================================================================

#: Notebook cell 1.  Every figure in this file was produced with these settings.
NOTEBOOK_RCPARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
}
plt.rcParams.update(NOTEBOOK_RCPARAMS)


def default_fluxcont_root() -> Path:
    """Return the portable root containing FLUXCONT training runs."""
    script_dir = Path(__file__).resolve().parent
    repository_root = Path(
        os.environ.get("FASTWIND_EMULATOR_ROOT", script_dir)
    ).expanduser().resolve()
    output_root = Path(
        os.environ.get("FASTWIND_OUTPUT_ROOT", repository_root / "outputs")
    ).expanduser()
    return Path(
        os.environ.get("FASTWIND_FLUXCONT_RUNS", output_root / "fluxcont_runs")
    ).expanduser()


def default_data_root() -> Path:
    """Return the portable root holding the external input datasets."""
    script_dir = Path(__file__).resolve().parent
    repository_root = Path(
        os.environ.get("FASTWIND_EMULATOR_ROOT", script_dir)
    ).expanduser().resolve()
    return Path(
        os.environ.get("FASTWIND_DATA_ROOT", repository_root / "data")
    ).expanduser()


#: The configuration the thesis adopted.  This is ``fluxcont_training``'s
#: ``DEFAULT_CONFIG_KEY = "v17a"``; the folder name is the run tag that run
#: wrote into its own ``emulator_FLUXCONT_meta.json``.  Notebook cell 1 opened
#: precisely this folder.
DEFAULT_RUN_NAME = (
    "fluxcont_v17a_binlogflam_balanced_loglambda_weighted_huber_bias_"
    "endpoint0p01_no_bn_control_1e3_1e5_fm64_drop0p02_13par"
)

#: Sub-folder of the run that the notebook wrote its figures into (cell 1).
PLOT_SUBDIR = "performance_plots"

#: Sub-folder of ``performance_plots`` used by ``fluxcont_chapter_plots.py``.
CORNER_SUBDIR = "corner_worst1pct"

#: Stem of the five primary artefacts.  This mirrors
#: ``fluxcont_training.ARTIFACT_STEM``; ``--verify-constants`` asserts that the
#: two are still identical, so the duplication cannot silently drift.
ARTIFACT_STEM = "emulator_FLUXCONT"

#: The remaining per-run files.  These are plain literals inside
#: ``fluxcont_training.train_one_config()`` rather than entries of
#: ``artifact_paths()``, so they are repeated here in the same spelling.
METRICS_CSV_NAME = "test_metrics_per_model.csv"
RUN_CONFIG_NAME = "run_config.json"
PARAM_NORM_NAME = "parameter_normalization.json"
SCALER_NAME = "sed_target_and_wavelength_scaler.json"
SUMMARY_NAME = "training_summary.json"

#: Written by ``--figure parameter-scatter`` (notebook cell 21) and consumed by
#: ``--figure worst-corner`` and ``--figure parameter-trends``.
METRICS_WITH_PARAMS_NAME = "test_metrics_with_parameters.csv"

#: (column in the metrics CSV / grid JSON, axis label, take log10 before plotting).
#: The column order is ``fluxcont_training.PARAM_COLS``; the labels and the
#: log10 flags come from ``fluxcont_chapter_plots.CORNER_PARAMS_13``, which is
#: itself identical to ``CORNER_PARAMS_13`` of the 13-parameter chapter script
#: so that the two corner figures look alike.
CORNER_PARAMS_13: Tuple[Tuple[str, str, bool], ...] = (
    ("teff",   r"$T_\mathrm{eff}$ [K]",   False),
    ("logg",   r"$\log g$",               False),
    ("radius", r"$R$ [$R_\odot$]",        False),
    ("mdot",   r"$\log_{10}\dot{M}$",     True),
    ("yhe",    r"$Y_\mathrm{He}$",        False),
    ("C",      r"$\epsilon_\mathrm{C}$",  False),
    ("N",      r"$\epsilon_\mathrm{N}$",  False),
    ("O",      r"$\epsilon_\mathrm{O}$",  False),
    ("beta",   r"$\beta$",                False),
    ("vinf",   r"$v_\infty$ [km/s]",      False),
    ("fic",    r"$f_\mathrm{ic}$",        False),
    ("fvel",   r"$f_\mathrm{vel}$",       False),
    ("fclump", r"$f_\mathrm{cl}$",        False),
)

#: The same 13 parameters with the labels ``plot_mean_relative_error_vs_parameters.py``
#: used.  They differ from the corner labels only in the radius and the clumping
#: factor, and in showing the CNO abundances as bare element symbols; both label
#: sets are kept verbatim so that each figure reproduces byte for byte.
TREND_PARAMS_13: Tuple[Tuple[str, str, bool], ...] = (
    ("teff",   r"$T_\mathrm{eff}$ [K]",        False),
    ("logg",   r"$\log g$",                    False),
    ("radius", r"$R_\ast$ [$R_\odot$]",        False),
    ("mdot",   r"$\log_{10}\dot{M}$",          True),
    ("yhe",    r"$Y_\mathrm{He}$",             False),
    ("C",      "C",                            False),
    ("N",      "N",                            False),
    ("O",      "O",                            False),
    ("beta",   r"$\beta$",                     False),
    ("vinf",   r"$v_\infty$ [km/s]",           False),
    ("fic",    r"$f_\mathrm{ic}$",             False),
    ("fvel",   r"$f_\mathrm{vel}$",            False),
    ("fclump", r"$f_\mathrm{clump}$",          False),
)

#: Plain column names used by the notebook's own parameter figure (cell 21).
PARAM_COLS: Tuple[str, ...] = tuple(name for name, _, _ in CORNER_PARAMS_13)

# --- signed-relative-error (plot_fluxcont_signed_relative_error.py) -------------------
SIGNED_BIN_EDGES = 180        # 180 edges -> 179 possible intervals
SIGNED_MAX_SAMPLES = 500      # individual model-wavelength points drawn for display
SIGNED_SEED = 42              # fixed seed of that draw
SIGNED_PERCENTILES = (2.5, 16.0, 84.0, 97.5)
FLUX_FLOOR = 1.0e-300         # denominator floor of the relative error

# --- binned-error (notebook cell 14) --------------------------------------------------
NOTEBOOK_BIN_EDGES = 180      # np.logspace(..., 180): the same 180 edges
NOTEBOOK_PERCENTILES = (16, 84, 95)

# --- calibration (notebook cell 12) ---------------------------------------------------
CALIBRATION_MAX_POINTS = 300_000
CALIBRATION_GRIDSIZE = 90
#: The notebook referred to an ``rng`` that it never created, so cell 12 raised a
#: NameError as written and the published figure was made after seeding by hand.
#: The seed is therefore *not* recoverable from the notebook; 42 is used because
#: it is the seed every other FLUXCONT script uses, and ``--calibration-seed``
#: exposes it.  See the "NOT REPRODUCED" note in the repository README.
CALIBRATION_SEED = 42

# --- per-model-metrics (notebook cells 16, 17) ----------------------------------------
METRIC_HIST_BINS = 80
METRIC_TABLE_ROWS = 12
WORST_SED_COLUMNS = 3

# --- worst-corner (fluxcont_chapter_plots.py) -----------------------------------------
WORST_FRACTION = 0.01         # 0.01 * 32,730 -> 327 models
KS_THRESHOLD = 0.15           # D >= 0.15 counts as a dependent parameter
KS_MIN_PARAMS = 3             # ... but never show fewer than three
CORNER_BACKGROUND_MAX = 6000  # grey points drawn, sub-sampled with CORNER_SEED
CORNER_SEED = 42
CORNER_DIAGONAL_BINS_ALL = 40
CORNER_DIAGONAL_BINS_WORST = 20
CORNER_PANEL_DEP = 2.2        # inches per panel, reduced (dependent) corner
CORNER_DPI_DEP = 160
CORNER_FORMAT_DEP = "pdf"
CORNER_PANEL_13 = 1.6         # inches per panel, all-13-parameter corner
CORNER_DPI_13 = 150
CORNER_FORMAT_13 = "png"

# --- parameter-trends (plot_mean_relative_error_vs_parameters.py) ---------------------
TREND_BINS = 25
TREND_MIN_PER_BIN = 20
TREND_PERCENTILE = 90.0
TREND_DPI = 170

#: Default per-model error column.  ``fluxcont_chapter_plots.py`` and
#: ``plot_mean_relative_error_vs_parameters.py`` both rank on this column, and it
#: is the quantity Chapter 5 reports.
DEFAULT_METRIC = "mean_relative_flux_error"

# --- arch-summary ---------------------------------------------------------------------
#: The 28 completed configurations in the order of the appendix table, as
#: (config key of ``fluxcont_training.CONFIGS``, run tag written by that run,
#: caption used in tab:fluxcont_architecture_metrics).  The run tags are the
#: folder names the historical scripts created and are the value of ``run_tag``
#: inside each folder's ``emulator_FLUXCONT_meta.json``.
ARCHITECTURE_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("v1", "fluxcont_logflam_loglambda_deep_dropout_13par",
     "Initial baseline"),
    ("v2", "fluxcont_v2_binlogflam_balanced_loglambda_weighted_huber_13par",
     "Weighted binned target"),
    ("v3", "fluxcont_v3_global_logsed_balanced_loglambda_weighted_huber_cut100_"
           "drop0p02_all_bn_13par",
     "Global, low dropout"),
    ("v4", "fluxcont_v4_global_logsed_balanced_loglambda_weighted_huber_cut100_"
           "drop0p10_all_bn_13par",
     "Global, high dropout"),
    ("v5", "fluxcont_v5_binlogflam_balanced_loglambda_weighted_huber_cut100_"
           "fm64_drop0p10_bn_13par",
     "Binned, 64 modes"),
    ("v6", "fluxcont_v6_binlogflam_balanced_loglambda_weighted_huber_deriv_"
           "cut100_fm64_drop0p10_bn_13par",
     "Binned + derivative term"),
    ("v7", "fluxcont_v7_global_logsed_balanced_loglambda_weighted_huber_bias_"
           "cut200_fm64_drop0p02_all_bn_13par",
     "Global, cut 200"),
    ("v8", "fluxcont_v8_binlogflam_balanced_loglambda_weighted_huber_bias_"
           "cut200_fm64_drop0p02_bn_13par",
     "Binned, cut 200"),
    ("v9", "fluxcont_v9_global_logsed_balanced_loglambda_weighted_huber_bias_"
           "cut100_fullsed_fm64_drop0p02_all_bn_13par",
     "Global full SED"),
    ("v10", "fluxcont_v10_global_logsed_balanced_loglambda_weighted_huber_bias_"
            "endpoint_cut100_fullsed_fm64_drop0p02_all_bn_13par",
     "Global full SED + endpoint"),
    ("v14a", "fluxcont_v14a_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "noendpoint_1e3_1e5_fm64_drop0p02_bn_13par",
     "Binned no endpoint"),
    ("v14b", "fluxcont_v14b_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_1e3_1e5_fm64_drop0p02_bn_13par",
     "Binned + endpoint"),
    ("v15a", "fluxcont_v15a_global_logsed_balanced_loglambda_weighted_huber_bias_"
             "noendpoint_1e3_1e5_fm64_drop0p02_all_bn_13par",
     "Global no endpoint"),
    ("v15b", "fluxcont_v15b_global_logsed_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_1e3_1e5_fm64_drop0p02_all_bn_13par",
     "Global + endpoint"),
    ("v16a", "fluxcont_v16a_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "Binned no BN control"),
    ("v16b", "fluxcont_v16b_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_layernorm_1e3_1e5_fm64_drop0p02_13par",
     "Binned layer norm"),
    ("v16c", "fluxcont_v16c_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_bn_lr1e4_1e3_1e5_fm64_drop0p02_13par",
     "Binned BN, lower learning rate"),
    ("v16d", "fluxcont_v16d_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p005_bn_1e3_1e5_fm64_drop0p02_13par",
     "Binned BN, weaker endpoint"),
    ("v17a", "fluxcont_v17a_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_no_bn_control_1e3_1e5_fm64_drop0p02_13par",
     "Adopted no BN control"),
    ("v17b", "fluxcont_v17b_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p005_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "No BN, weak endpoint"),
    ("v17c", "fluxcont_v17c_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p02_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "No BN, strong endpoint"),
    ("v17d", "fluxcont_v17d_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_tailanchors_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "No BN + tail anchors"),
    ("v17e", "fluxcont_v17e_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_residual_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "No BN + residual blocks"),
    ("v18a", "fluxcont_v18a_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_tailanchors0p0025_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "Tail anchors, weak weight"),
    ("v18b", "fluxcont_v18b_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_tailanchors0p005_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "Tail anchors, stronger weight"),
    ("v18c", "fluxcont_v18c_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_both_edgeanchors0p0025_no_bn_1e3_1e5_fm64_drop0p02_13par",
     "Both-edge anchors, 64 modes"),
    ("v18d", "fluxcont_v18d_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_both_edgeanchors0p0025_no_bn_1e3_1e5_fm96_drop0p02_13par",
     "Both-edge anchors, 96 modes"),
    ("v18e", "fluxcont_v18e_binlogflam_balanced_loglambda_weighted_huber_bias_"
             "endpoint0p01_no_bn_control_1e3_1e5_fm96_drop0p02_13par",
     "Wider Fourier basis"),
)

#: Mean relative flux errors above this are printed as "very large" in the
#: appendix table, because converting a large logarithmic residual back to
#: linear flux produces values with no useful magnitude.  The published table
#: prints 2.33e1 (v15a) and hides 1.76e5 (v7), so the cut-off it used lies
#: somewhere in (23.3, 1.76e5]; the exact value was never written down, and
#: 100.0 is the smallest round decade that reproduces the printed table.
VERY_LARGE_RELATIVE_ERROR = 100.0


# =====================================================================================
# SECTION 2.  SHARED HELPERS
# =====================================================================================

@lru_cache(maxsize=1)
def training_module():
    """Import ``fluxcont_training`` and check that this file still matches it.

    The import is deferred rather than done at module level because importing
    the training script pulls in torch, pandas and scikit-learn, while every
    figure except ``--loss-source checkpoint`` needs nothing but the JSON, NPZ
    and CSV artefacts.  Deferring it keeps ``--help`` and the whole notebook
    half of this file usable on a machine without torch.
    """
    try:
        import fluxcont_training  # noqa: F401  (imported for its constants)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "fluxcont_training.py could not be imported "
            f"({exc}).  It must sit next to this file, and it needs "
            "numpy, pandas, torch and scikit-learn."
        )

    if fluxcont_training.ARTIFACT_STEM != ARTIFACT_STEM:
        raise SystemExit(
            "Artefact stem drift: fluxcont_training.ARTIFACT_STEM is "
            f"{fluxcont_training.ARTIFACT_STEM!r}, this file expects "
            f"{ARTIFACT_STEM!r}."
        )
    if tuple(fluxcont_training.PARAM_COLS) != PARAM_COLS:
        raise SystemExit(
            "Parameter-column drift: fluxcont_training.PARAM_COLS is "
            f"{list(fluxcont_training.PARAM_COLS)}, this file expects "
            f"{list(PARAM_COLS)}."
        )
    expected = set(artifact_paths(Path(".")))
    produced = set(fluxcont_training.artifact_paths("."))
    if {Path(p).name for p in expected} != {Path(p).name for p in produced}:
        raise SystemExit(
            "Artefact-name drift between fluxcont_training.artifact_paths() "
            "and fluxcont_plots.artifact_paths()."
        )
    return fluxcont_training


def artifact_paths(run_dir: Path) -> Dict[str, Path]:
    """The five primary artefacts of a run, as ``fluxcont_training`` writes them."""
    run_dir = Path(run_dir)
    return {
        "model": run_dir / f"{ARTIFACT_STEM}.pth",
        "loss": run_dir / f"{ARTIFACT_STEM}_loss.json",
        "split": run_dir / f"{ARTIFACT_STEM}_split.json",
        "pred": run_dir / f"{ARTIFACT_STEM}_test_outputs.npz",
        "meta": run_dir / f"{ARTIFACT_STEM}_meta.json",
    }


def run_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Every input path a figure family may want, with the CLI overrides applied."""
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.fluxcont_root) / args.run_name
    primary = artifact_paths(run_dir)
    plot_dir = Path(args.output_dir) if args.output_dir else run_dir / PLOT_SUBDIR
    paths = {
        "run": run_dir,
        "plots": plot_dir,
        "checkpoint": Path(args.checkpoint) if args.checkpoint else primary["model"],
        "loss": Path(args.loss_json) if args.loss_json else primary["loss"],
        "split": primary["split"],
        "npz": Path(args.npz) if args.npz else primary["pred"],
        "meta": Path(args.meta_json) if args.meta_json else primary["meta"],
        "scaler": run_dir / SCALER_NAME,
        "summary": run_dir / SUMMARY_NAME,
        "run_config": run_dir / RUN_CONFIG_NAME,
        "param_norm": run_dir / PARAM_NORM_NAME,
        "metrics": Path(args.metrics_csv) if args.metrics_csv else run_dir / METRICS_CSV_NAME,
    }
    paths["metrics_with_params"] = (
        Path(args.metrics_with_params_csv) if args.metrics_with_params_csv
        else plot_dir / METRICS_WITH_PARAMS_NAME
    )
    return paths


def load_json(path: Path) -> Optional[dict]:
    """Notebook cell 2: a missing file is not an error, it is a missing panel."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> List[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, output_path: Path, dpi: Optional[float] = None,
                extra_dirs: Sequence[Path] = (), skip_existing: bool = False) -> List[Path]:
    """Write one figure to its own folder and, optionally, to the thesis folder."""
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"[skip] {output_path} exists")
        plt.close(fig)
        return []
    written: List[Path] = []
    for directory in (output_path.parent,) + tuple(Path(d) for d in extra_dirs):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / output_path.name
        if dpi is None:
            fig.savefig(target, bbox_inches="tight")
        else:
            fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
        print(f"[saved] {target}")
    plt.close(fig)
    return written


def save_table(frame: pd.DataFrame, output_path: Path,
               extra_dirs: Sequence[Path] = ()) -> List[Path]:
    """The CSV counterpart of :func:`save_figure`."""
    output_path = Path(output_path)
    written: List[Path] = []
    for directory in (output_path.parent,) + tuple(Path(d) for d in extra_dirs):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / output_path.name
        frame.to_csv(target, index=False)
        written.append(target)
        print(f"[saved] {target}")
    return written


class TestOutputs:
    """The flattened, ragged test partition of ``emulator_FLUXCONT_test_outputs.npz``.

    Notebook cell 3.  ``model_offsets`` turns the flat arrays back into one
    variable-length SED per test model: model ``i`` occupies the half-open slice
    ``[model_offsets[i], model_offsets[i + 1])``.  Each SED keeps its own native
    FASTWIND wavelengths, so the arrays are ragged and cannot be reshaped.

    The notebook recomputed the linear fluxes from the stored log fluxes rather
    than reading ``target_flux_flat`` / ``pred_flux_flat``, "because the log
    arrays are the actual training/evaluation target domain"; that behaviour is
    preserved for every notebook-derived family.  ``--figure
    signed-relative-error`` reads the stored linear arrays instead, because
    ``plot_fluxcont_signed_relative_error.py`` did.
    """

    def __init__(self, npz_path: Path):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.files = list(data.files)
        self.test_model_ids = data["test_model_ids"].astype(int)
        self.model_offsets = data["model_offsets"].astype(np.int64)
        self.model_ids_flat = data["model_ids_flat"].astype(int)
        self.wave = data["wavelengths_phys_flat"].astype(np.float64)
        self.target_log = data["target_log_flam_flat"].astype(np.float64)
        self.pred_log = data["pred_log_flam_flat"].astype(np.float64)
        self._data = data

        self.target_flux = np.power(10.0, self.target_log)
        self.pred_flux = np.power(10.0, self.pred_log)
        self.rel_err = (np.abs(self.pred_flux - self.target_flux)
                        / np.maximum(np.abs(self.target_flux), FLUX_FLOOR))
        self.abs_log_err = np.abs(self.pred_log - self.target_log)

    # -- notebook cell 7 ---------------------------------------------------------
    @property
    def n_test(self) -> int:
        return len(self.test_model_ids)

    def model_slice(self, model_index: int) -> slice:
        return slice(int(self.model_offsets[model_index]),
                     int(self.model_offsets[model_index + 1]))

    def model_arrays(self, model_index: int) -> Dict[str, np.ndarray]:
        sl = self.model_slice(model_index)
        order = np.argsort(self.wave[sl])
        return {
            "model_id": int(self.test_model_ids[model_index]),
            "wave": self.wave[sl][order],
            "target_log": self.target_log[sl][order],
            "pred_log": self.pred_log[sl][order],
            "target_flux": self.target_flux[sl][order],
            "pred_flux": self.pred_flux[sl][order],
            "rel_err": self.rel_err[sl][order],
            "abs_log_err": self.abs_log_err[sl][order],
        }

    def example_index(self) -> int:
        """Notebook cell 7: ``example_indices = [n_test // 2]``.

        "One representative test model is used for the profile and
        wavelength-error examples.  This avoids repeating the same visual
        comparison many times in the thesis."
        """
        return self.n_test // 2

    def describe(self) -> None:
        print("NPZ keys:")
        for key in self.files:
            arr = self._data[key]
            print(f"  {key:32s} shape={arr.shape} dtype={arr.dtype}")
        print("\nNumber of test models:", self.n_test)
        print("Number of flattened test wavelength points:", self.wave.size)
        print("Wavelength range [A]:", np.nanmin(self.wave), np.nanmax(self.wave))
        print("Median relative error:", np.nanmedian(self.rel_err))
        print("Mean relative error:", np.nanmean(self.rel_err))
        print("Median abs log10 error:", np.nanmedian(self.abs_log_err))


def loss_histories(paths: Dict[str, Path], source: str) -> Dict[str, object]:
    """Training / train-eval / validation histories, from JSON, NPZ or checkpoint.

    Notebook cell 5 preferred ``emulator_FLUXCONT_loss.json`` and fell back to
    the arrays that v14+ also stored inside the NPZ.  ``source="checkpoint"``
    adds the third place ``fluxcont_training`` writes them, the ``.pth`` file
    itself; it is the only path in this script that needs torch.
    """
    loss_data = load_json(paths["loss"]) if source in ("auto", "json") else None
    npz = None
    if source in ("auto", "npz") and Path(paths["npz"]).exists():
        npz = np.load(paths["npz"], allow_pickle=False)
    checkpoint = None
    if source == "checkpoint":
        import torch  # deferred: only this branch needs it

        training_module()  # assert that the artefact names still agree
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu",
                                weights_only=False)

    def loss_array(name: str, required: bool = False) -> Optional[np.ndarray]:
        if checkpoint is not None and name in checkpoint:
            values = checkpoint[name]
        elif loss_data is not None and name in loss_data:
            values = loss_data[name]
        elif npz is not None and name in npz.files:
            values = npz[name]
        else:
            if required:
                raise SystemExit(f"Missing required loss history: {name}")
            return None
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            if required:
                raise SystemExit(f"Loss history is empty: {name}")
            return None
        return values

    record: Dict[str, object] = {
        "train_losses": loss_array("train_losses", required=True),
        "val_losses": loss_array("val_losses", required=True),
        "train_eval_losses": loss_array("train_eval_losses", required=False),
        "best_epoch": np.nan,
        "best_val_loss": np.nan,
    }
    scalars = checkpoint if checkpoint is not None else loss_data
    if scalars is not None:
        raw_epoch = scalars.get("best_epoch", np.nan)
        if raw_epoch is not None and np.isfinite(float(raw_epoch)):
            record["best_epoch"] = int(raw_epoch)
        raw_val = scalars.get("best_val_loss", np.nan)
        if raw_val is not None and np.isfinite(float(raw_val)):
            record["best_val_loss"] = float(raw_val)
    return record


def per_model_metrics(outputs: TestOutputs, metrics_path: Path) -> pd.DataFrame:
    """Notebook cell 16: read the CSV the training script wrote, or rebuild it."""
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        print(f"[data] per-model metrics from {metrics_path}")
        return pd.read_csv(metrics_path)
    print(f"[data] {metrics_path} not found; recomputing from the NPZ")
    rows = []
    for index in range(outputs.n_test):
        model = outputs.model_arrays(index)
        rows.append({
            "model_id": model["model_id"],
            "n_points": model["wave"].size,
            "mse_log_flam": float(np.mean((model["pred_log"] - model["target_log"]) ** 2)),
            "mean_relative_flux_error": float(np.mean(model["rel_err"])),
            "median_relative_flux_error": float(np.median(model["rel_err"])),
        })
    return pd.DataFrame(rows)


def notebook_binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    """Notebook cell 14's ``binned_stats``, unchanged.

    ``np.digitize(x, bins) - 1`` gives ``len(bins) - 1`` for a value equal to the
    last edge, which the ``range(len(bins) - 1)`` loop never visits, so the
    single point at the maximum wavelength falls outside every interval.  That is
    the behaviour of the published figure and is deliberately kept.
    """
    rows = []
    inds = np.digitize(x, bins) - 1
    for i in range(len(bins) - 1):
        mask = inds == i
        if not np.any(mask):
            continue
        vals = y[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append({
            "lambda_left": bins[i],
            "lambda_right": bins[i + 1],
            "lambda_mid": np.sqrt(bins[i] * bins[i + 1]),
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p16": float(np.percentile(vals, NOTEBOOK_PERCENTILES[0])),
            "p84": float(np.percentile(vals, NOTEBOOK_PERCENTILES[1])),
            "p95": float(np.percentile(vals, NOTEBOOK_PERCENTILES[2])),
        })
    return pd.DataFrame(rows)


def log_wavelength_edges(wavelengths: np.ndarray, n_edges: int) -> np.ndarray:
    """``n_edges`` edges spaced uniformly in log10(lambda) over the retained range.

    With the thesis value of 180 this gives 179 possible intervals, each covering
    an equal *factor* in wavelength rather than an equal width in Angstrom.  For
    the adopted run 130 of them contain at least one native-grid test point.
    """
    finite = wavelengths[np.isfinite(wavelengths)]
    if finite.size == 0:
        raise SystemExit("The saved test output contains no finite wavelengths.")
    return np.logspace(np.log10(float(np.min(finite))),
                       np.log10(float(np.max(finite))), n_edges)


def signed_relative_error_percent(target_flux: np.ndarray,
                                  predicted_flux: np.ndarray) -> np.ndarray:
    """100 * (FASTWIND - prediction) / max(|FASTWIND|, 1e-300).

    Copied from ``plot_fluxcont_signed_relative_error.py``, including the in-place
    arithmetic that reuses ``target_flux`` as the denominator to avoid a second
    array of ~28 million doubles.  The caller must not use ``target_flux``
    afterwards.
    """
    signed_error = np.empty_like(target_flux, dtype=np.float64)
    np.subtract(target_flux, predicted_flux, out=signed_error)
    np.abs(target_flux, out=target_flux)
    np.maximum(target_flux, FLUX_FLOOR, out=target_flux)
    np.divide(signed_error, target_flux, out=signed_error)
    signed_error *= 100.0
    return signed_error


def signed_binned_statistics(wavelengths: np.ndarray, errors: np.ndarray,
                             edges: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact mean / median / 2.5 / 16 / 84 / 97.5 percentiles per populated interval."""
    finite = np.isfinite(wavelengths) & np.isfinite(errors)
    if not np.any(finite):
        raise SystemExit(
            "The saved test output contains no finite wavelength-error pairs.")

    bin_index = np.searchsorted(edges, wavelengths, side="right") - 1
    np.clip(bin_index, 0, len(edges) - 2, out=bin_index)

    statistics: Dict[str, List[float]] = {
        "wavelength": [], "mean": [], "median": [],
        "p2p5": [], "p16": [], "p84": [], "p97p5": [],
    }
    for index in range(len(edges) - 1):
        in_bin = finite & (bin_index == index)
        if not np.any(in_bin):
            continue
        values = errors[in_bin]
        percentiles = np.percentile(values, list(SIGNED_PERCENTILES))
        statistics["wavelength"].append(
            float(np.sqrt(edges[index] * edges[index + 1])))
        statistics["mean"].append(float(np.mean(values)))
        statistics["median"].append(float(np.median(values)))
        statistics["p2p5"].append(float(percentiles[0]))
        statistics["p16"].append(float(percentiles[1]))
        statistics["p84"].append(float(percentiles[2]))
        statistics["p97p5"].append(float(percentiles[3]))
    return {name: np.asarray(values) for name, values in statistics.items()}


def displayed_subset(wavelengths: np.ndarray, errors: np.ndarray,
                     seed: int, max_samples: int) -> np.ndarray:
    """The reproducible draw of individual model-wavelength points."""
    finite_indices = np.flatnonzero(np.isfinite(wavelengths) & np.isfinite(errors))
    if finite_indices.size == 0:
        raise SystemExit(
            "The saved test output contains no finite wavelength-error pairs.")
    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, finite_indices.size)
    return rng.choice(finite_indices, size=sample_size, replace=False)


# -------------------------------------------------------------------------------------
# Per-model error joined against the 13 input parameters
# -------------------------------------------------------------------------------------

def grid_parameters(model_ids: Sequence[int], grid_json: Path,
                    params: Sequence[Tuple[str, str, bool]]) -> np.ndarray:
    """Raw parameter values for a list of LHC model ids, from the grid JSON.

    ``fluxcont_chapter_plots.lhc_params``; only used when the joined CSV that
    ``--figure parameter-scatter`` writes is unavailable, because the grid file
    is large and takes one to two minutes to parse.
    """
    grid_json = Path(grid_json)
    if not grid_json.exists():
        raise SystemExit(f"grid file not found: {grid_json}")
    print(f"[params] loading {grid_json} (large, ~1-2 min)", flush=True)
    with open(grid_json) as handle:
        grid = json.load(handle)
    cols = [name for name, _, _ in params]
    return np.array(
        [[float(grid[str(int(mid))][col]) for col in cols] for mid in model_ids],
        dtype=np.float64,
    )


def load_metrics_with_parameters(paths: Dict[str, Path], grid_json: Path, metric: str,
                                 params: Sequence[Tuple[str, str, bool]]
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids, error, raw parameter values)`` for the FLUXCONT test partition.

    ``fluxcont_chapter_plots.load_test_models``, unchanged, including its order
    of preference and its screen.  The corrupt-row screen used for the line
    emulators is deliberately not applied: FLUXCONT targets are absolute
    logarithmic fluxes of order 1e28 to 1e37 rather than normalised fluxes near
    unity, so that screen would flag nothing and give a false impression of
    having filtered.
    """
    cols = [name for name, _, _ in params]
    joined = Path(paths["metrics_with_params"])
    plain = Path(paths["metrics"])

    if joined.is_file():
        rows = read_csv_rows(joined)
        if not rows:
            raise SystemExit(f"{joined} has no data rows")
        missing = [c for c in cols + [metric, "model_id"] if c not in rows[0]]
        if missing:
            raise SystemExit(f"{joined} lacks columns: {missing}")
        ids = np.array([int(r["model_id"]) for r in rows], dtype=np.int64)
        err = np.array([float(r[metric]) for r in rows], dtype=np.float64)
        vals = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
        print(f"[data] {ids.size} test models from {joined}")
    elif plain.is_file():
        print(f"[data] {joined} not found, falling back to {plain} joined "
              f"against the grid file")
        rows = read_csv_rows(plain)
        if not rows:
            raise SystemExit(f"{plain} has no data rows")
        missing = [c for c in (metric, "model_id") if c not in rows[0]]
        if missing:
            raise SystemExit(f"{plain} lacks columns: {missing}")
        ids = np.array([int(r["model_id"]) for r in rows], dtype=np.int64)
        err = np.array([float(r[metric]) for r in rows], dtype=np.float64)
        vals = grid_parameters(ids, grid_json, params)
        print(f"[data] {ids.size} test models")
    else:
        raise SystemExit(f"no per-model metrics found; looked for\n  {joined}\n  {plain}")

    good = np.isfinite(err) & np.isfinite(vals).all(axis=1)
    for k, (_, _, use_log) in enumerate(params):
        if use_log:
            good &= vals[:, k] > 0.0
    if not good.all():
        print(f"[screen] dropped {int((~good).sum())} models with a non-finite "
              f"error, a non-finite parameter, or a non-positive value in a "
              f"parameter shown logarithmically")
    return ids[good], err[good], vals[good]


def to_plot_values(vals: np.ndarray,
                   params: Sequence[Tuple[str, str, bool]]) -> np.ndarray:
    """Apply the log10 flag of each parameter, returning a copy."""
    out = vals.copy()
    for k, (_, _, use_log) in enumerate(params):
        if use_log:
            out[:, k] = np.log10(out[:, k])
    return out


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance, as in the 13-parameter script."""
    sa, sb = np.sort(a), np.sort(b)
    g = np.concatenate([sa, sb])
    return float(np.max(np.abs(
        np.searchsorted(sa, g, side="right") / sa.size
        - np.searchsorted(sb, g, side="right") / sb.size)))


def select_dependent(vals: np.ndarray, worst_idx: np.ndarray,
                     params: Sequence[Tuple[str, str, bool]],
                     threshold: float = KS_THRESHOLD,
                     min_params: int = KS_MIN_PARAMS
                     ) -> Tuple[List[int], np.ndarray, List[str]]:
    """Indices of the parameters whose worst-subset distribution differs.

    A parameter is kept when its KS distance from the full test distribution is
    at least ``threshold``; if fewer than ``min_params`` pass, the top
    ``min_params`` by KS are used instead.  Returns the selection sorted by
    decreasing KS, the full KS array, and a ``+``/``-`` sign per parameter saying
    whether the worst subset sits above or below the full-set median.
    """
    ks = np.array([ks_distance(vals[:, k], vals[worst_idx, k])
                   for k in range(len(params))])
    signs = ["+" if np.median(vals[worst_idx, k]) > np.median(vals[:, k]) else "-"
             for k in range(len(params))]
    sel = [k for k in range(len(params)) if ks[k] >= threshold]
    if len(sel) < min_params:
        sel = list(np.argsort(ks)[::-1][:min_params])
    sel.sort(key=lambda k: -ks[k])
    return sel, ks, signs


def worst_subset(err: np.ndarray, worst_frac: float = WORST_FRACTION) -> np.ndarray:
    """Indices of the worst ``worst_frac`` of models, largest error first.

    ``round(0.01 * 32730) = 327``, the number Chapter 5 quotes.
    """
    n_worst = max(1, int(round(worst_frac * err.size)))
    return np.argsort(err)[::-1][:n_worst]


# =====================================================================================
# SECTION 3.  FIGURE BLOCKS
# =====================================================================================

# -------------------------------------------------------------------------------------
# FIGURE: loss-curves
# -------------------------------------------------------------------------------------
# What it shows: the composite region-weighted Huber loss in standardised
# log10(F_lambda) on the training and validation partitions against epoch, on a
# logarithmic y axis.  The solid curve is the running average of the batches that
# updated the weights, so dropout was active while it was measured; the dashed
# curve re-evaluates the *same* training models at the end of the epoch with
# dropout disabled, which is why it can lie below the solid curve; the third
# curve is the validation set, which never updated the weights.  The dotted
# vertical line is the epoch whose weights were kept.  For the adopted run that
# is epoch 84, with a validation loss of 1.27e-3.
# Thesis: fig:fluxcont_loss.
# Source: notebook cell 5.
# -------------------------------------------------------------------------------------
def save_loss_plot(record: Dict[str, object], output_path: Path,
                   extra_dirs: Sequence[Path] = (), skip_existing: bool = False) -> None:
    train_losses = record["train_losses"]
    val_losses = record["val_losses"]
    train_eval_losses = record["train_eval_losses"]
    best_epoch = record["best_epoch"]
    best_val = record["best_val_loss"]

    epochs = np.arange(1, len(train_losses) + 1)
    val_epochs = np.arange(1, len(val_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_losses, label="train", lw=1.8)
    if train_eval_losses is not None:
        train_eval_epochs = np.arange(1, len(train_eval_losses) + 1)
        ax.plot(train_eval_epochs, train_eval_losses,
                label="train eval (dropout off)", lw=1.8, ls="--")
    ax.plot(val_epochs, val_losses, label="validation", lw=1.8)
    if np.isfinite(best_epoch):
        ax.axvline(best_epoch, color="k", ls=":", lw=1, alpha=0.7,
                   label=f"best epoch {best_epoch}")
        if np.isfinite(best_val):
            ax.scatter([best_epoch], [best_val], color="k", s=25, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE in standardized log10(F_lambda)")
    ax.set_title("FLUXCONT DeepONet training history")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: sed-profiles  (representative model, log-log)
# -------------------------------------------------------------------------------------
# What it shows: the FASTWIND continuum SED of one test model (blue) with the
# emulator prediction over-plotted (orange), both evaluated on the same native
# FASTWIND wavelength grid.  Log-log axes, because wavelength spans 1e3 to 1e5
# Angstrom and F_lambda spans roughly nine decades.
# Thesis: fig:fluxcont_profiles, sub-figure (a) of fig:fluxcont_representative_model.
# Source: notebook cell 9.
# -------------------------------------------------------------------------------------
def save_example_sed_plot(model: Dict[str, np.ndarray], output_path: Path,
                          extra_dirs: Sequence[Path] = (),
                          skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.loglog(model["wave"], model["target_flux"], color="tab:blue", lw=1.8,
              label="FASTWIND")
    ax.loglog(model["wave"], model["pred_flux"], color="tab:orange", lw=1.4,
              alpha=0.9, label="Emulator")
    ax.set_title("FLUXCONT SED: FASTWIND vs emulator, "
                 f"test model {model['model_id']}")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ Angstrom$^{-1}$]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=250, extra_dirs=extra_dirs,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: sed-profiles  (representative model, pointwise error)
# -------------------------------------------------------------------------------------
# What it shows: the pointwise absolute relative difference of the same two
# curves, as a percentage, on log-log axes.  This is the quantity that enters the
# MARE of Equation (mare_metric), evaluated at every native wavelength point
# instead of being averaged.
# Thesis: fig:fluxcont_example_relative_error, sub-figure (b) of
#         fig:fluxcont_representative_model.
# Source: notebook cell 10.
# -------------------------------------------------------------------------------------
def save_example_error_plot(model: Dict[str, np.ndarray], output_path: Path,
                            extra_dirs: Sequence[Path] = (),
                            skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.semilogx(model["wave"], 100.0 * model["rel_err"], color="tab:red", lw=1.1)
    ax.set_yscale("log")
    ax.set_title(f"Relative error vs wavelength, test model {model['model_id']}")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("Relative error [%]")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=250, extra_dirs=extra_dirs,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: sed-profiles  (the worst models)
# -------------------------------------------------------------------------------------
# What it shows: the same FASTWIND-versus-emulator comparison for the 12 test
# models with the largest mean relative flux error, in a 3-column grid.  It is
# the visual counterpart of the corner plot: the corner plot says *where* those
# models sit in parameter space, these panels say *what* their SEDs look like.
# Source: notebook cell 19, with the ranking of cell 17.
# -------------------------------------------------------------------------------------
def save_worst_sed_panels(outputs: TestOutputs, worst_indices: Sequence[int],
                          output_path: Path, extra_dirs: Sequence[Path] = (),
                          skip_existing: bool = False) -> None:
    ncols = WORST_SED_COLUMNS
    nrows = math.ceil(len(worst_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)

    for ax, idx in zip(axes.ravel(), worst_indices):
        model = outputs.model_arrays(idx)
        ax.loglog(model["wave"], model["target_flux"], color="tab:blue", lw=1.5,
                  label="FASTWIND")
        ax.loglog(model["wave"], model["pred_flux"], color="tab:orange", lw=1.1,
                  label="Emulator")
        ax.set_title(f"model {model['model_id']}")
        ax.set_xlabel("lambda [Angstrom]")
        ax.set_ylabel("F_lambda [erg/s/Angstrom]")
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(worst_indices):]:
        ax.axis("off")

    fig.suptitle("Worst test models by mean relative error", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: signed-relative-error
# -------------------------------------------------------------------------------------
# What it shows: the *signed* percentage error 100 (F_FW - F_emu) / |F_FW| against
# wavelength, which the absolute relative difference of the MARE cannot show
# because it discards the direction of the deviation.  The 180 bin edges are
# spaced uniformly in log lambda between the minimum and maximum retained test
# wavelengths, giving 179 possible intervals covering equal factors in
# wavelength; 130 of them contain at least one native-grid point.  Within each
# populated interval the mean, the median and the 2.5 / 16 / 84 / 97.5 empirical
# percentiles are computed from *all* points in that interval and plotted at its
# geometric centre.  The grey points are 500 individual model-wavelength pairs
# drawn with a fixed seed, so that the scatter is visible without obscuring the
# percentile bands.
# Thesis: fig:fluxcont_binned_relative_error.
# Source: plot_fluxcont_signed_relative_error.py.
# -------------------------------------------------------------------------------------
def save_signed_relative_error_plot(wavelengths: np.ndarray, errors: np.ndarray,
                                    statistics: Dict[str, np.ndarray],
                                    subset: np.ndarray, output_path: Path,
                                    extra_dirs: Sequence[Path] = (),
                                    skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.fill_between(statistics["wavelength"], statistics["p2p5"], statistics["p97p5"],
                    color="#f4a261", alpha=0.24, label="2.5--97.5 percentile", zorder=1)
    ax.fill_between(statistics["wavelength"], statistics["p16"], statistics["p84"],
                    color="#2a9d8f", alpha=0.42, label="16--84 percentile", zorder=2)
    ax.scatter(wavelengths[subset], errors[subset], s=5, alpha=0.35, color="#666666",
               label="Test-model samples", zorder=3)
    ax.plot(statistics["wavelength"], statistics["median"], color="red",
            linewidth=2.0, label="Median", zorder=4)
    ax.plot(statistics["wavelength"], statistics["mean"], color="darkorange",
            linewidth=1.5, linestyle="--", label="Mean", zorder=4)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Signed relative error (%)")
    ax.set_title("FLUXCONT signed relative error on the native grids")
    ax.grid(alpha=0.30)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"[skip] {output_path} exists")
        plt.close(fig)
        return
    for directory in (output_path.parent,) + tuple(Path(d) for d in extra_dirs):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / output_path.name
        # The source script saved without bbox_inches="tight"; kept verbatim so
        # that the figure's canvas matches the one in the thesis.
        fig.savefig(target, dpi=300)
        print(f"[saved] {target}")
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: binned-error
# -------------------------------------------------------------------------------------
# What it shows: the notebook's unsigned pair of the diagnostic above.  The first
# panel bins the absolute relative flux error by wavelength, the second bins the
# absolute error in log10(F_lambda), both with the median, the 16-84 band and the
# 95th percentile, on log-log axes.  The two CSV tables are written as well: they
# are what the 130-populated-intervals count is read off.
# Source: notebook cell 14.
# -------------------------------------------------------------------------------------
def save_binned_error_plot(stats: pd.DataFrame, colour: str, ylabel: str, title: str,
                           output_path: Path, extra_dirs: Sequence[Path] = (),
                           skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stats["lambda_mid"], stats["median"], color=colour, lw=1.8, label="median")
    ax.fill_between(stats["lambda_mid"], stats["p16"], stats["p84"], color=colour,
                    alpha=0.22, label="16-84%")
    ax.plot(stats["lambda_mid"], stats["p95"], color=colour, lw=1.0, alpha=0.7,
            ls="--", label="95%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda [Angstrom]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: calibration
# -------------------------------------------------------------------------------------
# What it shows: predicted against FASTWIND log10(F_lambda) for the whole test
# partition.  Each model-wavelength pair on the native grid is one point, but the
# ~28 million points are drawn as a hexbin whose colour is log10(count), so the
# figure shows density rather than individual markers.  The dashed white diagonal
# is perfect agreement.  The cloud follows it over roughly nine decades,
# log10(F_lambda) ~ 28 to 37, so the accuracy does not depend on the absolute
# flux level; the only departure is a sparse, individually resolved cloud of
# overpredicted points at the bright end.
# Thesis: fig:fluxcont_pred_true_log_flux.
# Source: notebook cell 12.
# -------------------------------------------------------------------------------------
def save_calibration_plot(target_log: np.ndarray, pred_log: np.ndarray,
                          sample: np.ndarray, gridsize: int, output_path: Path,
                          extra_dirs: Sequence[Path] = (),
                          skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    hexbin = ax.hexbin(target_log[sample], pred_log[sample], gridsize=gridsize,
                       bins="log", mincnt=1, cmap="viridis")
    lo = min(np.nanmin(target_log[sample]), np.nanmin(pred_log[sample]))
    hi = max(np.nanmax(target_log[sample]), np.nanmax(pred_log[sample]))
    ax.plot([lo, hi], [lo, hi], color="white", lw=1.4, ls="--")
    ax.set_xlabel("FASTWIND log10(F_lambda)")
    ax.set_ylabel("Emulator log10(F_lambda)")
    ax.set_title("Predicted vs true log flux")
    colourbar = fig.colorbar(hexbin, ax=ax)
    colourbar.set_label("log10(count)")
    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: per-model-metrics
# -------------------------------------------------------------------------------------
# What it shows: three histograms over the test models, of the per-model MSE in
# log10(F_lambda), the per-model mean relative flux error, and the per-model
# median relative flux error, all with a logarithmic count axis so that the tail
# of poor models stays visible next to the bulk.
# Source: notebook cell 16.
# -------------------------------------------------------------------------------------
def save_metric_histograms(metrics: pd.DataFrame, output_path: Path,
                           extra_dirs: Sequence[Path] = (),
                           skip_existing: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(metrics["mse_log_flam"], bins=METRIC_HIST_BINS, color="tab:blue",
                 alpha=0.8)
    axes[0].set_xlabel("MSE in log10(F_lambda)")
    axes[0].set_ylabel("test models")
    axes[0].set_title("Per-model log-flux MSE")

    axes[1].hist(metrics["mean_relative_flux_error"], bins=METRIC_HIST_BINS,
                 color="tab:red", alpha=0.8)
    axes[1].set_xlabel("mean relative error")
    axes[1].set_title("Per-model mean relative error")

    axes[2].hist(metrics["median_relative_flux_error"], bins=METRIC_HIST_BINS,
                 color="tab:green", alpha=0.8)
    axes[2].set_xlabel("median relative error")
    axes[2].set_title("Per-model median relative error")

    for ax in axes:
        ax.set_yscale("log")

    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: worst-corner
# -------------------------------------------------------------------------------------
# What it shows: where the worst one per cent of test models sit in the input
# parameter space.  Grey is the full test partition (sub-sampled to 6,000 points
# for the off-diagonal panels, with a fixed seed, so the file stays small), red is
# the worst subset: 327 of the 32,730 test models, every one of which has a mean
# relative flux error above 5.96 %, against a test-set mean of 2.0 %.  The
# diagonal panels are the two densities of that parameter, the panels below the
# diagonal are the pairwise projections.  Only the parameters whose worst-subset
# distribution differs from the full test set are shown, measured by the
# two-sample Kolmogorov-Smirnov distance with the threshold D >= 0.15 and a
# minimum of three parameters; for the adopted run the selection is
# logg(-0.31), mdot(+0.24), radius(-0.24), vinf(-0.19).  The all-13-parameter
# version is written next to it for reference.
# Thesis: fig:fluxcont_parameter_error.
# Source: fluxcont_chapter_plots.corner_plot.
# -------------------------------------------------------------------------------------
def save_corner_plot(vals: np.ndarray, err: np.ndarray,
                     params: Sequence[Tuple[str, str, bool]], out_dir: Path,
                     suffix: str, worst_idx: np.ndarray,
                     metric: str = DEFAULT_METRIC,
                     worst_frac: float = WORST_FRACTION,
                     bg_max: int = CORNER_BACKGROUND_MAX, seed: int = CORNER_SEED,
                     panel: float = CORNER_PANEL_DEP, dpi: int = CORNER_DPI_DEP,
                     fmt: str = CORNER_FORMAT_DEP,
                     extra_dirs: Sequence[Path] = (),
                     skip_existing: bool = False) -> float:
    """Layout, colours and marker sizes follow the 13-parameter chapter script."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_worst = worst_idx.size
    cutoff = 100.0 * err[worst_idx[-1]]

    rng = np.random.default_rng(seed)
    bg = (rng.choice(err.size, size=bg_max, replace=False)
          if err.size > bg_max else np.arange(err.size))

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(panel * n, panel * n), squeeze=False)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:
                ax.hist(vals[:, c], bins=CORNER_DIAGONAL_BINS_ALL, color="0.75",
                        density=True, label="all test models")
                ax.hist(vals[worst_idx, c], bins=CORNER_DIAGONAL_BINS_WORST,
                        color="tab:red", alpha=0.65, density=True,
                        label="worst-error models")
                ax.set_yticks([])
                if r == 0:
                    ax.legend(fontsize=max(5, int(3 * panel)), loc="upper left")
            else:
                ax.scatter(vals[bg, c], vals[bg, r], s=3, color="0.75", alpha=0.4,
                           rasterized=True)
                ax.scatter(vals[worst_idx, c], vals[worst_idx, r], s=10,
                           color="tab:red", alpha=0.85, rasterized=True)
            if r == n - 1:
                ax.set_xlabel(params[c][1], fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(params[r][1], fontsize=9)
            elif c > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(f"FLUXCONT continuum emulator: 13-parameter space\n"
                 f"Worst {100 * worst_frac:.1f}% by {metric.replace('_', ' ')} "
                 f"({n_worst} of {err.size}), cutoff = {cutoff:.2f}%",
                 y=0.995, fontsize=12)
    fig.tight_layout()

    name = f"corner_worst1pct_{suffix}_FLUXCONT.{fmt}"
    save_figure(fig, out_dir / name, dpi=dpi, extra_dirs=extra_dirs,
                skip_existing=skip_existing)
    return cutoff


# -------------------------------------------------------------------------------------
# FIGURE: parameter-trends
# -------------------------------------------------------------------------------------
# What it shows: one panel per input parameter, with the per-model mean relative
# flux error of every test model as a grey rasterized scatter on a logarithmic y
# axis, the binned median as a solid red curve and the binned 90th percentile as
# a dashed orange curve.  The parameter range is split into 25 equal bins and a
# bin is only drawn when it holds at least 20 models, so the curves are not set by
# one or two models at the edge of the hypercube.  This answers "how large does
# the error become as a parameter is varied", which the corner plot cannot:
# the corner plot answers "where do the worst models sit".
# Thesis: fig:fluxcont_parameter_trends (appendix).
# Source: plot_mean_relative_error_vs_parameters.py.
# -------------------------------------------------------------------------------------
def save_parameter_trend_plot(err: np.ndarray, vals: np.ndarray,
                              params: Sequence[Tuple[str, str, bool]],
                              n_bins: int, min_per_bin: int, percentile: float,
                              output_path: Path, extra_dirs: Sequence[Path] = (),
                              skip_existing: bool = False) -> None:
    ncols, nrows = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.1 * nrows))
    axes = axes.ravel()

    for k, (_name, label, _use_log) in enumerate(params):
        ax = axes[k]
        x = vals[:, k]

        ax.scatter(x, err, s=3, color="0.6", alpha=0.15, rasterized=True)

        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        med = np.full(n_bins, np.nan)
        pct = np.full(n_bins, np.nan)
        idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() >= min_per_bin:
                med[b] = np.median(err[mask])
                pct[b] = np.percentile(err[mask], percentile)
        ax.plot(centers, med, color="tab:red", lw=1.8, label="binned median")
        ax.plot(centers, pct, color="tab:orange", lw=1.4, ls="--",
                label=f"binned {int(percentile)}th percentile")

        ax.set_yscale("log")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("mean relative error", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    for k in range(len(params), nrows * ncols):
        axes[k].axis("off")
    axes[0].legend(fontsize=8, loc="upper right")

    fig.suptitle("FLUXCONT emulator: per-model mean relative flux error "
                 "vs input parameters", fontsize=14, y=0.995)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=TREND_DPI, extra_dirs=extra_dirs,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: parameter-scatter
# -------------------------------------------------------------------------------------
# What it shows: the notebook's first attempt at the same question, in which the
# colour bar encodes log10 of the very quantity already on the y axis and is
# therefore redundant.  It is superseded by --figure parameter-trends and is not
# printed in the thesis, but it is kept because it is the step that builds the
# per-model-metrics/parameter join the other two families read.
# Source: notebook cell 21.
# -------------------------------------------------------------------------------------
def save_parameter_scatter_plot(metrics_params: pd.DataFrame,
                                params: Sequence[str], output_path: Path,
                                extra_dirs: Sequence[Path] = (),
                                skip_existing: bool = False) -> None:
    ncols = 4
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
    color = np.log10(np.clip(metrics_params["mean_relative_flux_error"].values,
                             1e-12, None))

    scatter = None
    for ax, col in zip(axes.ravel(), params):
        x = metrics_params[col].values
        if col == "mdot":
            x = np.log10(x)
            xlabel = "log10(mdot)"
        else:
            xlabel = col
        scatter = ax.scatter(x, metrics_params["mean_relative_flux_error"], c=color,
                             s=5, alpha=0.35, cmap="viridis")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean relative error")

    for ax in axes.ravel()[len(params):]:
        ax.axis("off")

    fig.subplots_adjust(hspace=0.65, wspace=0.35, top=0.92, right=0.88)
    fig.colorbar(scatter, ax=axes.ravel().tolist(),
                 label="log10(mean relative error)", shrink=0.78, pad=0.025)
    fig.suptitle("FLUXCONT emulator error vs input parameters", y=0.985, fontsize=14)
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: arch-summary
# -------------------------------------------------------------------------------------
# What it shows: the best validation loss, the test MSE in log10(F_lambda) and the
# mean and median relative flux errors of every completed configuration of the
# architecture search, in the order of the appendix table.  The first three
# quantities come from each run's emulator_FLUXCONT_meta.json; the median is the
# mean over the test models of the median_relative_flux_error column of that
# run's test_metrics_per_model.csv, which is how the published table computed it.
# Configurations whose mean relative error exceeds the "very large" cut-off are
# left blank in the LaTeX body, because converting a large logarithmic residual
# back to linear flux produces a number with no useful magnitude.
# Thesis: tab:fluxcont_architecture_metrics (appendix).
# Source: not produced by the notebook; assembled here from the run folders.
# -------------------------------------------------------------------------------------
def save_arch_summary_plot(table: pd.DataFrame, output_path: Path,
                           extra_dirs: Sequence[Path] = (),
                           skip_existing: bool = False) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    positions = np.arange(len(table))

    axes[0].bar(positions, table["best_val_loss"], color="tab:blue", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("best validation loss")
    axes[0].set_title("FLUXCONT architecture search: completed configurations")
    axes[0].grid(alpha=0.25)

    axes[1].bar(positions, table["test_mean_relative_flux_error"], color="tab:red",
                alpha=0.8, label="mean")
    axes[1].bar(positions, table["test_median_relative_flux_error"], color="tab:green",
                alpha=0.6, label="median")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative flux error")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(table["label"], rotation=75, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, output_path, extra_dirs=extra_dirs, skip_existing=skip_existing)


def arch_summary_latex(table: pd.DataFrame, very_large: float) -> str:
    """The body of tab:fluxcont_architecture_metrics, one ``\\hline``-separated row."""
    def sci(value: float) -> str:
        if not np.isfinite(value):
            return "--"
        exponent = int(math.floor(math.log10(abs(value)))) if value != 0 else 0
        mantissa = value / (10.0 ** exponent)
        if exponent == 0:
            return f"${mantissa:.2f}$"
        return f"${mantissa:.2f}\\times10^{{{exponent}}}$"

    lines = []
    for row in table.itertuples(index=False):
        mean = getattr(row, "test_mean_relative_flux_error")
        mean_text = "very large" if mean > very_large else sci(mean)
        lines.append(
            f"{row.label} & {sci(row.best_val_loss)} & "
            f"{sci(getattr(row, 'test_mse_log_flam'))} & {mean_text} & "
            f"{sci(getattr(row, 'test_median_relative_flux_error'))} \\\\"
        )
    return "\n\\hline\n".join(lines) + "\n"


# =====================================================================================
# SECTION 4.  FIGURE DRIVERS, ONE PER FAMILY
# =====================================================================================

def _extra_dirs(args: argparse.Namespace, subdir: str = "") -> Tuple[Path, ...]:
    """The thesis copy that the preserved scripts also wrote, unless disabled."""
    if args.no_thesis_copy or not args.thesis_dir:
        return ()
    base = Path(args.thesis_dir)
    if not base.parent.is_dir():
        print(f"[note] thesis folder not found at {base}; copy the figure manually")
        return ()
    return (base / subdir,) if subdir else (base,)


def figure_loss_curves(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    record = loss_histories(paths, args.loss_source)
    print(f"[loss] {len(record['train_losses'])} epochs, best epoch "
          f"{record['best_epoch']}, best validation loss {record['best_val_loss']}")
    save_loss_plot(record, paths["plots"] / "training_validation_loss.png",
                   extra_dirs=_extra_dirs(args), skip_existing=args.skip_existing)


def figure_sed_profiles(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    outputs = TestOutputs(paths["npz"])
    if args.describe:
        outputs.describe()

    index = args.model_index if args.model_index is not None else outputs.example_index()
    if not 0 <= index < outputs.n_test:
        raise SystemExit(f"--model-index must lie in [0, {outputs.n_test}).")
    print(f"Example model index: {index}")
    print(f"Example model ID: {int(outputs.test_model_ids[index])}")
    model = outputs.model_arrays(index)

    if args.sed_scope in ("example", "both"):
        save_example_sed_plot(model, paths["plots"] / "example_sed_profiles_loglog.png",
                              extra_dirs=_extra_dirs(args),
                              skip_existing=args.skip_existing)
        save_example_error_plot(
            model, paths["plots"] / "example_relative_error_vs_wavelength.png",
            extra_dirs=_extra_dirs(args), skip_existing=args.skip_existing)

    if args.sed_scope in ("worst", "both"):
        metrics = per_model_metrics(outputs, paths["metrics"])
        worst = metrics.sort_values(args.metric, ascending=False).head(args.table_rows)
        model_id_to_index = {int(mid): i for i, mid in enumerate(outputs.test_model_ids)}
        worst_indices = [model_id_to_index[int(mid)] for mid in worst["model_id"].values
                         if int(mid) in model_id_to_index]
        save_worst_sed_panels(outputs, worst_indices,
                              paths["plots"] / "worst_model_sed_profiles.png",
                              extra_dirs=_extra_dirs(args),
                              skip_existing=args.skip_existing)


def figure_signed_relative_error(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    npz_path = Path(paths["npz"])
    if not npz_path.exists():
        raise SystemExit(f"{npz_path} not found")
    if args.signed_max_samples < 1:
        raise SystemExit("--signed-max-samples must be at least one.")
    if args.signed_bin_edges < 2:
        raise SystemExit("--signed-bin-edges must be at least two.")

    print(f"Reading saved test outputs: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as saved:
        wavelengths = np.asarray(saved["wavelengths_phys_flat"])
        target_flux = np.asarray(saved["target_flux_flat"], dtype=np.float64)
        predicted_flux = np.asarray(saved["pred_flux_flat"], dtype=np.float64)

    if not wavelengths.shape == target_flux.shape == predicted_flux.shape:
        raise SystemExit("Wavelength, FASTWIND-flux, and predicted-flux arrays "
                         "have different shapes.")

    errors = signed_relative_error_percent(target_flux, predicted_flux)
    del target_flux, predicted_flux

    edges = log_wavelength_edges(wavelengths, args.signed_bin_edges)
    statistics = signed_binned_statistics(wavelengths, errors, edges)
    subset = displayed_subset(wavelengths, errors, seed=args.signed_seed,
                              max_samples=args.signed_max_samples)

    save_signed_relative_error_plot(
        wavelengths, errors, statistics, subset,
        paths["plots"] / "relative_error_binned_by_wavelength.png",
        extra_dirs=_extra_dirs(args), skip_existing=args.skip_existing)
    print(f"Used {len(edges)} logarithmic bin edges, "
          f"{len(statistics['wavelength'])} non-empty intervals, and "
          f"{len(subset)} displayed samples.")


def figure_binned_error(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    outputs = TestOutputs(paths["npz"])
    bins = log_wavelength_edges(outputs.wave, args.binned_error_edges)

    rel_by_wave = notebook_binned_stats(outputs.wave, outputs.rel_err, bins)
    logerr_by_wave = notebook_binned_stats(outputs.wave, outputs.abs_log_err, bins)
    save_table(rel_by_wave,
               paths["plots"] / "relative_error_binned_by_wavelength.csv")
    save_table(logerr_by_wave,
               paths["plots"] / "abs_log_error_binned_by_wavelength.csv")
    print(f"[bins] {len(bins)} edges, {len(bins) - 1} possible intervals, "
          f"{len(rel_by_wave)} populated")

    save_binned_error_plot(
        rel_by_wave, "tab:red", "relative error",
        "Relative error binned by wavelength",
        paths["plots"] / "relative_error_binned_by_wavelength_unsigned.png",
        skip_existing=args.skip_existing)
    save_binned_error_plot(
        logerr_by_wave, "tab:purple", "abs error in log10(F_lambda)",
        "Absolute log-flux error binned by wavelength",
        paths["plots"] / "abs_log_error_binned_by_wavelength.png",
        extra_dirs=_extra_dirs(args), skip_existing=args.skip_existing)


def figure_calibration(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    outputs = TestOutputs(paths["npz"])

    max_points = args.calibration_max_points
    if args.calibration_all_points or outputs.target_log.size <= max_points:
        sample = np.arange(outputs.target_log.size)
        print(f"[hexbin] all {sample.size} model-wavelength points")
    else:
        rng = np.random.default_rng(args.calibration_seed)
        sample = rng.choice(outputs.target_log.size, size=max_points, replace=False)
        print(f"[hexbin] {max_points} of {outputs.target_log.size} points, "
              f"seed {args.calibration_seed}")

    save_calibration_plot(outputs.target_log, outputs.pred_log, sample,
                          args.calibration_gridsize,
                          paths["plots"] / "pred_vs_true_log_flux_hexbin.png",
                          extra_dirs=_extra_dirs(args),
                          skip_existing=args.skip_existing)


def figure_per_model_metrics(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    outputs = TestOutputs(paths["npz"])
    metrics = per_model_metrics(outputs, paths["metrics"])

    print(metrics.describe().to_string())
    save_table(metrics, paths["plots"] / "test_metrics_per_model_from_notebook.csv")

    save_metric_histograms(metrics, paths["plots"] / "per_model_metric_histograms.png",
                           extra_dirs=_extra_dirs(args),
                           skip_existing=args.skip_existing)

    worst = metrics.sort_values(args.metric, ascending=False).head(args.table_rows)
    best = metrics.sort_values(args.metric, ascending=True).head(args.table_rows)
    print(f"\nWorst {args.table_rows} test models by {args.metric.replace('_', ' ')}")
    print(worst.to_string(index=False))
    print(f"\nBest {args.table_rows} test models by {args.metric.replace('_', ' ')}")
    print(best.to_string(index=False))
    save_table(worst, paths["plots"] / "worst_test_models.csv")
    save_table(best, paths["plots"] / "best_test_models.csv")


def figure_worst_corner(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    params = list(CORNER_PARAMS_13)
    _ids, err, raw = load_metrics_with_parameters(paths, args.grid_json,
                                                  args.metric, params)
    vals = to_plot_values(raw, params)
    worst_idx = worst_subset(err, args.worst_fraction)
    out_dir = paths["plots"] / CORNER_SUBDIR
    extra = _extra_dirs(args, CORNER_SUBDIR)

    if not args.skip_corners_dep:
        sel, ks, signs = select_dependent(vals, worst_idx, params,
                                          threshold=args.ks_threshold,
                                          min_params=args.min_params)
        sel_params = [params[k] for k in sel]
        desc = ", ".join(f"{params[k][0]}({signs[k]}{ks[k]:.2f})" for k in sel)

        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "dependent_params_report.txt"
        with open(report, "w") as handle:
            handle.write(f"FLUXCONT: {desc}\n")
        print(f"[dep] FLUXCONT: {desc}")
        print(f"[saved] {report}")

        cutoff = save_corner_plot(vals[:, sel], err, sel_params, out_dir, "dep",
                                  worst_idx, metric=args.metric,
                                  worst_frac=args.worst_fraction,
                                  panel=CORNER_PANEL_DEP, dpi=CORNER_DPI_DEP,
                                  fmt=CORNER_FORMAT_DEP, extra_dirs=extra,
                                  skip_existing=args.skip_existing)
        print(f"[worst] {worst_idx.size} of {err.size} test models, "
              f"cutoff = {cutoff:.2f}%")

    if not args.skip_corners13:
        save_corner_plot(vals, err, params, out_dir, "13par", worst_idx,
                         metric=args.metric, worst_frac=args.worst_fraction,
                         panel=CORNER_PANEL_13, dpi=CORNER_DPI_13,
                         fmt=CORNER_FORMAT_13, extra_dirs=extra,
                         skip_existing=args.skip_existing)


def figure_parameter_trends(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    params = list(TREND_PARAMS_13)
    _ids, err, raw = load_metrics_with_parameters(paths, args.grid_json,
                                                  args.metric, params)
    vals = to_plot_values(raw, params)
    print(f"{err.size} test models loaded")
    save_parameter_trend_plot(
        err, vals, params, args.parameter_bins, args.parameter_min_per_bin,
        args.parameter_percentile,
        paths["plots"] / "mean_relative_error_vs_parameters.png",
        extra_dirs=_extra_dirs(args), skip_existing=args.skip_existing)


def figure_parameter_scatter(args: argparse.Namespace) -> None:
    paths = run_paths(args)
    outputs = TestOutputs(paths["npz"])
    metrics = per_model_metrics(outputs, paths["metrics"])

    grid_json = Path(args.grid_json)
    if not grid_json.exists():
        raise SystemExit(
            f"{grid_json} not found; --figure parameter-scatter needs the "
            "13-parameter grid file to build the metrics/parameter join.")
    print(f"Loading {grid_json}. This can take a little while because the "
          "file is large.")
    with open(grid_json, "r") as handle:
        grid = json.load(handle)

    rows = []
    for mid in metrics["model_id"].astype(int):
        vals = grid.get(str(mid))
        if vals is None:
            continue
        row = {"model_id": mid}
        for col in PARAM_COLS:
            row[col] = vals.get(col, np.nan)
        rows.append(row)

    params = pd.DataFrame(rows)
    metrics_params = metrics.merge(params, on="model_id", how="inner")
    save_table(metrics_params, paths["metrics_with_params"])
    print(metrics_params.head().to_string(index=False))

    save_parameter_scatter_plot(
        metrics_params, list(PARAM_COLS),
        paths["plots"] / "mean_relative_error_vs_parameters_colorbar.png",
        skip_existing=args.skip_existing)


def figure_arch_summary(args: argparse.Namespace) -> None:
    search_root = Path(args.search_root)
    if not search_root.is_dir():
        raise SystemExit(f"--search-root is not a directory: {search_root}")

    label_by_run_tag = {run_tag: (key, label)
                        for key, run_tag, label in ARCHITECTURE_TABLE}
    order = {run_tag: position
             for position, (_key, run_tag, _label) in enumerate(ARCHITECTURE_TABLE)}

    rows = []
    for meta_path in sorted(search_root.glob(f"*/{ARTIFACT_STEM}_meta.json")):
        meta = load_json(meta_path)
        if meta is None:
            continue
        run_dir = meta_path.parent
        run_tag = meta.get("run_tag", run_dir.name)
        key, label = label_by_run_tag.get(run_tag, (meta.get("subversion") or "", run_tag))

        median = np.nan
        metrics_path = run_dir / METRICS_CSV_NAME
        if metrics_path.exists():
            frame = pd.read_csv(metrics_path)
            if "median_relative_flux_error" in frame:
                median = float(frame["median_relative_flux_error"].mean())

        rows.append({
            "position": order.get(run_tag, len(order) + len(rows)),
            "config_key": key,
            "label": label,
            "run_tag": run_tag,
            "subversion": meta.get("subversion"),
            "target_scaling": meta.get("target_scaling"),
            "norm_type": meta.get("norm_type"),
            "learning_rate": meta.get("learning_rate"),
            "epochs_run": meta.get("epochs_run"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_loss": meta.get("best_val_loss"),
            "test_mse_scaled_log_flam": meta.get("test_mse_scaled_log_flam"),
            "test_mse_log_flam": meta.get("test_mse_log_flam"),
            "test_mean_relative_flux_error": meta.get("test_mean_relative_flux_error"),
            "test_median_relative_flux_error": median,
            "n_test": meta.get("n_test"),
        })

    if not rows:
        raise SystemExit(f"no {ARTIFACT_STEM}_meta.json found under {search_root}")

    table = (pd.DataFrame(rows).sort_values("position")
             .drop(columns="position").reset_index(drop=True))
    print(table[["config_key", "label", "best_val_loss", "test_mse_log_flam",
                 "test_mean_relative_flux_error",
                 "test_median_relative_flux_error"]].to_string(index=False))

    output_dir = Path(args.output_dir) if args.output_dir else search_root
    save_table(table, Path(output_dir) / "fluxcont_architecture_metrics.csv")

    latex_path = Path(output_dir) / "fluxcont_architecture_metrics.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, "w") as handle:
        handle.write(arch_summary_latex(table, args.very_large_threshold))
    print(f"[saved] {latex_path}")

    save_arch_summary_plot(table, Path(output_dir) / "fluxcont_architecture_metrics.png",
                           skip_existing=args.skip_existing)


# =====================================================================================
# SECTION 5.  COMMAND-LINE INTERFACE
# =====================================================================================

#: Every figure family, in the order of the FIGURE MAP in the module docstring.
FIGURE_DISPATCH = {
    "loss-curves": figure_loss_curves,
    "sed-profiles": figure_sed_profiles,
    "signed-relative-error": figure_signed_relative_error,
    "binned-error": figure_binned_error,
    "calibration": figure_calibration,
    "per-model-metrics": figure_per_model_metrics,
    "worst-corner": figure_worst_corner,
    "parameter-trends": figure_parameter_trends,
    "parameter-scatter": figure_parameter_scatter,
    "arch-summary": figure_arch_summary,
}

#: What ``--all`` runs: every family that needs nothing but the adopted run
#: folder.  ``parameter-scatter`` is left out because it needs the large
#: 13-parameter grid file, and ``arch-summary`` because it walks the whole
#: architecture-search output tree rather than one run.
ALL_FIGURES: Tuple[str, ...] = (
    "loss-curves", "sed-profiles", "signed-relative-error", "binned-error",
    "calibration", "per-model-metrics", "worst-corner", "parameter-trends",
)


def build_parser() -> argparse.ArgumentParser:
    root = default_fluxcont_root()
    data_root = default_data_root()
    p = argparse.ArgumentParser(
        prog="fluxcont_plots.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--figure", choices=tuple(FIGURE_DISPATCH),
                          help="Which figure family to produce.  See the FIGURE MAP "
                               "in the module docstring.")
    selector.add_argument("--all", action="store_true",
                          help="Produce every figure that reads only the adopted run "
                               "folder: " + ", ".join(ALL_FIGURES) + ".")

    # --- where the run lives -------------------------------------------------
    p.add_argument("--fluxcont-root", dest="fluxcont_root", type=Path, default=root,
                   help="Root holding one folder per FLUXCONT run.")
    p.add_argument("--run-name", dest="run_name", default=DEFAULT_RUN_NAME,
                   help="Folder name of the run to plot (default: the adopted v17a).")
    p.add_argument("--run-dir", dest="run_dir", type=Path, default=None,
                   help="Full path to the run folder, overriding "
                        "--fluxcont-root/--run-name.")
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=None,
                   help=f"Where the figures go (default: <run>/{PLOT_SUBDIR}, the "
                        "notebook's PLOT_DIR).")
    p.add_argument("--thesis-dir", dest="thesis_dir", type=Path,
                   default=None,
                   help="Optional second destination for figures, for example a "
                        "thesis project. Disabled by default.")
    p.add_argument("--no-thesis-copy", dest="no_thesis_copy", action="store_true",
                   help="Do not also write the figures into the thesis folder.")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                   help="Leave a figure alone if its file already exists.")
    p.add_argument("--describe", action="store_true",
                   help="Print the NPZ key/shape listing of notebook cell 3.")
    p.add_argument("--verify-constants", dest="verify_constants", action="store_true",
                   help="Import fluxcont_training.py and assert that the artefact "
                        "names and the 13 parameter columns still match.  Needs torch.")

    # --- explicit artefact overrides (defaults come from artifact_paths) ------
    p.add_argument("--npz", type=Path, default=None,
                   help=f"Override <run>/{ARTIFACT_STEM}_test_outputs.npz.")
    p.add_argument("--meta-json", dest="meta_json", type=Path, default=None,
                   help=f"Override <run>/{ARTIFACT_STEM}_meta.json.")
    p.add_argument("--loss-json", dest="loss_json", type=Path, default=None,
                   help=f"Override <run>/{ARTIFACT_STEM}_loss.json.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help=f"Override <run>/{ARTIFACT_STEM}.pth.")
    p.add_argument("--metrics-csv", dest="metrics_csv", type=Path, default=None,
                   help=f"Override <run>/{METRICS_CSV_NAME}.")
    p.add_argument("--metrics-with-params-csv", dest="metrics_with_params_csv",
                   type=Path, default=None,
                   help=f"Override <run>/{PLOT_SUBDIR}/{METRICS_WITH_PARAMS_NAME}, "
                        "the metrics/parameter join.")
    p.add_argument("--grid-json", dest="grid_json", type=Path,
                   default=data_root / "thirteen_parameter" / "grid_large_log_LHC.json",
                   help="The 13-parameter LHC grid file, needed by "
                        "parameter-scatter and by worst-corner / parameter-trends "
                        "when the joined CSV is missing.")

    # --- loss-curves ---------------------------------------------------------
    p.add_argument("--loss-source", dest="loss_source",
                   choices=("auto", "json", "npz", "checkpoint"), default="auto",
                   help="Where the loss histories are read from.  'auto' is the "
                        f"notebook's order, {ARTIFACT_STEM}_loss.json first and the "
                        "NPZ arrays second; 'checkpoint' reads them out of the .pth "
                        "and is the only option that needs torch.")

    # --- sed-profiles / per-model-metrics ------------------------------------
    p.add_argument("--sed-scope", dest="sed_scope",
                   choices=("example", "worst", "both"), default="both",
                   help="'example' is the representative model of notebook cell 7 "
                        "(index n_test // 2), 'worst' the ranked panels of cell 19.")
    p.add_argument("--model-index", dest="model_index", type=int, default=None,
                   help="Index into the test partition of the representative model "
                        "(default: n_test // 2, as the notebook chose it).")
    p.add_argument("--metric", default=DEFAULT_METRIC,
                   choices=("mean_relative_flux_error", "median_relative_flux_error",
                            "mse_log_flam", "mse_scaled_log_flam"),
                   help="Per-model error column used to rank the models.")
    p.add_argument("--table-rows", dest="table_rows", type=int,
                   default=METRIC_TABLE_ROWS,
                   help="How many worst and best models are listed and drawn.")

    # --- signed-relative-error -----------------------------------------------
    p.add_argument("--signed-bin-edges", dest="signed_bin_edges", type=int,
                   default=SIGNED_BIN_EDGES,
                   help="Number of logarithmic wavelength-bin edges; 180 gives 179 "
                        "intervals, of which 130 are populated for the adopted run.")
    p.add_argument("--signed-max-samples", dest="signed_max_samples", type=int,
                   default=SIGNED_MAX_SAMPLES,
                   help="Maximum number of individual model-wavelength points shown.")
    p.add_argument("--signed-seed", dest="signed_seed", type=int, default=SIGNED_SEED,
                   help="Seed of that draw, used for nothing else.")

    # --- binned-error --------------------------------------------------------
    p.add_argument("--binned-error-edges", dest="binned_error_edges", type=int,
                   default=NOTEBOOK_BIN_EDGES,
                   help="Number of logarithmic bin edges of the unsigned pair.")

    # --- calibration ---------------------------------------------------------
    p.add_argument("--calibration-max-points", dest="calibration_max_points", type=int,
                   default=CALIBRATION_MAX_POINTS,
                   help="Model-wavelength points fed to the hexbin.")
    p.add_argument("--calibration-all-points", dest="calibration_all_points",
                   action="store_true",
                   help="Use every point instead of sub-sampling.")
    p.add_argument("--calibration-gridsize", dest="calibration_gridsize", type=int,
                   default=CALIBRATION_GRIDSIZE, help="Hexbin gridsize.")
    p.add_argument("--calibration-seed", dest="calibration_seed", type=int,
                   default=CALIBRATION_SEED,
                   help="Seed of the sub-sample.  The notebook referred to an "
                        "unseeded 'rng' it never created, so its value is not "
                        "recoverable; 42 is used, as everywhere else here.")

    # --- worst-corner --------------------------------------------------------
    p.add_argument("--worst-fraction", dest="worst_fraction", type=float,
                   default=WORST_FRACTION,
                   help="Fraction of test models treated as the worst subset; "
                        "0.01 of 32,730 is the 327 models the thesis quotes.")
    p.add_argument("--ks-threshold", dest="ks_threshold", type=float,
                   default=KS_THRESHOLD,
                   help="Kolmogorov-Smirnov distance above which a parameter counts "
                        "as dependent and is kept in the reduced corner plot.")
    p.add_argument("--min-params", dest="min_params", type=int, default=KS_MIN_PARAMS,
                   help="Minimum number of parameters kept in the reduced corner "
                        "plot, even if fewer pass the threshold.")
    p.add_argument("--skip-corners-dep", dest="skip_corners_dep", action="store_true",
                   help="Do not draw the reduced (KS-selected) corner plot.")
    p.add_argument("--skip-corners13", dest="skip_corners13", action="store_true",
                   help="Do not draw the all-13-parameter reference corner plot.")

    # --- parameter-trends ----------------------------------------------------
    p.add_argument("--parameter-bins", dest="parameter_bins", type=int,
                   default=TREND_BINS,
                   help="Equal-width bins per parameter for the median and "
                        "percentile curves.")
    p.add_argument("--parameter-min-per-bin", dest="parameter_min_per_bin", type=int,
                   default=TREND_MIN_PER_BIN,
                   help="A bin is only drawn when it holds at least this many models.")
    p.add_argument("--parameter-percentile", dest="parameter_percentile", type=float,
                   default=TREND_PERCENTILE,
                   help="Upper percentile drawn beside the binned median.")

    # --- arch-summary --------------------------------------------------------
    p.add_argument("--search-root", dest="search_root", type=Path, default=root,
                   help="Root of the architecture-search output tree; every "
                        f"sub-folder holding {ARTIFACT_STEM}_meta.json is one row.")
    p.add_argument("--very-large-threshold", dest="very_large_threshold", type=float,
                   default=VERY_LARGE_RELATIVE_ERROR,
                   help="Mean relative errors above this are printed as 'very "
                        "large' in the LaTeX table body.  The published table's "
                        "exact cut-off was never recorded; any value in "
                        "(23.3, 1.76e5] reproduces it.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.table_rows < 1:
        raise SystemExit("--table-rows must be at least one.")
    if args.parameter_bins < 1:
        raise SystemExit("--parameter-bins must be at least one.")
    if not 0.0 < args.worst_fraction <= 1.0:
        raise SystemExit("--worst-fraction must lie in (0, 1].")
    if args.min_params < 1:
        raise SystemExit("--min-params must be at least one.")

    if args.verify_constants:
        training_module()
        print("[verify] fluxcont_training.py agrees on the artefact names and "
              "the 13 parameter columns.")

    figures = list(ALL_FIGURES) if args.all else [args.figure]
    paths = run_paths(args)
    Path(paths["plots"]).mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {paths['run']}")
    print(f"Output directory: {paths['plots']}")
    print(f"Figures: {', '.join(figures)}")

    for name in figures:
        print(f"\n--- {name} ---")
        FIGURE_DISPATCH[name](args)

    print(f"\nDone: {len(figures)} figure famil{'y' if len(figures) == 1 else 'ies'} "
          f"written under {paths['plots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
