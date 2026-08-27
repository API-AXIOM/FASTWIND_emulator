#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thirteen_parameter_plots.py
===========================

Every figure of Chapter 4 (the 13-parameter per-diagnostic line emulator) and
of the 13-parameter appendix, produced from saved training outputs.

This single file replaces the following research scripts and notebooks, without
changing any numerical behaviour:

    13_par/py_codes/13-par_chapter_plots.py
                                        -> loss-curves      (loss_curves)
                                           profiles         (line_profiles,
                                                             profiles_all_lines)
                                           relative-error   (rel_err_vs_wavelength)
                                           per-line-mare    (test_mare_per_line)
                                           worst-corner     (corner_one_line,
                                                             corner_dependent_params)
    13_par/py_codes/13_plot_after_training.py
                                        -> loss-curves --layout grid,
                                           per-line-mare (the MSE variant),
                                           profiles (the BLOeM overlay)
    13_par/py_codes/13_fw_emulator_per_line_comparison.ipynb
                                        -> loss-curves (cells 4, 5),
                                           per-line-mare (cell 6),
                                           worst-corner --corner-set 13par (cell 9),
                                           profiles (cells 11, 12),
                                           relative-error --binning uniform (cell 13)
    13_par/py_codes/13_fw_emulator_comparison_with_filtering.ipynb
                                        -> the local re-application of the
                                           training filter (cells 1, 2), reused by
                                           own-test; loss-curves (cells 5, 6);
                                           per-line-mare (cell 7); worst-corner
                                           (cell 14)
    13_par/py_codes/13_fw_val_loss_relative_error_filtered_unfiltered.ipynb
                                        -> the line-core table used to label the
                                           diagnostics; see the NOT REPRODUCED
                                           note below for its own figures
    13_par/convergence_parameter_space/plot_13par_convergence_parameter_space.py
                                        -> convergence
    analysis/plot_case_perfomance.py    -> the case table, the corrupt-row screen
                                           and the local filter used by own-test
    analysis/plot_lhc_sobol_merged_comparison.py
                                        -> own-test  (collect + bar_figure, with
                                           the merged case removed, see SCOPE)
    analysis/compare_lhc_sobol_datasets.py
                                        -> dataset-audit
    datasets/py_codes/compare_line_between_grids.py
                                        -> grid-consistency
    13_par_hetero/plot_hetero_grid_profiles.py
                                        -> hetero-grid-profiles
    13_par_hetero/plot_hetero_z_calibration.py
                                        -> hetero-z-calibration

Training and inference live in the companion file
``thirteen_parameter_training.py``.  The network is always rebuilt through that
file's ``build_model``, so a plot can never be made with a different
architecture than the one that was trained.  The import is deferred, because
only the three families that re-evaluate a checkpoint genuinely need torch;
everything else is a re-reading of the saved ``.npz`` and ``.json`` artefacts
and runs without it.


SCOPE
-----
The thesis contains no cross-evaluation between the two datasets and no merged
emulator.  ``analysis/cross_evaluate_lhc_sobol_merged.py`` and the merged case
of ``plot_lhc_sobol_merged_comparison.py`` are therefore **not** reproduced
here as figure families, and no merged case appears anywhere in this file: the
case table below lists the Latin-hypercube and Sobol campaigns only, and every
ensemble is evaluated on its own test partition and on nothing else.  The one
thing taken over from the cross-evaluation script is its checkpoint-evaluation
arithmetic (parameter normalisation with ``mdot`` in the log, wavelength
normalisation by ``lambda_min`` / ``lambda_max``, residual plus ``flux_offset``),
which is a metric helper and is written here directly on top of
``thirteen_parameter_training.build_model``.


INPUTS
------
Every family reads the five per-line artefacts that
``thirteen_parameter_training.write_line_artifacts`` writes into a run folder,
whose names come from ``thirteen_parameter_training.line_artifact_paths``::

    emulator_<LINE>.pth                 checkpoint: model_state_dict,
                                        param_cols, branch_widths, trunk_widths,
                                        latent_dim, activation /
                                        activation_sequence, dropout,
                                        dropout_after_layer_indices,
                                        use_batch_norm, fourier_modes,
                                        lambda_min, lambda_max,
                                        param_mins_after_mdot_log,
                                        param_maxs_after_mdot_log,
                                        param_range_after_mdot_log,
                                        use_residual, flux_offset,
                                        train_losses, val_losses,
                                        best_epoch, best_val_loss
    emulator_<LINE>_loss.json           train_losses, val_losses, best_epoch,
                                        best_val_loss, epochs_run, stopped_early
    emulator_<LINE>_split.json          train_model_ids, val_model_ids,
                                        test_model_ids
    emulator_<LINE>_test_outputs.npz    test_model_ids, wavelengths_phys,
                                        wavelengths_norm, target_residual,
                                        pred_residual, target_flux, pred_flux
    emulator_<LINE>_meta.json           n_total_models, n_train, n_val, n_test,
                                        n_wavelength_points, lambda_min,
                                        lambda_max, crop_range, best_epoch,
                                        best_val_loss, epochs_run, ...

``wavelengths_phys`` is present because the adopted configuration sets
``npz_includes_wavelengths_phys = True``; where a run predates that key the
physical wavelengths are rebuilt from ``wavelengths_norm`` and the ``lambda``
pair of the meta file, exactly as the notebooks did.

Four families need one further input:

    profiles, worst-corner   ``grid_large_log_LHC.json``, the 13 raw parameters
                             of every Latin-hypercube model
    convergence              that file plus ``grid_quality_large_lhc.json``, the
                             FASTWIND convergence flag of every model, and
                             pdflatex for the PGFPlots figure
    grid-consistency         ``datasets/lhc/out_lines.h5``,
                             ``datasets/sobol_v2/out_lines.h5`` and
                             ``datasets/sobol_v2/models.h5``; needs h5py
    dataset-audit            the same three files plus
                             ``datasets/lhc/grid_quality_large_lhc.json``
    hetero-*                 the heteroscedastic run folders under
                             ``13_par_hetero_h5/runs``, written by
                             ``13_par_hetero/13_emulator_hetero_h5.py``


THE NUMBERS THE THESIS QUOTES
-----------------------------
Every literal below is copied from the source listed beside it; none of them is
re-derived or tuned here, because the thesis text quotes them.

    161 adaptive equal-count wavelength bins   relative-error
                                               (13-par_chapter_plots.rel_err_vs_wavelength)
    at most 500 grey model-wavelength points   relative-error, same function
    seed 42 for that grey sample               relative-error, worst-corner,
                                               grid-consistency, hetero-*
    percentiles 2.5 / 16 / 84 / 97.5           relative-error
    band colours #2a9d8f (16-84, alpha 0.42)
    and #f4a261 (2.5-97.5, alpha 0.24),
    grey #666666 for the samples               relative-error
    worst fraction 0.01                        worst-corner
    KS threshold D >= 0.15, minimum 3 params   worst-corner
    6000 background points in the corners      worst-corner
    ten equal-width bins per parameter         convergence
    overall non-green fraction 37.408 %        convergence (a *result*, printed
                                               by the driver and drawn as the
                                               dashed reference line)
    twelve equal bins of T_eff, minimum 30
    models per bin per dataset                 grid-consistency
    15,000 models per dataset per diagnostic   grid-consistency (the thesis
                                               value; the research script's own
                                               default was 30,000 -- see the
                                               --max-models help)
    80 histogram bins over the central 99 %,
    up to 25 overlaid profiles per dataset     grid-consistency
    37 diagnostics                             per-line-mare, own-test


FIGURE MAP
----------
    --figure loss-curves
        Training and validation weighted-MSE versus epoch on a logarithmic
        axis, one figure per diagnostic (``--layout single``) or all of them as
        four-column small multiples (``--layout grid``).
        Thesis: fig:thirteen_parameter_halpha_loss and appendix
        fig:thirteen_parameter_loss_curves_1 ... _last (all 37 diagnostics).

    --figure profiles
        FASTWIND profile against the emulator prediction for one test model:
        the single-window pair for ``--profile-line`` on the native FASTWIND
        grid and on a regular 0.20 Angstrom grid, and the all-diagnostic panel
        grids in the same two variants.  The regular-grid panels are the direct
        demonstration that the trunk takes wavelength as a *coordinate*; they
        re-evaluate the checkpoint and therefore need torch.
        Thesis: fig:thirteen_parameter_halpha_native_profiles,
        fig:thirteen_parameter_halpha_regular_profiles and appendix
        fig:thirteen_parameter_native_profiles, _regular_profiles.

    --figure relative-error
        Signed relative error 100 (F_FW - F_emu)/(F_FW + eps) against
        wavelength over the whole test partition, in 161 adaptive equal-count
        quantile bins of pooled physical wavelength, with the median, the mean,
        the 16-84 and 2.5-97.5 percentile bands and a reproducible grey sample
        of at most 500 model-wavelength points.
        Thesis: fig:thirteen_parameter_halpha_relative_error and appendix
        fig:thirteen_parameter_relative_error_1 ... _last.

    --figure per-line-mare
        Test-set MARE of each of the 37 diagnostics, ordered by central
        wavelength, on a logarithmic axis; the MSE companion is written too.
        Thesis: the MARE panel of the line-by-line results section.

    --figure worst-corner
        Corner plot locating the worst one per cent of test models of each
        diagnostic, ranked by model-level MARE.  ``--corner-set dependent``
        shows only the parameters whose worst-subset distribution differs from
        the full test partition by a two-sample Kolmogorov-Smirnov distance
        D >= 0.15, with a minimum of three parameters retained; ``5par`` and
        ``13par`` write the fixed five- and thirteen-parameter versions.
        Thesis: fig:thirteen_parameter_halpha_parameter_error and appendix
        fig:thirteen_parameter_corner_1 ... _last.

    --figure convergence
        Percentage of FASTWIND calculations not labelled green, against each of
        the 13 sampled parameters in ten equal-width bins, with the overall
        non-green fraction as a dashed reference line.  Mass loss is binned in
        the log, as it was sampled.  Written as a standalone PGFPlots document
        and compiled with pdflatex, as the source did.
        Thesis: fig:thirteen_parameter_convergence_distribution.

    --figure own-test
        Per-diagnostic performance of the independently trained
        Latin-hypercube and Sobol ensembles, each on its own test partition:
        MARE in the upper panel, best validation loss in the lower panel, both
        logarithmic, over the 37 diagnostics.  The single-panel versions are
        written as well.  There is no merged case and no cross-evaluation.
        Thesis: fig:thirteen_parameter_own_test_by_line.

    --figure grid-consistency
        Three-panel comparison of the raw FASTWIND spectra of the two datasets
        for each diagnostic: the median of the integral of (1 - F) dlambda in
        twelve equal bins of effective temperature, the distribution of that
        integral, and up to 25 overlaid profiles from each dataset drawn from a
        tolerance box around the reference parameters.  No emulator is
        involved and no model of one dataset is paired with a model of the
        other.
        Thesis: fig:thirteen_parameter_grid_consistency and the appendix set.

    --figure dataset-audit
        Not printed in the thesis: the parameter-range, wavelength-window and
        nearest-neighbour-profile audit of the two datasets that the
        grid-consistency discussion rests on.

    --figure hetero-grid-profiles
        Not printed in the thesis: the heteroscedastic emulator evaluated on
        40, 80, 161, 322 and 644 wavelength points with its one-sigma band, the
        FASTWIND truth shown only at its own 161 native points.  Reads the
        separate ``13_par_hetero_h5`` run tree, not the adopted run.

    --figure hetero-z-calibration
        Not printed in the thesis: the z-score calibration check of those error
        bars, per diagnostic and as a std(z) overview.


NOT REPRODUCED HERE
-------------------
Three 13-parameter figures of the thesis are produced by scripts that are not
among the sources consolidated into this file, and are deliberately left alone
rather than guessed at:

    thirteen_parameter_results/architecture_avg_mare_filtered_unfiltered.png
        datasets/py_codes/compare_architectures_mare.py
    thirteen_parameter_results/all_lines_mare_filtered_unfiltered_points.png
        datasets/py_codes/plot_all_lines_mare_filtered_unfiltered.py
    thirteen_parameter_results/uv_absolute_error_summary.pdf
        13_par/plot_codes/uv_absolute_error_summary.tex (a standalone PGFPlots
        document, not a Python figure)

``13_par/py_codes/emulators_stat_analysis.ipynb`` is the *five*-parameter
architecture-search analysis (17 line windows, the ``don_finetune_run_*``
tree); it is already consolidated into ``five_parameter_plots.py`` and
contributes no 13-parameter figure.  ``plot_13par_raw_fastwind_test_profiles.ipynb``
and ``broad_emission_velocity_qc.ipynb`` are dataset-QC notebooks whose output
is the training-filter JSON consumed by ``thirteen_parameter_training.py``, not
a thesis figure.


USAGE
-----
    # every figure that needs nothing but the adopted run folders
    python thirteen_parameter_plots.py --all \
        --runs-root outputs/thirteen_parameter/runs \
        --grid-json data/thirteen_parameter/grid_large_log_LHC.json

    # one family
    python thirteen_parameter_plots.py --figure relative-error
    python thirteen_parameter_plots.py --figure relative-error \
        --relative-error-bins 161 --scatter-points 500 --seed 42
    python thirteen_parameter_plots.py --figure worst-corner \
        --corner-set dependent --ks-threshold 0.15 --min-params 3
    python thirteen_parameter_plots.py --figure worst-corner --corner-set 5par
    python thirteen_parameter_plots.py --figure loss-curves --layout single
    python thirteen_parameter_plots.py --figure per-line-mare

    # the profile pair of one window, for one chosen test model (needs torch)
    python thirteen_parameter_plots.py --figure profiles \
        --profile-line HALPHAHEII6527 --model-id 1964
    python thirteen_parameter_plots.py --figure profiles --which median

    # the convergence figure; --no-pdflatex stops after the .dat and .tex files
    python thirteen_parameter_plots.py --figure convergence \
        --quality-json data/thirteen_parameter/grid_quality_large_lhc.json

    # each ensemble on its own test partition, LHC and Sobol only
    python thirteen_parameter_plots.py --figure own-test
    python thirteen_parameter_plots.py --figure own-test --no-local-filter

    # the raw-spectra comparison of the two datasets (needs h5py)
    python thirteen_parameter_plots.py --figure grid-consistency \
        --grid-consistency-lines CIII4070 HEII6406 --max-models 15000
    python thirteen_parameter_plots.py --figure grid-consistency \
        --grid-consistency-lines all
    python thirteen_parameter_plots.py --figure dataset-audit \
        --audit-sections params windows profiles

    # the heteroscedastic side experiment (needs torch)
    python thirteen_parameter_plots.py --figure hetero-grid-profiles \
        --hetero-grids 40 80 161 322 644 --n-models 1
    python thirteen_parameter_plots.py --figure hetero-z-calibration

    # check that this file and thirteen_parameter_training.py still agree
    python thirteen_parameter_plots.py --figure per-line-mare --verify-constants
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: every figure is written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================================
# SECTION 1.  CONSTANTS
# =====================================================================================
# Every literal in this section is transcribed from the source named in the
# comment beside it.  Nothing is invented and nothing is re-derived.


def default_root() -> Path:
    """Return the portable data root used by the plotting commands.

    ``FASTWIND_DATA_ROOT`` may point anywhere on local or shared storage.  If it
    is unset, the documented ``data/`` directory beside these scripts is used.
    ``ML_EMULATOR_ROOT`` remains accepted as a backwards-compatible alias.
    """
    script_dir = Path(__file__).resolve().parent
    repository_root = Path(
        os.environ.get("FASTWIND_EMULATOR_ROOT", script_dir)
    ).expanduser().resolve()
    candidate = (
        os.environ.get("FASTWIND_DATA_ROOT")
        or os.environ.get("ML_EMULATOR_ROOT")
        or str(repository_root / "data")
    )
    return Path(candidate).expanduser()


def default_output_root() -> Path:
    """Return the portable output root used for checkpoints and figures."""
    script_dir = Path(__file__).resolve().parent
    repository_root = Path(
        os.environ.get("FASTWIND_EMULATOR_ROOT", script_dir)
    ).expanduser().resolve()
    return Path(
        os.environ.get("FASTWIND_OUTPUT_ROOT", repository_root / "outputs")
    ).expanduser()


#: The run-folder glob of the adopted deep, filtered, crop/edge-padded
#: Latin-hypercube campaign.  Source: 13-par_chapter_plots.DEEP_FILT_GLOB.
DEEP_FILTERED_GLOB = ("deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
                      "_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad_group*")

#: The same string without the group suffix, used as the plot-tree folder name.
#: Source: 13-par_chapter_plots.OUT_BASE.
DEEP_FILTERED_TAG = ("deep_b128_256_512_512_1024_2048_t128_256_512_512_1024_2048"
                     "_lat128_fm32_relu_nodrop_nobn_linefilter_cropedgepad")

#: The 13 branch inputs, in the checkpoint's order.  Must equal
#: thirteen_parameter_training.PARAM_COLS; --verify-constants asserts it.
PARAM_COLS: Tuple[str, ...] = ("teff", "logg", "radius", "mdot", "yhe",
                               "C", "N", "O", "beta", "vinf", "fic", "fvel",
                               "fclump")

#: Rows FASTWIND writes per line file.  Equals
#: thirteen_parameter_training.MAX_ROWS_PER_LINE_FILE.
MAX_ROWS_PER_LINE_FILE = 161

#: Denominator guard of the *unsigned* MARE.  Source: 13-par_chapter_plots.EPS,
#: 13_plot_after_training.EPS and both comparison notebooks.
EPS = 1e-8

#: Denominator guard of the *signed* relative error, and of the MARE in
#: plot_case_perfomance.  Source: 13-par_chapter_plots.rel_err_vs_wavelength
#: (eps=1e-6) and plot_case_perfomance.EPS.
EPS_SIGNED = 1e-6

#: Corrupt-row screen applied everywhere before a metric is computed: an
#: all-zero profile, a negative flux anywhere, or a non-finite value.
#: Source: 13-par_chapter_plots.load_screened, plot_case_perfomance.load_filtered,
#: compare_line_between_grids.screen_corrupt.
CORRUPT_ZERO_TOLERANCE = 1e-6
CORRUPT_NEGATIVE_TOLERANCE = -1e-3

#: Corner-plot parameter sets.  (grid column, axis label, take log10 first).
#: Source: 13-par_chapter_plots.CORNER_PARAMS_5 / CORNER_PARAMS_13.
CORNER_PARAMS_5: Tuple[Tuple[str, str, bool], ...] = (
    ("teff", r"$T_\mathrm{eff}$ [K]", False),
    ("logg", r"$\log g$", False),
    ("radius", r"$R$ [$R_\odot$]", False),
    ("mdot", r"$\log_{10}\dot{M}$", True),
    ("yhe", r"$Y_\mathrm{He}$", False),
)
CORNER_PARAMS_13: Tuple[Tuple[str, str, bool], ...] = CORNER_PARAMS_5 + (
    ("C", r"$\epsilon_\mathrm{C}$", False),
    ("N", r"$\epsilon_\mathrm{N}$", False),
    ("O", r"$\epsilon_\mathrm{O}$", False),
    ("beta", r"$\beta$", False),
    ("vinf", r"$v_\infty$ [km/s]", False),
    ("fic", r"$f_\mathrm{ic}$", False),
    ("fvel", r"$f_\mathrm{vel}$", False),
    ("fclump", r"$f_\mathrm{cl}$", False),
)

#: The relative-error figure.  Source: 13-par_chapter_plots.rel_err_vs_wavelength,
#: whose signature is (eps=1e-6, n_bins=161, max_points=500, seed=42).
RELATIVE_ERROR_BINS = 161
RELATIVE_ERROR_SCATTER_POINTS = 500
RELATIVE_ERROR_PERCENTILES: Tuple[float, float, float, float] = (2.5, 16.0, 84.0, 97.5)
BAND_COLOUR_2SIGMA = "#f4a261"      # 2.5-97.5 band
BAND_ALPHA_2SIGMA = 0.24
BAND_COLOUR_1SIGMA = "#2a9d8f"      # 16-84 band
BAND_ALPHA_1SIGMA = 0.42
SCATTER_COLOUR = "#666666"
#: The one diagnostic whose y axis is clamped instead of following the band,
#: because its outliers are enormous.  Source: the `if s == "CIV1169b"` branch.
RELATIVE_ERROR_CLAMPED_LINE = "CIV1169b"
RELATIVE_ERROR_CLAMP = (-100.0, 100.0)
RELATIVE_ERROR_YPAD_FRACTION = 0.25

#: The corner plots.  Source: 13-par_chapter_plots.corner_one_line defaults and
#: corner_dependent_params defaults.
WORST_FRACTION = 0.01
CORNER_BACKGROUND_MAX = 6000
CORNER_SEED = 42
KS_THRESHOLD = 0.15
KS_MIN_PARAMS = 3
CORNER_HIST_BINS_ALL = 40
CORNER_HIST_BINS_WORST = 20

#: The uniform grid the emulator is asked to predict on, in Angstrom.
#: Source: 13-par_chapter_plots.line_profiles / profiles_all_lines, and the
#: BLOEM_DLAM of 13_plot_after_training and the comparison notebooks.
UNIFORM_GRID_SPACING = 0.2
#: The BLOeM LR02 window, used only by the overlay variant of the panel grids.
#: Source: 13_plot_after_training.BLOEM_LMIN / BLOEM_LMAX.
BLOEM_LMIN = 3960.0
BLOEM_LMAX = 4570.0

#: Panel grids.  Source: 13-par_chapter_plots.profiles_all_lines.
PANEL_NCOLS = 4

#: The convergence figure.  Source:
#: plot_13par_convergence_parameter_space.N_BINS and .PARAMETERS.
CONVERGENCE_BINS = 10
#: (grid column, lower, upper, axis scale, x label, panel title).  The ranges
#: are the design limits quoted in the thesis; mdot is binned after log10.
CONVERGENCE_PARAMETERS: Tuple[Tuple[str, float, float, float, str, str], ...] = (
    ("teff", 25_000.0, 55_000.0, 1.0e-3, r"$T_{\mathrm{eff}}$ (kK)", r"$T_{\mathrm{eff}}$"),
    ("logg", 2.5, 4.5, 1.0, r"$\log g$", r"$\log g$"),
    ("radius", 5.0, 30.0, 1.0, r"$R_\ast\ (R_\odot)$", r"$R_\ast$"),
    ("mdot", -9.5, -5.0, 1.0,
     r"$\log_{10}(\dot M/[M_\odot\,\mathrm{yr}^{-1}])$", r"$\dot M$"),
    ("yhe", 0.08, 0.20, 1.0, r"$Y_{\mathrm{He}}$", r"$Y_{\mathrm{He}}$"),
    ("C", 6.4, 9.0, 1.0, r"$12+\log_{10}(\mathrm{C/H})$", r"C"),
    ("N", 6.5, 9.0, 1.0, r"$12+\log_{10}(\mathrm{N/H})$", r"N"),
    ("O", 6.5, 9.5, 1.0, r"$12+\log_{10}(\mathrm{O/H})$", r"O"),
    ("beta", 0.5, 3.0, 1.0, r"$\beta$", r"$\beta$"),
    ("vinf", 500.0, 4000.0, 1.0e-3,
     r"$v_\infty\ (10^3\,\mathrm{km\,s^{-1}})$", r"$v_\infty$"),
    ("fic", 0.0, 1.0, 1.0, r"$f_{\mathrm{ic}}$", r"$f_{\mathrm{ic}}$"),
    ("fvel", 0.0, 1.0, 1.0, r"$f_{\mathrm{vel}}$", r"$f_{\mathrm{vel}}$"),
    ("fclump", 1.01, 40.0, 1.0, r"$f_{\mathrm{clump}}$", r"$f_{\mathrm{clump}}$"),
)
#: The value the thesis quotes for the dashed reference line.  It is *computed*
#: by the driver from the quality file; this literal is only the expected value
#: that the driver checks itself against and prints.
CONVERGENCE_EXPECTED_NON_GREEN_PERCENT = 37.408

#: ``HALPHAHEII6527`` is spelled ``HALPHA`` in the Sobol line names.
#: Source: plot_case_perfomance.ALIASES, compare_lhc_sobol_datasets.ALIASES.
SHORT_NAME_ALIASES: Dict[str, str] = {"HALPHAHEII6527": "HALPHA"}

#: The two campaigns compared by --figure own-test.  Transcribed from
#: plot_case_perfomance.CASES with every merged entry removed; see SCOPE.
CASES: Dict[str, Dict[str, str]] = {
    "lhc": {
        "glob": os.path.join("runs",
                             "*relu_nodrop_nobn_linefilter_cropedgepad_group*",
                             "emulator_*_test_outputs.npz"),
        "out": "lhc_perfomance_plots",
        "label": "LHC",
    },
    "sobol": {
        "glob": os.path.join("sobol_runs", "*sobol_h5_line*",
                             "emulator_*_test_outputs.npz"),
        "out": "sobol_perfomance_plots",
        "label": "Sobol",
    },
    # clean retrain: corrupt rows excluded from the training filter
    "sobol_nc": {
        "glob": os.path.join("sobol_runs_nocorrupt", "*sobol_h5_nocorrupt_line*",
                             "emulator_*_test_outputs.npz"),
        "out": "sobol_nocorrupt_perfomance_plots",
        "label": "Sobol (clean)",
    },
    # v2: the final_release Sobol grid, corrupt screen applied throughout
    "sobol_v2": {
        "glob": os.path.join("sobol_runs_v2", "*sobol_v2_h5_nocorrupt_line*",
                             "emulator_*_test_outputs.npz"),
        "out": "sobol_v2_perfomance_plots",
        "label": "Sobol",
    },
}
#: The two ensembles of the own-test figure, in bar order.  There is no third
#: entry: the thesis has no merged emulator.
OWN_TEST_ORDER: Tuple[str, ...] = ("lhc", "sobol_v2")
OWN_TEST_COLOURS: Dict[str, str] = {"lhc": "tab:blue", "sobol_v2": "tab:orange"}

#: The local re-application of the server-side training filter.  Source:
#: plot_case_perfomance (identical literals in
#: 13_fw_emulator_comparison_with_filtering.ipynb cell 1).
ABS_FLUX_THRESHOLD = 15.0
LINE_WING_DEVIATION = 0.02
LINE_CORE_DEVIATION_MIN = 0.20
WIDTH_A_LIMIT = 50.0
VINF_MULTIPLIER = 1.5
C_KMS = 299792.458
SIGNAL_MODES: Tuple[str, ...] = ("emission", "absorption", "line_signal")

#: Line components used by the width QC, keyed by short name.
#: Source: plot_case_perfomance.QC_COMPONENTS.
QC_COMPONENTS: Dict[str, List[float]] = {
    "HALPHA": [6527.0, 6562.8, 6683.0],
    "HEII4686": [4685.7, 4713.1],
    "HEII1640": [1640.4],
    "CIV1169b": [1168.8, 1174.9, 1175.3, 1175.6, 1175.7, 1175.9, 1176.0,
                 1176.4, 1176.6],
    "CIV1550": [1548.2, 1550.8],
    "NV1240": [1238.8, 1242.8, 1175.7],
    "NIV1718": [1718.0],
    "OIV1340": [1338.6, 1342.9, 1343.5],
    "OIV1371": [1371.3],
    "CIII5696": [5695.9],
    "CIV5801": [5801.3, 5812.0],
    "NIV4058": [4057.8],
    "NIV6380": [6380.0],
    "NV4603": [4603.7, 4619.9],
    "HBETA": [4861.3, 4859.3, 4867.0],
    "HGAMMA": [4340.5, 4338.7],
    "HDELTA": [4101.7, 4100.0, 4103.4, 4116.1],
    "HEPS": [3970.1, 3968.4, 3967.5],
    "HEI4471": [4471.5],
    "HEI4922": [4921.9],
    "HEI5875": [5875.6],
    "HEI7065": [7065.2],
    "HEII5411": [5411.5],
    "HEII6406": [6406.4],
    "HEI4026": [4026.2, 4025.6],
    "HEI4387": [4387.9, 4379.1, 4383.0],
    "HEII4200": [4200.0, 4196.0, 4199.0, 4215.0],
    "HEII4541": [4541.6, 4534.6],
    "HEII6683": [6683.2, 6678.2],
    "OIII5592": [5592.3],
    "NIV3480": [3478.7, 3483.0, 3485.0],
    "NIII1750": [1748.6, 1749.7, 1752.2],
    "CIII4070": [4067.9, 4068.9, 4070.3, 4072.2, 4075.9, 4076.4],
    "CIIINIIIcombined": [4634.1, 4640.6, 4641.9, 4647.4, 4650.3, 4651.5],
    "PV1118": [1118.0, 1128.0, 1113.2, 1128.3],
    "SiIII4552": [4552.6, 4567.8, 4574.8, 4603.7, 4619.9],
    "SiIV1400": [1393.8, 1402.8],
}

#: ``plot_case_perfomance.SOBOL_ID_OFFSET`` (1,000,000) is deliberately *not*
#: carried over: it exists only so that a merged Latin-hypercube + Sobol id
#: space can be disambiguated, and this file never builds one.

#: The grid-consistency figure.  Source: compare_line_between_grids.
GRID_CONSISTENCY_TEFF_BIN_EDGES = 13          # np.linspace(..., 13) -> 12 bins
GRID_CONSISTENCY_MIN_PER_BIN = 30
GRID_CONSISTENCY_HIST_BINS = 80
GRID_CONSISTENCY_HIST_PERCENTILES = (0.5, 99.5)
GRID_CONSISTENCY_MAX_PROFILES = 25
GRID_CONSISTENCY_TOL_PRIMARY = 0.04           # fraction of the driver-1 range
GRID_CONSISTENCY_TOL_SECONDARY = 0.10         # fraction of the driver-2 range
#: The thesis quotes 15,000 models per dataset per diagnostic.  The research
#: script's own argparse default was 30,000; the thesis figure was produced
#: with the smaller sample, so that is the default here and the larger value
#: stays reachable through --max-models.
GRID_CONSISTENCY_MAX_MODELS = 15_000
GRID_CONSISTENCY_SEED = 42

#: Crop windows of the training pipeline, used only to set the x range of the
#: profile panel.  Source: compare_line_between_grids.REF_CROP_RANGES.
REF_CROP_RANGES: Dict[str, List[float]] = {
    "CIV1169b": [800.0, 1400.0],
    "HEII1640": [1600.0, 1680.0],
    "HEII4686": [4600.0, 4800.0],
    "HALPHAHEII6527": [6405.0, 6690.0],
    "CIV5801": [5650.0, 5960.0],
    "HDELTA": [4000.0, 4200.0],
    "NIV4058": [3920.0, 4200.0],
    "NIV6380": [6250.0, 6520.0],
    "HEII6406": [6280.0, 6540.0],
    "NIV3480": [3400.0, 3550.0],
    "NV1240": [1100.0, 1400.0],
    "HEII5411": [5300.0, 5525.0],
    "HEII4200": [4120.0, 4300.0],
    "CIIINIIIcombined": [4450.0, 4750.0],
    "CIV1550": [1450.0, 1700.0],
    "CIII5696": [5580.0, 5820.0],
    "HEPS": [3885.0, 4050.0],
    "CIII4070": [3950.0, 4200.0],
    "HBETA": [4770.0, 4965.0],
    "HEI4026": [3900.0, 4150.0],
    "HEI4387": [4250.0, 4520.0],
    "HEI4471": [4350.0, 4590.0],
    "HEI4922": [4820.0, 5020.0],
    "HEI5875": [5750.0, 6000.0],
    "HEI7065": [6900.0, 7200.0],
    "HEII4541": [4450.0, 4630.0],
    "HEII6683": [6550.0, 6825.0],
    "HGAMMA": [4250.0, 4450.0],
    "NIII1750": [1680.0, 1825.0],
    "NIV1718": [1640.0, 1795.0],
    "OIII5592": [5475.0, 5700.0],
    "OIV1340": [1260.0, 1425.0],
    "OIV1371": [1340.0, 1405.0],
    "NV4603": [4500.0, 4725.0],
    "PV1118": [970.0, 1270.0],
    "SiIII4552": [4415.0, 4715.0],
    "SiIV1400": [1250.0, 1550.0],
}

#: Parameters that most directly set the strength of each diagnostic; the first
#: drives the binned comparison, the second refines the profile cell.
#: Source: compare_line_between_grids.LINE_DRIVERS / DEFAULT_DRIVERS.
LINE_DRIVERS: Dict[str, Tuple[str, str]] = {
    "CIII4070": ("teff", "C"), "CIII5696": ("teff", "C"),
    "CIIINIIIcombined": ("teff", "C"), "CIV1169b": ("teff", "C"),
    "CIV1550": ("teff", "mdot"), "CIV5801": ("teff", "C"),
    "NIII1750": ("teff", "N"), "NIV1718": ("teff", "N"),
    "NIV3480": ("teff", "N"), "NIV4058": ("teff", "N"),
    "NIV6380": ("teff", "N"), "NV1240": ("teff", "mdot"),
    "NV4603": ("teff", "N"), "OIII5592": ("teff", "O"),
    "OIV1340": ("teff", "O"), "OIV1371": ("teff", "O"),
    "SiIII4552": ("teff", "logg"), "SiIV1400": ("teff", "mdot"),
    "PV1118": ("teff", "mdot"), "HALPHA": ("teff", "mdot"),
    "HALPHAHEII6527": ("teff", "mdot"),
}
DEFAULT_DRIVERS: Tuple[str, str] = ("teff", "logg")
#: The five diagnostics compare_line_between_grids runs when none is named.
GRID_CONSISTENCY_DEFAULT_LINES: Tuple[str, ...] = (
    "CIII4070", "CIII5696", "HBETA", "HEI4471", "SiIV1400")

#: The dataset audit.  Source: compare_lhc_sobol_datasets.
AUDIT_HIST_BINS = 60
AUDIT_WINDOW_SAMPLE = 500
AUDIT_RANGE_MATCH_TOLERANCE = 0.02
AUDIT_DEFAULT_PAIRS = 3
AUDIT_DEFAULT_PAIR_LINES: Tuple[str, ...] = ("HBETA", "HALPHA", "HEI4471", "CIV1550")
AUDIT_SEED = 42

#: The heteroscedastic side experiment.  Source: plot_hetero_grid_profiles and
#: plot_hetero_z_calibration.
HETERO_DIR_GLOB = "*hetero_bnll_h5_line??_*"
HETERO_GRIDS: Tuple[int, ...] = (40, 80, 161, 322, 644)
HETERO_SEED = 42
#: Grids at or below this many points get error bars, denser grids get a band.
HETERO_ERRORBAR_MAX_POINTS = 170
HETERO_Z_HIST_BINS = 200
HETERO_Z_HIST_RANGE = (-5.0, 5.0)

#: Output resolutions, per source.
DPI_PROFILE_SINGLE = 200      # 13-par_chapter_plots.line_profiles
DPI_PANEL = 170               # profiles_all_lines, loss_curves, test_mare_per_line
DPI_RELATIVE_ERROR_PDF = 300  # rel_err_vs_wavelength
DPI_RELATIVE_ERROR_PNG = 170
DPI_CORNER_DEPENDENT = 160    # corner_one_line default
DPI_CORNER_13PAR = 150        # the 13-parameter variant
DPI_HETERO = 150              # plot_hetero_*


# =====================================================================================
# SECTION 2.  SHARED HELPERS
# =====================================================================================

# -------------------------------------------------------------------------------------
# Deferred heavy imports
# -------------------------------------------------------------------------------------
# torch is needed only by the three families that re-evaluate a checkpoint
# (profiles on the uniform grid, hetero-grid-profiles) and by --verify-constants;
# h5py only by grid-consistency and dataset-audit.  Importing them lazily keeps
# --help and every npz-only family usable in an environment without them, which
# is exactly what 13-par_chapter_plots.py did for its uniform-grid panel.


@lru_cache(maxsize=1)
def torch_module():
    import torch  # noqa: E402  (deliberately deferred)
    return torch


@lru_cache(maxsize=1)
def h5py_module():
    try:
        import h5py  # noqa: E402  (deliberately deferred)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("h5py is required for this figure family.") from exc
    return h5py


@lru_cache(maxsize=1)
def training_module():
    """``thirteen_parameter_training``, imported from this file's folder.

    The network classes, ``build_model`` and ``line_artifact_paths`` all come
    from there, so a figure can never be drawn with a different architecture or
    a different artefact naming convention than the run it describes.
    """
    import importlib
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("thirteen_parameter_training")


def verify_constants() -> None:
    """Assert that this file and the training script still agree."""
    training = training_module()
    if tuple(training.PARAM_COLS) != PARAM_COLS:
        raise SystemExit(
            f"PARAM_COLS drifted: training has {tuple(training.PARAM_COLS)}, "
            f"this file has {PARAM_COLS}")
    if int(training.MAX_ROWS_PER_LINE_FILE) != MAX_ROWS_PER_LINE_FILE:
        raise SystemExit("MAX_ROWS_PER_LINE_FILE drifted from the training script.")
    if int(training.INPUT_DIM) != len(PARAM_COLS):
        raise SystemExit("INPUT_DIM drifted from the training script.")
    names = training.line_artifact_paths("", "X")
    expected = {"model": "emulator_X.pth", "loss": "emulator_X_loss.json",
                "split": "emulator_X_split.json",
                "pred": "emulator_X_test_outputs.npz",
                "meta": "emulator_X_meta.json"}
    for key, want in expected.items():
        if os.path.basename(names[key]) != want:
            raise SystemExit(
                f"Artefact name drifted for {key!r}: {names[key]!r} != {want!r}")
    print("[verify] thirteen_parameter_training.py agrees on the 13 parameter "
          "columns, the 161 rows per line file and the five artefact names.")


def resolve_device(spec: str = "auto"):
    """``auto`` | ``cpu`` | ``cuda`` | ``cuda:N`` -> a torch device.

    ``thirteen_parameter_training.resolve_device`` takes no argument and returns
    the triple ``(device, cuda_available, n_gpus)`` that the training loop needs;
    the plotting side wants a single overridable device, so the selection is
    written out here and delegated to torch.
    """
    torch = torch_module()
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# -------------------------------------------------------------------------------------
# Naming
# -------------------------------------------------------------------------------------

def short_name(line: str) -> str:
    """``OUT.HGAMMA_VTV010.zst`` -> ``HGAMMA``.

    Source: 13-par_chapter_plots.short_name.  The alias table of
    plot_case_perfomance.short_name is applied only where that script's case
    machinery is used, so this stays the identity on ``HALPHAHEII6527``.
    """
    name = str(line)
    if name.endswith(".zst"):
        name = name[:-4]
    if name.startswith("OUT."):
        name = name[len("OUT."):]
    for suffix in ("_VTV010", "_VT010"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def short_name_aliased(line: str) -> str:
    """As above, then ``HALPHAHEII6527`` -> ``HALPHA``.

    Source: plot_case_perfomance.short_name / compare_lhc_sobol_datasets.short_name.
    Used wherever Latin-hypercube and Sobol line names have to be matched.
    """
    return SHORT_NAME_ALIASES.get(short_name(line), short_name(line))


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")


def load_json(path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


# -------------------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------------------

def save_figure(fig, output_path, dpi: Optional[float] = None,
                extra_paths: Sequence[Path] = (), tight_bbox: bool = True,
                skip_existing: bool = False) -> List[Path]:
    """Write one figure, optionally also into the thesis tree, and close it.

    ``extra_paths`` are *full* paths, not directories, because the thesis tree
    renames several families (``relative_error_vs_wavelength_OUT.X.pdf`` becomes
    ``rel_X.pdf``, as 13-par_chapter_plots.rel_err_vs_wavelength did).
    """
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        print(f"[skip] {output_path} exists")
        plt.close(fig)
        return []
    written: List[Path] = []
    for target in (output_path,) + tuple(Path(p) for p in extra_paths):
        target.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {}
        if dpi is not None:
            kwargs["dpi"] = dpi
        if tight_bbox:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(target, **kwargs)
        written.append(target)
        print(f"[saved] {target}")
    plt.close(fig)
    return written


def save_rows(rows: Sequence[Sequence[object]], header: Sequence[str],
              output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    print(f"[saved] {output_path}")
    return output_path


def thesis_target(args, subdir: str, filename: str) -> List[Path]:
    """The second copy of a printed figure, inside the thesis figure tree."""
    if args.no_thesis_copy or not args.thesis_dir:
        return []
    root = Path(args.thesis_dir)
    if not root.parent.is_dir():
        return []
    return [root / subdir / filename]


# -------------------------------------------------------------------------------------
# Run discovery and artefact loading
# -------------------------------------------------------------------------------------

def discover_lines(runs_root, run_glob: str) -> Dict[str, Dict[str, str]]:
    """short name -> the five artefact paths of the adopted campaign.

    Source: 13-par_chapter_plots.discover_deep_filtered, extended with the
    ``_split.json`` and ``_meta.json`` members that
    ``thirteen_parameter_training.line_artifact_paths`` also writes.
    """
    found: Dict[str, Dict[str, str]] = {}
    pattern = os.path.join(str(runs_root), run_glob, "emulator_*_test_outputs.npz")
    for npz_path in sorted(glob.glob(pattern)):
        stem = npz_path[: -len("_test_outputs.npz")]
        line = os.path.basename(stem)[len("emulator_"):]
        found[short_name(line)] = {
            "line": line,
            "short": short_name(line),
            "run_dir": os.path.dirname(npz_path),
            "npz": npz_path,
            "pth": stem + ".pth",
            "loss": stem + "_loss.json",
            "split": stem + "_split.json",
            "meta": stem + "_meta.json",
        }
    if not found:
        raise SystemExit(f"no run folders matched {pattern}")
    return found


def load_screened(npz_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(ids, lambda, target, prediction)`` with the corrupt rows removed.

    Source: 13-par_chapter_plots.load_screened.  ``wavelengths_phys`` is the key
    that ``thirteen_parameter_training`` writes for the adopted configuration;
    a run without it is handled by rebuilding the physical wavelengths from
    ``wavelengths_norm`` and the meta file, as the notebooks did.
    """
    with np.load(npz_path) as data:
        ids = data["test_model_ids"].astype(np.int64)
        if "wavelengths_phys" in data:
            lam = data["wavelengths_phys"].astype(np.float64)
        else:
            meta_path = str(npz_path)[: -len("_test_outputs.npz")] + "_meta.json"
            meta = load_json(meta_path)
            lam_min = float(meta["lambda_min"])
            lam_max = float(meta["lambda_max"])
            lam = (lam_min + data["wavelengths_norm"].astype(np.float64)
                   * (lam_max - lam_min))
        tgt = data["target_flux"].astype(np.float64)
        prd = data["pred_flux"].astype(np.float64)
    bad = ((np.abs(tgt).max(axis=1) < CORRUPT_ZERO_TOLERANCE)
           | (tgt.min(axis=1) < CORRUPT_NEGATIVE_TOLERANCE)
           | ~np.isfinite(tgt).all(axis=1))
    return ids[~bad], lam[~bad], tgt[~bad], prd[~bad]


def shared_test_ids(lines: Dict[str, Dict[str, str]]) -> List[int]:
    """Model ids in the test partition of *every* diagnostic.

    Source: 13-par_chapter_plots.shared_test_ids.  The 70/15/15 split is applied
    per line to that line's own surviving model list, so the intersection is
    taken rather than assumed.
    """
    shared: Optional[set] = None
    for key in sorted(lines):
        with np.load(lines[key]["npz"]) as data:
            ids = set(data["test_model_ids"].astype(np.int64).tolist())
        shared = ids if shared is None else shared & ids
    return sorted(shared) if shared else []


def model_level_mare(target: np.ndarray, prediction: np.ndarray,
                     eps: float = EPS) -> np.ndarray:
    """Per-model MARE: the mean over wavelength of |dF| / (|F| + eps).

    Source: 13-par_chapter_plots (``rel.mean(axis=1)``), identical in
    plot_case_perfomance and both comparison notebooks.
    """
    return (np.abs(prediction - target) / (np.abs(target) + eps)).mean(axis=1)


# -------------------------------------------------------------------------------------
# The Latin-hypercube parameter table
# -------------------------------------------------------------------------------------

_GRID_CACHE: Dict[str, dict] = {}


def lhc_grid(grid_json) -> dict:
    """The whole ``grid_large_log_LHC.json``, cached; it is large and slow."""
    key = str(grid_json)
    if key not in _GRID_CACHE:
        print(f"[params] loading {key} (large, ~1-2 min)", flush=True)
        _GRID_CACHE[key] = load_json(key)
    return _GRID_CACHE[key]


def lhc_params(grid_json, ids: Sequence[int],
               params: Sequence[Tuple[str, str, bool]]) -> np.ndarray:
    """``(N, len(params))`` raw parameter values for the given model ids.

    Source: 13-par_chapter_plots.lhc_params.
    """
    grid = lhc_grid(grid_json)
    cols = [col for col, _, _ in params]
    return np.array([[float(grid[str(int(m))][c]) for c in cols] for m in ids],
                    dtype=np.float64)


def lhc_raw_parameter_rows(grid_json, ids: Sequence[int]) -> np.ndarray:
    """``(N, 13)`` raw parameters in ``PARAM_COLS`` order.

    The same lookup ``cross_evaluate_lhc_sobol_merged.params_for_ids`` performs
    for the ``lhc`` case; reproduced here as a metric helper so that this file
    never imports the cross-evaluation script.
    """
    grid = lhc_grid(grid_json)
    rows = np.empty((len(ids), len(PARAM_COLS)), dtype=np.float64)
    for i, model_id in enumerate(ids):
        record = grid[str(int(model_id))]
        rows[i] = [float(record[c]) for c in PARAM_COLS]
    return rows


def apply_corner_logs(values: np.ndarray,
                      params: Sequence[Tuple[str, str, bool]]) -> np.ndarray:
    """Take log10 of the columns flagged for it.  Source: corner_one_line."""
    out = values.copy()
    for k, (_, _, use_log) in enumerate(params):
        if use_log:
            out[:, k] = np.log10(out[:, k])
    return out


# -------------------------------------------------------------------------------------
# Checkpoint evaluation
# -------------------------------------------------------------------------------------

def _config_from_checkpoint(state: dict) -> dict:
    """The ``cfg`` dict ``thirteen_parameter_training.build_model`` expects.

    The fallbacks are those of 13_plot_after_training.load_model, including its
    expansion of the ``mixed_leaky_relu_silu`` activation label.
    """
    branch_widths = list(state.get("branch_widths", [128, 256, 512]))
    trunk_widths = list(state.get("trunk_widths", [256, 512, 512]))
    n_layers = max(len(branch_widths), len(trunk_widths))
    sequence = state.get("activation_sequence")
    if not (isinstance(sequence, list) and sequence):
        label = str(state.get("activation", "relu")).lower()
        if label == "mixed_leaky_relu_silu":
            sequence = ["leaky_relu", "leaky_relu"] + ["silu"] * max(0, n_layers - 2)
        else:
            sequence = [label] * n_layers
    return {
        "branch_widths": branch_widths,
        "trunk_widths": trunk_widths,
        "latent_dim": int(state.get("latent_dim", 128)),
        "fourier_modes": int(state.get("fourier_modes", 32)),
        "activation_sequence": list(sequence),
        "dropout": float(state.get("dropout", 0.0) or 0.0),
        "dropout_after_layer_indices": list(
            state.get("dropout_after_layer_indices", [])),
        "use_batch_norm": bool(state.get("use_batch_norm", False)),
    }


class LineEmulator:
    """One line checkpoint, ready to be asked for flux at arbitrary wavelengths.

    The network itself is built by ``thirteen_parameter_training.build_model``.
    The three scaling steps are the ones the training script applied, read back
    from the checkpoint: ``mdot`` in the log, then the min/range normalisation
    of the 13 branch inputs, the linear wavelength normalisation by
    ``lambda_min`` / ``lambda_max``, and the residual offset undone on output.
    """

    def __init__(self, pth_path, device=None, batch_size: int = 512):
        torch = torch_module()
        training = training_module()
        state = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = int(batch_size)
        self.model = training.build_model(_config_from_checkpoint(state))
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.param_cols = list(state.get("param_cols", PARAM_COLS))
        self.mdot_index = self.param_cols.index("mdot")
        self.param_min = np.asarray(state["param_mins_after_mdot_log"], np.float64)
        self.param_range = np.asarray(state["param_range_after_mdot_log"], np.float64)
        self.lambda_min = float(state["lambda_min"])
        self.lambda_max = float(state["lambda_max"])
        self.use_residual = bool(state.get("use_residual", True))
        self.flux_offset = float(state.get("flux_offset", 1.0))
        self.state = state

    def normalize_params(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, np.float64).copy()
        if values.ndim == 1:
            values = values[None, :]
        values[:, self.mdot_index] = np.log10(values[:, self.mdot_index])
        return (values - self.param_min) / self.param_range

    def predict(self, raw_params: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """``raw_params`` (N, 13) and ``wavelengths`` (N, K) -> flux (N, K)."""
        torch = torch_module()
        params_norm = self.normalize_params(raw_params)
        lam = np.atleast_2d(np.asarray(wavelengths, np.float64))
        coords = (lam - self.lambda_min) / (self.lambda_max - self.lambda_min)
        out = np.empty(lam.shape, dtype=np.float32)
        params_t = torch.as_tensor(params_norm, dtype=torch.float32,
                                   device=self.device)
        coords_t = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for start in range(0, params_t.shape[0], self.batch_size):
                stop = start + self.batch_size
                pred = self.model(params_t[start:stop], coords_t[start:stop])
                out[start:stop] = pred.cpu().numpy()
        if self.use_residual:
            out = out + self.flux_offset
        return out


def uniform_grid(lam_min: float, lam_max: float,
                 spacing: float = UNIFORM_GRID_SPACING) -> np.ndarray:
    """The regular grid the emulator was never trained on.

    Source: 13-par_chapter_plots (``np.arange(lam.min(), lam.max() + 1e-9, 0.2)``).
    """
    return np.arange(float(lam_min), float(lam_max) + 1e-9, float(spacing))


# -------------------------------------------------------------------------------------
# The local re-application of the training filter
# -------------------------------------------------------------------------------------
# The server-side filter JSON is not part of the analysis workspace, so the
# per-row decisions are rebuilt from the saved FASTWIND targets themselves.
# Every literal is that of plot_case_perfomance, which in turn copied them from
# 13_fw_emulator_comparison_with_filtering.ipynb cell 1 and from the filter
# builder that thirteen_parameter_training.py consumes.

def signal_mask(flux: np.ndarray, mode: str) -> np.ndarray:
    if mode == "emission":
        return flux > 1.0 + LINE_WING_DEVIATION
    if mode == "absorption":
        return flux < 1.0 - LINE_WING_DEVIATION
    return np.abs(flux - 1.0) > LINE_WING_DEVIATION


def signal_strength(flux: np.ndarray, mode: str) -> float:
    if mode == "emission":
        return float(np.nanmax(flux - 1.0))
    if mode == "absorption":
        return float(np.nanmax(1.0 - flux))
    return float(np.nanmax(np.abs(flux - 1.0)))


def longest_segment_width(wavelength: np.ndarray, flux: np.ndarray,
                          mode: str) -> float:
    good = np.isfinite(wavelength) & np.isfinite(flux) & signal_mask(flux, mode)
    if not np.any(good):
        return 0.0
    indices = np.flatnonzero(good)
    best = 0.0
    for segment in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
        if len(segment):
            best = max(best, float(wavelength[segment[-1]] - wavelength[segment[0]]))
    return best


def row_passes_filter(wavelength: np.ndarray, flux: np.ndarray,
                      components: Sequence[float], vinf: Optional[float]) -> bool:
    """The abs-flux cut and the physical width QC of one profile.

    Source: plot_case_perfomance.row_passes_filter.
    """
    if float(np.nanmax(np.abs(flux))) > ABS_FLUX_THRESHOLD:
        return False
    for mode in SIGNAL_MODES:
        if signal_strength(flux, mode) < LINE_CORE_DEVIATION_MIN:
            continue
        if longest_segment_width(wavelength, flux, mode) <= WIDTH_A_LIMIT:
            continue
        if vinf is None or not np.isfinite(vinf) or vinf <= 0:
            return False
        velocity_limit = VINF_MULTIPLIER * float(vinf)
        finite = np.isfinite(wavelength)
        allowed = np.zeros_like(wavelength, dtype=bool)
        for component in components:
            allowed |= finite & (
                np.abs(C_KMS * (wavelength / float(component) - 1.0)) <= velocity_limit)
        signal = finite & np.isfinite(flux) & signal_mask(flux, mode)
        if np.any(signal & ~allowed):
            return False
    return True


def case_paths(args, case: str) -> Dict[str, str]:
    """Absolute paths of one campaign, from the relative CASES table."""
    base = Path(args.base_dir)
    entry = CASES[case]
    return {"glob": str(base / entry["glob"]), "out": str(base / entry["out"]),
            "label": entry["label"]}


def discover_case(args, case: str) -> Dict[str, Dict[str, str]]:
    """short name (aliased) -> artefact paths of one campaign.

    Source: plot_case_perfomance.discover.
    """
    found: Dict[str, Dict[str, str]] = {}
    for npz_path in sorted(glob.glob(case_paths(args, case)["glob"])):
        stem = npz_path[: -len("_test_outputs.npz")]
        line = os.path.basename(stem)[len("emulator_"):]
        key = short_name_aliased(line)
        found[key] = {"line": line, "short": key, "npz": npz_path,
                      "pth": stem + ".pth", "loss": stem + "_loss.json",
                      "meta": stem + "_meta.json"}
    return found


_VINF_CACHE: Dict[str, Optional[Dict[int, float]]] = {}


def sobol_models_h5(args, case: str) -> str:
    """The ``models.h5`` holding the Sobol parameters of this campaign.

    Source: plot_case_perfomance.sobol_models_h5_for -- the ``_v2`` cases read
    the final_release grid, the others the earlier one.
    """
    sub = "sobol_v2" if str(case).endswith("_v2") else "sobol"
    return str(Path(args.datasets_dir) / sub / "models.h5")


def vinf_map_for_case(args, case: str) -> Optional[Dict[int, float]]:
    """``model_id -> vinf``, or ``None`` when the source is unavailable.

    Source: plot_case_perfomance.vinf_map_for_case, with the merged branch
    removed: there is no merged campaign in this file, so the id spaces of the
    two datasets are never combined and no id offset is ever applied.
    """
    if case in _VINF_CACHE:
        return _VINF_CACHE[case]

    result: Optional[Dict[int, float]] = None
    if case == "lhc":
        grid_json = str(args.grid_json)
        if os.path.exists(grid_json):
            grid = lhc_grid(grid_json)
            result = {int(k): float(v["vinf"]) for k, v in grid.items() if "vinf" in v}
    else:
        h5_path = sobol_models_h5(args, case)
        if os.path.exists(h5_path):
            h5py = h5py_module()
            with h5py.File(h5_path, "r") as handle:
                ids = handle["model_ids"][:].astype(np.int64)
                cols = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in handle["param_cols"][:]]
                vinf = handle["parameters"][:, cols.index("vinf")].astype(np.float64)
            result = {int(m): float(v) for m, v in zip(ids, vinf)}

    if result is None:
        print(f"[filter][WARN] no vinf source for case {case!r} -- the width QC is "
              "SKIPPED (abs-flux cut only).")
    _VINF_CACHE[case] = result
    return result


def load_case_filtered(info: Dict[str, str],
                       vinf_map: Optional[Dict[int, float]],
                       apply_filter: bool = True) -> Dict[str, object]:
    """One line of one campaign: corrupt screen, then the local filter.

    Source: plot_case_perfomance.load_filtered.  With ``apply_filter=False``
    only the corrupt-row screen runs, which is the screen
    13-par_chapter_plots.load_screened applies.
    """
    ids, lam, tgt, prd = load_screened(info["npz"])
    n_after_screen = int(ids.shape[0])
    if not apply_filter:
        return {"ids": ids, "lam": lam, "tgt": tgt, "prd": prd,
                "skipped": 0, "n_kept": n_after_screen}

    components = QC_COMPONENTS.get(info["short"], [])
    keep = np.ones(n_after_screen, dtype=bool)
    for i in range(n_after_screen):
        vinf = vinf_map.get(int(ids[i])) if vinf_map else None
        keep[i] = row_passes_filter(lam[i], tgt[i], components, vinf)
    return {"ids": ids[keep], "lam": lam[keep], "tgt": tgt[keep], "prd": prd[keep],
            "skipped": int((~keep).sum()), "n_kept": int(keep.sum())}


# =====================================================================================
# SECTION 3.  FIGURE BLOCKS
# =====================================================================================

# -------------------------------------------------------------------------------------
# FIGURE: loss-curves
# -------------------------------------------------------------------------------------
# What it shows: the amplitude-weighted mean-squared error of the training
# objective, w = 1 + 5|F - 1|, on the training and validation partitions against
# epoch on a logarithmic y axis, for one diagnostic.  The checkpoint that was
# kept is the minimum of the validation curve, not the last epoch, so the gap
# between the two curves at that minimum is the overfitting diagnostic.
# Thesis: fig:thirteen_parameter_halpha_loss and appendix
#         fig:thirteen_parameter_loss_curves_1 ... _last (all 37 diagnostics).
# Source: 13-par_chapter_plots.loss_curves for the single-panel version;
#         13_plot_after_training.plot_loss_curves and
#         13_fw_emulator_per_line_comparison.ipynb cell 5 for the grid version;
#         cell 4 of that notebook for the best-epoch marker.
# -------------------------------------------------------------------------------------
def save_loss_plot(short: str, loss: dict, output_path, file_format: str,
                   extra_paths: Sequence[Path] = (), mark_best_epoch: bool = False,
                   skip_existing: bool = False) -> None:
    train = np.asarray(loss.get("train_losses", []), dtype=float)
    validation = np.asarray(loss.get("val_losses", []), dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(np.arange(1, len(train) + 1), train, lw=1.2, color="tab:blue",
            label="train")
    ax.plot(np.arange(1, len(validation) + 1), validation, lw=1.2,
            color="tab:orange", label="validation")
    if mark_best_epoch and np.isfinite(loss.get("best_epoch", np.nan) or np.nan):
        ax.axvline(int(loss["best_epoch"]), color="k", lw=1, ls="--", alpha=0.5,
                   label="Best epoch")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted MSE loss")
    ax.set_title(f"{short}: training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


def save_loss_grid(lines: Dict[str, Dict[str, str]], output_path,
                   skip_existing: bool = False) -> None:
    """All diagnostics as four-column small multiples.

    Source: 13_plot_after_training.plot_loss_curves (figsize 4.3 x 2.9 per
    panel) and 13_fw_emulator_per_line_comparison.ipynb cell 5 (4.2 x 2.8).
    The panel size of the standalone script is used, as it is the one that ran
    over the full 37-diagnostic set.
    """
    shorts = sorted(lines)
    ncols = PANEL_NCOLS
    nrows = math.ceil(len(shorts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows),
                             squeeze=False)
    for ax, short in zip(axes.flat, shorts):
        loss = load_json(lines[short]["loss"])
        train = np.asarray(loss.get("train_losses", []), dtype=float)
        validation = np.asarray(loss.get("val_losses", []), dtype=float)
        if len(train):
            ax.plot(np.arange(1, len(train) + 1), train, label="train", lw=1)
        if len(validation):
            ax.plot(np.arange(1, len(validation) + 1), validation, label="val", lw=1)
        ax.set_yscale("log")
        ax.set_title(short, fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(shorts):]:
        fig.delaxes(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Training and validation loss per line", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.98))
    save_figure(fig, output_path, dpi=180, tight_bbox=False,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: profiles
# -------------------------------------------------------------------------------------
# What it shows: for one test model, the FASTWIND profile (blue, on its native
# irregular grid) with the emulator prediction over-plotted (orange), once with
# the emulator on that same native grid -- the predictions saved in the npz --
# and once with the checkpoint re-evaluated on a regular 0.20 Angstrom grid
# spanning the same window.  The regular-grid panel is the point of the whole
# coordinate-based construction: the trunk was never trained on a regular grid,
# yet the emulator can be queried at any wavelength inside the window, which is
# what an instrument-matched spectrum requires.
# Thesis: fig:thirteen_parameter_halpha_native_profiles,
#         fig:thirteen_parameter_halpha_regular_profiles and appendix
#         fig:thirteen_parameter_native_profiles, _regular_profiles.
# Source: 13-par_chapter_plots.line_profiles (single window) and
#         .profiles_all_lines (the panel grids); the BLOeM overlay variant is
#         13_plot_after_training.plot_profiles and
#         13_fw_emulator_per_line_comparison.ipynb cell 12.
# -------------------------------------------------------------------------------------
def save_single_profile_native(short: str, model_id: int, lam: np.ndarray,
                               target: np.ndarray, prediction: np.ndarray,
                               output_path, extra_paths: Sequence[Path] = (),
                               skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(lam, target, "o-", ms=3.5, lw=0.9, color="tab:blue",
            label="FASTWIND (native grid)")
    ax.plot(lam, prediction, "o--", ms=3, lw=1.0, color="tab:orange",
            label="Emulator (native grid)")
    ax.set_xlabel(r"wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"{short}, test model {model_id}: FASTWIND wavelength grid "
                 f"({lam.size} points)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PROFILE_SINGLE, extra_paths=extra_paths,
                skip_existing=skip_existing)


def save_single_profile_uniform(short: str, model_id: int, lam: np.ndarray,
                                target: np.ndarray, grid: np.ndarray,
                                prediction: np.ndarray, output_path,
                                spacing: float = UNIFORM_GRID_SPACING,
                                extra_paths: Sequence[Path] = (),
                                skip_existing: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(grid, prediction, "o-", ms=1.8, lw=1.1, color="tab:orange",
            label=rf"Emulator (uniform {spacing} $\mathrm{{\AA}}$ grid)")
    ax.plot(lam, target, "o", ms=3.5, color="tab:blue",
            label="FASTWIND (native grid)")
    ax.set_xlabel(r"wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"{short}, test model {model_id}: equally spaced "
                 rf"{spacing} $\mathrm{{\AA}}$ grid ({grid.size} points)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PROFILE_SINGLE, extra_paths=extra_paths,
                skip_existing=skip_existing)


def save_profile_panels(lines: Dict[str, Dict[str, str]], model_id: int,
                        output_path, uniform: bool,
                        emulator_factory=None, raw_params: Optional[np.ndarray] = None,
                        spacing: float = UNIFORM_GRID_SPACING,
                        extra_paths: Sequence[Path] = (),
                        skip_existing: bool = False) -> None:
    """One panel per diagnostic, for a single test model.

    Source: 13-par_chapter_plots.profiles_all_lines.  ``uniform=False`` draws
    the predictions saved in the npz; ``uniform=True`` re-evaluates each
    checkpoint on a regular grid, which needs torch.
    """
    shorts = sorted(lines)
    ncols = PANEL_NCOLS
    nrows = int(np.ceil(len(shorts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)

    for k, short in enumerate(shorts):
        ax = axes[k // ncols, k % ncols]
        ids, lam, tgt, prd = load_screened(lines[short]["npz"])
        rows = np.flatnonzero(ids == model_id)
        if rows.size == 0:
            ax.set_title(f"{short} (star not in test set)", fontsize=8)
            ax.axis("off")
            continue
        i = int(rows[0])
        ax.plot(lam[i], tgt[i], "o", ms=2, color="tab:blue", label="FASTWIND",
                rasterized=True)
        if uniform:
            emulator = emulator_factory(lines[short]["pth"])
            grid = uniform_grid(lam[i].min(), lam[i].max(), spacing)
            pred = emulator.predict(raw_params, grid[None, :])[0]
            ax.plot(grid, pred, "o-", ms=0.8, lw=0.9, color="tab:orange",
                    label=rf"emulator ({spacing} $\mathrm{{\AA}}$)", rasterized=True)
        else:
            ax.plot(lam[i], prd[i], "o--", ms=1.6, lw=0.9, color="tab:orange",
                    label="emulator (native)", rasterized=True)
        ax.set_title(short, fontsize=8)
        ax.tick_params(labelsize=6)

    for k in range(len(shorts), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    axes[0, 0].legend(fontsize=7)
    kind = (rf"equally spaced {spacing} $\mathrm{{\AA}}$ grid" if uniform
            else "native FASTWIND grid")
    fig.suptitle(f"FASTWIND vs emulator profiles, test model {model_id} ({kind})",
                 y=0.999, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


def save_bloem_panels(lines: Dict[str, Dict[str, str]], model_id: int,
                      output_path, emulator_factory, raw_params: np.ndarray,
                      spacing: float = UNIFORM_GRID_SPACING,
                      lmin: float = BLOEM_LMIN, lmax: float = BLOEM_LMAX,
                      skip_existing: bool = False) -> None:
    """The BLOeM LR02 overlay: only the diagnostics that reach 3960-4570 A.

    Source: 13_plot_after_training.plot_profiles (the ``bloem_rows`` branch) and
    13_fw_emulator_per_line_comparison.ipynb cell 12.
    """
    selected: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for short in sorted(lines):
        ids, lam, tgt, prd = load_screened(lines[short]["npz"])
        rows = np.flatnonzero(ids == model_id)
        if rows.size == 0:
            continue
        i = int(rows[0])
        if np.nanmax(lam[i]) >= lmin and np.nanmin(lam[i]) <= lmax:
            selected.append((short, lam[i], tgt[i], prd[i]))
    if not selected:
        print("[bloem] no diagnostic overlaps the LR02 window for this model")
        return

    ncols = PANEL_NCOLS
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows),
                             squeeze=False)
    for ax, (short, lam, tgt, prd) in zip(axes.flat, selected):
        ax.plot(lam, tgt, "o", ms=2, label="FASTWIND native")
        ax.plot(lam, prd, ".", ms=3, label="Emulator native saved")
        grid = uniform_grid(float(np.nanmin(lam)), float(np.nanmax(lam)), spacing)
        if grid.size >= 2:
            emulator = emulator_factory(lines[short]["pth"])
            pred = emulator.predict(raw_params, grid[None, :])[0]
            ax.plot(grid, pred, ".", ms=2.5,
                    label=f"Emulator BLOeM-like dlam={spacing}")
        ax.axvspan(lmin, lmax, color="0.9", zorder=-10)
        ax.set_xlim(max(np.nanmin(lam), lmin - 20), min(np.nanmax(lam), lmax + 20))
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("lambda [Angstrom]")
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(selected):]:
        fig.delaxes(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"BLOeM-style emulator profiles for test model {model_id}", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.98))
    save_figure(fig, output_path, dpi=200, skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: relative-error
# -------------------------------------------------------------------------------------
# What it shows: the signed relative error 100 (F_FW - F_emu) / (F_FW + 1e-6)
# of every test model at every one of its native wavelength points, pooled and
# then summarised in 161 adaptive equal-count bins of physical wavelength.  The
# bins are quantiles of the pooled wavelength array rather than a uniform grid,
# because the 13-parameter npz files store one wavelength row *per test model*
# and those rows are not identical: a fixed-index summary would label all errors
# with the grid of row 0.  The grey cloud is a reproducible sample of at most
# 500 individual model-wavelength points, drawn with seed 42, so the figure is
# byte-identical between runs.
# Thesis: fig:thirteen_parameter_halpha_relative_error and appendix
#         fig:thirteen_parameter_relative_error_1 ... _last.
# Source: 13-par_chapter_plots.rel_err_vs_wavelength, which restyled the
#         notebook version (13_fw_emulator_per_line_comparison.ipynb cell 13,
#         13_fw_emulator_comparison_with_filtering.ipynb cell 12) to match the
#         five-parameter chapter exactly.
# -------------------------------------------------------------------------------------
def adaptive_wavelength_bins(wavelength: np.ndarray, error: np.ndarray,
                             n_bins: int = RELATIVE_ERROR_BINS) -> Dict[str, np.ndarray]:
    """Equal-count quantile bins of the pooled wavelengths, and their statistics."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(wavelength, quantiles))
    if len(bin_edges) < 2:
        raise SystemExit("cannot construct wavelength bins: all points coincide")
    bin_index = np.searchsorted(bin_edges, wavelength, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(bin_edges) - 2)

    centres, median, mean = [], [], []
    p2p5, p16, p84, p97p5 = [], [], [], []
    for index in range(len(bin_edges) - 1):
        in_bin = bin_index == index
        if not np.any(in_bin):
            continue
        bin_errors = error[in_bin]
        centres.append(float(np.median(wavelength[in_bin])))
        median.append(float(np.median(bin_errors)))
        mean.append(float(np.mean(bin_errors)))
        percentiles = np.percentile(bin_errors, list(RELATIVE_ERROR_PERCENTILES))
        p2p5.append(float(percentiles[0]))
        p16.append(float(percentiles[1]))
        p84.append(float(percentiles[2]))
        p97p5.append(float(percentiles[3]))
    return {"lambda": np.asarray(centres), "median": np.asarray(median),
            "mean": np.asarray(mean), "p2p5": np.asarray(p2p5),
            "p16": np.asarray(p16), "p84": np.asarray(p84),
            "p97p5": np.asarray(p97p5), "n_populated": len(centres),
            "n_edges": len(bin_edges)}


def uniform_wavelength_bins(wavelength: np.ndarray, error: np.ndarray,
                            n_bins: int,
                            plot_percentiles: Tuple[float, float] = (0.5, 99.5)
                            ) -> Dict[str, np.ndarray]:
    """The notebook's uniform bins over the central 99 per cent of wavelengths.

    Source: 13_fw_emulator_per_line_comparison.ipynb cell 13
    (``compute_rel_err_stats_for_line_binned``), kept so the difference to the
    adaptive binning of the thesis figure stays reproducible.
    """
    lo, hi = np.nanpercentile(wavelength, list(plot_percentiles))
    in_range = (wavelength >= lo) & (wavelength <= hi)
    lam_use, err_use = wavelength[in_range], error[in_range]
    edges = np.linspace(lo, hi, n_bins + 1)
    centres_all = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.digitize(lam_use, edges) - 1

    centres, median, mean = [], [], []
    p2p5, p16, p84, p97p5 = [], [], [], []
    for i, centre in enumerate(centres_all):
        values = err_use[bin_id == i]
        if values.size == 0:
            continue
        centres.append(float(centre))
        median.append(float(np.median(values)))
        mean.append(float(np.mean(values)))
        p16.append(float(np.percentile(values, 16.0)))
        p84.append(float(np.percentile(values, 84.0)))
        p2p5.append(float(np.percentile(values, 2.5)))
        p97p5.append(float(np.percentile(values, 97.5)))
    return {"lambda": np.asarray(centres), "median": np.asarray(median),
            "mean": np.asarray(mean), "p2p5": np.asarray(p2p5),
            "p16": np.asarray(p16), "p84": np.asarray(p84),
            "p97p5": np.asarray(p97p5), "n_populated": len(centres),
            "plot_lambda_min": float(lo), "plot_lambda_max": float(hi)}


def save_relative_error_plot(short: str, wavelength: np.ndarray, error: np.ndarray,
                             stats: Dict[str, np.ndarray], output_paths: Sequence[Path],
                             dpis: Sequence[float],
                             max_points: int = RELATIVE_ERROR_SCATTER_POINTS,
                             seed: int = CORNER_SEED,
                             skip_existing: bool = False) -> None:
    rng = np.random.default_rng(seed)
    subset = rng.choice(error.size, size=min(max_points, error.size), replace=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.fill_between(stats["lambda"], stats["p2p5"], stats["p97p5"],
                    color=BAND_COLOUR_2SIGMA, alpha=BAND_ALPHA_2SIGMA,
                    label=r"2.5--97.5 percentile", zorder=1)
    ax.fill_between(stats["lambda"], stats["p16"], stats["p84"],
                    color=BAND_COLOUR_1SIGMA, alpha=BAND_ALPHA_1SIGMA,
                    label=r"16--84 percentile", zorder=2)
    ax.scatter(wavelength[subset], error[subset], s=5, alpha=0.35,
               color=SCATTER_COLOUR, label="Test-model samples", zorder=3)
    ax.plot(stats["lambda"], stats["median"], color="red", lw=2.0, label="Median",
            zorder=4)
    ax.plot(stats["lambda"], stats["mean"], color="darkorange", lw=1.5, ls="--",
            label="Mean", zorder=4)
    ax.axhline(0.0, color="black", lw=0.7)

    # Focused y limits: follow the 2.5-97.5 band, not the outliers.
    if short == RELATIVE_ERROR_CLAMPED_LINE:
        ax.set_ylim(*RELATIVE_ERROR_CLAMP)
    elif stats["p2p5"].size and stats["p97p5"].size:
        low = float(np.min(stats["p2p5"]))
        high = float(np.max(stats["p97p5"]))
        pad = RELATIVE_ERROR_YPAD_FRACTION * max(high - low, 1e-2)
        ax.set_ylim(low - pad, high + pad)

    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"{short} — relative error on the native grid")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    first, *rest = list(output_paths)
    first_dpi, *rest_dpis = list(dpis)
    if skip_existing and Path(first).exists():
        print(f"[skip] {first} exists")
        plt.close(fig)
        return
    for target, dpi in zip([first] + rest, [first_dpi] + rest_dpis):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi)
        print(f"[saved] {target}")
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: per-line-mare
# -------------------------------------------------------------------------------------
# What it shows: the test-set MARE of each of the 37 diagnostics on a
# logarithmic axis, ordered by the median wavelength of the window, with the
# corrupt-row screen applied.  The spread across the set is about three orders
# of magnitude, from He II 6406 at the accurate end to the ultraviolet
# resonance lines at the difficult end.
# Thesis: the MARE figure of the line-by-line results section.
# Source: 13-par_chapter_plots.test_mare_per_line; the MSE companion is
#         13_plot_after_training.plot_mse and
#         13_fw_emulator_per_line_comparison.ipynb cell 6.
# -------------------------------------------------------------------------------------
def save_metric_per_line_plot(labels: Sequence[str], values: Sequence[float],
                              ylabel: str, title: str, output_path,
                              extra_paths: Sequence[Path] = (),
                              skip_existing: bool = False) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, values, "o-", color="tab:blue", ms=5, lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=90, fontsize=8)
    ax.set_xlabel("Line")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: worst-corner
# -------------------------------------------------------------------------------------
# What it shows: a lower-triangle corner plot of the test partition of one
# diagnostic.  Grey points are the remaining test models, red points the worst
# one per cent ranked by model-level MARE, with the cutoff printed in the title.
# The default parameter selection is data-driven: only the parameters whose
# worst-subset distribution differs from the full test partition by a two-sample
# Kolmogorov-Smirnov distance D >= 0.15 are shown, with at least the three
# largest D retained if fewer than three pass.  That is what makes the panels
# readable at 13 parameters and is the selection the thesis describes.
# Thesis: fig:thirteen_parameter_halpha_parameter_error and appendix
#         fig:thirteen_parameter_corner_1 ... _last.
# Source: 13-par_chapter_plots.corner_one_line and .corner_dependent_params;
#         the fixed 13-parameter variant is also
#         13_fw_emulator_per_line_comparison.ipynb cell 9 and
#         13_fw_emulator_comparison_with_filtering.ipynb cell 14.
# -------------------------------------------------------------------------------------
def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance.  Source: _ks_distance."""
    sorted_a, sorted_b = np.sort(a), np.sort(b)
    pooled = np.concatenate([sorted_a, sorted_b])
    return float(np.max(np.abs(
        np.searchsorted(sorted_a, pooled, side="right") / sorted_a.size
        - np.searchsorted(sorted_b, pooled, side="right") / sorted_b.size)))


def select_dependent_parameters(values: np.ndarray, worst_index: np.ndarray,
                                params: Sequence[Tuple[str, str, bool]],
                                threshold: float = KS_THRESHOLD,
                                min_params: int = KS_MIN_PARAMS
                                ) -> Tuple[List[int], np.ndarray, List[str]]:
    """Indices of the parameters the worst subset actually depends on."""
    distances = np.array([ks_distance(values[:, k], values[worst_index, k])
                          for k in range(len(params))])
    signs = ["+" if np.median(values[worst_index, k]) > np.median(values[:, k])
             else "-" for k in range(len(params))]
    selected = [k for k in range(len(params)) if distances[k] >= threshold]
    if len(selected) < min_params:
        selected = list(np.argsort(distances)[::-1][:min_params])
    selected.sort(key=lambda k: -distances[k])
    return selected, distances, signs


def save_corner_plot(short: str, values: np.ndarray, worst_index: np.ndarray,
                     cutoff_percent: float, n_total: int,
                     params: Sequence[Tuple[str, str, bool]], output_path,
                     worst_fraction: float = WORST_FRACTION,
                     background_max: int = CORNER_BACKGROUND_MAX,
                     seed: int = CORNER_SEED, panel: float = 2.2,
                     dpi: float = DPI_CORNER_DEPENDENT,
                     extra_paths: Sequence[Path] = (),
                     skip_existing: bool = False) -> None:
    rng = np.random.default_rng(seed)
    background = (rng.choice(n_total, size=background_max, replace=False)
                  if n_total > background_max else np.arange(n_total))

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(panel * n, panel * n), squeeze=False)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(values[:, col], bins=CORNER_HIST_BINS_ALL, color="0.75",
                        density=True, label="all test models")
                ax.hist(values[worst_index, col], bins=CORNER_HIST_BINS_WORST,
                        color="tab:red", alpha=0.65, density=True,
                        label="worst-error models")
                ax.set_yticks([])
                if row == 0:
                    ax.legend(fontsize=max(5, int(3 * panel)), loc="upper left")
            else:
                ax.scatter(values[background, col], values[background, row], s=3,
                           color="0.75", alpha=0.4, rasterized=True)
                ax.scatter(values[worst_index, col], values[worst_index, row], s=10,
                           color="tab:red", alpha=0.85, rasterized=True)
            if row == n - 1:
                ax.set_xlabel(params[col][1], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(params[row][1], fontsize=9)
            elif col > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(f"{short}: 13-parameter space\n"
                 f"Worst {100 * worst_fraction:.1f}% by mean relative flux error "
                 f"({worst_index.size} of {n_total}), "
                 f"cutoff = {cutoff_percent:.2f}%",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi, extra_paths=extra_paths,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: own-test
# -------------------------------------------------------------------------------------
# What it shows: the Latin-hypercube and the Sobol ensemble side by side, each
# on its own test partition and on nothing else.  The upper panel is the MARE as
# a relative fraction, so 1e-2 is one per cent; the lower panel is the minimum
# weighted mean-squared validation loss reached during the corresponding
# training run, read from that run's own meta file.  Both axes are logarithmic
# and the horizontal axis lists the 37 diagnostics.
# The two bars of a diagnostic are independent experiments on different model
# populations and different FASTWIND formal-solution configurations, so they
# describe own-dataset performance; they are not a controlled comparison of the
# two sampling designs, and no merged case exists.
# Thesis: fig:thirteen_parameter_own_test_by_line.
# Source: plot_lhc_sobol_merged_comparison.collect and .bar_figure, with the
#         merged case removed; the per-case metric definitions and the local
#         filter come from plot_case_perfomance.
# -------------------------------------------------------------------------------------
def save_own_test_bar_figure(shorts: Sequence[str],
                             data: Dict[str, Dict[str, Dict[str, float]]],
                             key: str, ylabel: str, title: str, output_path,
                             order: Sequence[str] = OWN_TEST_ORDER,
                             extra_paths: Sequence[Path] = (),
                             skip_existing: bool = False) -> None:
    x = np.arange(len(shorts))
    width = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for k, case in enumerate(order):
        values = [data[case].get(s, {}).get(key, np.nan) for s in shorts]
        ax.bar(x + (k - (len(order) - 1) / 2) * width, values, width,
               label=CASES[case]["label"], color=OWN_TEST_COLOURS[case], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(list(shorts), rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(each ensemble evaluated on its OWN test partition; "
                 "the two bars are independent experiments, not a matched "
                 "comparison of sampling designs)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


def save_own_test_combined_figure(shorts: Sequence[str],
                                  data: Dict[str, Dict[str, Dict[str, float]]],
                                  output_path,
                                  order: Sequence[str] = OWN_TEST_ORDER,
                                  extra_paths: Sequence[Path] = (),
                                  skip_existing: bool = False) -> None:
    """The printed two-panel version: MARE above, best validation loss below.

    The bar geometry, the colours, the log axes and the 90-degree tick labels at
    font size 7 are those of ``plot_lhc_sobol_merged_comparison.bar_figure``.
    The only quantity that could not be read off a source is the overall figure
    height: 11.0 is two stacked copies of that function's 16 x 5.5 panel.
    """
    x = np.arange(len(shorts))
    width = 0.8 / len(order)
    fig, (ax_mare, ax_loss) = plt.subplots(2, 1, figsize=(16, 11.0), sharex=True)

    for k, case in enumerate(order):
        offset = (k - (len(order) - 1) / 2) * width
        mare = [data[case].get(s, {}).get("mare", np.nan) for s in shorts]
        loss = [data[case].get(s, {}).get("best_val_loss", np.nan) for s in shorts]
        ax_mare.bar(x + offset, mare, width, label=CASES[case]["label"],
                    color=OWN_TEST_COLOURS[case], alpha=0.85)
        ax_loss.bar(x + offset, loss, width, label=CASES[case]["label"],
                    color=OWN_TEST_COLOURS[case], alpha=0.85)

    ax_mare.set_yscale("log")
    ax_mare.set_ylabel("test MARE (relative)")
    ax_mare.legend()
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("best validation loss")
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(list(shorts), rotation=90, fontsize=7)
    ax_loss.set_xlabel("Line")
    fig.suptitle("Per-diagnostic performance of the independently trained "
                 "ensembles, each on its own test partition", y=0.995)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


# -------------------------------------------------------------------------------------
# FIGURE: grid-consistency
# -------------------------------------------------------------------------------------
# What it shows: whether the two datasets describe the same diagnostic at all,
# from the raw FASTWIND output and with no emulator involved.  Left panel: the
# median of the integral of (1 - F) dlambda over the wavelength interval common
# to both datasets, formed inside twelve equal bins of effective temperature and
# with a bin discarded when either dataset supplies fewer than 30 models to it.
# Middle panel: the distribution of that integral, as unfilled step histograms
# over the central 99 per cent of the pooled values.  Right panel: up to 25
# individual profiles from each dataset drawn from a rectangular tolerance box
# around the reference parameters, on the crop window the training pipeline uses.
# Each panel uses a random subsample of 15,000 models per dataset, and no model
# of one dataset is ever paired with a model of the other.
# Thesis: fig:thirteen_parameter_grid_consistency and the appendix set.
# Source: compare_line_between_grids.analyse_line.
# -------------------------------------------------------------------------------------
def read_dataset_line(h5_path, short: str, max_models: int,
                      seed: int = GRID_CONSISTENCY_SEED):
    """``(ids, lambda, flux)`` for one diagnostic, optionally subsampled.

    Source: compare_line_between_grids.read_line.
    """
    h5py = h5py_module()
    with h5py.File(str(h5_path), "r") as handle:
        groups = {short_name(k): k for k in handle["out_lines"].keys()}
        group_name = groups.get(short)
        if group_name is None:
            return None
        group = handle["out_lines"][group_name]
        ids = group["model_ids"][:].astype(np.int64)
        n = ids.size
        if max_models and n > max_models:
            rng = np.random.default_rng(seed)
            selection = np.sort(rng.choice(n, size=max_models, replace=False))
        else:
            selection = np.arange(n)
        lam = group["wavelength"][:][selection].astype(np.float64)
        flux = group["flux"][:][selection].astype(np.float64)
        ids = ids[selection]
    return ids, lam, flux


def screen_corrupt(ids: np.ndarray, lam: np.ndarray, flux: np.ndarray):
    """Source: compare_line_between_grids.screen_corrupt."""
    bad = ((np.abs(flux).max(axis=1) < CORRUPT_ZERO_TOLERANCE)
           | (flux.min(axis=1) < CORRUPT_NEGATIVE_TOLERANCE)
           | ~np.isfinite(flux).all(axis=1))
    return ids[~bad], lam[~bad], flux[~bad], int(bad.sum())


def shape_measures(lam: np.ndarray, flux: np.ndarray, lo: float, hi: float):
    """The EW-like integral of (1 - F) and the extreme fluxes inside [lo, hi].

    Source: compare_line_between_grids.shape_measures, including its five-point
    minimum and its ``np.trapezoid`` / ``np.trapz`` fallback.
    """
    ew = np.full(lam.shape[0], np.nan)
    fmin = np.full(lam.shape[0], np.nan)
    fmax = np.full(lam.shape[0], np.nan)
    for i in range(lam.shape[0]):
        w, y = lam[i], flux[i]
        mask = (w >= lo) & (w <= hi) & np.isfinite(w) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        order = np.argsort(w[mask])
        ww, yy = w[mask][order], y[mask][order]
        ew[i] = (float(np.trapezoid(1.0 - yy, ww)) if hasattr(np, "trapezoid")
                 else float(np.trapz(1.0 - yy, ww)))
        fmin[i] = float(yy.min())
        fmax[i] = float(yy.max())
    return ew, fmin, fmax


def save_grid_consistency_figure(short: str, driver_1: str, driver_2: str,
                                 centres: np.ndarray, median_a: np.ndarray,
                                 median_b: np.ndarray, ew_a: np.ndarray,
                                 ew_b: np.ndarray, lam_a: np.ndarray,
                                 flux_a: np.ndarray, lam_b: np.ndarray,
                                 flux_b: np.ndarray, index_a: np.ndarray,
                                 index_b: np.ndarray, centre_1: float,
                                 centre_2: float, window: Tuple[float, float],
                                 output_path, extra_paths: Sequence[Path] = (),
                                 skip_existing: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(centres, median_a, "o-", color="tab:blue", label="LHC")
    ax.plot(centres, median_b, "s--", color="tab:orange", label="Sobol v2")
    ax.set_xlabel(driver_1)
    ax.set_ylabel(r"median $\int (1-F)\,d\lambda$  [$\AA$]")
    ax.set_title(f"{short}: line strength at fixed {driver_1}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    pooled = np.concatenate([ew_a, ew_b])
    bins = np.linspace(np.nanpercentile(pooled, GRID_CONSISTENCY_HIST_PERCENTILES[0]),
                       np.nanpercentile(pooled, GRID_CONSISTENCY_HIST_PERCENTILES[1]),
                       GRID_CONSISTENCY_HIST_BINS)
    ax.hist(ew_a[np.isfinite(ew_a)], bins=bins, histtype="step", color="tab:blue",
            label="LHC", density=True)
    ax.hist(ew_b[np.isfinite(ew_b)], bins=bins, histtype="step", color="tab:orange",
            label="Sobol v2", density=True)
    ax.set_xlabel(r"$\int (1-F)\,d\lambda$  [$\AA$]")
    ax.set_ylabel("density")
    ax.set_title("distribution of line strength")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    for i in index_a:
        order = np.argsort(lam_a[i])
        ax.plot(lam_a[i][order], flux_a[i][order], color="tab:blue", lw=0.6, alpha=0.5)
    for i in index_b:
        order = np.argsort(lam_b[i])
        ax.plot(lam_b[i][order], flux_b[i][order], color="tab:orange", lw=0.6,
                alpha=0.5)
    ax.plot([], [], color="tab:blue", label=f"LHC ({index_a.size})")
    ax.plot([], [], color="tab:orange", label=f"Sobol v2 ({index_b.size})")
    ax.set_xlim(*window)
    ax.set_xlabel(r"wavelength [$\AA$]")
    ax.set_ylabel("normalised flux")
    ax.set_title(f"profiles near {driver_1}={centre_1:.4g}, {driver_2}={centre_2:.4g}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_path, dpi=DPI_PANEL, extra_paths=extra_paths,
                skip_existing=skip_existing)


# =====================================================================================
# SECTION 4.  FIGURE DRIVERS, ONE PER FAMILY
# =====================================================================================

def output_base(args) -> Path:
    """The plot tree of the adopted campaign.  Source: 13-par_chapter_plots.OUT_BASE."""
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.base_dir) / "13_line_emulator_plots" / DEEP_FILTERED_TAG


@lru_cache(maxsize=None)
def _cached_lines(runs_root: str, run_glob: str):
    return discover_lines(runs_root, run_glob)


def adopted_lines(args) -> Dict[str, Dict[str, str]]:
    lines = _cached_lines(str(args.runs_root), args.run_glob)
    if args.lines and args.lines != ["all"]:
        missing = [s for s in args.lines if s not in lines]
        if missing:
            raise SystemExit(f"lines not found: {missing}\n"
                             f"available: {sorted(lines)}")
        return {s: lines[s] for s in args.lines}
    return lines


def choose_star(lines: Dict[str, Dict[str, str]], profile_line: str,
                which: str = "median", model_id: Optional[int] = None) -> int:
    """Pick one test model, preferring models shared by every diagnostic.

    Source: 13-par_chapter_plots.choose_star.
    """
    ids, _, tgt, prd = load_screened(lines[profile_line]["npz"])
    star_mare = model_level_mare(tgt, prd)

    if model_id is not None:
        if not np.flatnonzero(ids == model_id).size:
            raise SystemExit(f"model id {model_id} not in the {profile_line} test set")
        return int(model_id)

    shared = set(shared_test_ids(lines))
    mask = (np.array([int(m) in shared for m in ids]) if shared
            else np.ones(ids.size, dtype=bool))
    candidates = np.flatnonzero(mask)
    order = candidates[np.argsort(star_mare[candidates])]
    if which == "worst":
        i = int(order[-1])
    elif which == "best":
        i = int(order[0])
    else:
        i = int(order[order.size // 2])
    chosen = int(ids[i])
    print(f"[star] model {chosen} ({profile_line} per-star MARE {star_mare[i]:.3e}, "
          f"selection={which}, shared across all lines={bool(shared)})")
    return chosen


def figure_loss_curves(args) -> None:
    lines = adopted_lines(args)
    out_dir = output_base(args) / "loss_curves"
    if args.layout in ("grid", "both"):
        save_loss_grid(lines,
                       output_base(args) / "train_val_loss_curves_all_lines.png",
                       skip_existing=args.skip_existing)
    if args.layout in ("single", "both"):
        for short in sorted(lines):
            try:
                loss = load_json(lines[short]["loss"])
            except (OSError, json.JSONDecodeError):
                print(f"[WARN] {short}: no readable loss json, skipping")
                continue
            name = f"train_val_loss_{short}.{args.format}"
            save_loss_plot(short, loss, out_dir / name, args.format,
                           extra_paths=thesis_target(
                               args, "deep_diagnostics/loss_curves", name),
                           mark_best_epoch=args.mark_best_epoch,
                           skip_existing=args.skip_existing)


def figure_profiles(args) -> None:
    lines = adopted_lines(args)
    profile_line = args.profile_line
    if profile_line not in lines and profile_line == "HALPHA":
        profile_line = "HALPHAHEII6527"
    if profile_line not in lines:
        raise SystemExit(f"line {args.profile_line!r} not found; "
                         f"available: {sorted(lines)}")

    model_id = choose_star(lines, profile_line, which=args.which,
                           model_id=args.model_id)

    ids, lam, tgt, prd = load_screened(lines[profile_line]["npz"])
    row = int(np.flatnonzero(ids == model_id)[0])
    lam_i, tgt_i, prd_i = lam[row], tgt[row], prd[row]

    tag = profile_line.lower()
    single_dir = output_base(args) / f"{tag}_profiles"
    native_name = f"{tag}_profile_fastwind_grid_model{model_id}.{args.format}"
    save_single_profile_native(
        profile_line, model_id, lam_i, tgt_i, prd_i, single_dir / native_name,
        extra_paths=thesis_target(args, "deep_diagnostics/profiles", native_name),
        skip_existing=args.skip_existing)

    panel_native = f"profiles_fastwind_grid_model_{model_id}.{args.format}"
    save_profile_panels(
        lines, model_id, output_base(args) / panel_native, uniform=False,
        extra_paths=thesis_target(args, "deep_diagnostics/profiles", panel_native),
        skip_existing=args.skip_existing)

    if args.skip_uniform:
        print("[profiles] --skip-uniform: the regular-grid panels were not drawn")
        return

    try:
        device = resolve_device(args.device)
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[WARN] cannot evaluate on the uniform grid (torch missing?): {exc}")
        return

    emulators: Dict[str, LineEmulator] = {}

    def emulator_factory(pth_path: str) -> LineEmulator:
        if pth_path not in emulators:
            emulators[pth_path] = LineEmulator(pth_path, device=device,
                                               batch_size=args.batch_size)
        return emulators[pth_path]

    raw_params = lhc_raw_parameter_rows(args.grid_json, [model_id])
    grid = uniform_grid(lam_i.min(), lam_i.max(), args.uniform_spacing)
    prediction = emulator_factory(lines[profile_line]["pth"]).predict(
        raw_params, grid[None, :])[0]
    print(f"[profiles] uniform grid: {grid.size} points at "
          f"{args.uniform_spacing} A")

    spacing_tag = str(args.uniform_spacing).replace(".", "p")
    uniform_name = f"{tag}_profile_uniform{spacing_tag}_model{model_id}.{args.format}"
    save_single_profile_uniform(
        profile_line, model_id, lam_i, tgt_i, grid, prediction,
        single_dir / uniform_name, spacing=args.uniform_spacing,
        extra_paths=thesis_target(args, "deep_diagnostics/profiles", uniform_name),
        skip_existing=args.skip_existing)

    panel_uniform = (f"profiles_uniform{spacing_tag}_grid_model_{model_id}"
                     f".{args.format}")
    save_profile_panels(
        lines, model_id, output_base(args) / panel_uniform, uniform=True,
        emulator_factory=emulator_factory, raw_params=raw_params,
        spacing=args.uniform_spacing,
        extra_paths=thesis_target(args, "deep_diagnostics/profiles", panel_uniform),
        skip_existing=args.skip_existing)

    if args.bloem_panels:
        save_bloem_panels(
            lines, model_id,
            output_base(args) / f"profiles_bloem_grid_model_{model_id}.{args.format}",
            emulator_factory, raw_params, spacing=args.uniform_spacing,
            lmin=args.bloem_lmin, lmax=args.bloem_lmax,
            skip_existing=args.skip_existing)


def figure_relative_error(args) -> None:
    lines = adopted_lines(args)
    out_dir = output_base(args) / "relative_error_vs_wavelength"
    for short in sorted(lines):
        ids, lam, tgt, prd = load_screened(lines[short]["npz"])
        # signed, in per cent, exactly as the five-parameter chapter defines it
        relative_error = 100.0 * (tgt - prd) / (tgt + args.relative_error_eps)

        flat_wavelength = lam.ravel()
        flat_error = relative_error.ravel()
        finite = np.isfinite(flat_wavelength) & np.isfinite(flat_error)
        flat_wavelength = flat_wavelength[finite]
        flat_error = flat_error[finite]

        if args.binning == "adaptive":
            stats = adaptive_wavelength_bins(flat_wavelength, flat_error,
                                             args.relative_error_bins)
        else:
            stats = uniform_wavelength_bins(flat_wavelength, flat_error,
                                            args.relative_error_bins)
        print(f"[{short}] {stats['n_populated']} populated bins of "
              f"{args.relative_error_bins}, {ids.size} screened test models")

        base = f"relative_error_vs_wavelength_{lines[short]['line']}"
        targets: List[Path] = [out_dir / f"{base}.pdf", out_dir / f"{base}.png"]
        dpis: List[float] = [DPI_RELATIVE_ERROR_PDF, DPI_RELATIVE_ERROR_PNG]
        for suffix, dpi in (("pdf", DPI_RELATIVE_ERROR_PDF),
                            ("png", DPI_RELATIVE_ERROR_PNG)):
            for extra in thesis_target(
                    args, "deep_diagnostics/relative_error_vs_wavelength",
                    f"rel_{short}.{suffix}"):
                targets.append(extra)
                dpis.append(dpi)
        save_relative_error_plot(short, flat_wavelength, flat_error, stats,
                                 targets, dpis,
                                 max_points=args.scatter_points, seed=args.seed,
                                 skip_existing=args.skip_existing)


def figure_per_line_mare(args) -> None:
    lines = adopted_lines(args)
    rows = []
    for short in sorted(lines):
        ids, lam, tgt, prd = load_screened(lines[short]["npz"])
        error = prd - tgt
        rows.append((float(np.median(lam)), lines[short]["line"], short,
                     float(np.mean(np.abs(error) / (np.abs(tgt) + EPS))),
                     float(np.mean(error ** 2)), int(ids.size)))
    rows.sort()  # by central wavelength, as the notebook version did

    labels = [r[1] for r in rows]
    mare = [r[3] for r in rows]
    mse = [r[4] for r in rows]

    name = f"test_mare_per_line.{args.format}"
    save_metric_per_line_plot(
        labels, mare, "Mean relative absolute error",
        "Test-set MARE per line - 13-parameter emulator",
        output_base(args) / name,
        extra_paths=thesis_target(args, "deep_diagnostics", name),
        skip_existing=args.skip_existing)
    save_metric_per_line_plot(
        labels, mse, "MSE in flux",
        "Test-set MSE per line - 13-parameter emulator",
        output_base(args) / f"test_mse_per_line.{args.format}",
        skip_existing=args.skip_existing)
    save_rows([[r[2], r[1], r[0], r[3], r[4], r[5]] for r in rows],
              ["line", "line_file", "median_lambda", "test_mare", "test_mse",
               "n_test_screened"],
              output_base(args) / "test_metrics_per_line.csv")
    print(f"[per-line-mare] {len(rows)} diagnostics; median MARE "
          f"{100.0 * float(np.median(mare)):.2f}%")


def figure_worst_corner(args) -> None:
    lines = adopted_lines(args)
    if args.corner_set == "5par":
        params: Sequence[Tuple[str, str, bool]] = CORNER_PARAMS_5
        out_dir = output_base(args) / "corner_per_line"
        suffix, panel, dpi, fmt = "5par", 2.2, DPI_CORNER_DEPENDENT, args.format
        thesis_subdir = None
    elif args.corner_set == "13par":
        params = CORNER_PARAMS_13
        out_dir = output_base(args) / "corner_per_line_13par"
        suffix, panel, dpi, fmt = "13par", 1.6, DPI_CORNER_13PAR, "png"
        thesis_subdir = None
    else:
        params = CORNER_PARAMS_13
        out_dir = output_base(args) / "corner_per_line_dependent"
        suffix, panel, dpi, fmt = "dep", 2.2, DPI_CORNER_DEPENDENT, args.format
        thesis_subdir = "deep_diagnostics/corner_per_line_dependent"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines: List[str] = []

    for short in sorted(lines):
        ids, lam, tgt, prd = load_screened(lines[short]["npz"])
        star_mare = model_level_mare(tgt, prd)
        n_worst = max(1, int(round(args.worst_fraction * ids.size)))
        worst_index = np.argsort(star_mare)[::-1][:n_worst]
        cutoff_percent = 100.0 * star_mare[worst_index[-1]]

        values = apply_corner_logs(lhc_params(args.grid_json, ids, params), params)

        if args.corner_set == "dependent":
            selected, distances, signs = select_dependent_parameters(
                values, worst_index, params, args.ks_threshold, args.min_params)
            description = ", ".join(
                f"{params[k][0]}({signs[k]}{distances[k]:.2f})" for k in selected)
            report_lines.append(f"{short}: {description}")
            print(f"[dep] {short}: {description}")
            plot_params = [params[k] for k in selected]
            plot_values = values[:, selected]
        else:
            plot_params = list(params)
            plot_values = values

        name = f"corner_worst1pct_{suffix}_{short}.{fmt}"
        extra = thesis_target(args, thesis_subdir, name) if thesis_subdir else []
        save_corner_plot(short, plot_values, worst_index, cutoff_percent,
                         int(ids.size), plot_params, out_dir / name,
                         worst_fraction=args.worst_fraction,
                         background_max=args.corner_background_max,
                         seed=args.seed, panel=panel, dpi=dpi,
                         extra_paths=extra, skip_existing=args.skip_existing)

    if report_lines:
        report_path = out_dir / "dependent_params_report.txt"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"[saved] {report_path}")


# -------------------------------------------------------------------------------------
# FIGURE: convergence
# -------------------------------------------------------------------------------------
# What it shows: where in the 13-dimensional Latin-hypercube design the FASTWIND
# calculations failed to converge.  For each parameter the sampled coordinate is
# divided into ten equal-width bins over the design limits and the percentage of
# models in that bin whose quality flag is not "green" is drawn as red points,
# with the overall non-green fraction as a dashed black reference line.  Mass
# loss is binned after log10, as it was sampled and as it is supplied to the
# emulator.  The overall fraction is computed here, not assumed; the thesis
# quotes 37.408 per cent and the driver prints what it found.
# Thesis: fig:thirteen_parameter_convergence_distribution.
# Source: plot_13par_convergence_parameter_space.py, standard library only.  The
#         figure is a standalone PGFPlots document compiled with pdflatex; this
#         is reproduced verbatim rather than redrawn in matplotlib.
# -------------------------------------------------------------------------------------
def iter_grid_entries(path: Path):
    """Yield ``(model_id, parameter_dict)`` from the pretty-printed grid JSON.

    Streaming, because the file is far too large to hold twice.
    Source: plot_13par_convergence_parameter_space.iter_grid_entries, including
    the exact six-space indentation the writer produced.
    """
    start_re = re.compile(r'^\s{6}"(\d+)":\s*\{\s*$')
    end_re = re.compile(r"^\s{6}\}[,]?\s*$")
    model_id = None
    buffer: List[str] = []

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if model_id is None:
                match = start_re.match(line)
                if match:
                    model_id = match.group(1)
                    buffer = ["{\n"]
            elif end_re.match(line):
                buffer.append("}\n")
                yield model_id, json.loads("".join(buffer))
                model_id = None
                buffer = []
            else:
                buffer.append(line)


def convergence_sampled_value(name: str, model: dict) -> float:
    value = float(model[name])
    return math.log10(value) if name == "mdot" else value


def figure_convergence(args) -> None:
    from collections import Counter

    out_dir = Path(args.convergence_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = load_json(args.quality_json)

    all_counts = {name: [0] * CONVERGENCE_BINS for name, *_ in CONVERGENCE_PARAMETERS}
    rejected_counts = {name: [0] * CONVERGENCE_BINS
                       for name, *_ in CONVERGENCE_PARAMETERS}
    flag_counts: "Counter[str]" = Counter()

    for model_id, model in iter_grid_entries(args.grid_json):
        flag = quality[model_id]["flag"]
        flag_counts[flag] += 1
        rejected = flag != "green"
        for name, lower, upper, _scale, _xlabel, _title in CONVERGENCE_PARAMETERS:
            value = convergence_sampled_value(name, model)
            index = int((value - lower) / (upper - lower) * CONVERGENCE_BINS)
            index = min(CONVERGENCE_BINS - 1, max(0, index))
            all_counts[name][index] += 1
            if rejected:
                rejected_counts[name][index] += 1

    total = sum(flag_counts.values())
    total_rejected = total - flag_counts["green"]
    overall_percent = 100.0 * total_rejected / total

    summary = {"models": total, "flags": dict(flag_counts),
               "non_green": total_rejected,
               "overall_non_green_percent": overall_percent, "bins": {}}

    for name, lower, upper, scale, _xlabel, _title in CONVERGENCE_PARAMETERS:
        data_path = out_dir / f"rejection_fraction_{name}.dat"
        width = (upper - lower) / CONVERGENCE_BINS
        rows = []
        with data_path.open("w", encoding="utf-8") as handle:
            handle.write("x rejected_percent all_count rejected_count\n")
            for index, (all_n, rejected_n) in enumerate(
                    zip(all_counts[name], rejected_counts[name])):
                centre = (lower + (index + 0.5) * width) * scale
                percent = 100.0 * rejected_n / all_n
                handle.write(f"{centre:.9g} {percent:.9g} {all_n} {rejected_n}\n")
                rows.append({"centre": centre, "non_green_percent": percent,
                             "all": all_n, "non_green": rejected_n})
        summary["bins"][name] = rows
        print(f"[saved] {data_path}")

    (out_dir / "convergence_parameter_space_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_lines = [
        r"\documentclass[tikz,border=3pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usetikzlibrary{calc}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 5, horizontal sep=1.25cm, "
        r"vertical sep=1.65cm},",
        r"  width=5.7cm, height=3.65cm,",
        r"  ymin=0, ymax=75,",
        r"  ytick={0,25,50,75},",
        r"  tick label style={font=\small},",
        r"  label style={font=\small},",
        r"  title style={font=\small, yshift=-1mm},",
        r"  grid=major, grid style={draw=gray!18},",
        r"  axis line style={black!70}, tick style={black!70},",
        r"  clip=false",
        r"]",
    ]
    for name, lower, upper, scale, xlabel, title in CONVERGENCE_PARAMETERS:
        xmin, xmax = lower * scale, upper * scale
        table = (out_dir / f"rejection_fraction_{name}.dat").as_posix()
        plot_lines.extend([
            rf"\nextgroupplot[title={{{title}}}, xlabel={{{xlabel}}}, "
            rf"xmin={xmin:g}, xmax={xmax:g}]",
            rf"\addplot+[very thick, color=red!75!black, mark=*, mark size=1.8pt] "
            rf"table[x=x,y=rejected_percent] {{{table}}};",
            rf"\addplot[densely dashed, black!65, thick] coordinates "
            rf"{{({xmin:g},{overall_percent:.8g}) ({xmax:g},{overall_percent:.8g})}};",
        ])
    plot_lines.extend([
        r"\end{groupplot}",
        r"\node[rotate=90, anchor=south, font=\small] at "
        r"([xshift=-1.15cm]group c1r3.west) "
        r"{Calculations not labelled green within bin (\%)};",
        r"\end{tikzpicture}",
        r"\end{document}",
    ])

    tex_path = out_dir / "thirteen_parameter_convergence_rejection_fraction.tex"
    tex_path.write_text("\n".join(plot_lines) + "\n", encoding="utf-8")
    print(f"[saved] {tex_path}")

    if args.no_pdflatex:
        print("[convergence] --no-pdflatex: the .tex was written but not compiled")
    else:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        tex_path.name], cwd=str(out_dir), check=True)
        pdf_path = out_dir / "thirteen_parameter_convergence_rejection_fraction.pdf"
        print(f"[saved] {pdf_path}")
        for extra in thesis_target(args, ".", pdf_path.name):
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(pdf_path.read_bytes())
            print(f"[saved] {extra}")

    print(json.dumps(summary["flags"], indent=2))
    print(f"Overall non-green fraction: {overall_percent:.3f}% "
          f"(the thesis quotes {CONVERGENCE_EXPECTED_NON_GREEN_PERCENT:.3f}%)")


def figure_own_test(args) -> None:
    """Each ensemble on its own test partition.  No merged case, no cross-eval."""
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    rows: List[List[object]] = []

    for case in OWN_TEST_ORDER:
        lines = discover_case(args, case)
        if not lines:
            raise SystemExit(f"no runs found for case {case!r} under "
                             f"{case_paths(args, case)['glob']}")
        if len(lines) != 37:
            print(f"[WARN] {case}: expected 37 diagnostics, found {len(lines)}")
        vinf_map = (vinf_map_for_case(args, case) if args.local_filter else None)
        collected: Dict[str, Dict[str, float]] = {}
        for short in sorted(lines):
            result = load_case_filtered(lines[short], vinf_map,
                                        apply_filter=args.local_filter)
            error = result["prd"] - result["tgt"]
            relative = np.abs(error) / (np.abs(result["tgt"]) + EPS_SIGNED)
            meta = load_json(lines[short]["meta"])
            collected[short] = {
                "mse": float(np.mean(error ** 2)),
                "mare": float(np.mean(relative)),
                "best_val_loss": float(meta["best_val_loss"]),
                "n_test_kept": int(result["ids"].shape[0]),
                "skipped_by_local_filter": int(result["skipped"]),
            }
            print(f"[{case}] {short}: mare={collected[short]['mare']:.3e} "
                  f"val_loss={collected[short]['best_val_loss']:.3e} "
                  f"kept={collected[short]['n_test_kept']} "
                  f"skipped={collected[short]['skipped_by_local_filter']}")
        data[case] = collected

    shorts = sorted(set().union(*[set(d) for d in data.values()]))
    for case, collected in data.items():
        missing = [s for s in shorts if s not in collected]
        if missing:
            print(f"[WARN] {case} is missing diagnostics: {missing}")

    out_dir = Path(args.own_test_dir or (Path(args.base_dir) / "comparison_plots"))
    save_own_test_combined_figure(
        shorts, data, out_dir / f"own_test_by_line.{args.format}",
        extra_paths=thesis_target(args, "sampling_comparison",
                                  f"own_test_by_line.{args.format}"),
        skip_existing=args.skip_existing)
    save_own_test_bar_figure(
        shorts, data, "mare", "test MARE (relative)",
        "Test MARE per line: each ensemble on its own test partition",
        out_dir / f"own_test_mare_by_line.{args.format}",
        extra_paths=thesis_target(args, "sampling_comparison",
                                  f"own_test_mare_by_line.{args.format}"),
        skip_existing=args.skip_existing)
    save_own_test_bar_figure(
        shorts, data, "best_val_loss", "best validation loss",
        "Best validation loss per line: each ensemble on its own validation "
        "partition",
        out_dir / f"own_test_val_loss_by_line.{args.format}",
        extra_paths=thesis_target(args, "sampling_comparison",
                                  f"own_test_val_loss_by_line.{args.format}"),
        skip_existing=args.skip_existing)

    keys = ("mse", "mare", "best_val_loss", "n_test_kept",
            "skipped_by_local_filter")
    for short in shorts:
        row: List[object] = [short]
        for case in OWN_TEST_ORDER:
            entry = data[case].get(short, {})
            row += [entry.get(k) for k in keys]
        rows.append(row)
    save_rows(rows,
              ["line"] + [f"{c}_{k}" for c in OWN_TEST_ORDER for k in keys],
              out_dir / "own_test_metrics.csv")

    for case in OWN_TEST_ORDER:
        values = [data[case][s]["mare"] for s in shorts if s in data[case]]
        print(f"[own-test] {CASES[case]['label']}: median MARE over "
              f"{len(values)} diagnostics = {100.0 * float(np.median(values)):.2f}%")


def _load_sobol_parameter_map(args) -> Dict[int, np.ndarray]:
    """``model_id -> (13,)`` raw Sobol parameters in ``PARAM_COLS`` order."""
    h5py = h5py_module()
    with h5py.File(str(args.sobol_models_h5), "r") as handle:
        ids = handle["model_ids"][:].astype(np.int64)
        cols = [c.decode() if isinstance(c, bytes) else str(c)
                for c in handle["param_cols"][:]]
        parameters = handle["parameters"][:].astype(np.float64)
    order = [cols.index(c) for c in PARAM_COLS]
    return {int(m): parameters[i, order] for i, m in enumerate(ids)}


def _load_lhc_parameter_map(args, quality_json: Optional[str] = None
                            ) -> Dict[int, np.ndarray]:
    """``model_id -> (13,)`` raw Latin-hypercube parameters.

    ``quality_json`` restricts the map to the green models, which is what
    compare_lhc_sobol_datasets.load_lhc_params does; compare_line_between_grids
    keeps every model with all 13 columns present.
    """
    grid = lhc_grid(args.grid_json)
    green: Optional[set] = None
    if quality_json and os.path.exists(quality_json):
        quality = load_json(quality_json)
        green = {k for k, v in quality.items() if v.get("flag") == "green"}
        print(f"LHC: restricting to {len(green)} GREEN models "
              "(the population actually trained on)")
    out: Dict[int, np.ndarray] = {}
    for key, record in grid.items():
        if green is not None and key not in green:
            continue
        if not all(c in record for c in PARAM_COLS):
            continue
        out[int(key)] = np.asarray([float(record[c]) for c in PARAM_COLS],
                                   dtype=np.float64)
    return out


def figure_grid_consistency(args) -> None:
    h5py = h5py_module()
    out_dir = Path(args.grid_consistency_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if list(args.grid_consistency_lines) == ["all"]:
        with h5py.File(str(args.sobol_out_lines), "r") as handle:
            wanted = sorted(short_name(k) for k in handle["out_lines"].keys())
    else:
        wanted = list(args.grid_consistency_lines)

    sobol_map = _load_sobol_parameter_map(args)
    lhc_map = _load_lhc_parameter_map(args)
    print(f"[params] LHC {len(lhc_map)} models, Sobol {len(sobol_map)} models")

    report: List[dict] = []
    for short in wanted:
        a = read_dataset_line(args.lhc_out_lines, short, args.max_models, args.seed)
        b = read_dataset_line(args.sobol_out_lines, short, args.max_models, args.seed)
        if a is None or b is None:
            print(f"[{short}] missing in one of the two datasets; skipped")
            continue
        ids_a, lam_a, flux_a = screen_corrupt(*a)[:3]
        ids_b, lam_b, flux_b = screen_corrupt(*b)[:3]

        lo = max(np.nanmin(lam_a), np.nanmin(lam_b))
        hi = min(np.nanmax(lam_a), np.nanmax(lam_b))
        print(f"[{short}] LHC   lambda {np.nanmin(lam_a):9.2f} - "
              f"{np.nanmax(lam_a):9.2f} ({lam_a.shape[1]} pts, {ids_a.size} models)")
        print(f"[{short}] Sobol lambda {np.nanmin(lam_b):9.2f} - "
              f"{np.nanmax(lam_b):9.2f} ({lam_b.shape[1]} pts, {ids_b.size} models)")
        print(f"[{short}] common window {lo:.2f} - {hi:.2f} A")

        ew_a, fmin_a, fmax_a = shape_measures(lam_a, flux_a, lo, hi)
        ew_b, fmin_b, fmax_b = shape_measures(lam_b, flux_b, lo, hi)

        keep_a = np.array([int(i) in lhc_map for i in ids_a])
        keep_b = np.array([int(i) in sobol_map for i in ids_b])
        par_a = np.array([lhc_map[int(i)] for i in ids_a if int(i) in lhc_map])
        par_b = np.array([sobol_map[int(i)] for i in ids_b if int(i) in sobol_map])
        ew_a, fmin_a, fmax_a = ew_a[keep_a], fmin_a[keep_a], fmax_a[keep_a]
        ew_b, fmin_b, fmax_b = ew_b[keep_b], fmin_b[keep_b], fmax_b[keep_b]
        lam_a, flux_a = lam_a[keep_a], flux_a[keep_a]
        lam_b, flux_b = lam_b[keep_b], flux_b[keep_b]

        driver_1, driver_2 = LINE_DRIVERS.get(short, DEFAULT_DRIVERS)
        i1, i2 = PARAM_COLS.index(driver_1), PARAM_COLS.index(driver_2)

        # twelve equal bins of the primary driver, both datasets binned alike
        x_a, x_b = par_a[:, i1], par_b[:, i1]
        edges = np.linspace(min(x_a.min(), x_b.min()), max(x_a.max(), x_b.max()),
                            GRID_CONSISTENCY_TEFF_BIN_EDGES)
        centres, median_a, median_b, counts_a, counts_b = [], [], [], [], []
        for k in range(len(edges) - 1):
            in_a = ((x_a >= edges[k]) & (x_a < edges[k + 1]) & np.isfinite(ew_a))
            in_b = ((x_b >= edges[k]) & (x_b < edges[k + 1]) & np.isfinite(ew_b))
            if in_a.sum() < GRID_CONSISTENCY_MIN_PER_BIN \
                    or in_b.sum() < GRID_CONSISTENCY_MIN_PER_BIN:
                continue
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            median_a.append(np.median(ew_a[in_a]))
            median_b.append(np.median(ew_b[in_b]))
            counts_a.append(int(in_a.sum()))
            counts_b.append(int(in_b.sum()))
        centres = np.asarray(centres)
        median_a = np.asarray(median_a)
        median_b = np.asarray(median_b)

        scale = np.nanmedian(np.abs(np.concatenate([ew_a, ew_b])))
        offset = (float(np.nanmedian(np.abs(median_b - median_a)) / scale)
                  if centres.size and scale > 0 else float("nan"))
        print(f"[{short}] median |EW(Sobol) - EW(LHC)| per bin / typical |EW| "
              f"= {offset:.3f}   (drivers: {driver_1}, {driver_2})")

        report.append({"line": short, "rel_ew_offset": offset,
                       "lhc_lo": float(np.nanmin(lam_a)),
                       "lhc_hi": float(np.nanmax(lam_a)),
                       "sob_lo": float(np.nanmin(lam_b)),
                       "sob_hi": float(np.nanmax(lam_b)),
                       "n_lhc": int(ew_a.size), "n_sobol": int(ew_b.size),
                       "ew_median_lhc": float(np.nanmedian(ew_a)),
                       "ew_median_sobol": float(np.nanmedian(ew_b)),
                       "fmin_median_lhc": float(np.nanmedian(fmin_a)),
                       "fmin_median_sobol": float(np.nanmedian(fmin_b)),
                       "fmax_median_lhc": float(np.nanmedian(fmax_a)),
                       "fmax_median_sobol": float(np.nanmedian(fmax_b))})

        centre_1 = float(np.nanmedian(np.concatenate([par_a[:, i1], par_b[:, i1]])))
        centre_2 = float(np.nanmedian(np.concatenate([par_a[:, i2], par_b[:, i2]])))
        width_1 = GRID_CONSISTENCY_TOL_PRIMARY * (np.nanmax(par_a[:, i1])
                                                  - np.nanmin(par_a[:, i1]))
        width_2 = GRID_CONSISTENCY_TOL_SECONDARY * (np.nanmax(par_a[:, i2])
                                                    - np.nanmin(par_a[:, i2]))
        index_a = np.flatnonzero((np.abs(par_a[:, i1] - centre_1) < width_1)
                                 & (np.abs(par_a[:, i2] - centre_2) < width_2)
                                 )[:GRID_CONSISTENCY_MAX_PROFILES]
        index_b = np.flatnonzero((np.abs(par_b[:, i1] - centre_1) < width_1)
                                 & (np.abs(par_b[:, i2] - centre_2) < width_2)
                                 )[:GRID_CONSISTENCY_MAX_PROFILES]

        crop = REF_CROP_RANGES.get(short)
        window = (crop[0], crop[1]) if crop is not None else (lo, hi)

        name = f"grid_consistency_{short}.{args.format}"
        save_grid_consistency_figure(
            short, driver_1, driver_2, centres, median_a, median_b, ew_a, ew_b,
            lam_a, flux_a, lam_b, flux_b, index_a, index_b, centre_1, centre_2,
            window, out_dir / name,
            extra_paths=thesis_target(args, "grid_consistency", name),
            skip_existing=args.skip_existing)

    if report:
        save_rows([[row[k] for k in report[0]] for row in report],
                  list(report[0]), out_dir / "grid_consistency_report.csv")
        print("\nlines ranked by disagreement between the two datasets:")
        for row in sorted(report, key=lambda z: -(z["rel_ew_offset"]
                                                  if np.isfinite(z["rel_ew_offset"])
                                                  else -1)):
            print(f"  {row['line']:20s} rel_EW_offset={row['rel_ew_offset']:.3f}")


def figure_dataset_audit(args) -> None:
    """The parameter, wavelength-window and nearest-neighbour audit.

    Source: compare_lhc_sobol_datasets.section_params / section_windows /
    section_profiles.  Not printed in the thesis; it is the measurement the
    grid-consistency discussion rests on.
    """
    h5py = h5py_module()
    out_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = list(args.audit_sections)

    lhc_map = sobol_map = None
    if {"params", "profiles"} & set(sections):
        lhc_map = _load_lhc_parameter_map(args, args.lhc_quality_json)
        sobol_map = _load_sobol_parameter_map(args)
        print(f"LHC models: {len(lhc_map)}   Sobol models: {len(sobol_map)}")

    if "params" in sections:
        lhc_par = np.asarray(list(lhc_map.values()))
        sobol_par = np.asarray(list(sobol_map.values()))
        print("\n===== PARAMETER RANGES =====")
        print(f"{'param':<8}{'LHC min':>12}{'LHC max':>12}"
              f"{'Sobol min':>12}{'Sobol max':>12}{'range match':>13}")
        rows = []
        for j, column in enumerate(PARAM_COLS):
            l0, l1 = lhc_par[:, j].min(), lhc_par[:, j].max()
            s0, s1 = sobol_par[:, j].min(), sobol_par[:, j].max()
            span = max(l1, s1) - min(l0, s0)
            match = ("SAME" if (abs(l0 - s0) < AUDIT_RANGE_MATCH_TOLERANCE * span
                                and abs(l1 - s1) < AUDIT_RANGE_MATCH_TOLERANCE * span)
                     else "DIFFERENT")
            print(f"{column:<8}{l0:>12.4g}{l1:>12.4g}{s0:>12.4g}{s1:>12.4g}"
                  f"{match:>13}")
            rows.append([column, l0, l1, s0, s1, match])
        save_rows(rows, ["param", "lhc_min", "lhc_max", "sobol_min", "sobol_max",
                         "range_match"], out_dir / "param_ranges.csv")

        fig, axes = plt.subplots(4, 4, figsize=(15, 11))
        flat = [ax for row in axes for ax in row]
        for j, column in enumerate(PARAM_COLS):
            ax = flat[j]
            lhs, rhs = lhc_par[:, j], sobol_par[:, j]
            label = column
            if column == "mdot":     # sampled log-uniformly, compared in the log
                lhs, rhs = np.log10(lhs), np.log10(rhs)
                label = "log10(mdot)"
            bins = np.linspace(min(lhs.min(), rhs.min()), max(lhs.max(), rhs.max()),
                               AUDIT_HIST_BINS)
            ax.hist(lhs, bins, density=True, alpha=0.55, label="LHC (green)",
                    color="tab:blue")
            ax.hist(rhs, bins, density=True, alpha=0.55, label="Sobol",
                    color="tab:orange")
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in flat[len(PARAM_COLS):]:
            ax.axis("off")
        flat[0].legend(fontsize=8)
        fig.suptitle("Parameter distributions: LHC (green models) vs Sobol", y=0.995)
        fig.tight_layout()
        save_figure(fig, out_dir / "param_distributions.png", dpi=DPI_PANEL,
                    skip_existing=args.skip_existing)

    if "windows" in sections:
        with h5py.File(str(args.lhc_out_lines), "r") as handle:
            lhc_groups = {short_name_aliased(k): k for k in handle["out_lines"]}
        with h5py.File(str(args.sobol_out_lines), "r") as handle:
            sobol_groups = {short_name_aliased(k): k for k in handle["out_lines"]}
        shared = sorted(set(lhc_groups) & set(sobol_groups))

        def window_stats(h5_path, group_name) -> Dict[str, float]:
            with h5py.File(str(h5_path), "r") as handle:
                group = handle["out_lines"][group_name]
                n = group["model_ids"].shape[0]
                selection = np.linspace(
                    0, n - 1, min(AUDIT_WINDOW_SAMPLE, n)).astype(int)
                n_points = group["n_points"][:][selection]
                waves = group["wavelength"][selection, :]
            mins, maxs, spacings = [], [], []
            for row, count in zip(waves, n_points):
                lam = row[: int(count)]
                lam = lam[np.isfinite(lam)]
                if lam.size < 2:
                    continue
                mins.append(float(lam.min()))
                maxs.append(float(lam.max()))
                spacings.append(float(np.median(np.diff(np.sort(lam)))))
            return {"lam_min": float(np.median(mins)),
                    "lam_max": float(np.median(maxs)),
                    "dlam": float(np.median(spacings)),
                    "n_points": float(np.median(n_points))}

        print(f"\n===== WAVELENGTH WINDOWS (median over {AUDIT_WINDOW_SAMPLE} "
              "sampled models) =====")
        rows = []
        for short in shared:
            a = window_stats(args.lhc_out_lines, lhc_groups[short])
            b = window_stats(args.sobol_out_lines, sobol_groups[short])
            print(f"{short:<18}{a['lam_min']:>10.1f}-{a['lam_max']:<10.1f}"
                  f"{b['lam_min']:>10.1f}-{b['lam_max']:<10.1f}"
                  f"{a['dlam']:>10.3f}{b['dlam']:>10.3f}")
            rows.append([short, a["lam_min"], a["lam_max"], a["dlam"], a["n_points"],
                         b["lam_min"], b["lam_max"], b["dlam"], b["n_points"]])
        save_rows(rows, ["line", "lhc_min", "lhc_max", "lhc_dlam", "lhc_npts",
                         "sobol_min", "sobol_max", "sobol_dlam", "sobol_npts"],
                  out_dir / "wavelength_windows.csv")

        fig, ax = plt.subplots(figsize=(9, 0.3 * len(rows) + 2))
        for i, row in enumerate(rows):
            ax.plot([row[1], row[2]], [i + 0.12, i + 0.12], lw=4, color="tab:blue",
                    alpha=0.8)
            ax.plot([row[5], row[6]], [i - 0.12, i - 0.12], lw=4, color="tab:orange",
                    alpha=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row[0] for row in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("wavelength [A]")
        ax.set_title("Native wavelength windows per line: LHC (blue) vs Sobol "
                     "(orange)", fontsize=10)
        fig.tight_layout()
        save_figure(fig, out_dir / "wavelength_windows.png", dpi=DPI_PANEL,
                    skip_existing=args.skip_existing)

    if "profiles" in sections:
        with h5py.File(str(args.lhc_out_lines), "r") as handle:
            lhc_groups = {short_name_aliased(k): k for k in handle["out_lines"]}
        with h5py.File(str(args.sobol_out_lines), "r") as handle:
            sobol_groups = {short_name_aliased(k): k for k in handle["out_lines"]}
        shared = sorted(set(lhc_groups) & set(sobol_groups))
        pair_lines = [s for s in args.audit_pair_lines if s in shared]
        if not pair_lines:
            raise SystemExit(f"none of {list(args.audit_pair_lines)} is in both "
                             "datasets")

        lhc_ids = np.asarray(sorted(lhc_map), dtype=np.int64)
        sobol_ids = np.asarray(sorted(sobol_map), dtype=np.int64)
        lhc_par = np.asarray([lhc_map[int(m)] for m in lhc_ids])
        sobol_par = np.asarray([sobol_map[int(m)] for m in sobol_ids])

        mdot = PARAM_COLS.index("mdot")
        p_lhc, p_sobol = lhc_par.copy(), sobol_par.copy()
        p_lhc[:, mdot] = np.log10(p_lhc[:, mdot])
        p_sobol[:, mdot] = np.log10(p_sobol[:, mdot])
        pooled = np.vstack([p_lhc, p_sobol])
        low, high = pooled.min(axis=0), pooled.max(axis=0)
        span = np.where(high - low == 0, 1.0, high - low)
        norm_lhc = (p_lhc - low) / span
        norm_sobol = (p_sobol - low) / span

        rng = np.random.default_rng(args.audit_seed)
        picks = rng.choice(len(sobol_ids), size=args.audit_pairs, replace=False)

        def row_map(h5_path, group) -> Dict[int, int]:
            with h5py.File(str(h5_path), "r") as handle:
                ids = handle["out_lines"][group]["model_ids"][:]
            out: Dict[int, int] = {}
            for i, m in enumerate(ids):
                out.setdefault(int(m), i)
            return out

        maps_lhc = {s: row_map(args.lhc_out_lines, lhc_groups[s]) for s in pair_lines}
        maps_sobol = {s: row_map(args.sobol_out_lines, sobol_groups[s])
                      for s in pair_lines}

        def read_row(h5_path, group, row):
            with h5py.File(str(h5_path), "r") as handle:
                g = handle["out_lines"][group]
                count = int(g["n_points"][row])
                return g["wavelength"][row, :count], g["flux"][row, :count]

        for k, pick in enumerate(picks, start=1):
            distance = np.sqrt(((norm_lhc - norm_sobol[pick]) ** 2).sum(axis=1))
            j = int(np.argmin(distance))
            sobol_id, lhc_id = int(sobol_ids[pick]), int(lhc_ids[j])
            print(f"\n[pair {k}] sobol {sobol_id} <-> lhc {lhc_id} "
                  f"(normalized 13-D distance {distance[j]:.4f})")
            for jj, column in enumerate(PARAM_COLS):
                print(f"    {column:<8} sobol={sobol_par[pick, jj]:<12.5g} "
                      f"lhc={lhc_par[j, jj]:<12.5g}")

            ncols = 2
            nrows = int(np.ceil(len(pair_lines) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6.2 * ncols, 3.0 * nrows),
                                     squeeze=False)
            flat = [ax for row in axes for ax in row]
            for ax, short in zip(flat, pair_lines):
                if sobol_id not in maps_sobol[short] or lhc_id not in maps_lhc[short]:
                    ax.set_title(f"{short}: row missing", fontsize=9)
                    continue
                wl, fl = read_row(args.lhc_out_lines, lhc_groups[short],
                                  maps_lhc[short][lhc_id])
                ws, fs = read_row(args.sobol_out_lines, sobol_groups[short],
                                  maps_sobol[short][sobol_id])
                ax.plot(wl, fl, "k-", lw=1.1, label=f"LHC {lhc_id}")
                ax.plot(ws, fs, "--", color="tab:red", lw=1.1,
                        label=f"Sobol {sobol_id}")
                ax.set_title(short, fontsize=9)
                ax.tick_params(labelsize=8)
            for ax in flat[len(pair_lines):]:
                ax.axis("off")
            flat[0].legend(fontsize=8)
            fig.suptitle("Nearest-neighbour FASTWIND profiles "
                         f"(13-D distance {distance[j]:.4f}) — similar params "
                         "should give similar profiles if settings match", y=0.999)
            fig.tight_layout()
            save_figure(fig, out_dir / f"nn_profiles_pair{k}.png", dpi=DPI_PANEL,
                        skip_existing=args.skip_existing)


# -------------------------------------------------------------------------------------
# FIGURES: hetero-grid-profiles, hetero-z-calibration
# -------------------------------------------------------------------------------------
# Not printed in the thesis, where calibrated emulator uncertainties appear only
# as future work.  These two families read the separate heteroscedastic run tree
# written by 13_par_hetero/13_emulator_hetero_h5.py, whose checkpoints carry the
# extra keys latent_dim_mu / latent_dim_total / log_var_min / log_var_max /
# sigma_scale_valcal and whose npz files carry pred_sigma / pred_sigma_cal.
# Those artefacts are *not* the ones thirteen_parameter_training.py writes, so
# the network is rebuilt locally here rather than through its build_model.
# Source: plot_hetero_grid_profiles.py, plot_hetero_z_calibration.py.
# -------------------------------------------------------------------------------------
def _hetero_line_dirs(runs_root) -> Dict[str, str]:
    """Source: plot_hetero_*.find_line_dirs."""
    found: Dict[str, str] = {}
    for directory in sorted(glob.glob(os.path.join(str(runs_root), HETERO_DIR_GLOB))):
        if os.path.isdir(directory):
            key = os.path.basename(directory).split("_line")[-1].split("_", 1)[-1]
            found[key] = directory
    return found


def _hetero_emulator(run_dir: str):
    """Load one heteroscedastic checkpoint.  Source: LineEmulator of the source."""
    torch = torch_module()
    nn = torch.nn

    def build_mlp(input_dim, output_dim, hidden_widths):
        layers, in_dim = [], input_dim
        for out_dim in hidden_widths:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    class TrunkNet(nn.Module):
        def __init__(self, output_dim, fourier_modes, hidden_widths):
            super().__init__()
            self.fourier_modes = fourier_modes
            self.net = build_mlp(2 * fourier_modes + 1, output_dim, hidden_widths)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            batch, npts, _ = x.shape
            freqs = 2.0 * math.pi * torch.arange(
                1, self.fourier_modes + 1, dtype=x.dtype).view(1, 1, -1)
            feats = torch.cat([torch.ones(batch, npts, 1, dtype=x.dtype),
                               torch.sin(freqs * x), torch.cos(freqs * x)], dim=-1)
            out = self.net(feats.view(-1, feats.shape[-1]))
            return out.view(batch, npts, -1)

    class BranchNet(nn.Module):
        """Wrapper so the state-dict keys match the training script (branch.net.*)."""

        def __init__(self, input_dim, output_dim, hidden_widths):
            super().__init__()
            self.net = build_mlp(input_dim, output_dim, hidden_widths)

        def forward(self, x):
            return self.net(x)

    class HeteroDeepONet(nn.Module):
        def __init__(self, ckpt):
            super().__init__()
            self.mu_dim = int(ckpt["latent_dim_mu"])
            total = int(ckpt["latent_dim_total"])
            self.branch = BranchNet(len(ckpt["param_cols"]), total,
                                    ckpt["branch_widths"])
            self.trunk = TrunkNet(total, int(ckpt["fourier_modes"]),
                                  ckpt["trunk_widths"])
            self.lv_min = float(ckpt.get("log_var_min", -14.0))
            self.lv_max = float(ckpt.get("log_var_max", 4.0))

        def forward(self, params, coords):
            branch = self.branch(params).unsqueeze(1)
            trunk = self.trunk(coords)
            mu = (trunk[..., :self.mu_dim] * branch[..., :self.mu_dim]).sum(-1)
            log_var = (trunk[..., self.mu_dim:] * branch[..., self.mu_dim:]).sum(-1)
            return mu, torch.clamp(log_var, self.lv_min, self.lv_max)

    class HeteroEmulator:
        def __init__(self, directory):
            pth = glob.glob(os.path.join(directory, "emulator_*.pth"))[0]
            ckpt = torch.load(pth, map_location="cpu", weights_only=False)
            self.model = HeteroDeepONet(ckpt)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.param_cols = ckpt["param_cols"]
            self.mdot_index = self.param_cols.index("mdot")
            self.p_min = np.asarray(ckpt["param_mins_after_mdot_log"], np.float64)
            self.p_rng = np.asarray(ckpt["param_range_after_mdot_log"], np.float64)
            self.lam_min = float(ckpt["lambda_min"])
            self.lam_max = float(ckpt["lambda_max"])
            self.sigma_scale = float(ckpt.get("sigma_scale_valcal", 1.0))
            self.flux_offset = float(ckpt.get("flux_offset", 1.0))

        def predict(self, raw_params, wavelengths):
            params = np.asarray(raw_params, np.float64).copy()
            params[self.mdot_index] = np.log10(params[self.mdot_index])
            params_norm = torch.tensor((params - self.p_min) / self.p_rng,
                                       dtype=torch.float32)
            coords = ((np.asarray(wavelengths, np.float64) - self.lam_min)
                      / (self.lam_max - self.lam_min))
            with torch.no_grad():
                mu, log_var = self.model(
                    params_norm.unsqueeze(0),
                    torch.tensor(coords, dtype=torch.float32).unsqueeze(0))
            flux = mu.squeeze(0).numpy() + self.flux_offset
            sigma = np.exp(0.5 * log_var.squeeze(0).numpy()) * self.sigma_scale
            return flux, sigma

    return HeteroEmulator(run_dir)


def figure_hetero_grid_profiles(args) -> None:
    line_dirs = _hetero_line_dirs(args.hetero_runs_root)
    if not line_dirs:
        raise SystemExit(f"no heteroscedastic runs under {args.hetero_runs_root}")
    selected = (sorted(line_dirs) if list(args.lines) in ([], ["all"])
                else list(args.lines))
    missing = [s for s in selected if s not in line_dirs]
    if missing:
        raise SystemExit(f"Lines not found: {missing}\nAvailable: {sorted(line_dirs)}")

    print(f"Loading {args.grid_json} ...", flush=True)
    grid = lhc_grid(args.grid_json)
    out_dir = Path(args.hetero_out_dir) / "grid_profiles"
    rng = np.random.default_rng(args.seed)

    for short in selected:
        print(f"=== {short} ===")
        emulator = _hetero_emulator(line_dirs[short])
        npz = np.load(glob.glob(os.path.join(line_dirs[short],
                                             "*_test_outputs.npz"))[0])
        n_test = npz["target_flux"].shape[0]
        for i in rng.choice(n_test, size=min(args.n_models, n_test), replace=False):
            i = int(i)
            model_id = int(npz["test_model_ids"][i])
            record = grid.get(str(model_id))
            if record is None:
                print(f"[skip] model {model_id} not in the grid JSON")
                continue
            raw_params = [record[c] for c in emulator.param_cols]

            lam = np.asarray(npz["wavelengths_phys"][i], np.float64)
            flux = np.asarray(npz["target_flux"][i], np.float64)
            order = np.argsort(lam)
            lam, flux = lam[order], flux[order]

            grids = list(args.hetero_grids)
            fig, axes = plt.subplots(len(grids), 1, figsize=(11, 2.9 * len(grids)),
                                     sharex=True, squeeze=False)
            for row, n_points in enumerate(grids):
                ax = axes[row][0]
                lam_grid = np.linspace(lam[0], lam[-1], int(n_points))
                flux_grid, sigma_grid = emulator.predict(raw_params, lam_grid)
                ax.plot(lam, flux, "k.", ms=4, zorder=3,
                        label=f"FASTWIND ({MAX_ROWS_PER_LINE_FILE} native points)")
                if n_points <= HETERO_ERRORBAR_MAX_POINTS:
                    ax.errorbar(lam_grid, flux_grid, yerr=sigma_grid, fmt="o",
                                color="tab:red", ms=3, lw=0.9, capsize=2, alpha=0.85,
                                zorder=2,
                                label=(r"emulator $\mu \pm 1\sigma$ "
                                       f"({int(n_points)} points)"))
                else:
                    ax.plot(lam_grid, flux_grid, "-", color="tab:red", lw=1.1,
                            zorder=2,
                            label=rf"emulator $\mu$ ({int(n_points)} points)")
                    ax.fill_between(lam_grid, flux_grid - sigma_grid,
                                    flux_grid + sigma_grid, color="tab:red",
                                    alpha=0.3, label=r"$\pm 1\sigma$")
                ax.set_ylabel("normalized flux")
                ax.legend(fontsize=8, loc="lower right")
            axes[-1][0].set_xlabel(r"wavelength [$\AA$]")
            fig.suptitle(f"{short}  model {model_id}: emulator on different "
                         "wavelength grids", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            save_figure(fig, out_dir / f"grid_profiles_{short}_model{model_id}.png",
                        dpi=DPI_HETERO, tight_bbox=False,
                        skip_existing=args.skip_existing)


def figure_hetero_z_calibration(args) -> None:
    line_dirs = _hetero_line_dirs(args.hetero_runs_root)
    if not line_dirs:
        raise SystemExit(f"no heteroscedastic runs under {args.hetero_runs_root}")
    selected = (sorted(line_dirs) if list(args.lines) in ([], ["all"])
                else list(args.lines))
    missing = [s for s in selected if s not in line_dirs]
    if missing:
        raise SystemExit(f"Lines not found: {missing}\nAvailable: {sorted(line_dirs)}")

    out_dir = Path(args.hetero_out_dir) / "z_calibration"
    std_by_line: Dict[str, float] = {}

    for short in selected:
        npz = np.load(glob.glob(os.path.join(line_dirs[short],
                                             "*_test_outputs.npz"))[0])
        sigma = npz["pred_sigma_cal"] if "pred_sigma_cal" in npz else npz["pred_sigma"]
        z = ((npz["target_residual"] - npz["pred_residual"]) / sigma).ravel()

        std_z = float(np.std(z))
        frac1 = float((np.abs(z) < 1).mean())
        frac2 = float((np.abs(z) < 2).mean())
        std_by_line[short] = std_z

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(z, bins=HETERO_Z_HIST_BINS, range=HETERO_Z_HIST_RANGE, density=True,
                color="lightsteelblue", edgecolor="none",
                label="measured z of all test points")
        xs = np.linspace(HETERO_Z_HIST_RANGE[0], HETERO_Z_HIST_RANGE[1], 400)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.5,
                label="ideal Gaussian (perfect error bars)")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(r"z = (F$_\mathrm{FASTWIND}$ - $\mu$) / $\sigma$")
        ax.set_ylabel("probability density")
        ax.set_title(f"{short}: are the error bars correct?\n"
                     f"std(z) = {std_z:.2f} (ideal 1)   |   "
                     f"|z|<1: {frac1:.0%} (ideal 68%)   |   "
                     f"|z|<2: {frac2:.0%} (ideal 95%)", fontsize=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_figure(fig, out_dir / f"z_hist_{short}.png", dpi=DPI_HETERO,
                    tight_bbox=False, skip_existing=args.skip_existing)
        print(f"[{short}] std(z)={std_z:.3f}  |z|<1: {frac1:.3f}  |z|<2: {frac2:.3f}")

    if len(std_by_line) > 1:
        names = list(std_by_line)
        values = [std_by_line[n] for n in names]
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.5))
        ax.barh(range(len(names)), values, color="lightsteelblue",
                edgecolor="tab:blue")
        ax.axvline(1.0, color="k", ls="--", lw=1.5)
        ax.text(1.0, len(names) - 0.2, " ideal: std(z) = 1", fontsize=9, va="bottom")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("std(z) on the test set")
        ax.set_title("error-bar calibration of all lines "
                     r"(bars near the dashed line = correct $\sigma$)", fontsize=10)
        fig.tight_layout()
        save_figure(fig, out_dir / "z_std_ALL_LINES.png", dpi=DPI_HETERO,
                    tight_bbox=False, skip_existing=args.skip_existing)


# =====================================================================================
# SECTION 5.  COMMAND-LINE INTERFACE
# =====================================================================================

#: Every figure family, in the order of the FIGURE MAP in the module docstring.
FIGURE_DISPATCH = {
    "loss-curves": figure_loss_curves,
    "profiles": figure_profiles,
    "relative-error": figure_relative_error,
    "per-line-mare": figure_per_line_mare,
    "worst-corner": figure_worst_corner,
    "convergence": figure_convergence,
    "own-test": figure_own_test,
    "grid-consistency": figure_grid_consistency,
    "dataset-audit": figure_dataset_audit,
    "hetero-grid-profiles": figure_hetero_grid_profiles,
    "hetero-z-calibration": figure_hetero_z_calibration,
}

#: What ``--all`` runs: every family that needs nothing but the adopted run
#: folders and the Latin-hypercube grid file.  ``convergence`` is left out
#: because it also needs the quality file and pdflatex, ``own-test`` because it
#: needs the second campaign's run tree, ``grid-consistency`` and
#: ``dataset-audit`` because they need the HDF5 datasets, and the two
#: ``hetero-*`` families because they read a different run tree altogether.
ALL_FIGURES: Tuple[str, ...] = (
    "loss-curves", "profiles", "relative-error", "per-line-mare", "worst-corner",
)


def build_parser() -> argparse.ArgumentParser:
    root = default_root()
    output_root = default_output_root()
    base = root / "thirteen_parameter"
    datasets = root / "datasets"
    run_base = output_root / "thirteen_parameter"

    p = argparse.ArgumentParser(
        prog="thirteen_parameter_plots.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--figure", choices=tuple(FIGURE_DISPATCH),
                          help="Which figure family to produce.  See the FIGURE MAP "
                               "in the module docstring.")
    selector.add_argument("--all", action="store_true",
                          help="Produce every figure that reads only the adopted run "
                               "folders: " + ", ".join(ALL_FIGURES) + ".")

    # --- where the runs live -------------------------------------------------
    p.add_argument("--base-dir", dest="base_dir", type=Path, default=base,
                   help="Input-data tree holding the grid and quality files.")
    p.add_argument("--runs-root", dest="runs_root", type=Path,
                   default=run_base / "runs",
                   help="Folder holding one sub-folder per training run.")
    p.add_argument("--run-glob", dest="run_glob", default=DEEP_FILTERED_GLOB,
                   help="Glob of the adopted campaign's run folders inside "
                        "--runs-root (default: the deep, filtered, crop/edge-padded "
                        "Latin-hypercube groups).")
    p.add_argument("--output-dir", dest="output_dir", type=Path,
                   default=run_base / "plots" / DEEP_FILTERED_TAG,
                   help="Root of the figure tree.")
    p.add_argument("--grid-json", dest="grid_json", type=Path,
                   default=base / "grid_large_log_LHC.json",
                   help="The 13 raw parameters of every Latin-hypercube model.")
    p.add_argument("--quality-json", dest="quality_json", type=Path,
                   default=base / "grid_quality_large_lhc.json",
                   help="The FASTWIND convergence flag of every model, for "
                        "--figure convergence.")
    p.add_argument("--datasets-dir", dest="datasets_dir", type=Path,
                   default=datasets,
                   help="The datasets tree holding lhc/ and sobol_v2/.")
    p.add_argument("--thesis-dir", dest="thesis_dir", type=Path,
                   default=None,
                   help="Optional second destination for figures, for example a "
                        "thesis project. Disabled by default.")
    p.add_argument("--no-thesis-copy", dest="no_thesis_copy", action="store_true",
                   help="Do not also write the figures into the thesis folder.")

    # --- general -------------------------------------------------------------
    p.add_argument("--lines", nargs="+", default=["all"],
                   help="Restrict the per-diagnostic families to these short names "
                        "(default: all 37).")
    p.add_argument("--format", choices=("pdf", "png"), default="pdf",
                   help="File format of the matplotlib figures.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=512,
                   help="Prediction batch size of the checkpoint evaluations.")
    p.add_argument("--seed", type=int, default=CORNER_SEED,
                   help="Seed of every reproducible random draw: the grey sample of "
                        "--figure relative-error, the background of --figure "
                        "worst-corner, the subsample of --figure grid-consistency "
                        "and the star choice of --figure hetero-grid-profiles "
                        f"(default: {CORNER_SEED}).")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                   help="Leave a figure alone if its file already exists.")
    p.add_argument("--verify-constants", dest="verify_constants", action="store_true",
                   help="Import thirteen_parameter_training.py and assert that the "
                        "13 parameter columns, the 161 rows per line file and the "
                        "five artefact names still match.  Needs torch.")

    # --- loss-curves ---------------------------------------------------------
    p.add_argument("--layout", choices=("single", "grid", "both"), default="single",
                   help="'single' writes one loss figure per diagnostic (the thesis "
                        "version), 'grid' the four-column small multiples of "
                        "13_plot_after_training.py, 'both' writes both.")
    p.add_argument("--mark-best-epoch", dest="mark_best_epoch", action="store_true",
                   help="Draw the dashed best-epoch line of "
                        "13_fw_emulator_per_line_comparison.ipynb cell 4.")

    # --- profiles ------------------------------------------------------------
    p.add_argument("--profile-line", dest="profile_line", default="CIV1169b",
                   help="Short name of the window drawn on its own (default: "
                        "CIV1169b, one of the hardest; HALPHA is accepted as an "
                        "alias for HALPHAHEII6527).")
    p.add_argument("--model-id", dest="model_id", type=int, default=None,
                   help="The test model the profiles show.  Without it the model is "
                        "chosen by --which among the models shared by every "
                        "diagnostic.")
    p.add_argument("--which", choices=("median", "worst", "best"), default="median",
                   help="How that model is chosen from the per-star MARE ranking.")
    p.add_argument("--uniform-spacing", dest="uniform_spacing", type=float,
                   default=UNIFORM_GRID_SPACING,
                   help="Spacing in Angstrom of the regular grid the emulator is "
                        "re-evaluated on (default: 0.2, the BLOeM LR02 setting).")
    p.add_argument("--skip-uniform", dest="skip_uniform", action="store_true",
                   help="Draw only the native-grid profiles, so that torch is not "
                        "needed.")
    p.add_argument("--bloem-panels", dest="bloem_panels", action="store_true",
                   help="Also draw the BLOeM LR02 overlay panels of "
                        "13_plot_after_training.py.")
    p.add_argument("--bloem-lmin", dest="bloem_lmin", type=float, default=BLOEM_LMIN,
                   help="Lower edge of the BLOeM LR02 window, in Angstrom.")
    p.add_argument("--bloem-lmax", dest="bloem_lmax", type=float, default=BLOEM_LMAX,
                   help="Upper edge of the BLOeM LR02 window, in Angstrom.")

    # --- relative-error ------------------------------------------------------
    p.add_argument("--binning", choices=("adaptive", "uniform"), default="adaptive",
                   help="'adaptive' uses equal-count quantile bins of the pooled "
                        "physical wavelengths (the thesis figure); 'uniform' uses "
                        "the notebook's equal-width bins over the central 99 per "
                        "cent.")
    p.add_argument("--relative-error-bins", dest="relative_error_bins", type=int,
                   default=RELATIVE_ERROR_BINS,
                   help=f"Number of wavelength bins (default: {RELATIVE_ERROR_BINS}).")
    p.add_argument("--relative-error-eps", dest="relative_error_eps", type=float,
                   default=EPS_SIGNED,
                   help="Denominator guard of the signed relative error "
                        f"(default: {EPS_SIGNED}).")
    p.add_argument("--scatter-points", dest="scatter_points", type=int,
                   default=RELATIVE_ERROR_SCATTER_POINTS,
                   help="Maximum number of grey model-wavelength samples drawn "
                        f"(default: {RELATIVE_ERROR_SCATTER_POINTS}).")

    # --- worst-corner --------------------------------------------------------
    p.add_argument("--corner-set", dest="corner_set",
                   choices=("dependent", "5par", "13par"), default="dependent",
                   help="'dependent' keeps only the KS-selected parameters (the "
                        "thesis figure), '5par' and '13par' the fixed sets.")
    p.add_argument("--worst-fraction", dest="worst_fraction", type=float,
                   default=WORST_FRACTION,
                   help="Fraction of the test partition marked as the worst cases "
                        f"(default: {WORST_FRACTION}).")
    p.add_argument("--ks-threshold", dest="ks_threshold", type=float,
                   default=KS_THRESHOLD,
                   help="Two-sample Kolmogorov-Smirnov distance above which a "
                        f"parameter counts as dependent (default: {KS_THRESHOLD}).")
    p.add_argument("--min-params", dest="min_params", type=int, default=KS_MIN_PARAMS,
                   help="Parameters retained even if fewer pass the threshold "
                        f"(default: {KS_MIN_PARAMS}).")
    p.add_argument("--corner-background-max", dest="corner_background_max", type=int,
                   default=CORNER_BACKGROUND_MAX,
                   help="Grey background points drawn per panel "
                        f"(default: {CORNER_BACKGROUND_MAX}).")

    # --- convergence ---------------------------------------------------------
    p.add_argument("--convergence-dir", dest="convergence_dir", type=Path,
                   default=base / "convergence_parameter_space",
                   help="Where the per-parameter .dat tables, the PGFPlots .tex and "
                        "the compiled .pdf are written.")
    p.add_argument("--no-pdflatex", dest="no_pdflatex", action="store_true",
                   help="Write the .dat tables and the .tex but do not run pdflatex.")

    # --- own-test ------------------------------------------------------------
    p.add_argument("--own-test-dir", dest="own_test_dir", type=Path, default=None,
                   help="Where the own-test figures go "
                        "(default: <base-dir>/comparison_plots).")
    p.add_argument("--no-local-filter", dest="local_filter", action="store_false",
                   help="Apply only the corrupt-row screen to each own test "
                        "partition, instead of also re-applying the training "
                        "filter's abs-flux cut and physical width QC.")
    p.set_defaults(local_filter=True)

    # --- grid-consistency / dataset-audit ------------------------------------
    p.add_argument("--grid-consistency-dir", dest="grid_consistency_dir", type=Path,
                   default=datasets / "grid_consistency_plots",
                   help="Where the three-panel dataset comparisons go.")
    p.add_argument("--grid-consistency-lines", dest="grid_consistency_lines",
                   nargs="+", default=list(GRID_CONSISTENCY_DEFAULT_LINES),
                   help="Short names, or 'all' for every diagnostic present in both "
                        "datasets.")
    p.add_argument("--lhc-out-lines", dest="lhc_out_lines", type=Path,
                   default=datasets / "lhc" / "out_lines.h5",
                   help="Raw Latin-hypercube profiles.")
    p.add_argument("--sobol-out-lines", dest="sobol_out_lines", type=Path,
                   default=datasets / "sobol_v2" / "out_lines.h5",
                   help="Raw Sobol profiles (the final_release grid).")
    p.add_argument("--sobol-models-h5", dest="sobol_models_h5", type=Path,
                   default=datasets / "sobol_v2" / "models.h5",
                   help="The 13 raw parameters of every Sobol model.")
    p.add_argument("--lhc-quality-json", dest="lhc_quality_json", type=Path,
                   default=datasets / "lhc" / "grid_quality_large_lhc.json",
                   help="Quality flags used by --figure dataset-audit to restrict "
                        "the Latin-hypercube population to the green models.")
    p.add_argument("--max-models", dest="max_models", type=int,
                   default=GRID_CONSISTENCY_MAX_MODELS,
                   help="Models sampled per dataset per diagnostic; 0 uses all of "
                        f"them.  Default {GRID_CONSISTENCY_MAX_MODELS}, the value "
                        "the thesis quotes.  The research script's own default was "
                        "30000.")
    p.add_argument("--audit-dir", dest="audit_dir", type=Path,
                   default=datasets / "comparison_plots_v2",
                   help="Where --figure dataset-audit writes.")
    p.add_argument("--audit-sections", dest="audit_sections", nargs="+",
                   default=["params", "windows", "profiles"],
                   choices=("params", "windows", "profiles"),
                   help="Which sections of the dataset audit to run.")
    p.add_argument("--audit-pairs", dest="audit_pairs", type=int,
                   default=AUDIT_DEFAULT_PAIRS,
                   help="Nearest-neighbour pairs drawn by the audit's profile "
                        "section.")
    p.add_argument("--audit-pair-lines", dest="audit_pair_lines", nargs="+",
                   default=list(AUDIT_DEFAULT_PAIR_LINES),
                   help="Diagnostics shown for each of those pairs.")
    p.add_argument("--audit-seed", dest="audit_seed", type=int, default=AUDIT_SEED,
                   help=f"Seed of the pair choice (default: {AUDIT_SEED}).")

    # --- the heteroscedastic side experiment ---------------------------------
    p.add_argument("--hetero-runs-root", dest="hetero_runs_root", type=Path,
                   default=output_root / "thirteen_parameter_hetero" / "runs",
                   help="Run tree of the heteroscedastic emulators.")
    p.add_argument("--hetero-out-dir", dest="hetero_out_dir", type=Path,
                   default=output_root / "thirteen_parameter_hetero" / "plots",
                   help="Where the two hetero-* families write.")
    p.add_argument("--hetero-grids", dest="hetero_grids", nargs="+", type=int,
                   default=list(HETERO_GRIDS),
                   help="Numbers of wavelength points the emulator is evaluated on.")
    p.add_argument("--n-models", dest="n_models", type=int, default=1,
                   help="Random test stars per diagnostic for "
                        "--figure hetero-grid-profiles.")
    return p


#: Options whose default is a path *inside* --base-dir, and the sub-path used.
#: When --base-dir is moved and the option itself was left at its default, the
#: option follows, so that a whole alternative tree can be pointed at with one
#: flag instead of six.
_BASE_DIR_DEPENDENTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("grid_json", ("grid_large_log_LHC.json",)),
    ("quality_json", ("grid_quality_large_lhc.json",)),
    ("convergence_dir", ("convergence_parameter_space",)),
)
#: The same, relative to --datasets-dir.
_DATASETS_DIR_DEPENDENTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("grid_consistency_dir", ("grid_consistency_plots",)),
    ("lhc_out_lines", ("lhc", "out_lines.h5")),
    ("sobol_out_lines", ("sobol_v2", "out_lines.h5")),
    ("sobol_models_h5", ("sobol_v2", "models.h5")),
    ("lhc_quality_json", ("lhc", "grid_quality_large_lhc.json")),
    ("audit_dir", ("comparison_plots_v2",)),
)


def rebase_default_paths(args) -> None:
    """Let --base-dir and --datasets-dir carry their dependent defaults along."""
    root = default_root()
    for parent_name, parent_default, table in (
            ("base_dir", root / "thirteen_parameter", _BASE_DIR_DEPENDENTS),
            ("datasets_dir", root / "datasets", _DATASETS_DIR_DEPENDENTS)):
        parent = Path(getattr(args, parent_name))
        if parent == parent_default:
            continue
        for option, parts in table:
            if Path(getattr(args, option)) == parent_default.joinpath(*parts):
                setattr(args, option, parent.joinpath(*parts))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rebase_default_paths(args)

    if args.relative_error_bins < 2:
        raise SystemExit("--relative-error-bins must be at least two.")
    if args.scatter_points < 1:
        raise SystemExit("--scatter-points must be at least one.")
    if not 0.0 < args.worst_fraction <= 1.0:
        raise SystemExit("--worst-fraction must lie in (0, 1].")
    if args.min_params < 1:
        raise SystemExit("--min-params must be at least one.")
    if args.n_models < 1:
        raise SystemExit("--n-models must be at least one.")
    if args.uniform_spacing <= 0.0:
        raise SystemExit("--uniform-spacing must be positive.")

    if args.verify_constants:
        verify_constants()

    figures = list(ALL_FIGURES) if args.all else [args.figure]

    print(f"Runs root: {args.runs_root}")
    print(f"Output directory: {output_base(args)}")
    print(f"Figures: {', '.join(figures)}")

    for name in figures:
        print(f"\n--- {name} ---")
        FIGURE_DISPATCH[name](args)

    print(f"\nDone: {len(figures)} figure famil{'y' if len(figures) == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
