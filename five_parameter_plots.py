#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
five_parameter_plots.py
=======================

Every figure of Chapter 3 (the five-parameter hydrogen and helium line
emulator), produced from saved training outputs.

This single file replaces the following research scripts, without changing any
numerical behaviour:

    plot_original_5par_emulator.py          -> loss-curves, profiles,
                                               relative-error, parameter-error,
                                               per-line-metrics
    plot_worst_1_percent_corner_5par.py     -> worst-corner
    parallel_plot_thesis.py                 -> parallel
    parallel_plot_from_metadata.py          -> parallel-interactive
    export_parallel_plot_metadata.py        -> export-metadata
    fw_emulator_per_line_comparison_hg.ipynb (plotting cells 11-17)
                                            -> loss-curves (single-line variant),
                                               family-mse, line-panels,
                                               relative-error --mode per-index,
                                               parameter-error

Training and inference live in the companion file
``five_parameter_training.py``; the network classes are imported from there so
that the two files can never drift apart.  The user-facing inference helper
(formerly ``emulator_inference.py``) was folded into the *training* script as
``--mode infer``, because rebuilding a checkpoint requires exactly the network
definitions that already live there.


INPUTS
------
Almost every figure reads only ``emulator_*.pth`` checkpoints.  Each checkpoint
stores the best-epoch weights *and*:

    train_loss_history, val_loss_history   the learning curves
    lambda_min, lambda_max                 the line window, for de-normalising
    param_mins, param_maxs                 the branch-input scaling
    model_ids_test, X_params_test,
    X_waves_test, Y_flux_test              the whole nominal test partition

so the plots below never have to re-read the 19,044-model FASTWIND grid.  The
exceptions are ``profiles`` and ``line-panels`` (which overlay the raw
FASTWIND ``OUT.*`` files) and ``parameter-error`` (which reads INDAT decks for
the physical, un-normalised parameters), both of which need ``--models-dir``.

The two parallel-coordinate figures read ``parallel_plot_metadata.json``, which
is produced by ``--figure export-metadata``.


FIGURE MAP
----------
    --figure loss-curves
        Training and validation weighted-MSE versus epoch, one panel per line.
        Thesis: fig:five_parameter_loss_curves (H-alpha + He II 6527) and
        appendix fig:five_parameter_all_training_1 ... _9 (all 17 lines).

    --figure profiles
        FASTWIND profile versus emulator prediction for randomly chosen common
        test models, on the native FASTWIND grid and on regular 0.20 / 0.10 /
        0.05 Angstrom grids.  The regular-grid panels are the direct
        demonstration that the trunk network takes wavelength as a *coordinate*.
        Thesis: fig:five_parameter_native_profiles,
        fig:five_parameter_regular_01_profiles, and appendix
        fig:five_parameter_native_848, _grid_010_848, _grid_020_848, _grid_005_848.

    --figure relative-error
        Signed relative error (F_FW - F_emu)/F_FW versus wavelength over the
        whole test partition, with median, mean and 16-84 / 2.5-97.5 percentile
        bands.
        Thesis: fig:five_parameter_relative_error and appendix
        fig:five_parameter_all_relative_1 ... _9.

    --figure per-line-metrics
        Test-set MSE and MARE for each of the 17 line emulators.
        Thesis: fig:five_parameter_mare (the MARE panel).

    --figure parameter-error
        Per-test-model MARE colour-coded in every pair of physical parameters.
        Supporting figure for the error-distribution discussion.

    --figure worst-corner
        Corner plot locating the worst 1 % of test models in the five
        independently varied parameters.
        Thesis: fig:five_parameter_halpha_parameter_error and appendix
        fig:five_parameter_worst_corner_* (all 17 lines).

    --figure parallel
        Thesis-styled parallel-coordinate comparison of all 460 evaluated
        configurations, plus the same plot with the best-MARE configuration
        highlighted.
        Thesis: fig:five_parameter_parallel_all, fig:five_parameter_parallel_best.

    --figure parallel-interactive
        The exploratory dark-themed Plotly version of the same data, at
        architecture level and once per spectral line.  Not printed in the
        thesis; used to choose the adopted architecture.

    --figure family-mse
        Test-set MSE per line for the original / latent128 / latent64 families
        on one axis.  Capacity-reduction check from the notebook.

    --figure line-panels
        Multi-panel FASTWIND-versus-emulator overview for one model, either for
        the lines overlapping the BLOeM LR02 window (3960-4570 A) or for all
        available lines, with the emulator evaluated on the native grid and/or
        on a BLOeM-like 0.2 A grid.

    --figure export-metadata
        Not a figure: walks the architecture-search output tree, benchmarks the
        evaluation time of every checkpoint, and writes
        ``parallel_plot_metadata.json``, which the two parallel-coordinate
        figures consume.  Run this before ``--figure parallel``.


USAGE
-----
    # every printed figure, from the adopted checkpoints
    python five_parameter_plots.py --all \
        --emulator-dir emulators_per_line_hg \
        --models-dir models_LHC \
        --output-dir 5_line_emulator_plot

    # one figure
    python five_parameter_plots.py --figure worst-corner \
        --emulator-dir emulators_per_line_hg --format pdf

    # the notebook's per-wavelength-index error statistics instead of the
    # adaptive physical-wavelength bins used in the thesis
    python five_parameter_plots.py --figure relative-error \
        --relative-error-mode per-index

    # the parallel-coordinate pipeline, in order
    python five_parameter_plots.py --figure export-metadata \
        --search-root outputs/five_parameter/fine_tuning_emulator_exploration \
        --hg-dir outputs/five_parameter/emulators_per_line_hg
    python five_parameter_plots.py --figure parallel \
        --metadata parallel_plot_metadata.json --output-dir thesis_figures

    # one command per figure family, with the defaults each preserved script used
    python five_parameter_plots.py --figure loss-curves
    python five_parameter_plots.py --figure profiles --num-models 3 --seed 42
    python five_parameter_plots.py --figure relative-error --relative-error-bins 161
    python five_parameter_plots.py --figure per-line-metrics
    python five_parameter_plots.py --figure parameter-error --models-dir models_LHC
    python five_parameter_plots.py --figure worst-corner --worst-fraction 0.01
    python five_parameter_plots.py --figure family-mse \
        --emulator-dir-128 emulators_per_line_128_hg \
        --emulator-dir-64 emulators_per_line_64_hg
    python five_parameter_plots.py --figure line-panels --target-model-id 1 \
        --panel-scope bloem --panel-grid-mode both --bloem-dlam 0.2
    python five_parameter_plots.py --figure line-panels --panel-scope all
    python five_parameter_plots.py --figure parallel-interactive \
        --metadata parallel_plot_metadata.json
    python five_parameter_plots.py --figure export-metadata
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: every figure is written to disk, never shown

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# The network definitions, the residual convention and the checkpoint loader all
# come from the training script, so a plot can never be made with a different
# architecture than the one that was trained.
from five_parameter_training import (  # noqa: E402
    ALPHA,
    BASELINE_ARCHITECTURES,
    DeepONetModel,
    FLUX_OFFSET,
    LINE_MAX_ROWS,
    PARAM_COLS,
    USE_RESIDUAL,
    _architecture_from_checkpoint,
    build_deeponet,
    load_fastwind_line,
    resolve_device,
)


# =====================================================================================
# SECTION 1.  CONSTANTS
# =====================================================================================

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
DEFAULT_ROOT = Path(
    os.environ.get("FASTWIND_5PAR_DATA", DATA_ROOT / "five_parameter")
).expanduser()
DEFAULT_RUN_ROOT = Path(
    os.environ.get("FASTWIND_5PAR_OUTPUT", OUTPUT_ROOT / "five_parameter")
).expanduser()
DEFAULT_EMULATORS = Path(
    os.environ.get(
        "FASTWIND_5PAR_CHECKPOINTS", DEFAULT_RUN_ROOT / "emulators_per_line_hg"
    )
).expanduser()
DEFAULT_MODELS = DEFAULT_ROOT / "models_LHC"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "plots"

#: Human-readable label for each of the 17 FASTWIND line windows.  H-alpha and
#: He II 6527 fall in one window and are therefore emulated together.
LINE_LABELS: Dict[str, str] = {
    "OUT.BETA_VTV010": r"H$\beta$",
    "OUT.HALPHA_HEII6527_VTV010": r"H$\alpha$ + He II 6527",
    "OUT.HDELTA_VTV010": r"H$\delta$",
    "OUT.HEI4026_VTV010": r"He I 4026",
    "OUT.HEI4387_VTV010": r"He I 4387",
    "OUT.HEI4471_VTV010": r"He I 4471",
    "OUT.HEI4922_VTV010": r"He I 4922",
    "OUT.HEI5875_VTV010": r"He I 5875",
    "OUT.HEI7065_VTV010": r"He I 7065",
    "OUT.HEII4200_VTV010": r"He II 4200",
    "OUT.HEII4541_VTV010": r"He II 4541",
    "OUT.HEII4686_VTV010": r"He II 4686",
    "OUT.HEII5411_VTV010": r"He II 5411",
    "OUT.HEII6406_VTV010": r"He II 6406",
    "OUT.HEII6683_VTV010": r"He II 6683",
    "OUT.HEPS_VTV010": r"H$\epsilon$",
    "OUT.HGAMMA_VTV010": r"H$\gamma$",
}

#: Wavelength grids on which the emulator is asked to predict.  ``None`` means
#: "the native, irregular FASTWIND grid of this model"; the numbers are regular
#: spacings in Angstrom, which the network was never trained on and which are
#: only reachable because wavelength is an input coordinate.
GRID_CASES: Dict[str, Tuple[Optional[float], str]] = {
    "native_grid": (None, "native grid"),
    "grid_0p20A": (0.20, r"0.20 $\AA$ grid"),
    "grid_0p10A": (0.10, r"0.10 $\AA$ grid"),
    "grid_0p05A": (0.05, r"0.05 $\AA$ grid"),
}

#: Physical parameters used for the error-versus-parameter scatter grids.
PARAMETER_LABELS: Dict[str, str] = {
    "Teff": r"$T_{\mathrm{eff}}$ (K)",
    "logg": r"$\log g$",
    "R": r"$R_\ast$ ($R_\odot$)",
    "Mdot": r"$\dot{M}$ ($M_\odot\,\mathrm{yr}^{-1}$)",
    "Y_He": r"$Y_\mathrm{He}$",
}

#: The five independently varied parameters, as used by the corner plots.
CORNER_PARAMETER_KEYS: Tuple[str, ...] = ("Teff", "logg", "R", "logMdot", "Y_He")
CORNER_PARAMETER_LABELS: Dict[str, str] = {
    "Teff": r"$T_{\mathrm{eff}}$ (kK)",
    "logg": r"$\log g$",
    "R": r"$R_\ast$ ($R_\odot$)",
    "logMdot": r"$\log_{10}\dot{M}$",
    "Y_He": r"$Y_{\mathrm{He}}$",
}

#: BLOeM LR02 setting: the observed window and sampling that motivated the
#: "predict on an arbitrary regular grid" test.
BLOEM_LMIN = 3960.0
BLOEM_LMAX = 4570.0
BLOEM_DLAM = 0.2

#: Portable defaults used by the parallel-coordinate metadata pipeline.
EXPORT_BASE_PATH = str(DEFAULT_RUN_ROOT / "fine_tuning_emulator_exploration")
EXPORT_CACHE_PATH = str(DEFAULT_RUN_ROOT / "emulator_metrics_cache.json")
EXPORT_OUTPUT_PATH = str(DEFAULT_RUN_ROOT / "parallel_plot_metadata.json")
EXPORT_TIMING_CACHE_PATH = str(DEFAULT_RUN_ROOT / "emulator_call_time_cache.json")
EXPORT_REFERENCE_MODEL_PARAMS_PATH = str(DEFAULT_ROOT / "all_parameters.txt")
HG_RUN_NAME = "fw_emulator_per_line_comparison_hg"
HG_ARCHITECTURE_NAME = "original"
HG_DIR = str(DEFAULT_EMULATORS)
TIMEOUT_RUNS = {"don_finetune_run_timeout", "don_finetune_run_timeout_extra"}
TIMING_DEVICE = os.environ.get("EMULATOR_TIMING_DEVICE", "cpu")
TIMING_WARMUP = 2
TIMING_REPEATS = 5
TIMING_NUM_POINTS = 161


# =====================================================================================
# SECTION 2.  SHARED HELPERS
# =====================================================================================

def checkpoint_line_name(path: Path) -> str:
    prefix = "emulator_"
    if not path.name.startswith(prefix) or path.suffix != ".pth":
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return path.name[len(prefix):-len(path.suffix)]


def safe_filename(line_file: str) -> str:
    """``OUT.HEII4686_VTV010`` -> ``heii4686``."""
    stem = line_file
    if stem.startswith("OUT."):
        stem = stem[len("OUT."):]
    if stem.endswith("_VTV010"):
        stem = stem[: -len("_VTV010")]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()


def to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location="cpu")


def discover_checkpoints(emulator_dir: Path, strict: bool = True) -> Dict[str, Path]:
    """Map line name -> checkpoint path, insisting that all 17 windows are present."""
    checkpoints = {
        checkpoint_line_name(path): path
        for path in sorted(Path(emulator_dir).glob("emulator_OUT.*_VTV010.pth"))
    }
    unknown = sorted(set(checkpoints) - set(LINE_LABELS))
    missing = sorted(set(LINE_LABELS) - set(checkpoints))
    if unknown:
        raise RuntimeError(f"No scientific label is defined for: {unknown}")
    if missing and strict:
        raise RuntimeError(f"Missing original emulator checkpoints for: {missing}")
    return checkpoints


def build_model(state: dict, device: torch.device) -> DeepONetModel:
    """Rebuild the network stored in a checkpoint and load its weights."""
    model = _architecture_from_checkpoint(state.get("config") or {}).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def common_test_model_ids(states: Dict[str, dict]) -> List[int]:
    """Model ids present in the test partition of *every* checkpoint.

    The split is deterministic in ``N``, and ``N`` can differ by one or two
    models between lines if a window is missing for a few models, so the
    intersection is taken rather than assumed.
    """
    common: Optional[set] = None
    for state in states.values():
        ids = set(to_numpy(state["model_ids_test"]).astype(int).tolist())
        common = ids if common is None else common & ids
    if not common:
        raise RuntimeError("The checkpoints have no common test models.")
    return sorted(common)


def parameters_for_model(state: dict, model_id: int) -> np.ndarray:
    model_ids = to_numpy(state["model_ids_test"]).astype(int)
    positions = np.flatnonzero(model_ids == model_id)
    if len(positions) != 1:
        raise RuntimeError(f"Model {model_id} occurs {len(positions)} times in a checkpoint.")
    return to_numpy(state["X_params_test"])[positions[0]].astype(np.float32)


def load_fastwind_profile(models_dir: Path, model_id: int, line_file: str):
    """Read the raw FASTWIND profile that the emulator is being compared against."""
    path = Path(models_dir) / str(model_id) / line_file
    data = np.loadtxt(path, max_rows=LINE_MAX_ROWS)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] != LINE_MAX_ROWS:
        raise RuntimeError(f"Expected {LINE_MAX_ROWS} rows in {path}, found {data.shape[0]}.")
    return data[:, 2].astype(np.float32), data[:, -1].astype(np.float32)


def predict_flux(model: DeepONetModel, state: dict, parameters: np.ndarray,
                 wavelengths: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict one profile on an arbitrary wavelength grid, in flux units."""
    lambda_min = float(state["lambda_min"])
    lambda_max = float(state["lambda_max"])
    coordinates = (wavelengths - lambda_min) / (lambda_max - lambda_min)
    parameter_tensor = torch.as_tensor(parameters[None, :], dtype=torch.float32, device=device)
    coordinate_tensor = torch.as_tensor(coordinates[None, :], dtype=torch.float32, device=device)
    with torch.inference_mode():
        residual = model(parameter_tensor, coordinate_tensor)[0]
    return residual.detach().cpu().numpy() + FLUX_OFFSET


def predict_full_test_set(model: DeepONetModel, state: dict, device: torch.device,
                          batch_size: int = 512):
    """Predict the whole test partition on its native grids.

    Returns ``(flux_true, flux_pred, coordinates, model_ids)`` with the residual
    offset already undone, so both fluxes are continuum-normalised.
    """
    parameters = to_numpy(state["X_params_test"]).astype(np.float32)
    coordinates = to_numpy(state["X_waves_test"]).astype(np.float32)
    residual_true = to_numpy(state["Y_flux_test"]).astype(np.float32)
    model_ids = to_numpy(state["model_ids_test"]).astype(int)

    prediction_batches = []
    with torch.inference_mode():
        for start in range(0, len(parameters), batch_size):
            stop = start + batch_size
            parameter_tensor = torch.as_tensor(parameters[start:stop], dtype=torch.float32, device=device)
            coordinate_tensor = torch.as_tensor(coordinates[start:stop], dtype=torch.float32, device=device)
            prediction_batches.append(model(parameter_tensor, coordinate_tensor).cpu().numpy())
    residual_prediction = np.concatenate(prediction_batches, axis=0)
    return residual_true + FLUX_OFFSET, residual_prediction + FLUX_OFFSET, coordinates, model_ids


@lru_cache(maxsize=None)
def parse_fastwind_parameters(models_dir: Path, model_id: int) -> Dict[str, float]:
    """The five physical parameters of one model, straight from its INDAT deck."""
    path = Path(models_dir) / str(model_id) / "INDAT.DAT"
    if not path.exists():
        path = Path(models_dir) / str(model_id) / "INDAT"
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    teff, logg, radius = map(float, lines[3].split()[:3])
    mass_loss = float(lines[5].split()[0])
    helium = float(lines[6].split()[0])
    return {"Teff": teff, "logg": logg, "R": radius, "Mdot": mass_loss, "Y_He": helium}


def regular_grid(wavelengths: np.ndarray, spacing: float) -> np.ndarray:
    """Uniform grid with the given spacing spanning the native window."""
    lower = float(np.min(wavelengths))
    upper = float(np.max(wavelengths))
    return np.arange(lower, upper + spacing * 1.0e-6, spacing, dtype=np.float32)


def mean_absolute_relative_error_percent(flux_true: np.ndarray, flux_prediction: np.ndarray,
                                         epsilon: float = 1.0e-8) -> np.ndarray:
    """Per-model MARE in per cent: mean over wavelength of |dF| / (|F| + eps)."""
    return 100.0 * np.mean(
        np.abs(flux_prediction - flux_true) / (np.abs(flux_true) + epsilon), axis=1)


def physical_parameters_from_state(state: dict) -> Dict[str, np.ndarray]:
    """Invert the stored normalised test inputs back to the five varied parameters.

    Column 4 (``v_inf``) is skipped because it is an affine function of Teff, and
    column 6 (``v_turb``) because it is identically zero after normalisation.
    Mdot is inverted in the log, matching how it was normalised.
    """
    x_params = to_numpy(state["X_params_test"]).astype(float)
    param_mins = to_numpy(state["param_mins"]).astype(float)
    param_maxs = to_numpy(state["param_maxs"]).astype(float)
    param_ranges = np.where(param_maxs - param_mins == 0.0, 1.0, param_maxs - param_mins)

    teff = x_params[:, 0] * param_ranges[0] + param_mins[0]
    logg = x_params[:, 1] * param_ranges[1] + param_mins[1]
    radius = x_params[:, 2] * param_ranges[2] + param_mins[2]
    log_mdot_min = np.log10(param_mins[3])
    log_mdot_max = np.log10(param_maxs[3])
    log_mdot = x_params[:, 3] * (log_mdot_max - log_mdot_min) + log_mdot_min
    helium = x_params[:, 5] * param_ranges[5] + param_mins[5]

    return {"Teff": teff / 1000.0, "logg": logg, "R": radius,
            "logMdot": log_mdot, "Y_He": helium}


# =====================================================================================
# SECTION 3.  FIGURE BLOCKS
# =====================================================================================

# -------------------------------------------------------------------------------------
# FIGURE: loss-curves
# -------------------------------------------------------------------------------------
# What it shows: the weighted mean-squared error of Equation (weighted line loss)
# on the training and validation partitions as a function of epoch, on a
# logarithmic y axis, for one line emulator.  The gap between the two curves is
# the overfitting diagnostic; the epoch at which the validation curve flattens is
# where early stopping (patience 40) eventually triggers, and the checkpoint that
# was kept is the minimum of the validation curve, not the last epoch.
# Thesis: fig:five_parameter_loss_curves (H-alpha + He II 6527) and appendix
#         fig:five_parameter_all_training_1 ... _9 for all 17 lines.
# Source: plot_original_5par_emulator.save_loss_plot; notebook cell 11 plotted
#         the same two histories for one chosen line in stacked subplots.
# -------------------------------------------------------------------------------------
def save_loss_plot(state: dict, line_file: str, output_dir: Path, file_format: str,
                   skip_existing: bool = False) -> None:
    output_path = output_dir / f"{safe_filename(line_file)}.{file_format}"
    if skip_existing and output_path.exists():
        return
    training = np.asarray(state["train_loss_history"], dtype=float)
    validation = np.asarray(state["val_loss_history"], dtype=float)
    epochs = np.arange(1, len(training) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(epochs, training, label="Training")
    ax.semilogy(epochs, validation, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted mean-squared error")
    ax.set_title(LINE_LABELS[line_file])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: profiles
# -------------------------------------------------------------------------------------
# What it shows: the raw FASTWIND profile of one test model (blue, on its native
# irregular grid) with the emulator prediction over-plotted (orange).  Four
# variants per line and model: on the native grid, and on regular 0.20, 0.10 and
# 0.05 Angstrom grids.  The regular-grid panels are the point of the whole
# coordinate-based construction: the network was never trained on a regular grid,
# yet it can be queried at any wavelength inside the window, which is what an
# instrument-matched spectrum requires.
# Thesis: fig:five_parameter_native_profiles, fig:five_parameter_regular_01_profiles,
#         and appendix fig:five_parameter_native_848 / _grid_010_848 /
#         _grid_020_848 / _grid_005_848 (test model 848).
# Source: plot_original_5par_emulator.save_profile_plot.
# -------------------------------------------------------------------------------------
def save_profile_plot(raw_wavelength: np.ndarray, raw_flux: np.ndarray,
                      prediction_wavelength: np.ndarray, prediction_flux: np.ndarray,
                      line_file: str, model_id: int, case_directory: Path,
                      case_title: str, file_format: str,
                      skip_existing: bool = False) -> None:
    output_path = case_directory / f"model_{model_id}_{safe_filename(line_file)}.{file_format}"
    if skip_existing and output_path.exists():
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(raw_wavelength, raw_flux, "o", ms=3.0, linestyle="none", color="tab:blue",
            label="FASTWIND raw data")
    ax.plot(prediction_wavelength, prediction_flux, ".", color="tab:orange", ms=3.0,
            linestyle="none", label="Emulator prediction")
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Normalised flux")
    ax.set_title(f"{LINE_LABELS[line_file]} - {case_title}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: relative-error  (mode "adaptive", the thesis version)
# -------------------------------------------------------------------------------------
# What it shows: the signed relative error 100*(F_FW - F_emu)/(F_FW + 1e-6) of
# every test model at every native wavelength point, as a thin scatter cloud,
# summarised by the median and mean and by the 16-84 and 2.5-97.5 percentile
# bands.  Errors concentrate in the line core, where the profile changes fastest
# and where the diagnostic information sits.
#
# Binning: the wavelength arrays differ slightly between models, so grid index j
# is NOT the same wavelength in every model.  The statistics are therefore taken
# in equal-count (quantile) bins of the pooled *physical* wavelength, which also
# preserves the dense FASTWIND sampling near the core instead of smearing it.
# Thesis: fig:five_parameter_relative_error and appendix
#         fig:five_parameter_all_relative_1 ... _9.
# Source: plot_original_5par_emulator.save_relative_error_wavelength_plot.
# -------------------------------------------------------------------------------------
def save_relative_error_wavelength_plot(state: dict, flux_true: np.ndarray,
                                        flux_prediction: np.ndarray, coordinates: np.ndarray,
                                        line_file: str, output_dir: Path, file_format: str,
                                        n_bins: int, skip_existing: bool = False) -> None:
    output_path = output_dir / f"{safe_filename(line_file)}.{file_format}"
    if skip_existing and output_path.exists():
        return
    epsilon = 1.0e-6
    relative_error = 100.0 * (flux_true - flux_prediction) / (flux_true + epsilon)
    lambda_min = float(state["lambda_min"])
    lambda_max = float(state["lambda_max"])
    wavelengths = lambda_min + coordinates * (lambda_max - lambda_min)

    flat_wavelength = wavelengths.ravel()
    flat_error = relative_error.ravel()
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(flat_wavelength, quantiles))
    if len(bin_edges) < 2:
        raise RuntimeError(f"Cannot construct wavelength bins for {line_file}.")

    bin_index = np.searchsorted(bin_edges, flat_wavelength, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(bin_edges) - 2)
    wavelength, median, mean, p16, p84, p2p5, p97p5 = [], [], [], [], [], [], []
    for index in range(len(bin_edges) - 1):
        in_bin = bin_index == index
        if not np.any(in_bin):
            continue
        bin_wavelengths = flat_wavelength[in_bin]
        bin_errors = flat_error[in_bin]
        wavelength.append(float(np.median(bin_wavelengths)))
        median.append(float(np.median(bin_errors)))
        mean.append(float(np.mean(bin_errors)))
        percentiles = np.percentile(bin_errors, [2.5, 16.0, 84.0, 97.5])
        p2p5.append(float(percentiles[0]))
        p16.append(float(percentiles[1]))
        p84.append(float(percentiles[2]))
        p97p5.append(float(percentiles[3]))

    wavelength = np.asarray(wavelength)
    median = np.asarray(median)
    mean = np.asarray(mean)
    p16, p84 = np.asarray(p16), np.asarray(p84)
    p2p5, p97p5 = np.asarray(p2p5), np.asarray(p97p5)

    total_points = flat_error.size
    max_points = 8000
    rng = np.random.default_rng(42)   # fixed seed: the scatter cloud is reproducible
    subset = rng.choice(total_points, size=min(max_points, total_points), replace=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.scatter(flat_wavelength[subset], flat_error[subset], s=3, alpha=0.12,
               color="tab:blue", label="Test-model samples")
    ax.plot(wavelength, median, color="red", lw=2.0, label="Median")
    ax.plot(wavelength, mean, color="darkorange", lw=1.5, ls="--", label="Mean")
    ax.fill_between(wavelength, p16, p84, alpha=0.30, label=r"16--84 percentile")
    ax.fill_between(wavelength, p2p5, p97p5, alpha=0.18, label=r"2.5--97.5 percentile")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"{LINE_LABELS[line_file]} - relative error on the native grid")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: relative-error  (mode "per-index", the notebook version)
# -------------------------------------------------------------------------------------
# The earlier notebook version of the same figure.  It computes the percentiles
# along the model axis at each fixed grid *index* j and plots them against the
# wavelength array of the first test model.  That is simpler but implicitly
# assumes index j is the same wavelength in every model, which is only
# approximately true - which is exactly why the thesis version above rebins in
# physical wavelength.  Both are kept so the difference is reproducible.
# Note this variant uses (F_FW - F_emu)/(F_FW + 1e-6) as a *fraction*, not a
# percentage, matching notebook cells 16 and 17.
# Source: fw_emulator_per_line_comparison_hg.ipynb cells 16-17.
# -------------------------------------------------------------------------------------
def compute_rel_err_stats_per_index(state: dict, flux_true: np.ndarray,
                                    flux_prediction: np.ndarray) -> Dict[str, np.ndarray]:
    lam_min = float(state["lambda_min"])
    lam_max = float(state["lambda_max"])
    waves_norm_1d = to_numpy(state["X_waves_test"])[0]
    lam_phys = lam_min + waves_norm_1d * (lam_max - lam_min)

    eps = 1e-6
    rel_err = (flux_true - flux_prediction) / (flux_true + eps)
    p16, p84 = np.percentile(rel_err, [16.0, 84.0], axis=0)
    p2p5, p97p5 = np.percentile(rel_err, [2.5, 97.5], axis=0)
    return {
        "lambda": lam_phys,
        "mean": np.mean(rel_err, axis=0),
        "median": np.median(rel_err, axis=0),
        "p16": p16, "p84": p84, "p2p5": p2p5, "p97p5": p97p5,
        "rel_err_samples": rel_err,
        "model_ids": to_numpy(state["model_ids_test"]).astype(int),
    }


def save_relative_error_per_index_plot(state: dict, flux_true: np.ndarray,
                                       flux_prediction: np.ndarray, line_file: str,
                                       output_dir: Path, file_format: str,
                                       skip_existing: bool = False) -> None:
    output_path = output_dir / f"{safe_filename(line_file)}.{file_format}"
    if skip_existing and output_path.exists():
        return
    stats = compute_rel_err_stats_per_index(state, flux_true, flux_prediction)
    lam = stats["lambda"]
    rel_err = stats["rel_err_samples"]
    n_models, n_wave = rel_err.shape

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    max_points = 8000
    total_points = n_models * n_wave
    if total_points <= max_points:
        for i in range(n_models):
            ax.scatter(lam, rel_err[i], s=3, alpha=0.12)
    else:
        rng = np.random.default_rng(42)
        subset = rng.choice(total_points, size=max_points, replace=False)
        m_idx = subset // n_wave
        w_idx = subset % n_wave
        ax.scatter(lam[w_idx], rel_err[m_idx, w_idx], s=3, alpha=0.12)

    ax.plot(lam, stats["median"], lw=2, label="median", color="red")
    ax.plot(lam, stats["mean"], lw=1.5, ls="--", label="mean", color="orange")
    ax.fill_between(lam, stats["p16"], stats["p84"], alpha=0.30, label="1$\\sigma$ (16-84%)")
    ax.fill_between(lam, stats["p2p5"], stats["p97p5"], alpha=0.18, label="2$\\sigma$ (2.5-97.5%)")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlabel(r"$\lambda$ [$\AA$]")
    ax.set_ylabel(r"$(F_{\rm FW} - F_{\rm emu}) / F_{\rm FW}$")
    ax.set_title(f"Relative error vs wavelength\n{line_file} (FASTWIND-native grid)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: parameter-error
# -------------------------------------------------------------------------------------
# What it shows: for one line, the per-test-model mean absolute relative error
# colour-coded on every pair of the five physical parameters (10 panels).  This
# is where the thesis conclusion "the largest errors occur preferentially in cool
# models with high mass loss" is read off.  Mdot axes are logarithmic because the
# grid samples log10(Mdot) uniformly.
# Source: plot_original_5par_emulator.save_average_relative_error_parameter_plot;
#         notebook cell 15 is the same quantity with a per-model-id accumulation.
# -------------------------------------------------------------------------------------
def save_average_relative_error_parameter_plot(models_dir: Path, model_ids: np.ndarray,
                                               flux_true: np.ndarray,
                                               flux_prediction: np.ndarray, line_file: str,
                                               output_dir: Path, file_format: str,
                                               skip_existing: bool = False) -> None:
    output_path = output_dir / f"{safe_filename(line_file)}.{file_format}"
    if skip_existing and output_path.exists():
        return
    average_error = mean_absolute_relative_error_percent(flux_true, flux_prediction)
    parameter_names = list(PARAMETER_LABELS)
    parameter_rows = [parse_fastwind_parameters(Path(models_dir), int(mid)) for mid in model_ids]
    parameter_values = {
        name: np.asarray([row[name] for row in parameter_rows], dtype=float)
        for name in parameter_names
    }
    pairs = [(parameter_names[i], parameter_names[j])
             for i in range(len(parameter_names))
             for j in range(i + 1, len(parameter_names))]

    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.0), constrained_layout=True)
    used_axes = []
    scatter = None
    for ax, (x_name, y_name) in zip(axes.flat, pairs):
        scatter = ax.scatter(parameter_values[x_name], parameter_values[y_name],
                             c=average_error, s=22, alpha=0.45, cmap="viridis_r")
        ax.set_xlabel(PARAMETER_LABELS[x_name])
        ax.set_ylabel(PARAMETER_LABELS[y_name])
        if x_name == "Mdot":
            ax.set_xscale("log")
        if y_name == "Mdot":
            ax.set_yscale("log")
        ax.grid(alpha=0.15)
        used_axes.append(ax)

    for ax in axes.flat[len(pairs):]:
        ax.remove()
    if scatter is not None:
        colour_bar = fig.colorbar(scatter, ax=used_axes, shrink=0.82, pad=0.02)
        colour_bar.set_label("Mean absolute relative error (%)")
    fig.suptitle(f"{LINE_LABELS[line_file]} - mean absolute relative error per test model",
                 fontsize=15)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: per-line-metrics
# -------------------------------------------------------------------------------------
# What it shows: one point per spectral line giving the test-set MSE (in per cent
# squared) and the test-set MARE (in per cent), averaged over the 2,857 test
# models and the 161 native wavelength points of that line.  These are the
# per-line versions of the combined metrics used to rank the 460 configurations.
# Thesis: fig:five_parameter_mare (MARE panel).
# Source: plot_original_5par_emulator.save_test_metric_per_line_plots.
# -------------------------------------------------------------------------------------
def save_test_metric_per_line_plots(line_files: List[str], mse_percent_squared: List[float],
                                    mare_percent: List[float], output_dir: Path,
                                    file_format: str) -> None:
    x_positions = np.arange(len(line_files))
    labels = [LINE_LABELS[line_file] for line_file in line_files]
    plot_specs = (
        (mse_percent_squared, r"MSE (\%$^2$)", "Test-set MSE per line", "test_set_mse_per_line"),
        (mare_percent, "MARE (%)", "Test-set MARE per line", "test_set_mare_per_line"),
    )
    for values, ylabel, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(x_positions, values, "o", ms=5.0, linestyle="none", color="tab:blue")
        ax.set_xticks(x_positions, labels, rotation=55, ha="right")
        ax.set_xlabel("Spectral line")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename}.{file_format}", dpi=300)
        plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: worst-corner
# -------------------------------------------------------------------------------------
# What it shows: a lower-triangle corner plot of the five independently varied
# parameters.  Grey points are the nominal test partition; red points are the
# worst 1 % of models ranked by model-level MARE, with the cutoff printed in the
# title.  If the red points clustered nowhere in particular the errors would be
# random; in practice they concentrate at low Teff and high Mdot, which is the
# regime where H-alpha goes into wind emission and the profile is most sensitive.
# Thesis: fig:five_parameter_halpha_parameter_error (H-alpha + He II 6527) and
#         appendix fig:five_parameter_worst_corner_* for the remaining lines.
# Source: plot_worst_1_percent_corner_5par.py.
# -------------------------------------------------------------------------------------
def worst_indices(errors: np.ndarray, fraction: float) -> Tuple[np.ndarray, float]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--worst-fraction must lie between 0 and 1.")
    n_worst = max(1, int(math.ceil(fraction * len(errors))))
    indices = np.argsort(errors)[-n_worst:]
    cutoff = float(np.min(errors[indices]))
    return indices, cutoff


def _set_axis_limits(ax, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        pad = 0.5 if lower == 0.0 else abs(lower) * 0.05
    else:
        pad = 0.05 * (upper - lower)
    ax.set_xlim(lower - pad, upper + pad)


def save_corner_plot(parameters: Dict[str, np.ndarray], errors: np.ndarray,
                     worst: np.ndarray, cutoff: float, line_file: str, output_dir: Path,
                     file_format: str, fraction: float) -> None:
    output_path = output_dir / f"{safe_filename(line_file)}.{file_format}"
    n_params = len(CORNER_PARAMETER_KEYS)
    worst_mask = np.zeros(len(errors), dtype=bool)
    worst_mask[worst] = True

    fig, axes = plt.subplots(n_params, n_params, figsize=(11.0, 11.0))
    for row, y_key in enumerate(CORNER_PARAMETER_KEYS):
        for col, x_key in enumerate(CORNER_PARAMETER_KEYS):
            ax = axes[row, col]
            x_values = parameters[x_key]
            y_values = parameters[y_key]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                bins = 24
                ax.hist(x_values[~worst_mask], bins=bins, color="0.72", alpha=0.85)
                ax.hist(x_values[worst_mask], bins=bins, color="tab:red", alpha=0.75)
                _set_axis_limits(ax, x_values)
                ax.set_yticks([])
            else:
                ax.scatter(x_values[~worst_mask], y_values[~worst_mask], s=8,
                           color="0.65", alpha=0.35, linewidths=0)
                ax.scatter(x_values[worst_mask], y_values[worst_mask], s=18,
                           color="tab:red", alpha=0.90, edgecolors="none")
                _set_axis_limits(ax, x_values)
                finite_y = y_values[np.isfinite(y_values)]
                y_lower = float(np.min(finite_y))
                y_upper = float(np.max(finite_y))
                y_pad = 0.05 * (y_upper - y_lower) if y_upper != y_lower else 0.5
                ax.set_ylim(y_lower - y_pad, y_upper + y_pad)
                ax.grid(alpha=0.15)

            if row == n_params - 1:
                ax.set_xlabel(CORNER_PARAMETER_LABELS[x_key], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(CORNER_PARAMETER_LABELS[y_key], fontsize=10)
            elif row != col:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    percent = 100.0 * fraction
    fig.suptitle(
        f"{LINE_LABELS[line_file]} - worst {percent:.0f}% test models in parameter space\n"
        f"red: {len(worst)} models with MARE >= {cutoff:.3f}%",
        fontsize=15, y=0.985,
    )
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="0.65", markersize=6,
                   label="remaining test models"),
        plt.Line2D([], [], marker="o", linestyle="", color="tab:red", markersize=6,
                   label="worst 1%"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.965, 0.955),
               frameon=False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------------------
# FIGURE: family-mse
# -------------------------------------------------------------------------------------
# What it shows: the test-set flux-space MSE per line for the three hand-written
# capacity variants - "original" (branch 128-256-512, trunk 256-512-512, latent
# 128), "latent128" (thinner, same latent size) and "latent64" (thinner, latent
# halved).  This is the capacity-reduction check that motivated keeping the wider
# original network.  Lines for which a family has no checkpoint are left as NaN.
# Source: fw_emulator_per_line_comparison_hg.ipynb cell 12.
# -------------------------------------------------------------------------------------
def figure_family_mse(args: argparse.Namespace, device: torch.device) -> None:
    families = {
        "original": Path(args.emulator_dir),
        "latent128": Path(args.emulator_dir_128) if args.emulator_dir_128 else None,
        "latent64": Path(args.emulator_dir_64) if args.emulator_dir_64 else None,
    }
    rows = []
    for line_file in LINE_LABELS:
        row: Dict[str, object] = {"line": line_file}
        for family, directory in families.items():
            key = f"MSE_{family}"
            if directory is None:
                row[key] = np.nan
                continue
            path = directory / f"emulator_{line_file}.pth"
            try:
                state = load_checkpoint(path)
                model = build_model(state, device)
                flux_true, flux_pred, _, _ = predict_full_test_set(model, state, device,
                                                                   args.batch_size)
                row[key] = float(np.mean((flux_pred - flux_true) ** 2))
            except Exception as e:
                print(f"[{family}] Could not compute MSE for {line_file}: {e}")
                row[key] = np.nan
        rows.append(row)

    mse_df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "family_mse_per_line.csv"
    mse_df.to_csv(csv_path, index=False)

    x = np.arange(len(mse_df))
    labels = [LINE_LABELS[lf] for lf in mse_df["line"]]
    fig, ax = plt.subplots(figsize=(16, 6))
    for family in families:
        ax.plot(x, mse_df[f"MSE_{family}"], "o-", label=family.capitalize())
    ax.set_xticks(x, labels, rotation=90)
    ax.set_ylabel("MSE (flux)")
    ax.set_xlabel("Line")
    ax.set_title("Test-set MSE per line for the three emulator capacity families")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"family_mse_per_line.{args.format}", dpi=300)
    plt.close(fig)
    print(f"Wrote {csv_path} and family_mse_per_line.{args.format}")


# -------------------------------------------------------------------------------------
# FIGURE: line-panels
# -------------------------------------------------------------------------------------
# What it shows: a grid of small panels, one per spectral line, for a single
# stellar model.  Each panel over-plots the native FASTWIND profile with the
# emulator evaluated on the native grid and/or on a BLOeM-like uniform grid of
# 0.2 Angstrom (the LR02 setting of the BLOeM survey).  ``--panel-scope bloem``
# keeps only the lines whose FASTWIND window overlaps 3960-4570 A, which is what
# an LR02 observation would actually contain; ``--panel-scope all`` keeps every
# available line.
# Source: fw_emulator_per_line_comparison_hg.ipynb cells 13 (bloem) and 14 (all).
# -------------------------------------------------------------------------------------
def make_bloem_segment_from_fw_range(lam_min_fw: float, lam_max_fw: float,
                                     dlam: float = BLOEM_DLAM) -> Optional[np.ndarray]:
    """Uniform grid of step ``dlam`` covering the native FASTWIND range."""
    if lam_max_fw <= lam_min_fw or dlam <= 0:
        return None
    n_pts = int(np.floor((lam_max_fw - lam_min_fw) / dlam)) + 1
    return lam_min_fw + dlam * np.arange(n_pts, dtype=np.float32)


def figure_line_panels(args: argparse.Namespace, device: torch.device) -> None:
    emulator_dir = Path(args.emulator_dir)
    models_dir = Path(args.models_dir)
    model_id = int(args.target_model_id)
    grid_mode = args.panel_grid_mode
    if grid_mode not in {"native", "bloem", "both"}:
        raise ValueError(f"--panel-grid-mode must be native|bloem|both, got {grid_mode}")

    saved_paths = sorted(emulator_dir.glob("emulator_*.pth"))
    if not saved_paths:
        raise RuntimeError(f"No saved emulator_*.pth files found in {emulator_dir}.")
    saved_line_names = [p.name[len("emulator_"):-len(".pth")] for p in saved_paths]

    selected: List[str] = []
    for line_name in saved_line_names:
        line_path = models_dir / str(model_id) / line_name
        if not line_path.exists():
            continue
        obs_wave, _ = load_fastwind_line(str(line_path), max_rows=LINE_MAX_ROWS)
        if args.panel_scope == "bloem":
            # Keep only lines whose FASTWIND range overlaps the BLOeM LR02 window.
            if float(obs_wave.max()) < BLOEM_LMIN or float(obs_wave.min()) > BLOEM_LMAX:
                continue
        selected.append(line_name)

    if args.panel_scope == "bloem":
        print(f"{len(selected)} / {len(saved_line_names)} lines overlap the BLOeM window "
              f"[{BLOEM_LMIN:.1f}, {BLOEM_LMAX:.1f}] A AND have FASTWIND data for "
              f"model_id = {model_id}.")
    else:
        print(f"{len(selected)} / {len(saved_line_names)} lines have FASTWIND data for "
              f"model_id = {model_id}.")
    print("Lines used:", selected)
    if not selected:
        raise RuntimeError("No lines available for this model and scope.")

    ncols = 4
    nrows = math.ceil(len(selected) / ncols)
    width, height = (4.0, 3.0) if args.panel_scope == "bloem" else (4.2, 3.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)

    used_axes = 0
    for ax, line_name in zip(axes.flat, selected):
        try:
            line_path = models_dir / str(model_id) / line_name
            obs_wave, obs_flux = load_fastwind_line(str(line_path), max_rows=LINE_MAX_ROWS)
            state = load_checkpoint(emulator_dir / f"emulator_{line_name}.pth")
            model = build_model(state, device)
            parameters = parameters_for_model(state, model_id) if args.panel_params_from_checkpoint \
                else _normalised_params_from_indat(models_dir, model_id, state)

            ax.plot(obs_wave, obs_flux, "o", markersize=2, label="FASTWIND (native)")
            if grid_mode in {"native", "both"}:
                flux_native = predict_flux(model, state, parameters, obs_wave, device)
                ax.plot(obs_wave, flux_native, ".", linewidth=1.0,
                        label="Emulator (native FW grid)")
            if grid_mode in {"bloem", "both"}:
                lam_bloem = make_bloem_segment_from_fw_range(
                    float(obs_wave.min()), float(obs_wave.max()), dlam=args.bloem_dlam)
                if lam_bloem is not None and len(lam_bloem) >= 2:
                    flux_bloem = predict_flux(model, state, parameters, lam_bloem, device)
                    ax.plot(lam_bloem, flux_bloem, ".", markersize=3,
                            label=f"Emulator BLOeM-like (dlam={args.bloem_dlam} A)")

            ax.set_title(line_name, fontsize=9)
            ax.set_xlabel("lambda [Angstrom]")
            ax.set_ylabel("Flux")
            ax.grid(True, alpha=0.3)
            if used_axes == 0:
                ax.legend(fontsize=7)
            used_axes += 1
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=7)
            ax.set_title(line_name, fontsize=9)

    for ax in axes.flat[used_axes:]:
        ax.axis("off")

    grid_map = {"native": "native FASTWIND grid", "bloem": "BLOeM-like grid",
                "both": "native FASTWIND + BLOeM-like grids"}
    scope_map = {"bloem": "lines overlapping BLOeM window", "all": "ALL available lines"}
    fig.suptitle(f"Original emulator - {scope_map[args.panel_scope]} - model_id = {model_id}\n"
                 f"Emulator on {grid_map[grid_mode]}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"line_panels_{args.panel_scope}_{grid_mode}_model{model_id}.{args.format}"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def _normalised_params_from_indat(models_dir: Path, model_id: int, state: dict) -> np.ndarray:
    """Normalise a model's INDAT parameters with the checkpoint's own scaling.

    Used when the requested model is not in the test partition stored in the
    checkpoint (the notebook read them from ``model_param_norm``, which was built
    over the whole grid).
    """
    from five_parameter_training import normalize_params, read_indat_params
    params = read_indat_params(str(Path(models_dir) / str(model_id)))
    return normalize_params(params, state["param_mins"], state["param_maxs"])


# =====================================================================================
# SECTION 4.  FIGURE DRIVERS
# =====================================================================================
# The blocks of Section 3 are deliberately "one call, one file".  The drivers below are
# what the preserved scripts did around them inside their ``main``: open every
# checkpoint once, rebuild each network once, and write one file per line into the same
# sub-directory the original script used, so an existing ``5_line_emulator_plot`` tree
# keeps exactly the layout the thesis \includegraphics paths expect.
# ``figure_family_mse`` and ``figure_line_panels`` (Section 3) are already drivers of
# this kind, because the notebook cells they come from were themselves whole-figure
# cells rather than reusable functions.
# -------------------------------------------------------------------------------------

def selected_line_files(args: argparse.Namespace) -> List[str]:
    """Which line windows to plot: all 17, unless ``--lines`` restricts the set."""
    if not args.lines:
        return list(LINE_LABELS)
    unknown = [line_file for line_file in args.lines if line_file not in LINE_LABELS]
    if unknown:
        raise SystemExit(f"No scientific label is defined for: {unknown}")
    return list(args.lines)


def load_states(args: argparse.Namespace, line_files: Sequence[str]) -> Dict[str, dict]:
    """Load the checkpoints of the requested lines.

    All 17 windows must be present unless ``--lines`` asked for a subset, which is
    the strictness ``plot_original_5par_emulator.discover_checkpoints`` enforced.
    """
    checkpoints = discover_checkpoints(Path(args.emulator_dir), strict=not args.lines)
    missing = [line_file for line_file in line_files if line_file not in checkpoints]
    if missing:
        raise SystemExit(f"No checkpoint in {args.emulator_dir} for: {missing}")
    return {line_file: load_checkpoint(checkpoints[line_file]) for line_file in line_files}


def figure_loss_curves(args: argparse.Namespace, device: torch.device) -> None:
    """One training/validation curve per line, into ``train_validation_loss/``."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    loss_dir = Path(args.output_dir) / "train_validation_loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    for line_file in line_files:
        save_loss_plot(states[line_file], line_file, loss_dir, args.format, args.skip_existing)
    print(f"Wrote {len(line_files)} loss curves to {loss_dir}")


def figure_profiles(args: argparse.Namespace, device: torch.device) -> None:
    """FASTWIND versus emulator for ``--num-models`` common test models, four grids each."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    available_ids = common_test_model_ids(states)
    if args.num_models > len(available_ids):
        raise SystemExit(f"Requested {args.num_models} models, but only "
                         f"{len(available_ids)} are common to all checkpoints.")
    selected_ids = sorted(random.Random(args.seed).sample(available_ids, args.num_models))

    prediction_root = Path(args.output_dir) / "predictions"
    case_dirs: Dict[str, Path] = {}
    for case_name in GRID_CASES:
        case_dirs[case_name] = prediction_root / case_name
        case_dirs[case_name].mkdir(parents=True, exist_ok=True)

    print(f"Random seed: {args.seed}")
    print(f"Selected common test models: {selected_ids}")

    for line_file in line_files:
        state = states[line_file]
        model = build_model(state, device)
        for model_id in selected_ids:
            raw_wave, raw_flux = load_fastwind_profile(Path(args.models_dir), model_id, line_file)
            parameters = parameters_for_model(state, model_id)
            for case_name, (spacing, case_title) in GRID_CASES.items():
                profile_path = (case_dirs[case_name]
                                / f"model_{model_id}_{safe_filename(line_file)}.{args.format}")
                if args.skip_existing and profile_path.exists():
                    continue
                prediction_wave = raw_wave if spacing is None else regular_grid(raw_wave, spacing)
                prediction_flux = predict_flux(model, state, parameters, prediction_wave, device)
                save_profile_plot(raw_wave, raw_flux, prediction_wave, prediction_flux,
                                  line_file, model_id, case_dirs[case_name], case_title,
                                  args.format, args.skip_existing)
        print(f"Finished {line_file}")

    total_predictions = len(line_files) * len(selected_ids) * len(GRID_CASES)
    print(f"Wrote {total_predictions} profile plots under {prediction_root}")


def figure_relative_error(args: argparse.Namespace, device: torch.device) -> None:
    """Signed relative error versus wavelength, in either binning mode."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    adaptive = args.relative_error_mode == "adaptive"
    output_dir = Path(args.output_dir) / ("rel_error_wav" if adaptive
                                          else "rel_error_wav_per_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    for line_file in line_files:
        state = states[line_file]
        model = build_model(state, device)
        flux_true, flux_prediction, coordinates, _ = predict_full_test_set(
            model, state, device, args.batch_size)
        if adaptive:
            save_relative_error_wavelength_plot(state, flux_true, flux_prediction, coordinates,
                                                line_file, output_dir, args.format,
                                                args.relative_error_bins, args.skip_existing)
        else:
            save_relative_error_per_index_plot(state, flux_true, flux_prediction, line_file,
                                               output_dir, args.format, args.skip_existing)
        print(f"Finished {line_file}")

    print(f"Wrote {len(line_files)} relative-error plots "
          f"({args.relative_error_mode} binning) to {output_dir}")


def figure_per_line_metrics(args: argparse.Namespace, device: torch.device) -> None:
    """Test-set MSE and MARE of every line, on one axis each."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mse_percent_squared: List[float] = []
    mare_percent: List[float] = []
    for line_file in line_files:
        state = states[line_file]
        model = build_model(state, device)
        flux_true, flux_prediction, _, _ = predict_full_test_set(
            model, state, device, args.batch_size)
        mse_percent_squared.append(float(np.mean((100.0 * (flux_prediction - flux_true)) ** 2)))
        mare_percent.append(float(100.0 * np.mean(
            np.abs(flux_prediction - flux_true) / (np.abs(flux_true) + 1.0e-8))))
        print(f"Finished {line_file}")

    save_test_metric_per_line_plots(line_files, mse_percent_squared, mare_percent,
                                    output_dir, args.format)
    print(f"Wrote test_set_mse_per_line.{args.format} and "
          f"test_set_mare_per_line.{args.format} to {output_dir}")


def figure_parameter_error(args: argparse.Namespace, device: torch.device) -> None:
    """Per-model MARE colour-coded on every pair of physical parameters."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    output_dir = Path(args.output_dir) / "avg_rel_err_per_line_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for line_file in line_files:
        state = states[line_file]
        model = build_model(state, device)
        flux_true, flux_prediction, _, test_model_ids = predict_full_test_set(
            model, state, device, args.batch_size)
        save_average_relative_error_parameter_plot(Path(args.models_dir), test_model_ids,
                                                   flux_true, flux_prediction, line_file,
                                                   output_dir, args.format, args.skip_existing)
        print(f"Finished {line_file}")

    print(f"Wrote {len(line_files)} parameter-error plots to {output_dir}")


def figure_worst_corner(args: argparse.Namespace, device: torch.device) -> None:
    """Corner plot of the worst ``--worst-fraction`` of the test partition."""
    line_files = selected_line_files(args)
    states = load_states(args, line_files)
    output_dir = Path(args.output_dir) / "worst_1_percent_corner_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for line_file in line_files:
        state = states[line_file]
        model = build_model(state, device)
        flux_true, flux_prediction, _, _ = predict_full_test_set(
            model, state, device, args.batch_size)
        errors = mean_absolute_relative_error_percent(flux_true, flux_prediction)
        worst, cutoff = worst_indices(errors, args.worst_fraction)
        parameters = physical_parameters_from_state(state)
        save_corner_plot(parameters, errors, worst, cutoff, line_file, output_dir,
                         args.format, args.worst_fraction)
        print(f"{line_file}: {len(worst)} worst models, "
              f"cutoff MARE = {cutoff:.4f}%, max MARE = {float(np.max(errors)):.4f}%")

    print(f"Wrote {len(line_files)} corner plots to {output_dir}")


# =====================================================================================
# SECTION 5.  THE PARALLEL-COORDINATE PIPELINE
# =====================================================================================
# These three families are the only ones that never open an ``emulator_*.pth`` of the
# adopted run: they walk the whole architecture-search output tree (export-metadata), or
# they read the ``parallel_plot_metadata.json`` that the walk produces (parallel,
# parallel-interactive).  Their bodies are the one part of the consolidation that is not
# yet inlined here; they are loaded from the preserved scripts next to this file so that
# fig:five_parameter_parallel_all and fig:five_parameter_parallel_best cannot drift while
# that port is outstanding.  Every default below is the module constant of the script it
# drives, repeated in Section 1.
# -------------------------------------------------------------------------------------

def load_preserved_script(name: str, legacy_dir: Path):
    """Import one of the preserved ``5_par`` scripts as a module, by file path."""
    import sys
    from importlib import util as importlib_util

    path = Path(legacy_dir) / f"{name}.py"
    if not path.is_file():
        raise SystemExit(f"Cannot find {path}.  Point --legacy-dir at the folder that "
                         f"holds {name}.py.")
    spec = importlib_util.spec_from_file_location(f"_preserved_{name}", path)
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def figure_parallel(args: argparse.Namespace, device: torch.device) -> None:
    """The two thesis-styled parallel-coordinate figures, all and best-highlighted."""
    thesis = load_preserved_script("parallel_plot_thesis", args.legacy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = thesis.load_data(args.metadata)
    for stem, highlighted in (("parallel_hyperparams_thesis_all", False),
                              ("parallel_hyperparams_thesis_best", True)):
        output = output_dir / stem
        thesis.draw_static(data, output, highlighted)
        thesis.draw_html(data, output, highlighted)
        print(f"Wrote {output}.pdf, {output}.png and {output}.html")
    best = data.loc[data["mare"].idxmin()]
    print(f"rows={len(data)} best={best['architecture']} MARE={best['mare']:.12g}")


def figure_parallel_interactive(args: argparse.Namespace, device: torch.device) -> None:
    """The exploratory dark-themed Plotly version, at architecture and line level."""
    interactive = load_preserved_script("parallel_plot_from_metadata", args.legacy_dir)
    metadata = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_directory = Path.cwd()
    os.chdir(output_dir)  # the preserved script writes its HTML into the working directory
    try:
        html_path, n_architectures = interactive.plot_architecture_level(str(metadata))
        print(f"Wrote {html_path} ({n_architectures} architectures)")
        _, saved_paths, manifest = interactive.plot_line_level(str(metadata), verbose=True)
        print(f"Wrote {len(saved_paths)} per-line HTML files and {manifest}")
    finally:
        os.chdir(previous_directory)


def figure_export_metadata(args: argparse.Namespace, device: torch.device) -> None:
    """Not a figure: benchmark every checkpoint and write parallel_plot_metadata.json."""
    export = load_preserved_script("export_parallel_plot_metadata", args.legacy_dir)
    export.BASE_PATH = str(args.search_root)
    export.HG_DIR = str(args.hg_dir)
    export.CACHE_PATH = str(args.metrics_cache)
    export.OUTPUT_PATH = str(args.metadata)
    export.TIMING_CACHE_PATH = str(args.timing_cache)
    export.REFERENCE_MODEL_PARAMS_PATH = str(args.reference_params)
    # ``load_reference_model_params`` captured the old constant as a default argument,
    # so the module attribute alone is not enough to redirect it.
    export.load_reference_model_params.__defaults__ = (str(args.reference_params),)
    export.main()
    print(f"Wrote {args.metadata} and {args.timing_cache}")


# =====================================================================================
# SECTION 6.  COMMAND-LINE INTERFACE
# =====================================================================================

#: Every figure family, in the order of the FIGURE MAP in the module docstring.
FIGURE_DISPATCH = {
    "loss-curves": figure_loss_curves,
    "profiles": figure_profiles,
    "relative-error": figure_relative_error,
    "per-line-metrics": figure_per_line_metrics,
    "parameter-error": figure_parameter_error,
    "worst-corner": figure_worst_corner,
    "parallel": figure_parallel,
    "parallel-interactive": figure_parallel_interactive,
    "family-mse": figure_family_mse,
    "line-panels": figure_line_panels,
    "export-metadata": figure_export_metadata,
}

#: What ``--all`` runs: every family that needs nothing but the checkpoints and the
#: FASTWIND models.  The parallel-coordinate families are left out because they need
#: ``parallel_plot_metadata.json``, which ``--figure export-metadata`` produces in a
#: separate, hours-long pass over the architecture-search tree.
ALL_FIGURES: Tuple[str, ...] = (
    "loss-curves", "profiles", "relative-error", "per-line-metrics",
    "parameter-error", "worst-corner", "family-mse", "line-panels",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="five_parameter_plots.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--figure", choices=tuple(FIGURE_DISPATCH),
                          help="Which figure family to produce.  See the FIGURE MAP in "
                               "the module docstring.")
    selector.add_argument("--all", action="store_true",
                          help="Produce every figure that reads only the checkpoints: "
                               + ", ".join(ALL_FIGURES) + ".")

    # --- data / output -------------------------------------------------------
    p.add_argument("--emulator-dir", type=Path, default=DEFAULT_EMULATORS,
                   help="Folder of adopted emulator_*.pth checkpoints (17 windows).")
    p.add_argument("--emulator-dir-128", dest="emulator_dir_128", type=Path,
                   default=DEFAULT_ROOT / "emulators_per_line_128_hg",
                   help="The notebook's EMULATOR_DIR_128, for --figure family-mse.")
    p.add_argument("--emulator-dir-64", dest="emulator_dir_64", type=Path,
                   default=DEFAULT_ROOT / "emulators_per_line_64_hg",
                   help="The notebook's EMULATOR_DIR_64, for --figure family-mse.")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS,
                   help="Directory holding one numbered subdirectory per FASTWIND model. "
                        "Needed by profiles, parameter-error and line-panels.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Root of the figure tree; each family writes into its own "
                        "sub-directory of it.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--format", choices=("pdf", "png"), default="pdf",
                   help="File format of the matplotlib figures.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Leave a figure alone if its file already exists.")
    p.add_argument("--lines", nargs="+", default=None,
                   help="Restrict the per-line families to these OUT.* windows "
                        "(default: all 17).")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=512,
                   help="Test-set prediction batch size.")

    # --- profiles ------------------------------------------------------------
    p.add_argument("--num-models", dest="num_models", type=int, default=3,
                   help="How many common test models to draw profiles for.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed of the random choice of those test models.")

    # --- relative-error ------------------------------------------------------
    p.add_argument("--relative-error-mode", dest="relative_error_mode",
                   choices=("adaptive", "per-index"), default="adaptive",
                   help="'adaptive' rebins in physical wavelength (the thesis figure); "
                        "'per-index' takes the statistics at fixed grid index, as the "
                        "notebook did.")
    p.add_argument("--relative-error-bins", dest="relative_error_bins", type=int,
                   default=161,
                   help="Number of adaptive physical-wavelength bins used for the "
                        "wavelength-dependent relative-error summaries (default: 161).")

    # --- worst-corner --------------------------------------------------------
    p.add_argument("--worst-fraction", dest="worst_fraction", type=float, default=0.01,
                   help="Fraction of the test partition marked as the worst cases.")

    # --- line-panels ---------------------------------------------------------
    p.add_argument("--target-model-id", dest="target_model_id", type=int, default=1,
                   help="The FASTWIND model whose spectrum the panels show.")
    p.add_argument("--panel-scope", dest="panel_scope", choices=("bloem", "all"),
                   default="bloem",
                   help="'bloem' keeps only the lines overlapping the BLOeM LR02 window, "
                        "'all' keeps every available line.")
    p.add_argument("--panel-grid-mode", dest="panel_grid_mode",
                   choices=("native", "bloem", "both"), default="both",
                   help="Which grids the emulator is evaluated on in the panels.")
    p.add_argument("--panel-params-from-checkpoint", dest="panel_params_from_checkpoint",
                   action="store_true",
                   help="Take the branch inputs from the checkpoint's test partition "
                        "instead of normalising the model's INDAT deck; only possible "
                        "if --target-model-id is in the test partition of every line.")
    p.add_argument("--bloem-dlam", dest="bloem_dlam", type=float, default=BLOEM_DLAM,
                   help="Spacing in Angstrom of the BLOeM-like uniform grid.")

    # --- the parallel-coordinate pipeline ------------------------------------
    p.add_argument("--metadata", type=Path, default=Path(EXPORT_OUTPUT_PATH),
                   help="parallel_plot_metadata.json: written by --figure "
                        "export-metadata, read by --figure parallel and "
                        "parallel-interactive.")
    p.add_argument("--search-root", dest="search_root", type=Path,
                   default=Path(EXPORT_BASE_PATH),
                   help="Root of the architecture-search output tree to walk.")
    p.add_argument("--hg-dir", dest="hg_dir", type=Path, default=Path(HG_DIR),
                   help="Folder of the adopted 'original' run, added to the metadata as "
                        f"run {HG_RUN_NAME!r}.")
    p.add_argument("--metrics-cache", dest="metrics_cache", type=Path,
                   default=Path(EXPORT_CACHE_PATH),
                   help="Cache of the already computed test metrics.")
    p.add_argument("--timing-cache", dest="timing_cache", type=Path,
                   default=Path(EXPORT_TIMING_CACHE_PATH),
                   help="Cache of the measured emulator call times.")
    p.add_argument("--reference-params", dest="reference_params", type=Path,
                   default=Path(EXPORT_REFERENCE_MODEL_PARAMS_PATH),
                   help="Parameter table used to build the timing input vector.")
    p.add_argument("--legacy-dir", dest="legacy_dir", type=Path,
                   default=Path(__file__).resolve().parent,
                   help="Folder holding the preserved parallel_plot_thesis.py, "
                        "parallel_plot_from_metadata.py and "
                        "export_parallel_plot_metadata.py.  These three ship "
                        "alongside this file, so the default is this folder.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.num_models < 1:
        raise SystemExit("--num-models must be at least one.")
    if args.relative_error_bins < 2:
        raise SystemExit("--relative-error-bins must be at least two.")

    device = resolve_device(args.device)
    figures = list(ALL_FIGURES) if args.all else [args.figure]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figures: {', '.join(figures)}")

    for name in figures:
        print(f"\n--- {name} ---")
        FIGURE_DISPATCH[name](args, device)

    print(f"\nDone: {len(figures)} figure famil{'y' if len(figures) == 1 else 'ies'} "
          f"written under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
