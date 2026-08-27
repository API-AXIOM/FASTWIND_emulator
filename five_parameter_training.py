#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
five_parameter_training.py
==========================

Training, architecture search and inference for the **five-parameter hydrogen
and helium line emulator** of Chapter 3 of the MSc thesis.

This single file replaces the following research scripts, without changing any
numerical behaviour:

    don_emulator_5_par.py                -> SEARCH_STAGES["stage1_base"]
    don_emulator_5_par_extra.py          -> SEARCH_STAGES["stage2_extra"]
    don_emulator_5_par_timeout.py        -> SEARCH_STAGES["stage3_timeout"]
    don_emulator_5_par_timeout_extra.py  -> SEARCH_STAGES["stage4_timeout_extra"]
    check_of_initials_arch.py            -> BASELINE_ARCHITECTURES (+ --mode baseline)
    fw_emulator_per_line_comparison_hg.ipynb (training cells 0-10)
                                         -> BASELINE_ARCHITECTURES (identical code path)
    emulator_inference.py                -> --mode infer  and the public API at the
                                            bottom of this file

The plotting cells of the notebook and the five plotting scripts live in the
companion file ``five_parameter_plots.py``.


WHAT THE EMULATOR IS
--------------------
FASTWIND is a NLTE model-atmosphere and radiative-transfer code for hot,
massive stars with winds.  One FASTWIND run takes of order tens of minutes of
CPU time and produces, among other things, a set of continuum-normalised
spectral line profiles.  Fitting an observed spectrum requires many thousands
of such evaluations, which is why a fast surrogate ("emulator") is useful.

The surrogate used here is a **DeepONet** (deep operator network).  A DeepONet
approximates an operator - a map from an input *function or parameter vector*
to an output *function* - by factorising it into two subnetworks whose outputs
are combined with an inner product:

        F(theta)(lambda)  ~=  sum_k  b_k(theta) * t_k(lambda)

  * The **branch network** ``b`` sees only the stellar parameters ``theta``.
    Physically it encodes *which star* we are looking at: its output is a
    latent vector of ``latent_dim`` coefficients that describe the atmosphere.
  * The **trunk network** ``t`` sees only the wavelength coordinate
    ``lambda``.  Physically it is a learned, line-specific basis of profile
    shapes evaluated at that wavelength - the analogue of a set of basis
    functions in a spectral expansion.
  * The dot product of the two mixes "which star" with "where in the profile",
    producing the flux at that one wavelength.

WHY WAVELENGTH IS AN INPUT COORDINATE, NOT AN OUTPUT INDEX
----------------------------------------------------------
A naive emulator would output a fixed-length vector of 161 fluxes, i.e. it
would treat the wavelength as an *index* j = 0..160.  That is wrong here for
two physical reasons:

  1. FASTWIND does not use a uniform wavelength grid.  It concentrates points
     near the line centre, where the profile has the most structure, and
     spreads them out in the neighbouring continuum.
  2. The actual wavelength array is **not identical between stellar models**,
     because the sampling adapts to the computed profile.  So "index 80" does
     not correspond to the same physical wavelength in two different models.

Treating lambda as a genuine *input coordinate* of the trunk network solves
both problems at once: every flux value stays paired with its own physical
wavelength, the network learns a continuous function of lambda, and at
prediction time the user may ask for **any** wavelength grid inside the
line window - the native FASTWIND grid, or a regular 0.05/0.10/0.20 Angstrom
grid matching an instrument.  This is the central "coordinate-based" claim
that Chapter 3 sets out to test.

The trunk input is not the raw scalar lambda_norm but a random-free Fourier
feature expansion ``[c, sin(2*pi*n*x), cos(2*pi*n*x)]`` for n = 1..M with
M = ``fourier_modes``.  Plain MLPs have a strong spectral bias and learn
smooth, low-frequency functions much faster than sharp ones; the explicit
Fourier features give the network high-frequency basis functions for free, so
it can represent the narrow line core without needing extreme depth.

WHY ONE EMULATOR PER SPECTRAL LINE
----------------------------------
17 separate DeepONets are trained, one per FASTWIND line window (18 hydrogen
and helium diagnostics; H-alpha and He II 6527 share a window and therefore
share one emulator).  Reasons:

  * The wavelength normalisation is per line: ``lambda`` is mapped to [0,1]
    using that line's own lambda_min/lambda_max.  A single global network
    would have to spend capacity resolving the enormous empty gaps between
    windows.
  * Different lines respond to different physics (H-alpha to the mass-loss
    rate through wind emission, He I/He II ratios to temperature, the Balmer
    wings to gravity).  Per-line networks let each one specialise, and a badly
    behaved line cannot degrade the others.
  * It makes the error budget diagnosable line by line, which is what the
    thesis reports.

WHAT THE CONVERGENCE SELECTION DOES
-----------------------------------
Every FASTWIND run writes a ``CONVERG`` file with one row per atmospheric
(NLTE) iteration; the second column of the last row is the largest remaining
relative correction.  The input decks allow at most 100 iterations.
``classify_model`` reproduces the labelling used to build the grid:

    green  : run stopped before the 100-iteration limit (any final correction),
             OR reached the limit with final correction < 0.01
    yellow : reached the limit with 0.01 <= correction < 0.2
    orange : reached the limit with correction >= 0.2
    red    : CONVERG missing/unreadable, or the H-gamma output file
             (``OUT.HGAMMA_VTV010``) is missing or empty

The H-gamma file is used as a *completeness sentinel*: it is one of the 17
windows written by the spectral-synthesis stage, so an empty or absent H-gamma
means the synthesis did not finish even if the atmosphere iteration did.
Only **green** models are kept.  Out of 19,621 FASTWIND calculations this
leaves **19,044** models (56 yellow, 0 orange, 521 red).  The purpose is to
stop the network from being trained to reproduce unconverged numerical
solutions, i.e. from learning the code's failures as if they were physics.

TRAINING TARGET AND LOSS
------------------------
The network predicts the **residual** ``flux - 1.0`` rather than the flux
itself (``USE_RESIDUAL = True``, ``FLUX_OFFSET = 1.0``).  A continuum-normalised
profile is 1.0 almost everywhere, so predicting the residual removes a large
constant offset and lets all of the network's dynamic range be spent on the
line itself.

The loss is a **core-weighted mean squared error**

    L = mean[ (pred - target)^2 * (1 + ALPHA * |target|) ],  ALPHA = 5.0

so wavelengths where the profile departs strongly from the continuum - the
line core and the wings, i.e. exactly the diagnostic parts - are weighted up
to six times more heavily than the flat continuum.  Without this the optimiser
would happily drive the loss down by fitting the continuum perfectly and
ignoring the core.

SPLIT
-----
``sklearn.model_selection.train_test_split`` is applied twice with
``random_state = SPLIT_SEED = 42``: first ``test_size = 0.30`` to peel off a
temporary block, then ``test_size = 0.50`` on that block.  This gives
**70 % train / 15 % validation / 15 % test** = 13,330 / 2,857 / 2,857 models.
Because the split is a function of the seed and of ``N`` only, every line
emulator receives the *same* stellar models in the same partition.


COMMAND-LINE USAGE
------------------
List every configuration this file can reproduce, with per-stage counts::

    python five_parameter_training.py --mode list-configs

Train the single adopted emulator for one line (this is the code path that
produced the released ``emulators_per_line_hg`` checkpoints)::

    python five_parameter_training.py --mode line \
        --line OUT.HGAMMA_VTV010 \
        --models-root models_LHC \
        --output-dir emulators_per_line_hg

Train all 17 lines for one of the four hand-written baseline families
(this reproduces ``check_of_initials_arch.py`` / the notebook cells)::

    python five_parameter_training.py --mode baseline --baseline original
    python five_parameter_training.py --mode baseline --baseline all

Run one stage of the architecture search (the four ``don_emulator_*`` scripts)::

    python five_parameter_training.py --mode search --stage stage1_base
    python five_parameter_training.py --mode search --stage stage2_extra
    python five_parameter_training.py --mode search --stage stage3_timeout
    python five_parameter_training.py --mode search --stage stage4_timeout_extra
    python five_parameter_training.py --mode search --stage all

The two "timeout" stages were written for a SLURM cluster.  They stop cleanly
before the job walltime and can be resubmitted to continue where they left
off; that behaviour is on by default for those stages and can be forced or
disabled anywhere::

    DON_STOP_BEFORE_TIMEOUT_SECONDS=600 \
    DON_MAX_RUNTIME_SECONDS=7200 \
    python five_parameter_training.py --mode search --stage stage3_timeout \
        --timeout-guard --resume

Predict a spectrum from a set of trained checkpoints (replaces
``emulator_inference.py``)::

    python five_parameter_training.py --mode infer \
        --emulator-dir emulators_per_line_hg \
        --indat models_LHC/848/INDAT \
        --output-npz predicted_spectrum.npz

or, from Python::

    from five_parameter_training import emulate_hg_spectrum_from_indat
    result = emulate_hg_spectrum_from_indat(
        "models_LHC/848/INDAT",
        emulator_dir="emulators_per_line_hg",
        output_wavelength=np.linspace(4000.0, 7000.0, 5000),
    )
    wavelength = result["spectrum"]["wavelength"]
    flux       = result["spectrum"]["flux"]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# The original scripts silenced this warning; keeping it avoids noisy logs on
# multi-GPU nodes where torch probes cuBLAS before a CUDA context exists.
warnings.filterwarnings("ignore", message=".*cuBLAS.*no current CUDA context.*")


# =====================================================================================
# SECTION 1.  FIXED PHYSICS / DATA CONSTANTS
# =====================================================================================
# These never varied in any run.  They are collected here so that a reader can
# see the whole data contract in one place.

#: Portable repository and storage roots.  Command-line options always take
#: precedence, while these environment variables make it possible to keep the
#: large FASTWIND datasets and training outputs outside the Git checkout.
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

#: Root directory holding one subdirectory per FASTWIND model, named by integer id.
DEFAULT_MODELS_ROOT = str(
    Path(os.environ.get("FASTWIND_5PAR_DATA", DATA_ROOT / "five_parameter"))
    / "models_LHC"
)

#: Where the architecture-search runs write their per-configuration subfolders.
DEFAULT_OUTPUT_DIR = str(
    Path(os.environ.get("FASTWIND_5PAR_OUTPUT", OUTPUT_ROOT / "five_parameter"))
    / "fine_tuning_emulator_exploration"
)

#: The seven columns handed to the branch network.  Only five vary independently
#: (Teff, logg, R, Mdot, Y_He).  v_inf is fixed by the LMC relation
#: v_inf = 0.088*Teff - 1200 km/s, so after per-column min-max scaling the
#: normalised v_inf column is *identical* to the normalised Teff column and
#: carries no extra information.  v_turb is fixed at 10 km/s for every model,
#: so its range is zero, is replaced by 1.0 during normalisation, and the whole
#: normalised column is exactly zero.  Both were kept so that the same data
#: pipeline can be reused for datasets in which they do vary (Chapter 4).
PARAM_COLS: Tuple[str, ...] = ("Teff", "logg", "R", "Mdot", "v_inf", "Y_He", "v_turb")

#: Index of Mdot inside PARAM_COLS.  Mdot spans more than two decades, so it is
#: log10-scaled *before* being min-max normalised, unlike the other six columns.
MDOT_INDEX = 3

#: Number of rows read from each ``OUT.*`` line file.  FASTWIND writes 161
#: wavelength/flux pairs followed by a footer that must not be parsed.
LINE_MAX_ROWS = 161

#: A model contributing fewer than this many wavelength points to a line is dropped.
MIN_POINTS_PER_LINE = 20

#: Sentinel file used by the convergence screening (see the module docstring).
CONVERGENCE_SENTINEL_LINE = "OUT.HGAMMA_VTV010"

#: Residual-target settings.  The network predicts (flux - FLUX_OFFSET).
USE_RESIDUAL = True
FLUX_OFFSET = 1.0

#: Core-weighting strength of the loss:  w = 1 + ALPHA * |residual|.
ALPHA = 5.0

#: Train / validation / test split.  Applied as two successive calls to
#: sklearn.train_test_split with this seed: 0.30 then 0.50 of the remainder,
#: i.e. 70 % / 15 % / 15 % -> 13330 / 2857 / 2857 models.
SPLIT_SEED = 42
SPLIT_TEST_SIZE_FIRST = 0.30
SPLIT_TEST_SIZE_SECOND = 0.50

#: Global torch seed.  ``torch.manual_seed(0)`` was executed in the notebook and
#: in check_of_initials_arch.py (cell 2) before any model was built, and it
#: therefore applies to the four BASELINE_ARCHITECTURES.  The four
#: don_emulator_5_par*.py search scripts never called torch.manual_seed, so
#: their weight initialisation and DataLoader shuffling used torch's default
#: nondeterministic seeding.  This asymmetry is preserved: each config below
#: carries its own ``torch_manual_seed`` (0 or None).
BASELINE_TORCH_SEED = 0

#: LR scheduler: ReduceLROnPlateau on the validation loss.
LR_SCHEDULER = {"mode": "min", "factor": 0.5, "patience": 10}

#: Optimiser.  Adam with the config's learning rate; weight decay was always 0.
OPTIMISER = "adam"
WEIGHT_DECAY = 0.0

#: An epoch counts as an improvement only if it beats the best validation loss
#: by more than this absolute margin.
EARLY_STOP_MIN_DELTA = 1e-6

#: Whether nn.DataParallel is used is decided at runtime from the GPU count.
USE_DATA_PARALLEL_IF_MULTI_GPU = True


# =====================================================================================
# SECTION 2.  THE FULL SEARCH SPACE
# =====================================================================================
# Every architecture and hyperparameter combination that was actually run is
# recorded below.  Nothing here is invented: each entry is transcribed from one
# of the six source files.  The adopted network is ADOPTED_CONFIG.
#
# ------------------------------------------------------------------------------------
# 2a.  The four hand-written "baseline" families.
#      Source: check_of_initials_arch.py (notebook cells 0-10 of
#      fw_emulator_per_line_comparison_hg.ipynb, exported to a script).
#      These four share BATCH_SIZE = 1024, LEARNING_RATE = 1e-3,
#      NUM_EPOCHS = 200, EARLY_STOP_PATIENCE = 40, fourier_modes = 32,
#      activation relu, dropout 0.0 and torch.manual_seed(0).
#
#      NOTE on ``first_fourier_term``.  This is the single genuine architectural
#      difference between the baseline code and the search code:
#        "ones" -> trunk input = [1, sin(2 pi n x), cos(2 pi n x)]   (baselines,
#                  and therefore the *adopted* emulator)
#        "x"    -> trunk input = [x, sin(2 pi n x), cos(2 pi n x)]   (all four
#                  don_emulator_5_par*.py search scripts, and the
#                  ``original_xterm`` baseline which was run precisely to test
#                  whether that choice mattered)
#      The input dimension (2*M + 1) and hence the parameter count is identical
#      either way, so checkpoints are state-dict compatible; only the value of
#      the first feature differs.
# ------------------------------------------------------------------------------------
BASELINE_COMMON = {
    "latent_dim": 128,
    "activation": "relu",
    "dropout": 0.0,
    "fourier_modes": 32,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "max_epochs": 200,
    "early_stop_patience": 40,
    "split_seed": SPLIT_SEED,
    "torch_manual_seed": BASELINE_TORCH_SEED,
    "weight_decay": WEIGHT_DECAY,
    "alpha": ALPHA,
    "use_residual": USE_RESIDUAL,
}

BASELINE_ARCHITECTURES: Dict[str, Dict[str, object]] = {
    # ---- THE ADOPTED ARCHITECTURE -------------------------------------------------
    # Ranked 2nd of the 460 configurations by combined test MARE (0.15861 % vs a
    # formal minimum of 0.15857 %) but ~3.7x faster to evaluate (0.018 s vs
    # 0.066 s for the full set of 17 emulators under the same benchmark), so it
    # was retained for every result in Chapter 3.  Ranked 21st by validation
    # loss alone, and the 20 configurations above it beat it by only 0.3-4.4 %
    # while being 1.2-10x slower.
    "original": {
        **BASELINE_COMMON,
        "architecture_name": "original",
        "architecture_tag": "original_lat128_relu_drop0p0_fm32",
        "branch_widths": [128, 256, 512],
        "trunk_widths": [256, 512, 512],
        "first_fourier_term": "ones",
        "adopted": True,
        "source": "check_of_initials_arch.py cells 2-4 / notebook cells 2-4",
    },
    # Same topology, but the first trunk feature is x instead of the constant 1.
    # Run specifically to check whether that choice matters.
    "original_xterm": {
        **BASELINE_COMMON,
        "architecture_name": "original_xterm",
        "architecture_tag": "original_xterm_lat128_relu_drop0p0_fm32",
        "branch_widths": [128, 256, 512],
        "trunk_widths": [256, 512, 512],
        "first_fourier_term": "x",
        "adopted": False,
        "source": "check_of_initials_arch.py cell 4 (xterm block)",
    },
    # Deliberately thinner network at the same latent dimension: does the extra
    # width of "original" actually buy anything?
    "latent128": {
        **BASELINE_COMMON,
        "architecture_name": "latent128",
        "architecture_tag": "latent128_lat128_relu_drop0p0_fm32",
        "branch_widths": [64, 128, 128],
        "trunk_widths": [128, 256, 256],
        "first_fourier_term": "ones",
        "adopted": False,
        "source": "check_of_initials_arch.py cells 5-7 / notebook cells 5-7",
    },
    # Same thin topology but with the latent (shared) dimension halved to 64:
    # how small can the shared representation be before the profiles degrade?
    "latent64": {
        **BASELINE_COMMON,
        "latent_dim": 64,
        "architecture_name": "latent64",
        "architecture_tag": "latent64_lat64_relu_drop0p0_fm32",
        "branch_widths": [64, 128, 128],
        "trunk_widths": [128, 256, 256],
        "first_fourier_term": "ones",
        "adopted": False,
        "source": "check_of_initials_arch.py cells 8-10 / notebook cells 8-10",
    },
}

#: Convenience handle used as the default everywhere.
ADOPTED_CONFIG: Dict[str, object] = BASELINE_ARCHITECTURES["original"]


# ------------------------------------------------------------------------------------
# 2b.  The four grid-search stages.
#      Each stage is one of the don_emulator_5_par*.py scripts, transcribed
#      verbatim.  A stage is a Cartesian product over the listed axes; the total
#      number of *configurations* is the product of the axis lengths, and each
#      configuration trains all 17 line emulators.
#
#      Common to all four stages:
#        FOURIER_MODES = 32, MAX_EPOCHS = 300, EARLY_STOP_PATIENCE = 40,
#        WEIGHT_DECAY = 0.0, ALPHA = 5.0, USE_RESIDUAL = True,
#        FLUX_OFFSET = 1.0, SPLIT_SEED = 42, first_fourier_term = "x",
#        torch_manual_seed = None (never set in these scripts).
# ------------------------------------------------------------------------------------
SEARCH_COMMON = {
    "fourier_modes": 32,
    "max_epochs": 300,
    "early_stop_patience": 40,
    "weight_decay": WEIGHT_DECAY,
    "alpha": ALPHA,
    "use_residual": USE_RESIDUAL,
    "flux_offset": FLUX_OFFSET,
    "split_seed": SPLIT_SEED,
    "first_fourier_term": "x",
    "torch_manual_seed": None,
}

SEARCH_STAGES: Dict[str, Dict[str, object]] = {
    # -------------------------------------------------------------------------------
    # Stage 1 - the opening sweep: 3- and 4-hidden-layer branch/trunk topologies
    # of moderate width, both latent dimensions, ReLU vs GELU, dropout 0 vs 0.1.
    # 8 * 2 * 2 * 2 = 64 configurations.
    # Source: don_emulator_5_par.py
    # -------------------------------------------------------------------------------
    "stage1_base": {
        **SEARCH_COMMON,
        "source": "don_emulator_5_par.py",
        "description": (
            "Explicit HG-like branch/trunk topologies with 3 or 4 hidden layers, "
            "sweeping latent dimension, activation and dropout."
        ),
        "architectures": [
            {"name": "arch01_b128-256-512_t256-512-512", "branch_widths": [128, 256, 512], "trunk_widths": [256, 512, 512]},
            {"name": "arch02_b128-256-512_t128-256-512", "branch_widths": [128, 256, 512], "trunk_widths": [128, 256, 512]},
            {"name": "arch03_b256-512-512_t256-512-512", "branch_widths": [256, 512, 512], "trunk_widths": [256, 512, 512]},
            {"name": "arch04_b128-192-384_t192-384-384", "branch_widths": [128, 192, 384], "trunk_widths": [192, 384, 384]},
            {"name": "arch05_b128-256-512-512_t256-512-512-512", "branch_widths": [128, 256, 512, 512], "trunk_widths": [256, 512, 512, 512]},
            {"name": "arch06_b128-192-384-512_t192-384-512-512", "branch_widths": [128, 192, 384, 512], "trunk_widths": [192, 384, 512, 512]},
            {"name": "arch07_b256-384-512-512_t256-384-512-512", "branch_widths": [256, 384, 512, 512], "trunk_widths": [256, 384, 512, 512]},
            {"name": "arch08_b128-256-512-768_t256-512-768-768", "branch_widths": [128, 256, 512, 768], "trunk_widths": [256, 512, 768, 768]},
        ],
        "latent_dims": [128, 256],
        "activations": ["relu", "gelu"],
        "dropout_rates": [0.0, 0.1],
        "learning_rates": [1e-3],
        "batch_sizes": [2048],
        # don_emulator_5_par.py has no deadline guard and no resume logic, and it
        # names the output folder without the lr/bs suffix.
        "timeout_guard": False,
        "resume": False,
        "tag_includes_lr_bs": False,
    },
    # -------------------------------------------------------------------------------
    # Stage 2 - "extra": does adding two more hidden layers help?  All candidates
    # have 5 hidden layers in both branch and trunk.  Activation and dropout were
    # frozen at the stage-1 winners (ReLU, no dropout).
    # 8 * 2 * 1 * 1 = 16 configurations.
    # Source: don_emulator_5_par_extra.py
    # -------------------------------------------------------------------------------
    "stage2_extra": {
        **SEARCH_COMMON,
        "source": "don_emulator_5_par_extra.py",
        "description": (
            "Architectures with +2 hidden layers versus the original 3-layer setup; "
            "5 hidden layers in both branch and trunk."
        ),
        "architectures": [
            {"name": "xarch01_b128-256-512-512-512_t256-512-512-512-512", "branch_widths": [128, 256, 512, 512, 512], "trunk_widths": [256, 512, 512, 512, 512]},
            {"name": "xarch02_b128-256-512-768-768_t256-512-512-768-768", "branch_widths": [128, 256, 512, 768, 768], "trunk_widths": [256, 512, 512, 768, 768]},
            {"name": "xarch03_b128-192-256-384-512_t192-256-384-512-512", "branch_widths": [128, 192, 256, 384, 512], "trunk_widths": [192, 256, 384, 512, 512]},
            {"name": "xarch04_b192-256-384-512-512_t256-384-512-512-512", "branch_widths": [192, 256, 384, 512, 512], "trunk_widths": [256, 384, 512, 512, 512]},
            {"name": "xarch05_b128-256-384-512-640_t256-384-512-640-640", "branch_widths": [128, 256, 384, 512, 640], "trunk_widths": [256, 384, 512, 640, 640]},
            {"name": "xarch06_b256-384-512-512-512_t256-512-512-512-512", "branch_widths": [256, 384, 512, 512, 512], "trunk_widths": [256, 512, 512, 512, 512]},
            {"name": "xarch07_b128-256-512-640-768_t256-512-640-768-768", "branch_widths": [128, 256, 512, 640, 768], "trunk_widths": [256, 512, 640, 768, 768]},
            {"name": "xarch08_b160-320-512-512-512_t256-512-512-512-512", "branch_widths": [160, 320, 512, 512, 512], "trunk_widths": [256, 512, 512, 512, 512]},
        ],
        "latent_dims": [128, 256],
        "activations": ["relu"],
        "dropout_rates": [0.0],
        "learning_rates": [1e-3],
        "batch_sizes": [2048],
        "timeout_guard": False,
        "resume": False,
        "tag_includes_lr_bs": False,
    },
    # -------------------------------------------------------------------------------
    # Stage 3 - "timeout": increasing-width 5x5 and 6x6 branch/trunk families,
    # now also sweeping dropout, learning rate and batch size.  This is the first
    # stage that was long enough to need the SLURM walltime guard and the
    # resume-from-disk logic.
    # 6 * 2 * 1 * 3 * 2 * 2 = 144 configurations.
    # Source: don_emulator_5_par_timeout.py
    # -------------------------------------------------------------------------------
    "stage3_timeout": {
        **SEARCH_COMMON,
        "source": "don_emulator_5_par_timeout.py",
        "description": (
            "Increasing-width 5x5 and 6x6 branch/trunk families, evaluated over "
            "multiple dropout rates, batch sizes and learning rates."
        ),
        "architectures": [
            {"name": "inc5x5_128_256_384_512_640", "branch_widths": [128, 256, 384, 512, 640], "trunk_widths": [128, 256, 384, 512, 640]},
            {"name": "inc5x5_128_256_512_768_1024", "branch_widths": [128, 256, 512, 768, 1024], "trunk_widths": [128, 256, 512, 768, 1024]},
            {"name": "inc5x5_256_384_512_768_1024", "branch_widths": [256, 384, 512, 768, 1024], "trunk_widths": [256, 384, 512, 768, 1024]},
            {"name": "inc6x6_128_256_384_512_640_768", "branch_widths": [128, 256, 384, 512, 640, 768], "trunk_widths": [128, 256, 384, 512, 640, 768]},
            {"name": "inc6x6_128_256_512_768_1024_1280", "branch_widths": [128, 256, 512, 768, 1024, 1280], "trunk_widths": [128, 256, 512, 768, 1024, 1280]},
            {"name": "inc6x6_256_384_512_768_1024_1536", "branch_widths": [256, 384, 512, 768, 1024, 1536], "trunk_widths": [256, 384, 512, 768, 1024, 1536]},
        ],
        "latent_dims": [128, 256],
        "activations": ["relu"],
        "dropout_rates": [0.0, 0.1, 0.2],
        "learning_rates": [1e-3, 5e-4],
        "batch_sizes": [512, 256],
        "timeout_guard": True,
        "resume": True,
        "tag_includes_lr_bs": True,
    },
    # -------------------------------------------------------------------------------
    # Stage 4 - "timeout extra": equal-width, deeper (4x4, 5x5, 6x6) families at a
    # single batch size, sweeping three learning rates.
    # 9 * 2 * 1 * 1 * 3 * 1 = 54 configurations.
    # Source: don_emulator_5_par_timeout_extra.py
    # -------------------------------------------------------------------------------
    "stage4_timeout_extra": {
        **SEARCH_COMMON,
        "source": "don_emulator_5_par_timeout_extra.py",
        "description": (
            "Equal-width branch/trunk families with deeper networks (4x4, 5x5, 6x6) "
            "evaluated over learning-rate variants."
        ),
        "architectures": [
            {"name": "eq4x4_w384", "branch_widths": [384, 384, 384, 384], "trunk_widths": [384, 384, 384, 384]},
            {"name": "eq4x4_w512", "branch_widths": [512, 512, 512, 512], "trunk_widths": [512, 512, 512, 512]},
            {"name": "eq4x4_w768", "branch_widths": [768, 768, 768, 768], "trunk_widths": [768, 768, 768, 768]},
            {"name": "eq5x5_w384", "branch_widths": [384, 384, 384, 384, 384], "trunk_widths": [384, 384, 384, 384, 384]},
            {"name": "eq5x5_w512", "branch_widths": [512, 512, 512, 512, 512], "trunk_widths": [512, 512, 512, 512, 512]},
            {"name": "eq5x5_w768", "branch_widths": [768, 768, 768, 768, 768], "trunk_widths": [768, 768, 768, 768, 768]},
            {"name": "eq6x6_w512", "branch_widths": [512, 512, 512, 512, 512, 512], "trunk_widths": [512, 512, 512, 512, 512, 512]},
            {"name": "eq6x6_w768", "branch_widths": [768, 768, 768, 768, 768, 768], "trunk_widths": [768, 768, 768, 768, 768, 768]},
            {"name": "eq6x6_w1024", "branch_widths": [1024, 1024, 1024, 1024, 1024, 1024], "trunk_widths": [1024, 1024, 1024, 1024, 1024, 1024]},
        ],
        "latent_dims": [128, 256],
        "activations": ["relu"],
        "dropout_rates": [0.0],
        "learning_rates": [1e-3, 2e-4, 5e-4],
        "batch_sizes": [512],
        "timeout_guard": True,
        "resume": True,
        "tag_includes_lr_bs": True,
    },
}

# ------------------------------------------------------------------------------------
# 2c.  Hyperparameter ranges as reported in the thesis
#      (Table "Hyperparameter ranges represented in the combined architecture
#      search", five_parameter_emulator.tex).  Kept here for cross-checking.
#
#      IMPORTANT DISCREPANCY, recorded rather than hidden:
#      the thesis table lists Fourier modes {32, 64}, batch sizes
#      {256, 512, 1024, 2048} and maximum epochs {200, 300}.  The four
#      don_emulator_5_par*.py scripts preserved in the repository all fix
#      FOURIER_MODES = 32 and MAX_EPOCHS = 300; the 64-mode comparison described
#      in the text ("These results showed no consistent improvement from using
#      64 Fourier modes") was run from a driver script that is not present in
#      this directory.  Batch size 1024 and max_epochs 200 come from the
#      BASELINE_ARCHITECTURES above; batch sizes 2048/512/256 come from stages
#      1-2 / 3-4.  Similarly, the stages enumerated here sum to
#      64 + 16 + 144 + 54 = 278 configurations plus the 4 baseline families
#      = 282, whereas the thesis reports 460 evaluated configurations in total:
#      the balance comes from earlier sweeps (including the 64-mode variants)
#      whose driver scripts were not kept.  ``fourier_modes`` is a first-class
#      argument everywhere in this file, so a 64-mode run is reachable with
#      ``--fourier-modes 64``.
# ------------------------------------------------------------------------------------
THESIS_TABLE_RANGES: Dict[str, object] = {
    "branch_trunk_hidden_layers": [3, 4, 5, 6],
    "hidden_layer_widths": [128, 256, 384, 512, 640, 768, 1024, 1280, 1536],
    "latent_dimension": [128, 256],
    "activation": ["relu", "gelu"],
    "dropout": [0.0, 0.1, 0.2],
    "learning_rate": [2e-4, 5e-4, 1e-3],
    "batch_size": [256, 512, 1024, 2048],
    "fourier_modes": [32, 64],
    "max_epochs": [200, 300],
    "total_evaluated_configurations_reported_in_thesis": 460,
}

#: The 17 FASTWIND line windows.  One DeepONet is trained per entry.
LINE_FILES: Tuple[str, ...] = (
    "OUT.BETA_VTV010",                 # H beta
    "OUT.HALPHA_HEII6527_VTV010",      # H alpha + He II 6527 (shared window)
    "OUT.HDELTA_VTV010",               # H delta
    "OUT.HEI4026_VTV010",
    "OUT.HEI4387_VTV010",
    "OUT.HEI4471_VTV010",
    "OUT.HEI4922_VTV010",
    "OUT.HEI5875_VTV010",
    "OUT.HEI7065_VTV010",
    "OUT.HEII4200_VTV010",
    "OUT.HEII4541_VTV010",
    "OUT.HEII4686_VTV010",
    "OUT.HEII5411_VTV010",
    "OUT.HEII6406_VTV010",
    "OUT.HEII6683_VTV010",
    "OUT.HEPS_VTV010",                 # H epsilon
    "OUT.HGAMMA_VTV010",               # H gamma  (also the convergence sentinel)
)


# =====================================================================================
# SECTION 3.  WALLTIME GUARD AND RESUME BOOKKEEPING
# =====================================================================================
# Transcribed from don_emulator_5_par_timeout.py.  The two "timeout" stages were
# submitted to a SLURM queue whose jobs are killed at the walltime.  Being killed
# mid-``torch.save`` corrupts a checkpoint, so the script instead watches the
# clock and raises DeadlineReached at a safe point (between epochs, or between
# configurations), writes the summary, and exits cleanly.  Resubmitting the same
# job then skips every (config, line) pair whose .pth and _loss.json already
# exist on disk.

#: Safety margin subtracted from the SLURM walltime, in seconds (env-overridable,
#: exactly as in the original script).
STOP_BEFORE_TIMEOUT_SECONDS = int(os.environ.get("DON_STOP_BEFORE_TIMEOUT_SECONDS", "300"))

#: Optional hard cap on this process's runtime, in seconds (env-overridable).
_MAX_RUNTIME_ENV = os.environ.get("DON_MAX_RUNTIME_SECONDS")
MAX_RUNTIME_SECONDS: Optional[int] = None if _MAX_RUNTIME_ENV is None else int(_MAX_RUNTIME_ENV)

RUN_START_TIME = time.time()


class DeadlineReached(RuntimeError):
    """Raised when the remaining walltime budget is too small to continue safely."""


def _slurm_time_limit_seconds() -> Optional[int]:
    """Parse ``$SLURM_TIMELIMIT`` (``D-HH:MM:SS`` / ``HH:MM:SS`` / ``HH:MM``)."""
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


TIME_LIMIT_SECONDS = _slurm_time_limit_seconds()
_slurm_deadline_ts = (
    None if TIME_LIMIT_SECONDS is None
    else RUN_START_TIME + TIME_LIMIT_SECONDS - STOP_BEFORE_TIMEOUT_SECONDS
)
_runtime_deadline_ts = (
    None if MAX_RUNTIME_SECONDS is None else RUN_START_TIME + MAX_RUNTIME_SECONDS
)
_deadline_candidates = [ts for ts in (_slurm_deadline_ts, _runtime_deadline_ts) if ts is not None]
DEADLINE_TS: Optional[float] = min(_deadline_candidates) if _deadline_candidates else None


def time_remaining_seconds() -> Optional[float]:
    if DEADLINE_TS is None:
        return None
    return DEADLINE_TS - time.time()


def ensure_time_budget(context: str, minimum_seconds: float = 0, enabled: bool = True) -> None:
    """Raise :class:`DeadlineReached` if less than ``minimum_seconds`` remain.

    ``minimum_seconds`` was 30 s inside the epoch loop and 180 s before starting
    a new configuration in the original script.
    """
    if not enabled:
        return
    remaining = time_remaining_seconds()
    if remaining is not None and remaining <= minimum_seconds:
        raise DeadlineReached(
            f"Stopping before walltime during {context}. Remaining budget: {remaining:.1f}s"
        )


def load_existing_summary(summary_path: str) -> List[dict]:
    if not os.path.exists(summary_path):
        return []
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:  # pragma: no cover - defensive, as in the original
        print(f"[summary] Could not read existing summary at {summary_path}: {e}", flush=True)
        return []


def save_summary(summary_path: str, records: Sequence[dict]) -> None:
    with open(summary_path, "w") as f:
        json.dump(list(records), f, indent=4)


def summary_key(record: Mapping[str, object]) -> tuple:
    return (
        record.get("architecture_tag"),
        record.get("latent_dim"),
        record.get("activation"),
        record.get("dropout"),
        record.get("learning_rate"),
        record.get("batch_size"),
    )


def is_line_complete(output_dir: str, architecture_tag: str, line_file: str) -> bool:
    arch_dir = os.path.join(output_dir, architecture_tag)
    model_path = os.path.join(arch_dir, f"emulator_{line_file}.pth")
    loss_path = model_path.replace(".pth", "_loss.json")
    return os.path.exists(model_path) and os.path.exists(loss_path)


def load_existing_line_val_loss(output_dir: str, architecture_tag: str, line_file: str) -> Optional[float]:
    loss_path = os.path.join(output_dir, architecture_tag, f"emulator_{line_file}_loss.json")
    if not os.path.exists(loss_path):
        return None
    try:
        with open(loss_path, "r") as f:
            data = json.load(f)
        vals = data.get("val_losses", [])
        return float(vals[-1]) if vals else None
    except Exception:
        return None


# =====================================================================================
# SECTION 4.  DATA LOADING AND PREPROCESSING  (one shared implementation)
# =====================================================================================
# Identical in all six source files; consolidated here.

def parse_indat_for_model(model_id: int, model_dir: str) -> Dict[str, float]:
    """Read a FASTWIND ``INDAT``/``INDAT.DAT`` deck and return the seven parameters.

    Only non-blank lines are counted, so the indices below refer to the
    documented FASTWIND deck layout:
        line 3: Teff  logg  R/Rsun
        line 5: Mdot  vmin  vinf  beta  vtrans
        line 6: N(He)/N(H)  XIHE
        line 9: vturb  metallicity  LINES  LIM
    """
    indat_path = None
    for fname in ["INDAT", "INDAT.DAT"]:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            indat_path = p
            break
    if indat_path is None:
        raise FileNotFoundError(f"No INDAT file for model {model_id} in {model_dir}")
    with open(indat_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    Teff, logg, R = map(float, lines[3].split()[:3])
    Mdot, vmin, vinf, beta, vtrans = map(float, lines[5].split()[:5])
    Y_He = float(lines[6].split()[0])
    vturb = np.nan
    if len(lines) > 8:
        parts = lines[8].split()
        try:
            vturb = float(parts[0])
        except ValueError:
            vturb = np.nan
    return {"model": model_id, "Teff": Teff, "logg": logg, "R": R,
            "Mdot": Mdot, "v_inf": vinf, "Y_He": Y_He, "v_turb": vturb}


def classify_model(model_id: int, model_dir: str) -> Dict[str, object]:
    """Assign the green/yellow/orange/red convergence flag (see module docstring).

    The two ``converg_steps < 100`` branches are written out separately, exactly
    as in the sources, even though both evaluate to the same expression: the run
    finished before the iteration cap, so it converged regardless of the size of
    the final correction.  (``don_emulator_5_par*.py`` writes ``>= 0.0045`` and
    ``check_of_initials_arch.py`` writes ``> 0.0045`` in the second branch; the
    outcome is identical either way, so ``>=`` is used here.)
    """
    conv_path = os.path.join(model_dir, "CONVERG")
    hg_path = os.path.join(model_dir, CONVERGENCE_SENTINEL_LINE)
    result: Dict[str, object] = {"model": model_id, "flag": "red"}
    try:
        df = pd.read_csv(conv_path, sep=r"\s+", header=None)
        converg_steps = len(df)                 # one row per NLTE atmosphere iteration
        correction = float(df.iloc[-1, 1])      # largest remaining relative update
        result["converg_steps"] = converg_steps
        result["max_corr"] = correction

        has_hgamma = os.path.exists(hg_path) and os.path.getsize(hg_path) > 0
        if converg_steps < 100 and correction < 0.0045:
            flag = "green" if has_hgamma else "red"
        elif converg_steps < 100 and correction >= 0.0045:
            flag = "green" if has_hgamma else "red"
        elif correction < 0.01:
            flag = "green" if has_hgamma else "red"
        elif correction < 0.2:
            flag = "yellow" if has_hgamma else "red"
        else:
            flag = "orange" if has_hgamma else "red"
        result["flag"] = flag
    except Exception as e:
        result["error"] = str(e)
    return result


def load_fastwind_line(path: str, max_rows: int = LINE_MAX_ROWS):
    """Return (wavelength [Angstrom], continuum-normalised flux) from one ``OUT.*`` file.

    Column 2 is the wavelength and the last column is the emergent normalised
    flux.  Only the first ``max_rows`` rows are read so that the trailing footer
    written by FASTWIND is never parsed as data.
    """
    arr = np.loadtxt(path, max_rows=max_rows)
    if arr.ndim == 1:
        arr = arr[None, :]
    waves = arr[:, 2].astype(np.float32)
    flux = arr[:, -1].astype(np.float32)
    return waves, flux


class GridData:
    """The convergence-selected FASTWIND grid, parsed once and shared by every run.

    Attributes
    ----------
    model_dir_map : dict[int, str]   green models only
    params_df     : DataFrame        physical parameters of the green models
    param_mins, param_maxs : ndarray (7,)  scaling limits, stored in every checkpoint
    model_param_norm : dict[int, ndarray]  normalised 7-vector per model
    line_files    : list[str]        the ``OUT.*`` windows found in the reference model
    """

    def __init__(self, models_root: str, verbose: bool = True):
        self.models_root = models_root

        # 1) Discover model directories (integer-named subfolders only, so that
        #    tarballs and stray files are ignored).
        model_dirs = sorted(
            [d for d in glob.glob(os.path.join(models_root, "*"))
             if os.path.isdir(d) and os.path.basename(d).isdigit()],
            key=lambda x: int(os.path.basename(x)),
        )
        model_dir_map = {int(os.path.basename(d)): d for d in model_dirs}
        if verbose:
            print(f"Found {len(model_dir_map)} model directories under {models_root}.")

        # 2) Physical parameters from the INDAT decks.
        param_rows = []
        for mid, mdir in model_dir_map.items():
            try:
                param_rows.append(parse_indat_for_model(mid, mdir))
            except Exception as e:
                print(f"[PARAMS] Skipping model {mid}: {e}")
        params_df = pd.DataFrame(param_rows).sort_values("model").reset_index(drop=True)
        if verbose:
            print(f"Parsed INDAT parameters for {len(params_df)} models.")

        # 3) Convergence screening: keep green only.
        quality_records = [classify_model(mid, mdir) for mid, mdir in model_dir_map.items()]
        quality_df = pd.DataFrame(quality_records).set_index("model")
        if verbose:
            print("\nQuality flag counts:")
            print(quality_df["flag"].value_counts())
        good_models = quality_df.index[quality_df["flag"] == "green"]
        if verbose:
            print(f"\nUsing {len(good_models)} models after removing red-flagged ones.")
        params_df = params_df[params_df["model"].isin(good_models)].copy().reset_index(drop=True)
        model_dir_map = {mid: model_dir_map[mid] for mid in good_models}

        # 4) Min-max normalisation of the branch inputs to [0, 1].  The limits are
        #    computed once over the whole retained dataset (NOT per batch and NOT
        #    per partition) and are stored in every checkpoint so that inference
        #    can reproduce them exactly.
        params_arr = params_df[list(PARAM_COLS)].values.astype(np.float32)
        param_mins = params_arr.min(axis=0)
        param_maxs = params_arr.max(axis=0)
        # A zero range (v_turb, which is fixed at 10 km/s) is replaced by 1.0, so
        # that column becomes identically zero instead of NaN.
        param_range = np.where(param_maxs - param_mins == 0, 1.0, param_maxs - param_mins)
        params_norm = (params_arr - param_mins) / param_range
        # Mdot spans >2 decades, so it is log10-scaled first and then renormalised.
        log_mdot = np.log10(params_arr[:, MDOT_INDEX])
        log_mdot_norm = (log_mdot - log_mdot.min()) / (log_mdot.max() - log_mdot.min())
        params_norm[:, MDOT_INDEX] = log_mdot_norm

        self.model_dir_map = model_dir_map
        self.params_df = params_df
        self.quality_df = quality_df
        self.params_arr = params_arr
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.params_norm = params_norm
        self.model_param_norm = {
            int(params_df.loc[i, "model"]): params_norm[i] for i in range(len(params_df))
        }
        if verbose:
            print(f"\nFinal: {len(self.model_param_norm)} usable models with normalized parameters.")

        # 5) Line windows, taken from the lowest-numbered surviving model.
        if len(model_dir_map) == 0:
            raise RuntimeError("No valid model directories found after filtering!")
        ref_model_id = min(model_dir_map.keys())
        ref_dir = model_dir_map[ref_model_id]
        ref_line_paths = sorted(
            f for f in glob.glob(os.path.join(ref_dir, "OUT.*"))
            if not f.endswith("Zone.Identifier")  # Windows alternate-data-stream files
        )
        self.line_files = [os.path.basename(p) for p in ref_line_paths]
        if verbose:
            print(f"Reference model {ref_model_id} has {len(self.line_files)} spectral line files.")
            print("Example line files:", self.line_files[:10])

    # ---------------------------------------------------------------------------
    def build_dataset_for_line(self, line_file: str, verbose: bool = True) -> Dict[str, object]:
        """Stack (parameters, wavelengths, flux) over every green model for one line."""
        all_params, all_waves, all_flux, all_model_ids = [], [], [], []
        for model_id, model_dir in self.model_dir_map.items():
            if model_id not in self.model_param_norm:
                continue
            path = os.path.join(model_dir, line_file)
            if not os.path.exists(path):
                continue  # not every model necessarily has every window
            try:
                waves, flux = load_fastwind_line(path, max_rows=LINE_MAX_ROWS)
            except Exception as e:
                if verbose:
                    print(f"[{line_file}] Could not read {path}: {e}")
                continue
            if waves.shape[0] < MIN_POINTS_PER_LINE:
                if verbose:
                    print(f"[{line_file}] Model {model_id} has too few points; skipping.")
                continue
            all_params.append(self.model_param_norm[model_id])
            all_waves.append(waves)
            all_flux.append(flux)
            all_model_ids.append(model_id)

        if not all_params:
            raise RuntimeError(f"No data found for spectral line '{line_file}'.")

        X_params = np.stack(all_params).astype(np.float32)   # (N, 7)
        X_waves = np.stack(all_waves).astype(np.float32)     # (N, P)
        Y_flux = np.stack(all_flux).astype(np.float32)       # (N, P)
        model_ids_line = np.array(all_model_ids, dtype=int)

        # Per-line wavelength normalisation to [0, 1].  Each emulator therefore
        # only ever sees its own window, and the trunk's Fourier features have a
        # well-defined period across that window.
        lambda_min = X_waves.min()
        lambda_max = X_waves.max()
        X_waves_norm = (X_waves - lambda_min) / (lambda_max - lambda_min)

        if verbose:
            print(f"[{line_file}] Using {X_params.shape[0]} models, {X_waves.shape[1]} points, "
                  f"lambda-range {lambda_min:.2f}-{lambda_max:.2f} A")

        return {
            "X_params": X_params,
            "X_waves_norm": X_waves_norm,
            "Y_flux": Y_flux,
            "model_ids": model_ids_line,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
        }

    def preload(self, line_files: Optional[Sequence[str]] = None, verbose: bool = True) -> Dict[str, dict]:
        """Build and cache the dataset for every requested line."""
        line_files = list(self.line_files if line_files is None else line_files)
        print("\n[DEBUG] Preloading datasets for all spectral lines...\n", flush=True)
        preloaded: Dict[str, dict] = {}
        for i, line_file in enumerate(line_files):
            print(f"[DEBUG] Preloading line {i+1}/{len(line_files)}: {line_file}", flush=True)
            try:
                preloaded[line_file] = self.build_dataset_for_line(line_file, verbose=verbose)
                print(f"[DEBUG] Done: {line_file}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Skipped {line_file} - {e}", flush=True)
        return preloaded


# =====================================================================================
# SECTION 5.  THE MODEL
# =====================================================================================
# One generic implementation covering every architecture in Section 2.  With
# dropout = 0 the generated ``nn.Sequential`` has exactly the same submodule
# indices as the hand-written baseline classes in the notebook, so checkpoints
# written by either code path load into this class without renaming.

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class BranchNet(nn.Module):
    """Stellar parameters -> latent coefficients.

    Input : (batch, 7) normalised parameters.
    Output: (batch, latent_dim).  Physically, "which atmosphere is this".
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_widths: Sequence[int],
                 activation_fn=nn.ReLU, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_widths:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    """Normalised wavelength -> latent basis functions.

    Input : (batch, P) or (batch, P, 1) coordinates already scaled to [0, 1].
    Output: (batch, P, latent_dim).  Physically, a learned profile-shape basis
            evaluated at each requested wavelength.

    The coordinate is expanded into ``2*fourier_modes + 1`` features:
        first_fourier_term == "ones" -> [1, sin(2 pi n x), cos(2 pi n x)]
        first_fourier_term == "x"    -> [x, sin(2 pi n x), cos(2 pi n x)]
    with n = 1 .. fourier_modes.  See the note in Section 2a: the adopted
    emulator uses "ones"; the grid-search scripts used "x".
    """

    def __init__(self, latent_dim: int, hidden_widths: Sequence[int], activation_fn=nn.ReLU,
                 dropout: float = 0.0, fourier_modes: int = 32,
                 first_fourier_term: str = "ones"):
        super().__init__()
        if first_fourier_term not in ("ones", "x"):
            raise ValueError(f"first_fourier_term must be 'ones' or 'x', got {first_fourier_term!r}")
        self.fourier_modes = fourier_modes
        self.first_fourier_term = first_fourier_term
        in_dim = 2 * fourier_modes + 1
        layers: List[nn.Module] = []
        for out_dim in hidden_widths:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(-1)
        batch_size, P, _ = coords.shape
        n = self.fourier_modes
        freqs = 2.0 * math.pi * torch.arange(
            1, n + 1, device=coords.device, dtype=coords.dtype
        ).view(1, 1, n)
        sin_feats = torch.sin(freqs * coords)
        cos_feats = torch.cos(freqs * coords)
        if self.first_fourier_term == "ones":
            first = torch.ones(batch_size, P, 1, device=coords.device, dtype=coords.dtype)
        else:
            first = coords
        features = torch.cat([first, sin_feats, cos_feats], dim=-1)
        features = features.view(-1, features.shape[-1])
        out = self.net(features)
        return out.view(batch_size, P, -1)


class DeepONetModel(nn.Module):
    """The DeepONet:  flux(theta, lambda) = sum_k branch_k(theta) * trunk_k(lambda)."""

    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B = self.branch(params)              # (batch, k)
        T = self.trunk(coords)               # (batch, P, k)
        return (T * B.unsqueeze(1)).sum(dim=-1)   # (batch, P)


def build_deeponet(config: Mapping[str, object], input_dim: int = 7) -> DeepONetModel:
    """Instantiate a DeepONet from a config dict of the form used in Section 2."""
    activation_name = str(config.get("activation", "relu")).lower()
    act_fn = ACTIVATIONS[activation_name]
    branch = BranchNet(
        input_dim=input_dim,
        latent_dim=int(config["latent_dim"]),
        hidden_widths=list(config["branch_widths"]),
        activation_fn=act_fn,
        dropout=float(config.get("dropout", 0.0)),
    )
    trunk = TrunkNet(
        latent_dim=int(config["latent_dim"]),
        hidden_widths=list(config["trunk_widths"]),
        activation_fn=act_fn,
        dropout=float(config.get("dropout", 0.0)),
        fourier_modes=int(config.get("fourier_modes", 32)),
        first_fourier_term=str(config.get("first_fourier_term", "ones")),
    )
    return DeepONetModel(branch, trunk)


def weighted_line_loss(preds: torch.Tensor, targets: torch.Tensor,
                       alpha: float = ALPHA, use_residual: bool = USE_RESIDUAL,
                       flux_offset: float = FLUX_OFFSET) -> torch.Tensor:
    """Core-weighted mean squared error,  mean[(p - t)^2 * (1 + alpha*|line|)].

    ``|line|`` is the departure of the target from the continuum: it equals
    ``|target|`` when the network is trained on residuals (flux - 1) and
    ``|target - flux_offset|`` otherwise.  Weighting by it forces the optimiser
    to spend its capacity on the diagnostic core and wings rather than on the
    flat continuum, which occupies most of every window.
    """
    line_mag = targets.abs() if use_residual else (targets - flux_offset).abs()
    weights = 1.0 + alpha * line_mag
    return ((preds - targets) ** 2 * weights).mean()


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device string, preferring CUDA when available."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was requested but torch.cuda.is_available() is False; using CPU.")
        return torch.device("cpu")
    return resolved


# =====================================================================================
# SECTION 6.  THE TRAINING LOOP  (one shared implementation)
# =====================================================================================

def make_config_tag(architecture_name: str, latent_dim: int, activation: str, dropout: float,
                    fourier_modes: int, learning_rate: Optional[float] = None,
                    batch_size: Optional[int] = None, include_lr_bs: bool = False) -> str:
    """Reproduce the folder-name convention of the original scripts.

    Stages 1-2 (``don_emulator_5_par.py`` / ``_extra.py``)::

        {arch}_lat{L}_{act}_drop{d with . -> p}_fm{M}

    Stages 3-4 (``*_timeout*.py``) append the learning rate and batch size::

        ..._lr{lr:.0e with '+0' removed}_bs{B}
    """
    tag = (f"{architecture_name}_lat{latent_dim}_{activation.lower()}_"
           f"drop{str(dropout).replace('.', 'p')}_fm{fourier_modes}")
    if include_lr_bs:
        lr_tag = f"{learning_rate:.0e}".replace("+0", "")
        tag = f"{tag}_lr{lr_tag}_bs{batch_size}"
    return tag


def train_emulator_for_line(
    line_file: str,
    data: Mapping[str, object],
    config: Mapping[str, object],
    grid: "GridData",
    output_dir: str,
    architecture_tag: Optional[str] = None,
    device: Optional[torch.device] = None,
    timeout_guard: bool = False,
    verbose: bool = True,
):
    """Train one DeepONet for one spectral line window.

    Returns ``(best_val_loss, checkpoint_path, loss_json_path)``.

    The routine is a verbatim consolidation of ``train_emulator_for_line`` from
    ``don_emulator_5_par*.py`` and of the four ``train_emulator_for_line*``
    variants in ``check_of_initials_arch.py``.  The only behavioural switches
    are supplied through ``config``.
    """
    device = resolve_device(device)

    architecture_name = str(config["architecture_name"])
    latent_dim = int(config["latent_dim"])
    branch_widths = list(config["branch_widths"])
    trunk_widths = list(config["trunk_widths"])
    activation_name = str(config.get("activation", "relu"))
    dropout = float(config.get("dropout", 0.0))
    fourier_modes = int(config.get("fourier_modes", 32))
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    max_epochs = int(config["max_epochs"])
    early_stop_patience = int(config["early_stop_patience"])
    weight_decay = float(config.get("weight_decay", WEIGHT_DECAY))
    alpha = float(config.get("alpha", ALPHA))
    use_residual = bool(config.get("use_residual", USE_RESIDUAL))
    flux_offset = float(config.get("flux_offset", FLUX_OFFSET))
    split_seed = int(config.get("split_seed", SPLIT_SEED))
    first_fourier_term = str(config.get("first_fourier_term", "ones"))
    torch_manual_seed = config.get("torch_manual_seed", None)

    if architecture_tag is None:
        architecture_tag = str(config.get("architecture_tag") or make_config_tag(
            architecture_name, latent_dim, activation_name, dropout, fourier_modes,
            learning_rate, batch_size, include_lr_bs=False,
        ))

    X_params_all = np.asarray(data["X_params"])
    X_waves_all = np.asarray(data["X_waves_norm"])
    Y_flux_all = np.asarray(data["Y_flux"])
    model_ids_all = np.asarray(data["model_ids"])
    lam_min = data["lambda_min"]
    lam_max = data["lambda_max"]

    # Residual target: the network learns (flux - 1) rather than the flux, so the
    # constant continuum level does not consume network capacity.
    if use_residual:
        Y_flux_all = Y_flux_all - flux_offset
        if verbose:
            print(f"[{line_file}] Using residual targets (flux - {flux_offset}).")

    # 70 / 15 / 15 split, fixed seed, identical for every line because it depends
    # only on N and the seed.
    N = X_params_all.shape[0]
    indices = np.arange(N)
    train_idx, temp_idx = train_test_split(indices, test_size=SPLIT_TEST_SIZE_FIRST,
                                           random_state=split_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=SPLIT_TEST_SIZE_SECOND,
                                         random_state=split_seed)
    if verbose:
        print(f"[{line_file}] N={N} -> Train={len(train_idx)}, Val={len(val_idx)}, "
              f"Test={len(test_idx)}", flush=True)

    train_dataset = TensorDataset(
        torch.tensor(X_params_all[train_idx], dtype=torch.float32),
        torch.tensor(X_waves_all[train_idx], dtype=torch.float32),
        torch.tensor(Y_flux_all[train_idx], dtype=torch.float32))
    val_dataset = TensorDataset(
        torch.tensor(X_params_all[val_idx], dtype=torch.float32),
        torch.tensor(X_waves_all[val_idx], dtype=torch.float32),
        torch.tensor(Y_flux_all[val_idx], dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # torch.manual_seed(0) was executed once, before any model construction, in
    # the notebook / check_of_initials_arch.py.  Setting it here reproduces the
    # deterministic initialisation for the baseline families; the search stages
    # pass None and therefore keep torch's default seeding.
    if torch_manual_seed is not None:
        torch.manual_seed(int(torch_manual_seed))

    model = build_deeponet({
        "latent_dim": latent_dim,
        "branch_widths": branch_widths,
        "trunk_widths": trunk_widths,
        "activation": activation_name,
        "dropout": dropout,
        "fourier_modes": fourier_modes,
        "first_fourier_term": first_fourier_term,
    }, input_dim=X_params_all.shape[1])

    if USE_DATA_PARALLEL_IF_MULTI_GPU and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=LR_SCHEDULER["mode"],
        factor=LR_SCHEDULER["factor"],
        patience=LR_SCHEDULER["patience"],
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # Walltime guard: refuse to start an epoch we cannot safely finish.
        ensure_time_budget(f"epoch loop for {line_file}", minimum_seconds=30,
                           enabled=timeout_guard)

        model.train()
        running_loss = 0.0
        for batch_params, batch_coords, batch_flux in train_loader:
            batch_params = batch_params.to(device)
            batch_coords = batch_coords.to(device)
            batch_targets = batch_flux.to(device)
            optimizer.zero_grad()
            preds = model(batch_params, batch_coords)
            loss = weighted_line_loss(preds, batch_targets, alpha, use_residual, flux_offset)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_params, batch_coords, batch_flux in val_loader:
                batch_params = batch_params.to(device)
                batch_coords = batch_coords.to(device)
                batch_targets = batch_flux.to(device)
                preds = model(batch_params, batch_coords)
                loss = weighted_line_loss(preds, batch_targets, alpha, use_residual, flux_offset)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Checkpoint the *best* epoch, not the last one.  The saved state carries
        # everything downstream analysis needs, including the test partition, so
        # that the plotting script never has to re-read the FASTWIND grid.
        if avg_val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = avg_val_loss
            model_to_save = model.module if hasattr(model, "module") else model
            best_state = {
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss_history": train_losses,
                "val_loss_history": val_losses,
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "param_mins": grid.param_mins,
                "param_maxs": grid.param_maxs,
                "model_ids_test": model_ids_all[test_idx],
                "X_params_test": X_params_all[test_idx],
                "X_waves_test": X_waves_all[test_idx],
                "Y_flux_test": Y_flux_all[test_idx],
                "config": {
                    "line_file": line_file,
                    "architecture_name": architecture_name,
                    "architecture_tag": architecture_tag,
                    "latent_dim": latent_dim,
                    "branch_depth": len(branch_widths),
                    "trunk_depth": len(trunk_widths),
                    "branch_widths": branch_widths,
                    "trunk_widths": trunk_widths,
                    "activation": activation_name,
                    "dropout": dropout,
                    "fourier_modes": fourier_modes,
                    "use_residual": use_residual,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "split_seed": split_seed,
                    # The three keys below were written only by the baseline code
                    # path in the sources; they are always written here because
                    # they are pure metadata and make a checkpoint self-describing.
                    "max_epochs": max_epochs,
                    "early_stop_patience": early_stop_patience,
                    "first_fourier_term": first_fourier_term,
                },
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[{line_file}] Epoch {epoch}/{max_epochs} - Train: {avg_train_loss:.3e}, "
                  f"Val: {avg_val_loss:.3e}, No improve: {epochs_no_improve}", flush=True)
        if epochs_no_improve >= early_stop_patience:
            print(f"[{line_file}] Early stopping at epoch {epoch}.", flush=True)
            break

    arch_dir = os.path.join(output_dir, architecture_tag) if architecture_tag else output_dir
    os.makedirs(arch_dir, exist_ok=True)
    save_path = os.path.join(arch_dir, f"emulator_{line_file}.pth")

    if best_state:
        torch.save(best_state, save_path)
        print(f"[{line_file}] Model saved at: {save_path}", flush=True)
    else:
        print(f"[{line_file}] No best_state captured; something went wrong.", flush=True)

    loss_path = save_path.replace(".pth", "_loss.json")
    loss_data = {
        "line_file": line_file,
        "train_losses": train_losses,
        "val_losses": val_losses,
        # Extra provenance fields, written by check_of_initials_arch.py.  Harmless
        # for the readers in five_parameter_plots.py, which key off "val_losses".
        "architecture_name": architecture_name,
        "architecture_tag": architecture_tag,
        "latent_dim": latent_dim,
        "branch_widths": branch_widths,
        "trunk_widths": trunk_widths,
        "fourier_modes": fourier_modes,
        "activation": activation_name,
        "dropout": dropout,
        "use_residual": use_residual,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stop_patience": early_stop_patience,
        "epochs_ran": len(val_losses),
        "stopped_early": len(val_losses) < max_epochs,
    }
    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)

    return best_val_loss, save_path, loss_path


# =====================================================================================
# SECTION 7.  RUN MODES
# =====================================================================================

def enumerate_stage(stage: Mapping[str, object]) -> List[dict]:
    """Expand one search stage into its explicit list of configurations.

    Loop order is the original one: architecture, latent dim, activation,
    dropout, learning rate, batch size.
    """
    configs: List[dict] = []
    for arch in stage["architectures"]:
        for latent_dim in stage["latent_dims"]:
            for activation in stage["activations"]:
                for dropout in stage["dropout_rates"]:
                    for lr in stage["learning_rates"]:
                        for bs in stage["batch_sizes"]:
                            tag = make_config_tag(
                                arch["name"], latent_dim, activation, dropout,
                                int(stage["fourier_modes"]), lr, bs,
                                include_lr_bs=bool(stage["tag_includes_lr_bs"]),
                            )
                            configs.append({
                                "architecture_name": arch["name"],
                                "architecture_tag": tag,
                                "branch_widths": list(arch["branch_widths"]),
                                "trunk_widths": list(arch["trunk_widths"]),
                                "latent_dim": latent_dim,
                                "activation": activation,
                                "dropout": dropout,
                                "learning_rate": lr,
                                "batch_size": bs,
                                "fourier_modes": int(stage["fourier_modes"]),
                                "max_epochs": int(stage["max_epochs"]),
                                "early_stop_patience": int(stage["early_stop_patience"]),
                                "weight_decay": stage["weight_decay"],
                                "alpha": stage["alpha"],
                                "use_residual": stage["use_residual"],
                                "flux_offset": stage["flux_offset"],
                                "split_seed": stage["split_seed"],
                                "first_fourier_term": stage["first_fourier_term"],
                                "torch_manual_seed": stage["torch_manual_seed"],
                            })
    return configs


def mode_list_configs() -> None:
    """Print the complete search space without touching any data."""
    print("=" * 88)
    print("ADOPTED CONFIGURATION")
    print("=" * 88)
    for key in ("architecture_name", "architecture_tag", "branch_widths", "trunk_widths",
                "latent_dim", "fourier_modes", "first_fourier_term", "activation", "dropout",
                "batch_size", "learning_rate", "max_epochs", "early_stop_patience",
                "split_seed", "torch_manual_seed"):
        print(f"  {key:24s} : {ADOPTED_CONFIG[key]}")
    print(f"  {'loss':24s} : weighted MSE, w = 1 + {ALPHA} * |flux - 1|")
    print(f"  {'optimiser':24s} : Adam(lr={ADOPTED_CONFIG['learning_rate']}, "
          f"weight_decay={WEIGHT_DECAY})")
    print(f"  {'lr schedule':24s} : ReduceLROnPlateau({LR_SCHEDULER})")
    print(f"  {'split':24s} : 70/15/15 via two train_test_split calls, seed {SPLIT_SEED}")

    print()
    print("=" * 88)
    print("BASELINE FAMILIES (check_of_initials_arch.py / notebook cells 0-10)")
    print("=" * 88)
    for name, cfg in BASELINE_ARCHITECTURES.items():
        marker = "  <-- ADOPTED" if cfg.get("adopted") else ""
        print(f"  {name:16s} branch={cfg['branch_widths']} trunk={cfg['trunk_widths']} "
              f"latent={cfg['latent_dim']} first_term={cfg['first_fourier_term']}{marker}")
    print(f"  -> {len(BASELINE_ARCHITECTURES)} configurations x {len(LINE_FILES)} lines")

    total = len(BASELINE_ARCHITECTURES)
    print()
    print("=" * 88)
    print("GRID-SEARCH STAGES (don_emulator_5_par*.py)")
    print("=" * 88)
    for stage_name, stage in SEARCH_STAGES.items():
        n = len(enumerate_stage(stage))
        total += n
        print(f"\n  {stage_name}  [{stage['source']}]  -> {n} configurations")
        print(f"    {stage['description']}")
        print(f"    architectures : {len(stage['architectures'])}  "
              f"({', '.join(a['name'] for a in stage['architectures'])})")
        print(f"    latent_dims   : {stage['latent_dims']}")
        print(f"    activations   : {stage['activations']}")
        print(f"    dropout       : {stage['dropout_rates']}")
        print(f"    learning_rate : {stage['learning_rates']}")
        print(f"    batch_size    : {stage['batch_sizes']}")
        print(f"    fourier_modes : {stage['fourier_modes']}   max_epochs: {stage['max_epochs']}   "
              f"patience: {stage['early_stop_patience']}")
        print(f"    walltime guard: {stage['timeout_guard']}   resume: {stage['resume']}")

    print()
    print("=" * 88)
    print(f"TOTAL configurations reproducible from this file : {total}")
    print(f"Total reported in the thesis                     : "
          f"{THESIS_TABLE_RANGES['total_evaluated_configurations_reported_in_thesis']}")
    print("The difference comes from earlier sweeps (including the 64-Fourier-mode")
    print("comparison) whose driver scripts were not preserved; see THESIS_TABLE_RANGES.")
    print("=" * 88)


def mode_line(args: argparse.Namespace) -> None:
    """Train a single line with a single configuration (default: the adopted one)."""
    grid = GridData(args.models_root)
    config = dict(BASELINE_ARCHITECTURES[args.baseline])
    _apply_cli_overrides(config, args)

    line_file = args.line
    data = grid.build_dataset_for_line(line_file, verbose=True)
    device = resolve_device(args.device)
    print(f"\nUsing device: {device} (GPUs available: {torch.cuda.device_count()})")

    best_val, ckpt, loss_json = train_emulator_for_line(
        line_file=line_file,
        data=data,
        config=config,
        grid=grid,
        output_dir=args.output_dir,
        architecture_tag=args.architecture_tag,
        device=device,
        timeout_guard=args.timeout_guard is True,
    )
    print(f"\n{line_file}: best val loss = {best_val:.6e}")
    print(f"  checkpoint : {ckpt}")
    print(f"  loss curve : {loss_json}")


def mode_baseline(args: argparse.Namespace) -> None:
    """Train all 17 lines for one (or all) of the four hand-written families.

    This is the ``check_of_initials_arch.py`` code path, which is also the one
    that produced the released ``emulators_per_line_hg`` checkpoints when run
    with ``--baseline original``.
    """
    grid = GridData(args.models_root)
    line_files = args.lines or grid.line_files
    device = resolve_device(args.device)
    print(f"\nUsing device: {device} (GPUs available: {torch.cuda.device_count()})")

    names = list(BASELINE_ARCHITECTURES) if args.baseline == "all" else [args.baseline]
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_records = load_existing_summary(summary_path) if args.resume else []

    for name in names:
        config = dict(BASELINE_ARCHITECTURES[name])
        _apply_cli_overrides(config, args)
        tag = args.architecture_tag or str(config["architecture_tag"])
        print("\n" + "#" * 80)
        print(f"##### Baseline family: {name}  (tag {tag}) #####")
        print("#" * 80, flush=True)

        results: Dict[str, object] = {
            "architecture_tag": tag,
            "architecture_name": config["architecture_name"],
            "branch_widths": config["branch_widths"],
            "trunk_widths": config["trunk_widths"],
            "latent_dim": config["latent_dim"],
            "activation": config["activation"],
            "dropout": config["dropout"],
            "fourier_modes": config["fourier_modes"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "first_term": "constant_1" if config["first_fourier_term"] == "ones" else "x",
        }

        for line_file in line_files:
            print("\n" + "=" * 80)
            print(f"Training emulator for spectral line: {line_file}")
            print(f"Config: {tag}")
            print("=" * 80, flush=True)
            if args.resume and is_line_complete(args.output_dir, tag, line_file):
                print(f"[resume] Found existing artifacts for {tag} / {line_file}", flush=True)
                results[line_file] = load_existing_line_val_loss(args.output_dir, tag, line_file)
                continue
            try:
                data = grid.build_dataset_for_line(line_file, verbose=True)
                best_val, _, _ = train_emulator_for_line(
                    line_file=line_file, data=data, config=config, grid=grid,
                    output_dir=args.output_dir, architecture_tag=tag, device=device,
                    timeout_guard=args.timeout_guard is True,
                )
                results[line_file] = float(best_val)
            except Exception as e:
                print(f"Skipping {line_file} due to error: {e}", flush=True)
                results[line_file] = None

        summary_records.append(results)
        save_summary(summary_path, summary_records)
        print(f"\nFinished family {name}. Validation losses:")
        for lf in line_files:
            val = results.get(lf)
            if isinstance(val, (int, float)):
                print(f"  {lf}: {val:.3e}")
            else:
                print(f"  {lf}: ({val})")

    print(f"\nFinal summary written to {summary_path}")


def mode_search(args: argparse.Namespace) -> None:
    """Run one (or every) stage of the architecture search."""
    stage_names = list(SEARCH_STAGES) if args.stage == "all" else [args.stage]

    grid = GridData(args.models_root)
    line_files = args.lines or grid.line_files
    preloaded = grid.preload(line_files)
    device = resolve_device(args.device)
    print(f"\nUsing device: {device} (GPUs available: {torch.cuda.device_count()})")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "training_summary.json")

    for stage_name in stage_names:
        stage = SEARCH_STAGES[stage_name]
        # Per-stage defaults, overridable from the command line.
        timeout_guard = bool(stage["timeout_guard"]) if args.timeout_guard is None else args.timeout_guard
        resume = bool(stage["resume"]) if args.resume is None else args.resume

        configs = enumerate_stage(stage)
        print("\n" + "#" * 88)
        print(f"##### STAGE {stage_name}: {len(configs)} configurations "
              f"({stage['source']}) #####")
        print(f"##### walltime guard: {timeout_guard}   resume: {resume} #####")
        print("#" * 88, flush=True)
        if timeout_guard and TIME_LIMIT_SECONDS is not None:
            print(f"SLURM time limit detected: {TIME_LIMIT_SECONDS/3600:.2f} h | "
                  f"stop buffer: {STOP_BEFORE_TIMEOUT_SECONDS/60:.1f} min", flush=True)

        summary_records = load_existing_summary(summary_path) if resume else []
        summary_map = {summary_key(r): r for r in summary_records}

        try:
            for config in configs:
                ensure_time_budget("config scheduling", minimum_seconds=180, enabled=timeout_guard)
                _apply_cli_overrides(config, args)
                tag = str(config["architecture_tag"])
                key = (tag, config["latent_dim"], config["activation"], config["dropout"],
                       config["learning_rate"], config["batch_size"])
                config_results = summary_map.get(key, {
                    "architecture_tag": tag,
                    "architecture_name": config["architecture_name"],
                    "branch_widths": config["branch_widths"],
                    "trunk_widths": config["trunk_widths"],
                    "latent_dim": config["latent_dim"],
                    "activation": config["activation"],
                    "dropout": config["dropout"],
                    "fourier_modes": config["fourier_modes"],
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                })

                pending_lines = [
                    lf for lf in preloaded
                    if config_results.get(lf) is None
                    and not (resume and is_line_complete(args.output_dir, tag, lf))
                ]
                if resume and not pending_lines:
                    print(f"[resume] Skipping completed config: {tag}", flush=True)
                    summary_map[key] = config_results
                    continue

                print(f"\n##### Training config: {tag} #####", flush=True)
                print(f"  Branch widths: {config['branch_widths']}", flush=True)
                print(f"  Trunk widths : {config['trunk_widths']}", flush=True)
                print(f"  Latent dim   : {config['latent_dim']}", flush=True)
                print(f"  Activation   : {config['activation']}", flush=True)
                print(f"  Dropout      : {config['dropout']}", flush=True)
                print(f"  Learning rate: {config['learning_rate']}", flush=True)
                print(f"  Batch size   : {config['batch_size']}", flush=True)
                print(f"  Pending lines: {len(pending_lines)}", flush=True)

                for line_file in preloaded:
                    if config_results.get(line_file) is not None:
                        continue
                    if resume and is_line_complete(args.output_dir, tag, line_file):
                        print(f"[resume] Found existing artifacts for {tag} / {line_file}", flush=True)
                        existing_val = load_existing_line_val_loss(args.output_dir, tag, line_file)
                        config_results[line_file] = (
                            existing_val if existing_val is not None else "completed_existing"
                        )
                        summary_map[key] = config_results
                        save_summary(summary_path, list(summary_map.values()))
                        continue

                    print("\n" + "=" * 80, flush=True)
                    print(f"Training emulator for spectral line: {line_file}", flush=True)
                    print(f"Config: {tag}", flush=True)
                    print("=" * 80, flush=True)
                    try:
                        best_val, _, _ = train_emulator_for_line(
                            line_file=line_file,
                            data=preloaded[line_file],
                            config=config,
                            grid=grid,
                            output_dir=args.output_dir,
                            architecture_tag=tag,
                            device=device,
                            timeout_guard=timeout_guard,
                        )
                        config_results[line_file] = float(best_val)
                    except DeadlineReached:
                        summary_map[key] = config_results
                        save_summary(summary_path, list(summary_map.values()))
                        raise
                    except Exception as e:
                        print(f"Skipping {line_file} due to error: {e}", flush=True)
                        config_results[line_file] = None
                    finally:
                        summary_map[key] = config_results
                        save_summary(summary_path, list(summary_map.values()))

                summary_map[key] = config_results
                save_summary(summary_path, list(summary_map.values()))

                print(f"\nFinished config {tag}. Validation losses:", flush=True)
                for lf in line_files:
                    val = config_results.get(lf)
                    if isinstance(val, (int, float)):
                        print(f"  {lf}: {val:.3e}", flush=True)
                    elif val is not None:
                        print(f"  {lf}: {val}", flush=True)
                    else:
                        print(f"  {lf}: (pending/error)", flush=True)
        except DeadlineReached as e:
            print(f"[deadline] {e}", flush=True)
            print(f"[deadline] Summary saved to {summary_path}", flush=True)
            save_summary(summary_path, list(summary_map.values()))
            return
        else:
            print(f"\nStage {stage_name} completed.", flush=True)
        finally:
            save_summary(summary_path, list(summary_map.values()))
            print(f"Final summary written to {summary_path}", flush=True)

    print("\nAll requested training runs completed.", flush=True)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Apply the optional single-hyperparameter overrides from the command line.

    Nothing here changes any recorded default; the overrides exist so that a
    reader can, for example, rerun the adopted architecture with 64 Fourier
    modes (``--fourier-modes 64``) as described in THESIS_TABLE_RANGES.
    """
    for cli_name, cfg_name, caster in (
        ("latent_dim", "latent_dim", int),
        ("fourier_modes", "fourier_modes", int),
        ("dropout", "dropout", float),
        ("activation", "activation", str),
        ("batch_size", "batch_size", int),
        ("learning_rate", "learning_rate", float),
        ("max_epochs", "max_epochs", int),
        ("patience", "early_stop_patience", int),
        ("alpha", "alpha", float),
        ("split_seed", "split_seed", int),
        ("first_fourier_term", "first_fourier_term", str),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            config[cfg_name] = caster(value)
    if getattr(args, "torch_seed", None) is not None:
        config["torch_manual_seed"] = None if args.torch_seed < 0 else int(args.torch_seed)


# =====================================================================================
# SECTION 8.  INFERENCE API   (was emulator_inference.py)
# =====================================================================================
# emulator_inference.py was the user-facing helper shipped to colleagues together
# with a folder of ``.pth`` checkpoints: PyTorch needs the Python class
# definitions to rebuild a network before its weights can be loaded, so the
# checkpoints alone are not enough.  Those definitions now live in Section 5 of
# this file, so the helper was folded in here rather than into the plotting
# script.  The public names and signatures are unchanged, so
#
#     from five_parameter_training import emulate_hg_spectrum_from_indat
#
# is a drop-in replacement for the old
#
#     from emulator_inference import emulate_hg_spectrum_from_indat
#
# Default behaviour: if no wavelength grid is supplied, each line is evaluated on
# a uniform grid of ``num_points`` points between the checkpoint's saved
# ``lambda_min`` and ``lambda_max``.  Custom grids are passed per line through
# ``wavelengths_by_line``, keyed by line name (e.g. ``"OUT.HGAMMA_VTV010"``).

DEFAULT_PARAM_ORDER: Tuple[str, ...] = PARAM_COLS
DEFAULT_EMULATOR_DIR = str(
    Path(
        os.environ.get(
            "FASTWIND_5PAR_CHECKPOINTS",
            OUTPUT_ROOT / "five_parameter" / "emulators_per_line_hg",
        )
    )
)

_CHECKPOINT_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, Mapping[str, object], torch.device]] = {}


def clear_emulator_cache() -> None:
    """Clear cached per-line emulator checkpoints."""
    _CHECKPOINT_CACHE.clear()


def _as_param_vector(params, param_order: Sequence[str] = DEFAULT_PARAM_ORDER) -> np.ndarray:
    if isinstance(params, Mapping):
        return np.asarray([params[name] for name in param_order], dtype=np.float32)
    return np.asarray(params, dtype=np.float32)


def _resolve_indat_path(indat_path_or_model_dir: str) -> str:
    """Return the INDAT file path from either a file path or a model directory."""
    if os.path.isfile(indat_path_or_model_dir):
        return indat_path_or_model_dir
    if not os.path.isdir(indat_path_or_model_dir):
        raise FileNotFoundError(
            f"INDAT path or model directory not found: {indat_path_or_model_dir}")
    for filename in ("INDAT", "INDAT.DAT"):
        candidate = os.path.join(indat_path_or_model_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No INDAT or INDAT.DAT found in {indat_path_or_model_dir}")


def read_indat_params(indat_path_or_model_dir: str,
                      default_v_turb: Optional[float] = None) -> Dict[str, float]:
    """Read the seven emulator parameters from a FASTWIND INDAT/INDAT.DAT file.

    Values are returned in physical units under the names expected by
    :func:`emulate_hg_spectrum`.
    """
    indat_path = _resolve_indat_path(indat_path_or_model_dir)
    with open(indat_path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    try:
        teff, logg, radius = map(float, lines[3].split()[:3])
        mdot, _vmin, v_inf, _beta, _vtrans = map(float, lines[5].split()[:5])
        y_he = float(lines[6].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse FASTWIND parameters from {indat_path}") from exc

    v_turb = default_v_turb
    if len(lines) > 8:
        try:
            v_turb = float(lines[8].split()[0])
        except (IndexError, ValueError):
            pass
    if v_turb is None:
        raise ValueError(
            f"Could not parse v_turb from {indat_path}; pass default_v_turb if it is fixed.")

    return {"Teff": teff, "logg": logg, "R": radius, "Mdot": mdot,
            "v_inf": v_inf, "Y_He": y_he, "v_turb": v_turb}


def normalize_params(params, param_mins: Sequence[float], param_maxs: Sequence[float],
                     param_order: Sequence[str] = DEFAULT_PARAM_ORDER) -> np.ndarray:
    """Normalise the seven stellar parameters exactly as during training.

    ``param_mins``/``param_maxs`` come from the checkpoint, so the scaling is
    guaranteed to match the one used when the network was fitted.
    """
    params_vec = _as_param_vector(params, param_order=param_order)
    param_mins = np.asarray(param_mins, dtype=np.float32)
    param_maxs = np.asarray(param_maxs, dtype=np.float32)
    if params_vec.shape[0] != len(param_order):
        raise ValueError(f"Expected {len(param_order)} parameters, got {params_vec.shape[0]}.")

    param_range = np.where(param_maxs - param_mins == 0.0, 1.0, param_maxs - param_mins)
    params_norm = (params_vec - param_mins) / param_range

    mdot_idx = list(param_order).index("Mdot")
    log_mdot = np.log10(params_vec[mdot_idx])
    log_mdot_min = np.log10(param_mins[mdot_idx])
    log_mdot_max = np.log10(param_maxs[mdot_idx])
    log_mdot_range = 1.0 if log_mdot_max == log_mdot_min else (log_mdot_max - log_mdot_min)
    params_norm[mdot_idx] = (log_mdot - log_mdot_min) / log_mdot_range
    return params_norm.astype(np.float32)


def _architecture_from_checkpoint(config: Mapping[str, object]) -> DeepONetModel:
    """Rebuild the network described by a checkpoint's ``config`` block.

    Newer checkpoints carry explicit ``branch_widths``/``trunk_widths``.  The
    earliest ones only recorded ``architecture_name``; for those the widths are
    looked up in BASELINE_ARCHITECTURES, which is what emulator_inference.py's
    ``_pick_architecture`` hard-coded.
    """
    architecture_name = str(config.get("architecture_name", "original"))
    latent_dim = int(config.get("latent_dim", 128))
    fourier_modes = int(config.get("fourier_modes", 32))
    activation = str(config.get("activation", "relu"))
    dropout = float(config.get("dropout", 0.0))
    branch_widths = config.get("branch_widths")
    trunk_widths = config.get("trunk_widths")
    first_fourier_term = config.get("first_fourier_term")

    if not branch_widths or not trunk_widths:
        base = BASELINE_ARCHITECTURES.get(architecture_name)
        if base is None:
            raise ValueError(
                f"Checkpoint does not record branch/trunk widths and "
                f"architecture_name={architecture_name!r} is not a known baseline.")
        branch_widths = base["branch_widths"]
        trunk_widths = base["trunk_widths"]
    if first_fourier_term is None:
        base = BASELINE_ARCHITECTURES.get(architecture_name)
        # Baseline checkpoints predate the key; they were all trained with "ones"
        # except original_xterm.  Search-stage checkpoints used "x".
        first_fourier_term = base["first_fourier_term"] if base is not None else "x"

    return build_deeponet({
        "latent_dim": latent_dim,
        "branch_widths": list(branch_widths),
        "trunk_widths": list(trunk_widths),
        "activation": activation,
        "dropout": dropout,
        "fourier_modes": fourier_modes,
        "first_fourier_term": first_fourier_term,
    }, input_dim=len(DEFAULT_PARAM_ORDER))


def load_emulator_checkpoint(checkpoint_path: str,
                             device: Optional[Union[str, torch.device]] = None):
    """Load one saved per-line emulator checkpoint and rebuild its network."""
    device = resolve_device(device)
    checkpoint_path = os.path.abspath(checkpoint_path)
    cache_key = (checkpoint_path, str(device))
    if cache_key in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[cache_key]

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch older than the weights_only argument
        state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config") or {}
    model = _architecture_from_checkpoint(config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    cached = (model, state, device)
    _CHECKPOINT_CACHE[cache_key] = cached
    return cached


def default_wavelength_grid(lambda_min: float, lambda_max: float,
                            num_points: int = LINE_MAX_ROWS) -> np.ndarray:
    """Return a uniform wavelength grid in Angstrom."""
    return np.linspace(lambda_min, lambda_max, num_points, dtype=np.float32)


def emulate_line(checkpoint_path: str, params, wavelengths: Optional[Iterable[float]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 num_points: int = LINE_MAX_ROWS,
                 param_order: Sequence[str] = DEFAULT_PARAM_ORDER) -> Dict[str, np.ndarray]:
    """Emulate one spectral line from one checkpoint.

    ``params`` is either a 7-element sequence in PARAM_COLS order or a mapping
    with those keys.  ``wavelengths`` may be any grid inside the line window -
    this is the whole point of the coordinate-based formulation.
    """
    model, state, device = load_emulator_checkpoint(checkpoint_path, device=device)

    params_norm = normalize_params(params, state["param_mins"], state["param_maxs"],
                                   param_order=param_order)
    lambda_min = float(state["lambda_min"])
    lambda_max = float(state["lambda_max"])
    if wavelengths is None:
        wavelengths = default_wavelength_grid(lambda_min, lambda_max, num_points=num_points)

    wavelengths = np.asarray(list(wavelengths), dtype=np.float32)
    wavelengths_norm = (wavelengths - lambda_min) / (lambda_max - lambda_min)

    params_tensor = torch.tensor(params_norm[None, :], dtype=torch.float32, device=device)
    waves_tensor = torch.tensor(wavelengths_norm[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        flux = model(params_tensor, waves_tensor).detach().cpu().numpy().ravel()

    if USE_RESIDUAL:
        flux = flux + FLUX_OFFSET   # undo the residual target

    default_name = os.path.basename(checkpoint_path)
    if default_name.startswith("emulator_"):
        default_name = default_name[len("emulator_"):]
    if default_name.endswith(".pth"):
        default_name = default_name[: -len(".pth")]

    return {
        "line_name": (state.get("config") or {}).get("line_file", default_name),
        "wavelength": wavelengths,
        "flux": flux.astype(np.float32),
    }


def emulate_hg_spectrum(params, emulator_dir: str = DEFAULT_EMULATOR_DIR,
                        wavelengths_by_line: Optional[Mapping[str, Iterable[float]]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        num_points: int = LINE_MAX_ROWS,
                        param_order: Sequence[str] = DEFAULT_PARAM_ORDER
                        ) -> Dict[str, Dict[str, np.ndarray]]:
    """Emulate every line whose checkpoint is found in ``emulator_dir``."""
    checkpoint_paths = sorted(glob.glob(os.path.join(emulator_dir, "emulator_*.pth")))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No emulator_*.pth files found in {emulator_dir}")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for checkpoint_path in checkpoint_paths:
        line_name = os.path.basename(checkpoint_path)[len("emulator_"):-len(".pth")]
        wavelengths = None if wavelengths_by_line is None else wavelengths_by_line.get(line_name)
        results[line_name] = emulate_line(
            checkpoint_path=checkpoint_path, params=params, wavelengths=wavelengths,
            device=device, num_points=num_points, param_order=param_order,
        )
    return results


def merge_line_spectra(line_results: Mapping[str, Mapping[str, np.ndarray]],
                       output_wavelength: Optional[Iterable[float]] = None,
                       fill_value: float = 1.0) -> Dict[str, np.ndarray]:
    """Merge per-line emulator outputs onto one wavelength grid.

    Each line is interpolated only inside its own window; points covered by no
    window are set to ``fill_value`` (the continuum), and points covered by more
    than one window are averaged.  ``coverage`` records how many windows
    contributed to each point.
    """
    if not line_results:
        raise ValueError("line_results is empty.")

    if output_wavelength is None:
        output_wavelength = np.unique(np.concatenate(
            [np.asarray(r["wavelength"], dtype=np.float32) for r in line_results.values()]))

    wavelength = np.asarray(list(output_wavelength), dtype=np.float32)
    if wavelength.ndim != 1:
        raise ValueError("output_wavelength must be one-dimensional.")

    order = np.argsort(wavelength)
    wavelength_sorted = wavelength[order]
    flux_sum = np.zeros_like(wavelength_sorted, dtype=np.float64)
    count = np.zeros_like(wavelength_sorted, dtype=np.float64)

    for result in line_results.values():
        line_wavelength = np.asarray(result["wavelength"], dtype=np.float32)
        line_flux = np.asarray(result["flux"], dtype=np.float32)
        if line_wavelength.ndim != 1 or line_flux.ndim != 1:
            raise ValueError("Each line result must contain 1-D wavelength and flux arrays.")
        if line_wavelength.size != line_flux.size:
            raise ValueError("Line wavelength and flux arrays must have the same length.")
        if line_wavelength.size == 0:
            continue
        line_order = np.argsort(line_wavelength)
        line_wavelength = line_wavelength[line_order]
        line_flux = line_flux[line_order]
        mask = (wavelength_sorted >= line_wavelength[0]) & (wavelength_sorted <= line_wavelength[-1])
        if not np.any(mask):
            continue
        flux_sum[mask] += np.interp(wavelength_sorted[mask], line_wavelength, line_flux)
        count[mask] += 1.0

    flux_sorted = np.full_like(wavelength_sorted, fill_value, dtype=np.float64)
    covered = count > 0
    flux_sorted[covered] = flux_sum[covered] / count[covered]

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    return {
        "wavelength": wavelength_sorted[inverse_order],
        "flux": flux_sorted[inverse_order].astype(np.float32),
        "coverage": count[inverse_order].astype(np.int16),
    }


def emulate_hg_spectrum_from_indat(indat_path_or_model_dir: str,
                                   emulator_dir: str = DEFAULT_EMULATOR_DIR,
                                   output_wavelength: Optional[Iterable[float]] = None,
                                   wavelengths_by_line: Optional[Mapping[str, Iterable[float]]] = None,
                                   device: Optional[Union[str, torch.device]] = None,
                                   num_points: int = LINE_MAX_ROWS,
                                   default_v_turb: Optional[float] = None,
                                   fill_value: float = 1.0,
                                   return_lines: bool = True) -> Dict[str, object]:
    """Read parameters from INDAT, run every per-line emulator, and merge them.

    Returns ``{"params": ..., "spectrum": {...}, "lines": {...}}``.
    """
    params = read_indat_params(indat_path_or_model_dir, default_v_turb=default_v_turb)
    line_results = emulate_hg_spectrum(params=params, emulator_dir=emulator_dir,
                                       wavelengths_by_line=wavelengths_by_line,
                                       device=device, num_points=num_points)
    spectrum = merge_line_spectra(line_results=line_results,
                                  output_wavelength=output_wavelength,
                                  fill_value=fill_value)
    result: Dict[str, object] = {"params": params, "spectrum": spectrum}
    if return_lines:
        result["lines"] = line_results
    return result


def mode_infer(args: argparse.Namespace) -> None:
    """``--mode infer``: predict a spectrum from a folder of trained checkpoints."""
    if args.indat:
        output_wavelength = None
        if args.wavelength_range:
            lo, hi, n = args.wavelength_range
            output_wavelength = np.linspace(float(lo), float(hi), int(n))
        result = emulate_hg_spectrum_from_indat(
            args.indat, emulator_dir=args.emulator_dir,
            output_wavelength=output_wavelength, device=args.device,
            num_points=args.num_points, default_v_turb=args.default_v_turb,
        )
        params = result["params"]
        spectrum = result["spectrum"]
        lines = result["lines"]
    else:
        if args.params is None or len(args.params) != len(DEFAULT_PARAM_ORDER):
            raise SystemExit(
                "--mode infer needs either --indat PATH or --params "
                + " ".join(DEFAULT_PARAM_ORDER))
        params = dict(zip(DEFAULT_PARAM_ORDER, [float(v) for v in args.params]))
        lines = emulate_hg_spectrum(params, emulator_dir=args.emulator_dir,
                                    device=args.device, num_points=args.num_points)
        output_wavelength = None
        if args.wavelength_range:
            lo, hi, n = args.wavelength_range
            output_wavelength = np.linspace(float(lo), float(hi), int(n))
        spectrum = merge_line_spectra(lines, output_wavelength=output_wavelength)

    print("Parameters:")
    for name in DEFAULT_PARAM_ORDER:
        print(f"  {name:8s} = {params[name]:g}")
    print(f"\nEmulated {len(lines)} line windows from {args.emulator_dir}")
    print(f"Merged spectrum: {spectrum['wavelength'].size} points, "
          f"{int((spectrum['coverage'] > 0).sum())} covered by at least one window")

    if args.output_npz:
        payload = {"wavelength": spectrum["wavelength"], "flux": spectrum["flux"],
                   "coverage": spectrum["coverage"]}
        for line_name, res in lines.items():
            payload[f"{line_name}__wavelength"] = res["wavelength"]
            payload[f"{line_name}__flux"] = res["flux"]
        np.savez_compressed(args.output_npz, **payload)
        print(f"Wrote {args.output_npz}")


# =====================================================================================
# SECTION 9.  COMMAND-LINE INTERFACE
# =====================================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="five_parameter_training.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=("list-configs", "line", "baseline", "search", "infer"),
                   help="What to do.  See the module docstring for examples.")

    # --- data / output -------------------------------------------------------
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                   help="Directory holding one numbered subdirectory per FASTWIND model.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where checkpoints, loss curves and training_summary.json are written.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")

    # --- which lines ---------------------------------------------------------
    p.add_argument("--line", default="OUT.HGAMMA_VTV010",
                   help="Single line window for --mode line.")
    p.add_argument("--lines", nargs="+", default=None,
                   help="Restrict --mode baseline/search to these line windows "
                        "(default: every OUT.* window found in the reference model).")

    # --- which configuration -------------------------------------------------
    p.add_argument("--baseline", default="original",
                   choices=list(BASELINE_ARCHITECTURES) + ["all"],
                   help="Hand-written family for --mode line / --mode baseline. "
                        "'original' is the adopted architecture.")
    p.add_argument("--stage", default="all", choices=list(SEARCH_STAGES) + ["all"],
                   help="Which architecture-search stage to run for --mode search.")
    p.add_argument("--architecture-tag", default=None,
                   help="Override the output subfolder name.")

    # --- optional single-hyperparameter overrides ---------------------------
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=None)
    p.add_argument("--fourier-modes", dest="fourier_modes", type=int, default=None,
                   help="32 in every preserved script; 64 was also explored (see "
                        "THESIS_TABLE_RANGES).")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--activation", default=None, choices=list(ACTIVATIONS))
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="Core-weighting strength of the loss (5.0 in every run).")
    p.add_argument("--split-seed", dest="split_seed", type=int, default=None)
    p.add_argument("--first-fourier-term", dest="first_fourier_term", default=None,
                   choices=("ones", "x"))
    p.add_argument("--torch-seed", dest="torch_seed", type=int, default=None,
                   help="torch.manual_seed value; pass a negative number to leave "
                        "torch unseeded, as the search scripts did.")

    # --- run management (the 'timeout' variants) -----------------------------
    guard = p.add_mutually_exclusive_group()
    guard.add_argument("--timeout-guard", dest="timeout_guard", action="store_true",
                       default=None,
                       help="Stop cleanly before the SLURM walltime "
                            "($SLURM_TIMELIMIT minus $DON_STOP_BEFORE_TIMEOUT_SECONDS, "
                            "default 300 s) or $DON_MAX_RUNTIME_SECONDS. Default: on "
                            "for stage3/stage4, off elsewhere.")
    guard.add_argument("--no-timeout-guard", dest="timeout_guard", action="store_false",
                       help="Disable the walltime guard.")
    res = p.add_mutually_exclusive_group()
    res.add_argument("--resume", dest="resume", action="store_true", default=None,
                     help="Skip (config, line) pairs whose .pth and _loss.json already "
                          "exist. Default: on for stage3/stage4, off elsewhere.")
    res.add_argument("--no-resume", dest="resume", action="store_false",
                     help="Retrain everything from scratch.")

    # --- inference -----------------------------------------------------------
    p.add_argument("--emulator-dir", default=DEFAULT_EMULATOR_DIR,
                   help="Folder of emulator_*.pth checkpoints for --mode infer.")
    p.add_argument("--indat", default=None,
                   help="FASTWIND INDAT file (or model directory) for --mode infer.")
    p.add_argument("--params", nargs=7, default=None, metavar=tuple(DEFAULT_PARAM_ORDER),
                   help="Physical parameters for --mode infer, in place of --indat.")
    p.add_argument("--num-points", dest="num_points", type=int, default=LINE_MAX_ROWS,
                   help="Points per line when no explicit grid is given.")
    p.add_argument("--wavelength-range", nargs=3, default=None,
                   metavar=("LAMBDA_MIN", "LAMBDA_MAX", "N"),
                   help="Merge the emulated lines onto this uniform grid.")
    p.add_argument("--default-v-turb", dest="default_v_turb", type=float, default=None,
                   help="Fallback microturbulence if the INDAT deck does not list it.")
    p.add_argument("--output-npz", dest="output_npz", default=None,
                   help="Write the emulated spectrum to this .npz file.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.mode == "list-configs":
        mode_list_configs()
    elif args.mode == "line":
        if args.baseline == "all":
            raise SystemExit("--mode line needs a single --baseline, not 'all'.")
        mode_line(args)
    elif args.mode == "baseline":
        mode_baseline(args)
    elif args.mode == "search":
        mode_search(args)
    elif args.mode == "infer":
        mode_infer(args)
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"Unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
