#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
FASTWIND FLUXCONT (continuum / SED) DeepONet emulator -- consolidated training script
=====================================================================================

This single script reproduces *every* configuration of the FLUXCONT architecture
search that was carried out for the MSc thesis.  It replaces the 18 near-duplicate
files ``FLUXCONT_emulator.py`` and ``FLUXCONT_emulator_v2.py`` ... ``_v18.py``
(23,720 lines in total) by one implementation plus a ``CONFIGS`` dictionary.

Nothing was tuned, simplified or re-derived while consolidating: every numerical
value below was copied from the corresponding source file, and every behavioural
difference between two source files is expressed as a configuration key, never as
a code change.  Where two versions did genuinely different things (for example the
plain mean-squared-error loss of v1 versus the region-weighted Huber loss of v2+,
or the two different derivative penalties of v6 and v13), *both* code paths are
present and are selected by configuration.


--------------------------------------------------------------------------------
1.  What is FLUXCONT and what does this emulator predict?
--------------------------------------------------------------------------------

``FLUXCONT`` is a FASTWIND output file.  FASTWIND is a non-LTE, spherically
symmetric, line-blanketed model-atmosphere and wind code for hot massive stars.
For every converged model it writes, among many other products, the emergent
*continuum* spectral energy distribution (SED) sampled on the code's own
frequency grid.  The line emulators of the other thesis chapters predict
continuum-*normalised* line profiles inside narrow wavelength windows; this
emulator instead predicts the broad continuum itself, from the ultraviolet
through the optical and into the near/mid infrared.

The continuum matters because it sets the ionising photon output, the infrared
excess produced by the wind, and the absolute flux scale against which the
normalised line profiles are measured.  Its shape varies smoothly with wavelength
and carries no narrow line morphology, which is why no five-parameter warm-up
emulator was needed and the continuum emulator was trained directly on the
13-parameter Latin-hypercube grid.

Branch input  (13 parameters, the "operator" input of the DeepONet):

    theta = (teff, logg, radius, mdot, yhe, C, N, O, beta, vinf, fic, fvel, fclump)

    - teff, logg, radius : photospheric structure
    - mdot, beta, vinf   : wind mass-loss rate, velocity-law exponent, terminal speed
    - yhe, C, N, O       : helium abundance and CNO abundances
    - fic, fvel, fclump  : the three clumping parameters (interclump density
                           contrast, velocity filling, clumping factor)

    ``mdot`` is transformed to log10(mdot) *first* because mass-loss rates span
    several decades; all 13 components are then min-max scaled to [0, 1] using
    the minimum and maximum over the models that survive the quality selection.

Trunk input (the "coordinate" input of the DeepONet):

    a single scalar, the normalised logarithmic wavelength (see section 3).

Target:

    the standardised log10 of the converted flux density F_lambda (see section 2).

``rmax`` (the outer radius of the FASTWIND atmosphere in units of the stellar
radius) is read from the grid JSON and used **only** in the flux conversion.  It
is deliberately *not* a network input: it is a property of the numerical model
setup, not a physical parameter of the star.


--------------------------------------------------------------------------------
2.  Why the flux is handled in log space
--------------------------------------------------------------------------------

FASTWIND stores log10(F_nu), a flux per unit *frequency*, in the third column of
FLUXCONT (the second column is the wavelength in Angstrom).  This script converts
it to a flux per unit *wavelength* at the stellar surface and rescales it to the
outer boundary of the model atmosphere:

    F_nu     = 10 ** (LOG F-NUE)
    F_lambda = 3.00e18 * F_nu / lambda_Angstrom**2          # nu-to-lambda Jacobian
    F_lambda = F_lambda * 4 * pi * (R_sun_cm * radius)**2   # stellar surface area
    F_lambda = F_lambda * rmax**2                           # outer-boundary scaling

The factor 3.00e18 is the speed of light in Angstrom/s, which is the Jacobian
|dnu/dlambda| = c / lambda**2 of the change of variable.  The exact literal
``3.00e18`` (not 2.99792458e18) is what the original code used and is preserved.

Over the retained domain F_lambda spans roughly 10**28 to 10**37 erg/s/Angstrom.
Regressing on such a quantity directly would be hopeless: the loss would be
dominated by the few brightest wavelengths, the optimiser would see gradients
spanning nine decades, and a 1% error in the ultraviolet would be numerically
invisible next to a 1% error at the peak.  Working with

    q(lambda) = log10(F_lambda)

turns the multiplicative dynamic range into an additive one of order ten, makes
the *relative* flux error (the quantity actually reported in the thesis) an
approximately uniform function of the residual, and matches the way the physics
behaves -- opacities, optical depths and the Planck function all vary
multiplicatively with the stellar parameters.

Only finite, strictly positive (lambda, F_lambda) pairs survive; a non-positive
or non-finite flux cannot be represented in log space and would be a numerical
artefact of the model calculation rather than a physical prediction.


--------------------------------------------------------------------------------
3.  Why the wavelength coordinate is logarithmic and normalised
--------------------------------------------------------------------------------

The retained SED covers 10**3 to 10**5 Angstrom in the final configuration, and
the early configurations covered roughly 20 to 10**7 Angstrom.  On a *linear*
wavelength axis almost all of the trunk input range would be occupied by the
infrared tail, and the entire ultraviolet -- where the continuum changes fastest
and where the ionising flux lives -- would be squeezed into a sliver near zero.
A network fed a linear coordinate would have to resolve structure over five
decades of scale with a single set of weights.

The coordinate is therefore

    x = ( log10(lambda) - log10(lambda)_min ) / ( log10(lambda)_max - log10(lambda)_min )

so that x is in [0, 1] with roughly equal resolution per decade.  The min and max
are computed once over *all* retained records, so x is a fixed, dataset-level
coordinate and the same value of x always means the same wavelength.

The trunk then expands x into Fourier features

    [1, sin(2*pi*k*x), cos(2*pi*k*x)]  for k = 1 .. FOURIER_MODES

giving 2*FOURIER_MODES + 1 inputs.  Plain MLPs have a strong spectral bias: they
learn the lowest-frequency component of a target function far faster than the
higher-frequency ones.  Feeding an explicit Fourier basis removes that bias and
lets a modest network represent the curvature of the SED (the Balmer/Paschen
jumps, the shape of the free-free infrared excess) without needing to be deep in
the coordinate direction.  The architecture search tested 32, 64 and 96 modes.


--------------------------------------------------------------------------------
4.  Sampling intervals versus standardisation bins  (two different things)
--------------------------------------------------------------------------------

These two numbers are easy to confuse; they are unrelated and serve different
purposes.

``sampling_bins``  (64 in every configuration that uses balanced sampling)
    Controls *which native points are handed to the network in a training
    example*.  Each training example is one SED, from which
    ``wave_samples_per_model`` entries are drawn.  The [0, 1] coordinate is split
    into ``sampling_bins`` equal intervals and points are drawn from every
    occupied interval, so a batch is not dominated by whichever part of the
    spectrum happens to be most densely sampled on the native FASTWIND grid.

    Note that entries are *not* distinct points.  After the 10**3-10**5 Angstrom
    restriction each SED retains between 743 and 860 native points, i.e. fewer
    than the 1024 entries requested, so some native pairs appear more than once
    in a training example.  This is resampling with replacement of *existing*
    FASTWIND pairs.  No wavelength or flux value is ever interpolated, invented,
    or moved onto a common grid: every model keeps its own native wavelengths.

``target_norm_bins``  (200, only when ``target_scaling == "binned"``)
    Controls *how the target value is standardised*.  The [0, 1] coordinate is
    split into ``target_norm_bins`` equal intervals and the mean and standard
    deviation of q = log10(F_lambda) are accumulated per interval over the
    *training* records only.  Each target is then transformed with the statistics
    of its own bin.  Empty bins are filled by linear interpolation of the
    neighbouring bin centres, and standard deviations are floored at 1e-4.

    The alternative, ``target_scaling == "global"``, uses one mean and one
    standard deviation for the whole SED.  Because the typical continuum level
    falls by several decades from the ultraviolet to the infrared, a single
    global sigma makes the residual in the faint parts of the spectrum
    numerically negligible; the binned treatment lets the target scale follow the
    characteristic flux level and spread at each wavelength.  The search found
    this to be the single most important choice: on the same restricted domain
    the global model reached a mean relative error of ~2330% (driven by rare but
    enormous flux-space outliers) against ~13.2% for the binned model.

Target standardisation must not be confused with the [0, 1] mapping of the
wavelength *input* in section 3.  Both use bins of the same coordinate; they act
on different quantities.


--------------------------------------------------------------------------------
5.  The loss terms and why each exists
--------------------------------------------------------------------------------

The composite loss actually used is

    L = L_huber
        + a_bias  * L_bias
        + a_end   * L_end
        + a_tail  * L_tail          (only in the anchor experiments)
        + a_deriv * L_deriv         (only in v13-style experiments)
        + a_amp   * L_amp           (only in v13-style experiments)

and, separately, v6 used

    L = L_huber + a_edgederiv * L_edgederiv

while v1 used plain unweighted mean-squared error.

``L_huber`` -- region-weighted Huber penalty on the standardised target.
    rho_delta(e) = 0.5*e**2                for |e| <= delta
                 = delta*(|e| - 0.5*delta)  for |e| >  delta
    (implemented as ``0.5*q**2 + delta*l`` with ``q = min(|e|, delta)`` and
    ``l = |e| - q``, which is the same function).

    *Why Huber and not MSE*: the FLUXCONT curves occasionally contain a handful
    of points where the model atmosphere itself is noisy, and near the domain
    edges the emulator error is intrinsically larger.  Under MSE a single such
    residual contributes quadratically for ever and drags the whole fit; the
    Huber penalty becomes linear beyond ``delta`` so those points still push the
    solution in the right direction but cannot dominate the gradient.  ``delta``
    is 1.0 throughout, i.e. one standard deviation of the standardised target.

    *Why region weights*: on the broad early domain the ionising continuum below
    the Lyman limit, the mid-range and the far infrared tail have very different
    scientific value and very different numerical behaviour.  The weights (see
    ``REGION_WEIGHTS_*`` below) de-emphasise the sub-Lyman region, which is not
    the scientific target here and is the part most sensitive to the details of
    the atmosphere calculation, and up-weight the long-wavelength tail whose
    contribution after conversion back to linear flux would otherwise be
    negligible.  On the final 10**3-10**5 Angstrom domain *all* retained points
    fall in the unit-weight region, so the weighting is inert for the adopted
    model and is retained only so that the earlier runs remain reproducible.

``L_bias`` -- squared weighted mean signed residual of each sampled SED.
    L_bias = mean_over_SEDs[ ( sum_j w_j (yhat_j - y_j) / sum_j w_j )**2 ]

    *Why*: the pointwise Huber term is happy with a curve that is systematically
    2% high everywhere as long as no individual residual is large.  But a
    constant offset in log10(F_lambda) is exactly the error that survives
    conversion back to linear flux as a constant *relative* flux error, which is
    the headline metric of the thesis.  Squaring the *mean* residual over a whole
    curve penalises that coherent offset specifically, without penalising
    zero-mean scatter twice.  Coefficient 0.10 throughout.

``L_end`` -- Huber penalty repeated at the two extreme sampled coordinates.
    *Why*: the trunk's Fourier basis is periodic on [0, 1], and both the
    coordinate normalisation and the binned standardisation are least constrained
    at the very edges of the domain, where there is data on one side only.  The
    empirical symptom was a systematic droop of the predicted SED at the shortest
    and longest retained wavelengths.  The endpoint term re-applies the Huber
    penalty at the smallest and largest *sampled* coordinate of every SED in the
    batch, giving those two boundaries extra influence.

    Crucially these are existing FASTWIND points, chosen from the sample by
    ``argsort`` of the coordinates; the term introduces no synthetic wavelength
    or flux values outside the native grid.  The coefficient was scanned over
    0.005 / 0.01 / 0.02 and 0.01 was adopted.

``L_tail`` -- the same Huber penalty at the sampled points *nearest* a list of
    fixed normalised coordinates (v17d, v18a-d).  Used to test whether pinning
    3e4, 6e4 and 1e5 Angstrom -- or both domain edges at once -- reduced the
    residual structure in the infrared.  None of these improved the validation
    loss and the relative errors together, so the adopted model has a_tail = 0.

``L_deriv`` / ``L_edgederiv`` -- Huber penalty on the point-to-point *slope* of
    the curve, intended to encourage smooth predictions.  v6 applied it to the
    de-standardised log-flux slope, restricted to adjacent pairs whose midpoint
    lies below 900 Angstrom or above 1e5 Angstrom (a_edgederiv = 0.05); v13
    applied it to the standardised values over all adjacent pairs
    (a_deriv = 0.001).  The v6 variant destroyed the optimisation -- validation
    loss ~2.6e2, three orders of magnitude worse than anything else -- because
    consecutive sampled coordinates can be arbitrarily close, so the finite
    difference divides by a tiny dx and the gradient explodes.  Kept here purely
    for reproducibility of that negative result.

``L_amp`` -- squared difference of the mean level of the predicted and target
    curve (v13, a_amp = 0.01).  A weaker, unweighted cousin of ``L_bias``.


--------------------------------------------------------------------------------
6.  Which configuration is the adopted one?
--------------------------------------------------------------------------------

    ``v17a``  --  source file ``fluxcont/FLUXCONT_emulator_v17.py``, first entry
                  of its ``EXPERIMENTS`` list.

This is the configuration the thesis calls **"Adopted no BN control"**:
binned target standardisation over 10**3-10**5 Angstrom, dropout 0.02, no batch
or layer normalisation, 64 Fourier modes, Huber + bias(0.10) + endpoint(0.01),
no tail anchors, no residual blocks, learning rate 3e-4.

Reported metrics: best validation loss 1.27e-3 at epoch 84, test MSE in
log10(F_lambda) 3.49e-4, mean relative error 2.01e-2, median 1.40e-2.

``v17a`` is the default of ``--config``.


--------------------------------------------------------------------------------
7.  Version-to-change table
--------------------------------------------------------------------------------

The thesis appendix (Table "FLUXCONT architectures tested before selecting the
adopted model") lists 28 configurations under descriptive names.  The mapping to
the source files is below.  The appendix explicitly notes that "Versions 11 to 13
were not available and are therefore not included" -- their result folders were
missing -- so v11a-c, v12a-b and v13a appear here but have no thesis row.

  config  source file  thesis name                    what changed vs. the previous step
  ------  -----------  -----------------------------  ------------------------------------------------
  v1      _emulator    Initial baseline               first DeepONet continuum test: global target
                       .py                            standardisation, plain unweighted MSE, uniform
                                                      random wavelength sampling, dropout 0.30 after
                                                      hidden layers 3 and 5, 32 Fourier modes, batch
                                                      norm, no wavelength cut (full ~20-1e7 A file),
                                                      FLUXCONT parsed with numpy.loadtxt
  v2      _v2.py       Weighted binned target         wavelength-binned target standardisation (200
                                                      bins); region-weighted Huber loss replaces MSE;
                                                      balanced sampling over 64 log-lambda intervals;
                                                      column-wise FLUXCONT parser replaces loadtxt
  v3      _v3.py       Global, low dropout            back to global standardisation; dropout
                                                      0.30 -> 0.02 and now after all six layers;
                                                      lambda >= 100 A; UV weight 1.25 -> 2.0
  v4      _v4.py       Global, high dropout           dropout 0.02 -> 0.10 (only change)
  v5      _v5.py       Binned, 64 modes               binned standardisation again; Fourier modes
                                                      32 -> 64; dropout 0.10; UV weight back to 1.25
  v6      _v6.py       Binned + derivative term       added edge-restricted derivative Huber term
                                                      (alpha 0.05) on de-standardised log-flux slopes
                                                      below 900 A and above 1e5 A
  v7      _v7.py       Global, cut 200                global standardisation; lambda >= 200 A;
                                                      dropout 0.02; derivative term dropped and the
                                                      curve-bias term (alpha 0.10) introduced;
                                                      UV weight 2.0
  v8      _v8.py       Binned, cut 200                same as v7 but binned standardisation and
                                                      UV weight 1.25
  v9      _v9.py       Global full SED                global; lambda >= 100 A; 1024 -> 2500 entries per
                                                      SED together with the "full SED" sampling branch
                                                      that takes every native point and pads;
                                                      train-evaluation loss curve added
  v10     _v10.py      Global full SED + endpoint     added the endpoint anchor term, alpha 0.05
  v11     _v11.py      (results unavailable)          converted to a multi-experiment runner
          v11a/b/c                                    (EXPERIMENTS list, one shared data-preparation
                                                      pass); lambda >= 200 A; endpoint alpha scanned
                                                      over 0.005 / 0.010 / 0.020
  v12     _v12.py      (results unavailable)          binned standardisation, lambda >= 200 A,
          v12a/b                                      1024 entries per SED, endpoint 0.005 / 0.010
  v13     _v13.py      (results unavailable)          global, lambda >= 200 A, endpoint 0.010 plus a
          v13a                                        standardised-space derivative term (0.001) and
                                                      a curve-amplitude term (0.010)
  v14     _v14.py      Binned no endpoint /           wavelength domain finally restricted to
          v14a/b       Binned + endpoint              1e3-1e5 A; binned standardisation, 1024 entries;
                                                      endpoint 0.0 vs 0.010
  v15     _v15.py      Global no endpoint /           same restricted domain with global
          v15a/b       Global + endpoint              standardisation and 2500 entries;
                                                      endpoint 0.0 vs 0.010
  v16     _v16.py      Binned no BN control /         hidden-layer normalisation ablation on the
          v16a/b/c/d   Binned layer norm /            restricted binned setup: none / layer norm /
                       Binned BN, lower learning      batch norm with lr 1e-4 / batch norm with
                       rate / Binned BN, weaker       endpoint 0.005
                       endpoint
  v17     _v17.py      Adopted no BN control /        added the residual-MLP-block option and the
          v17a..e      No BN, weak endpoint /         fixed tail-anchor loss to the code base; endpoint
                       No BN, strong endpoint /       scanned 0.010 / 0.005 / 0.020 without
                       No BN + tail anchors /         normalisation, then tail anchors (0.010) and
                       No BN + residual blocks        residual blocks.  v17a is the ADOPTED model.
  v18     _v18.py      Tail anchors, weak weight /    made the Fourier-mode count a per-experiment
          v18a..e      Tail anchors, stronger         setting; weak/mild tail anchors (0.0025 / 0.005),
                       weight / Both-edge anchors,    anchors at both domain edges (0.0025) with 64 and
                       64 modes / Both-edge anchors,  96 modes, and a 96-mode control with no anchors
                       96 modes / Wider Fourier basis


--------------------------------------------------------------------------------
8.  Usage
--------------------------------------------------------------------------------

Train the adopted model (the default)::

    python fluxcont_training.py

Train it explicitly, or any other historical configuration::

    python fluxcont_training.py --config v17a
    python fluxcont_training.py --config v1
    python fluxcont_training.py --config v16b

Reproduce a whole multi-experiment file in one go, sharing the (expensive) data
preparation exactly as the original ``EXPERIMENTS`` runner did::

    python fluxcont_training.py --group v17          # v17a .. v17e
    python fluxcont_training.py --config v18a,v18b   # any explicit subset

Inspect without training::

    python fluxcont_training.py --list
    python fluxcont_training.py --config v17a --print-config
    python fluxcont_training.py --config v17a --dry-run

Point at the data and choose an output location::

    python fluxcont_training.py --config v17a \
        --storage-dir data/fluxcont \
        --output-root outputs/fluxcont_runs

Every path and every hyper-parameter can also be overridden through the same
``ML13_*`` environment variables the original scripts used, e.g.
``ML13_SED_MAX_EPOCHS``, ``ML13_SED_BATCH_SIZE``, ``ML13_SED_LEARNING_RATE``,
``ML13_STORAGE_DIR``.  Environment overrides are applied *after* the chosen
CONFIGS entry, so an unset environment reproduces the published run exactly.

Outputs written to ``<output-root>/<run_tag>/`` (identical names and contents to
the originals, which is what ``fluxcont_plots.py`` consumes)::

    emulator_FLUXCONT.pth                   checkpoint + full configuration
    emulator_FLUXCONT_loss.json             loss histories
    emulator_FLUXCONT_split.json            train / val / test model ids
    emulator_FLUXCONT_test_outputs.npz      flattened native-grid test predictions
    emulator_FLUXCONT_meta.json             run metadata and aggregate test metrics
    test_metrics_per_model.csv              per-test-model metrics
    run_config.json                         resolved configuration
    parameter_normalization.json            13-parameter min-max scaling
    sed_target_and_wavelength_scaler.json   target scaler + wavelength range
    training_summary.json                   status / resume record

Requirements: python 3.8+, numpy, pandas, torch, scikit-learn.  ``zstd`` must be
on the PATH if the FLUXCONT files are zstd-compressed.


--------------------------------------------------------------------------------
9.  Config-key reference: one line per key, the single thing it changes
--------------------------------------------------------------------------------

Read top to bottom; each entry states only what differs from the entry above it
(or, for the lettered variants, from the first letter of the same group).

  v1     baseline: global target scaling, plain MSE, no region weights, uniform
         random sampling, 32 Fourier modes, dropout 0.30 after layers 3 and 5,
         batch norm, no wavelength cut, FLUXCONT read with ``numpy.loadtxt``
  v2     -> binned target scaling (200 bins) + region-weighted Huber (UV 1.25) +
         balanced sampling over 64 intervals + column-wise FLUXCONT parser
  v3     -> global target scaling; dropout 0.02 after all six layers;
         lambda >= 100 A; UV weight 2.0
  v4     -> dropout 0.10 (nothing else)
  v5     -> binned target scaling; 64 Fourier modes; dropout 0.10; UV weight 1.25
  v6     -> adds the edge-restricted de-standardised derivative Huber term,
         alpha 0.05, below 900 A and at/above 1e5 A
  v7     -> global scaling; lambda >= 200 A; dropout 0.02; derivative term
         removed; curve-bias term alpha 0.10 added; UV weight 2.0
  v8     -> binned scaling and UV weight 1.25 (otherwise identical to v7)
  v9     -> global scaling; lambda >= 100 A; 2500 entries per SED with the
         "full SED" sampling branch; per-epoch train-evaluation loss recorded
  v10    -> adds the endpoint anchor term, alpha 0.05
  v11a   -> lambda >= 200 A; endpoint alpha 0.005; multi-experiment runner
  v11b   -> endpoint alpha 0.010
  v11c   -> endpoint alpha 0.020
  v12a   -> binned scaling, 1024 entries per SED, no full-SED branch,
         UV weight 1.25, endpoint alpha 0.005
  v12b   -> endpoint alpha 0.010
  v13a   -> global scaling, 2500 entries, full-SED branch, UV weight 2.0,
         endpoint 0.010 plus standardised-space derivative 0.001 and
         curve-amplitude 0.010
  v14a   -> domain restricted to 1e3-1e5 A; binned scaling; 1024 entries;
         endpoint alpha 0.0; loss arrays also written into the .npz
  v14b   -> endpoint alpha 0.010
  v15a   -> global scaling with 2500 entries and the full-SED branch,
         UV weight 2.0, endpoint alpha 0.0
  v15b   -> endpoint alpha 0.010
  v16a   -> binned scaling again; no hidden-layer normalisation at all
  v16b   -> layer normalisation instead of none
  v16c   -> batch normalisation with learning rate 1.0e-4
  v16d   -> batch normalisation with endpoint alpha 0.005
  v17a   -> no normalisation, endpoint 0.010, no anchors, no residual blocks
         (the ADOPTED model)
  v17b   -> endpoint alpha 0.005
  v17c   -> endpoint alpha 0.020
  v17d   -> tail anchors at 3e4 / 6e4 / 1e5 A with alpha 0.010
  v17e   -> residual MLP blocks instead of plain layers
  v18a   -> tail anchors at 3e4 / 6e4 / 1e5 A with alpha 0.0025
  v18b   -> tail anchor alpha 0.005
  v18c   -> anchors at both domain edges (1e3, 1.2e3, 1.5e3, 3e4, 6e4, 1e5),
         alpha 0.0025
  v18d   -> the same both-edge anchors with 96 Fourier modes
  v18e   -> 96 Fourier modes with no tail anchors at all (control)

Group keys ``v11`` ... ``v18`` expand to all their lettered members and share a
single data-preparation pass, exactly as the original ``EXPERIMENTS`` runners
did.  ``v1`` ... ``v10`` are single runs and have no letters.
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
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Physical and grid constants (identical in all 18 source files)
# =============================================================================

#: Speed of light in Angstrom/s as it appears literally in the source scripts.
#: This is the Jacobian factor of the F_nu -> F_lambda change of variable,
#: F_lambda = (c / lambda**2) * F_nu.  The rounded value 3.00e18 rather than
#: 2.99792458e18 is what every historical run used and is preserved verbatim.
C_ANGSTROM_PER_S = 3.00e18

#: Solar radius in cm, used to turn ``radius`` (in solar radii) into a surface area.
RSUN_CM = 6.96e10

#: Branch-network input parameters, in the exact order used by every version.
PARAM_COLS = [
    "teff",    # effective temperature [K]
    "logg",    # surface gravity, log10(cgs)
    "radius",  # stellar radius [R_sun]      -- also used in the flux conversion
    "mdot",    # mass-loss rate [M_sun/yr]   -- log10 is taken before scaling
    "yhe",     # helium abundance N(He)/N(H)
    "C",       # carbon abundance
    "N",       # nitrogen abundance
    "O",       # oxygen abundance
    "beta",    # velocity-law exponent
    "vinf",    # terminal wind velocity [km/s]
    "fic",     # interclump density contrast
    "fvel",    # velocity filling factor
    "fclump",  # clumping factor
]

RADIUS_COL = "radius"
RMAX_COL = "rmax"  # outer radius of the FASTWIND grid; NOT a network input

#: Hidden-layer widths of both subnetworks.  Identical in all 18 versions.
BRANCH_WIDTHS = [128, 256, 512, 512, 1024, 2048]
TRUNK_WIDTHS = [128, 256, 512, 512, 1024, 2048]

#: Activation sequence, one entry per hidden layer.  Identical in all 18 versions.
#: LeakyReLU in the two narrow early layers keeps a non-zero gradient for negative
#: pre-activations and avoids dead units during the initial feature transform;
#: SiLU in the four wide layers is smooth and non-monotonic, which suits the
#: gradual construction of the latent representation.  This sequence was never
#: varied in the search, so its individual contribution cannot be isolated.
ACTIVATION_SEQUENCE = ["leaky_relu", "leaky_relu", "silu", "silu", "silu", "silu"]

#: Length of the branch/trunk latent vectors whose dot product is the prediction.
LATENT_DIM = 128

#: Minimum number of usable wavelength points for a FLUXCONT file to be accepted.
MIN_FLUXCONT_POINTS = 500

#: Wavelength points evaluated at once when predicting a full native grid.
EVAL_WAVE_CHUNK_SIZE = 4096

# --- Optimiser / schedule / split, identical in all 18 versions ---------------
OPTIMIZER = "adam"          # torch.optim.Adam
BATCH_SIZE = 32             # SEDs (curves) per batch, not points
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30    # epochs without validation improvement before stopping
SCHEDULER = "reduce_on_plateau"
SCHEDULER_FACTOR = 0.5      # lr multiplied by this on plateau
SCHEDULER_PATIENCE = 10     # scheduler patience, in epochs
SPLIT_SEED = 42             # sklearn random_state for both train/val/test splits
TEST_SIZE_FIRST_SPLIT = 0.30   # train vs (val+test)
TEST_SIZE_SECOND_SPLIT = 0.50  # val vs test, applied to the 30% remainder

#: Note on seeding: none of the source scripts called ``torch.manual_seed`` or
#: ``numpy.random.seed``.  Only the *data split* is seeded (SPLIT_SEED = 42, used
#: as ``random_state`` in both ``train_test_split`` calls).  Weight initialisation
#: and the per-epoch wavelength resampling are therefore not bit-reproducible.
#: This is faithful to the originals and is stated here rather than "fixed",
#: because adding a global seed would change the published numbers.


# =============================================================================
# Region-weight sets for the Huber term
# =============================================================================
#
# The weights multiply the pointwise Huber penalty as a function of the *physical*
# wavelength.  Two sets appear in the sources; they differ only in the weight of
# the 100-900 Angstrom band (1.25 in the wavelength-binned runs, 2.0 in the
# globally standardised runs).  v1 used no weighting at all.
#
# On the final 10**3-10**5 Angstrom domain every retained point lies in the
# ``mid`` band, so both sets reduce to unit weight and the choice is inert.

REGION_WEIGHTS_UV125 = {
    "uv_extreme_max": 100.0,    # lambda <  100 A          -> uv_extreme_weight
    "uv_max": 900.0,            # 100 <= lambda < 900 A    -> uv_weight
    "tail_start": 1.0e5,        # 900 <= lambda < 1e5 A    -> mid_weight
    "far_tail_start": 1.0e6,    # 1e5 <= lambda < 1e6 A    -> tail_weight
    "uv_extreme_weight": 1.0,   # lambda >= 1e6 A          -> far_tail_weight
    "uv_weight": 1.25,
    "mid_weight": 1.0,
    "tail_weight": 2.0,
    "far_tail_weight": 3.0,
}

REGION_WEIGHTS_UV200 = {
    "uv_extreme_max": 100.0,
    "uv_max": 900.0,
    "tail_start": 1.0e5,
    "far_tail_start": 1.0e6,
    "uv_extreme_weight": 1.0,
    "uv_weight": 2.0,
    "mid_weight": 1.0,
    "tail_weight": 2.0,
    "far_tail_weight": 3.0,
}

#: Normalised trunk coordinates of the fixed anchor wavelengths used by the
#: v17d / v18a-e experiments.  These were hard-coded in the sources as decimal
#: literals (they are log-normalised positions of 1e3, 1.2e3, 1.5e3, 3e4, 6e4 and
#: 1e5 Angstrom within the 1e3-1e5 domain) and are reproduced digit for digit.
#: Note the deliberate discrepancy at 6e4: v17d wrote 0.889075625191822 while
#: v18a-d wrote 0.8890756251918219.  Both literals are kept as they were.
TAIL_ANCHORS_V17D = [0.7385606273598312, 0.889075625191822, 1.0]
TAIL_ANCHORS_V18 = [0.7385606273598312, 0.8890756251918219, 1.0]
BOTH_EDGE_ANCHORS_V18 = [
    0.0,
    0.03959062302381233,
    0.08804562952784067,
    0.7385606273598312,
    0.8890756251918219,
    1.0,
]

#: Physical wavelengths (Angstrom) the anchor coordinates above correspond to.
#: Written into the run metadata exactly as the sources wrote them.
TAIL_ANCHOR_LAMBDAS_TAIL = [3.0e4, 6.0e4, 1.0e5]
TAIL_ANCHOR_LAMBDAS_BOTH_EDGES = [1.0e3, 1.2e3, 1.5e3, 3.0e4, 6.0e4, 1.0e5]


# =============================================================================
# Fixed strings that the sources wrote into their output files
# =============================================================================

#: Description of the flux conversion, copied verbatim into every scaler file.
CONVERSION_NOTE = (
    "Fnu=10**LOG_F_NUE; F_lambda=3e18*Fnu/lambda_A**2; "
    "multiply by 4*pi*(Rsun*radius)**2 and rmax**2"
)

#: The train-evaluation note.  v9-v15 wrote the first form, v16-v18 the second
#: (the clause "when batch norm is enabled" was added once the normalisation
#: ablation made batch norm optional).  Both are preserved.
TRAIN_EVAL_NOTE_V9 = (
    "Computed in eval mode after each epoch, with dropout disabled and batch "
    "norm running statistics."
)
TRAIN_EVAL_NOTE_V16 = (
    "Computed in eval mode after each epoch, with dropout disabled and batch "
    "norm running statistics when batch norm is enabled."
)

#: The width fragment shared by every single-run architecture tag of v1-v10.
_DEEP_TAG = (
    "deep_b128_256_512_512_1024_2048_"
    "t128_256_512_512_1024_2048_"
    "lat128_"
)

#: Default learning rate.  Every source read it as
#: ``float(os.environ.get("ML13_SED_LEARNING_RATE", "3e-4"))``; only v16c
#: overrode it, to 1.0e-4.
DEFAULT_LEARNING_RATE = 3e-4

#: Default loss name written into the metadata of the multi-experiment runners
#: when an experiment entry did not carry its own ``loss_name`` (v12 only).
DEFAULT_LOSS_NAME = "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor"


# =============================================================================
# Portable paths and environment defaults
# =============================================================================

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
DEFAULT_BASE_DIR = os.environ.get("ML13_BASE_DIR", str(OUTPUT_ROOT))
DEFAULT_STORAGE_DIR = os.environ.get(
    "ML13_STORAGE_DIR", str(DATA_ROOT / "fluxcont")
)
DEFAULT_NUM_WORKERS = int(os.environ.get("ML13_NUM_WORKERS", "4"))
DEFAULT_ZSTD_BIN = os.environ.get("ML13_ZSTD_BIN", "zstd")
DEFAULT_STOP_BEFORE_TIMEOUT_SECONDS = int(
    os.environ.get("ML13_STOP_BEFORE_TIMEOUT_SECONDS", "300")
)
_MAX_RUNTIME_ENV = os.environ.get("ML13_MAX_RUNTIME_SECONDS")
DEFAULT_MAX_RUNTIME_SECONDS = None if _MAX_RUNTIME_ENV is None else int(_MAX_RUNTIME_ENV)


# =============================================================================
# CONFIGS -- one entry per historical run
# =============================================================================
#
# Every behavioural difference between the 18 source files is a key here.  The
# defaults below are the values that the majority of the later versions used;
# each entry states only what it changes.  Nothing is derived or interpolated:
# a key that is not listed in an entry genuinely had the default value in that
# source file.

_CONFIG_DEFAULTS = {
    # --- identity -------------------------------------------------------
    "subversion": None,           # sub-run label of a multi-experiment file
    "group": None,                # group key ("v11" .. "v18"), None if single
    "group_run_tag": None,        # run tag of the group-level summary file
    "group_architecture_name": None,
    "group_architecture_tag": None,
    "architecture_name": None,
    "architecture_tag": None,
    "run_tag": None,              # defaults to architecture_tag when None
    "loss_name": None,            # string written into the metadata files

    # --- data -----------------------------------------------------------
    "reader": "columns",          # "loadtxt" (v1) or "columns" (v2+)
    "lambda_min_filter": None,    # Angstrom, applied to the native grid
    "lambda_max_filter": None,
    "target_scaling": "binned",   # "global" or "binned"
    "target_norm_bins": 200,      # only used when target_scaling == "binned"

    # --- sampling of training entries ------------------------------------
    "sampling": "balanced",       # "uniform_random" (v1) or "balanced" (v2+)
    "sampling_bins": 64,
    "full_sed_sampling": False,   # the v9 "take every point, then pad" branch
    "wave_samples_per_model": 1024,

    # --- architecture -----------------------------------------------------
    "fourier_modes": 64,
    "dropout": 0.02,
    "dropout_after_layer_indices": [0, 1, 2, 3, 4, 5],
    "use_batch_norm": True,
    "norm_type": "batch_norm",    # "batch_norm", "layer_norm" or "none"
    "residual_blocks": False,
    "learning_rate": DEFAULT_LEARNING_RATE,

    # --- loss --------------------------------------------------------------
    "loss_kind": "weighted",      # "mse" (v1) or "weighted" (v2+)
    "region_weights": REGION_WEIGHTS_UV125,   # None means unit weights
    "huber_delta": 1.0,
    "curve_bias_alpha": 0.0,
    "endpoint_alpha": 0.0,
    "tail_anchor_alpha": 0.0,
    "tail_anchor_lambdas": [],
    "tail_anchor_coords_norm": [],
    "derivative_alpha": 0.0,      # v13: standardised-space slope penalty
    "amplitude_alpha": 0.0,       # v13: curve-level penalty
    "edge_derivative_alpha": 0.0,  # v6: de-standardised edge slope penalty
    "derivative_edge_uv_max": 900.0,
    "derivative_edge_tail_start": 1.0e5,

    # --- bookkeeping --------------------------------------------------------
    "track_train_eval": False,    # per-epoch eval-mode pass over the train set
    "train_eval_loss_note": TRAIN_EVAL_NOTE_V9,
    "npz_loss_arrays": False,     # v14+ also stored the loss curves in the npz
}


def _config(**overrides):
    """Return a full configuration: the defaults updated by ``overrides``."""
    cfg = copy.deepcopy(_CONFIG_DEFAULTS)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise KeyError(f"Unknown configuration keys: {unknown}")
    cfg.update(copy.deepcopy(overrides))
    if cfg["run_tag"] is None:
        cfg["run_tag"] = cfg["architecture_tag"]
    return cfg


CONFIGS = {
    # -------------------------------------------------------------------
    # v1 -- Initial baseline (single run)
    # -------------------------------------------------------------------
    "v1": _config(
        architecture_name="fluxcont_sed_anjas_deep_dropout_deeponet",
        architecture_tag="fluxcont_logflam_loglambda_" + _DEEP_TAG + "fm32_drop0p3_bn",
        reader="loadtxt",
        target_scaling="global",
        sampling="uniform_random",
        fourier_modes=32,
        dropout=0.30,
        dropout_after_layer_indices=[3, 5],
        loss_kind="mse",
        region_weights=None,
    ),

    # -------------------------------------------------------------------
    # v2 -- Weighted binned target (single run)
    # -------------------------------------------------------------------
    "v2": _config(
        architecture_name="fluxcont_sed_v2_weighted_huber_bin_norm_deeponet",
        architecture_tag=(
            "fluxcont_v2_binlogflam_balanced_loglambda_weighted_huber_"
            + _DEEP_TAG
            + "fm32_drop0p3_bn"
        ),
        loss_name="region_weighted_huber",
        fourier_modes=32,
        dropout=0.30,
        dropout_after_layer_indices=[3, 5],
        region_weights=REGION_WEIGHTS_UV125,
    ),

    # -------------------------------------------------------------------
    # v3 -- Global, low dropout (single run)
    # -------------------------------------------------------------------
    "v3": _config(
        architecture_name="fluxcont_sed_v3_global_logsed_weighted_huber_drop002_bn_deeponet",
        architecture_tag=(
            "fluxcont_v3_global_logsed_balanced_loglambda_weighted_huber_cut100_"
            + _DEEP_TAG
            + "fm32_drop0p02_all_bn"
        ),
        loss_name="region_weighted_huber",
        target_scaling="global",
        lambda_min_filter=100.0,
        fourier_modes=32,
        dropout=0.02,
        region_weights=REGION_WEIGHTS_UV200,
    ),

    # -------------------------------------------------------------------
    # v4 -- Global, high dropout (single run)
    # -------------------------------------------------------------------
    "v4": _config(
        architecture_name="fluxcont_sed_v4_global_logsed_weighted_huber_drop010_bn_deeponet",
        architecture_tag=(
            "fluxcont_v4_global_logsed_balanced_loglambda_weighted_huber_cut100_"
            + _DEEP_TAG
            + "fm32_drop0p10_all_bn"
        ),
        loss_name="region_weighted_huber",
        target_scaling="global",
        lambda_min_filter=100.0,
        fourier_modes=32,
        dropout=0.10,
        region_weights=REGION_WEIGHTS_UV200,
    ),

    # -------------------------------------------------------------------
    # v5 -- Binned, 64 modes (single run)
    # -------------------------------------------------------------------
    "v5": _config(
        architecture_name="fluxcont_sed_v5_weighted_huber_bin_norm_cut100_fm64_drop010_deeponet",
        architecture_tag=(
            "fluxcont_v5_binlogflam_balanced_loglambda_weighted_huber_cut100_"
            + _DEEP_TAG
            + "fm64_drop0p10_bn"
        ),
        loss_name="region_weighted_huber",
        lambda_min_filter=100.0,
        dropout=0.10,
        region_weights=REGION_WEIGHTS_UV125,
    ),

    # -------------------------------------------------------------------
    # v6 -- Binned + derivative term (single run)
    #
    # Note the ``loss_name`` really is the bare "region_weighted_huber" in the
    # source: the derivative term was added to the code but the label written
    # into the metadata was never updated.  Reproduced as-is.
    # -------------------------------------------------------------------
    "v6": _config(
        architecture_name=(
            "fluxcont_sed_v6_weighted_huber_bin_norm_cut100_fm64_drop010_deriv_deeponet"
        ),
        architecture_tag=(
            "fluxcont_v6_binlogflam_balanced_loglambda_weighted_huber_deriv_cut100_"
            + _DEEP_TAG
            + "fm64_drop0p10_bn"
        ),
        loss_name="region_weighted_huber",
        lambda_min_filter=100.0,
        dropout=0.10,
        region_weights=REGION_WEIGHTS_UV125,
        edge_derivative_alpha=0.05,
        derivative_edge_uv_max=900.0,
        derivative_edge_tail_start=1.0e5,
    ),

    # -------------------------------------------------------------------
    # v7 -- Global, cut 200 (single run)
    # -------------------------------------------------------------------
    "v7": _config(
        architecture_name=(
            "fluxcont_sed_v7_global_logsed_weighted_huber_bias_cut200_fm64_drop002_bn_deeponet"
        ),
        architecture_tag=(
            "fluxcont_v7_global_logsed_balanced_loglambda_weighted_huber_bias_cut200_"
            + _DEEP_TAG
            + "fm64_drop0p02_all_bn"
        ),
        loss_name="region_weighted_huber_plus_curve_bias",
        target_scaling="global",
        lambda_min_filter=200.0,
        region_weights=REGION_WEIGHTS_UV200,
        curve_bias_alpha=0.10,
    ),

    # -------------------------------------------------------------------
    # v8 -- Binned, cut 200 (single run)
    # -------------------------------------------------------------------
    "v8": _config(
        architecture_name=(
            "fluxcont_sed_v8_weighted_huber_bias_bin_norm_cut200_fm64_drop002_deeponet"
        ),
        architecture_tag=(
            "fluxcont_v8_binlogflam_balanced_loglambda_weighted_huber_bias_cut200_"
            + _DEEP_TAG
            + "fm64_drop0p02_bn"
        ),
        loss_name="region_weighted_huber_plus_curve_bias",
        lambda_min_filter=200.0,
        region_weights=REGION_WEIGHTS_UV125,
        curve_bias_alpha=0.10,
    ),

    # -------------------------------------------------------------------
    # v9 -- Global full SED (single run)
    # -------------------------------------------------------------------
    "v9": _config(
        architecture_name=(
            "fluxcont_sed_v9_global_logsed_weighted_huber_bias_cut100_fullsed_"
            "fm64_drop002_bn_deeponet"
        ),
        architecture_tag=(
            "fluxcont_v9_global_logsed_balanced_loglambda_weighted_huber_bias_cut100_fullsed_"
            + _DEEP_TAG
            + "fm64_drop0p02_all_bn"
        ),
        loss_name="region_weighted_huber_plus_curve_bias",
        target_scaling="global",
        lambda_min_filter=100.0,
        region_weights=REGION_WEIGHTS_UV200,
        curve_bias_alpha=0.10,
        wave_samples_per_model=2500,
        full_sed_sampling=True,
        track_train_eval=True,
    ),

    # -------------------------------------------------------------------
    # v10 -- Global full SED + endpoint (single run)
    # -------------------------------------------------------------------
    "v10": _config(
        architecture_name=(
            "fluxcont_sed_v10_global_logsed_weighted_huber_bias_endpoint_cut100_fullsed_"
            "fm64_drop002_bn_deeponet"
        ),
        architecture_tag=(
            "fluxcont_v10_global_logsed_balanced_loglambda_weighted_huber_bias_endpoint_"
            "cut100_fullsed_"
            + _DEEP_TAG
            + "fm64_drop0p02_all_bn"
        ),
        loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
        target_scaling="global",
        lambda_min_filter=100.0,
        region_weights=REGION_WEIGHTS_UV200,
        curve_bias_alpha=0.10,
        endpoint_alpha=0.05,
        wave_samples_per_model=2500,
        full_sed_sampling=True,
        track_train_eval=True,
    ),
}


# -----------------------------------------------------------------------------
# The multi-experiment files, v11 .. v18
# -----------------------------------------------------------------------------
#
# From v11 on, each source file contained an ``EXPERIMENTS`` list and a shared
# data-preparation pass.  Everything outside the list was identical for all
# members of a group, so it is factored into ``_GROUP_COMMON`` below; the
# per-letter overrides are then exactly the dictionaries of the original list.

#: Group-level metadata, one entry per multi-experiment source file.
GROUP_META = {
    "v11": {
        "architecture_name": (
            "fluxcont_sed_v11_global_logsed_weighted_huber_bias_endpoint_cut200_"
            "fm64_drop002_bn_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v11_global_logsed_weighted_huber_bias_endpoint_cut200_"
            "fm64_drop002_bn_multi"
        ),
        "run_tag": (
            "fluxcont_v11_global_logsed_balanced_loglambda_weighted_huber_bias_endpoint_"
            "cut200_fm64_drop0p02_all_bn_13par_multi"
        ),
    },
    "v12": {
        "architecture_name": (
            "fluxcont_sed_v12_weighted_huber_bias_endpoint_bin_norm_cut200_"
            "fm64_drop002_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v12_binlogflam_weighted_huber_bias_endpoint_cut200_"
            "fm64_drop002_bn_multi"
        ),
        "run_tag": (
            "fluxcont_v12_binlogflam_balanced_loglambda_weighted_huber_bias_endpoint_"
            "cut200_fm64_drop0p02_bn_13par_multi"
        ),
    },
    "v13": {
        "architecture_name": (
            "fluxcont_sed_v13_global_logsed_weighted_huber_bias_endpoint_deriv_amp_"
            "cut200_fm64_drop002_bn_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v13_global_logsed_weighted_huber_bias_endpoint_deriv_amp_"
            "cut200_fm64_drop002_bn_multi"
        ),
        "run_tag": (
            "fluxcont_v13_global_logsed_balanced_loglambda_weighted_huber_bias_endpoint_"
            "deriv_amp_cut200_fm64_drop0p02_all_bn_13par_multi"
        ),
    },
    "v14": {
        "architecture_name": (
            "fluxcont_sed_v14_weighted_huber_bias_endpoint_bin_norm_cut1e3_1e5_"
            "fm64_drop002_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v14_binlogflam_weighted_huber_bias_endpoint_cut1e3_1e5_"
            "fm64_drop002_bn_multi"
        ),
        "run_tag": (
            "fluxcont_v14_binlogflam_balanced_loglambda_weighted_huber_bias_endpoint_"
            "cut1e3_1e5_fm64_drop0p02_bn_13par_multi"
        ),
    },
    "v15": {
        "architecture_name": (
            "fluxcont_sed_v15_global_logsed_weighted_huber_bias_endpoint_cut1e3_1e5_"
            "fm64_drop002_bn_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v15_global_logsed_weighted_huber_bias_endpoint_cut1e3_1e5_"
            "fm64_drop002_bn_multi"
        ),
        "run_tag": (
            "fluxcont_v15_global_logsed_balanced_loglambda_weighted_huber_bias_endpoint_"
            "cut1e3_1e5_fm64_drop0p02_all_bn_13par_multi"
        ),
    },
    "v16": {
        "architecture_name": (
            "fluxcont_sed_v16_binlogflam_norm_ablation_1e3_1e5_fm64_drop002_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v16_binlogflam_norm_ablation_1e3_1e5_fm64_drop002_multi"
        ),
        "run_tag": (
            "fluxcont_v16_binlogflam_balanced_loglambda_weighted_huber_bias_norm_ablation_"
            "1e3_1e5_fm64_drop0p02_13par_multi"
        ),
    },
    "v17": {
        "architecture_name": (
            "fluxcont_sed_v17_binlogflam_tail_anchor_residual_1e3_1e5_"
            "fm64_drop002_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v17_binlogflam_tail_anchor_residual_1e3_1e5_fm64_drop002_multi"
        ),
        "run_tag": (
            "fluxcont_v17_binlogflam_balanced_loglambda_weighted_huber_bias_tail_anchor_"
            "residual_1e3_1e5_fm64_drop0p02_13par_multi"
        ),
    },
    "v18": {
        "architecture_name": (
            "fluxcont_sed_v18_binlogflam_edge_anchor_fourier_1e3_1e5_"
            "fm64_drop002_deeponet_multi"
        ),
        "architecture_tag": (
            "fluxcont_v18_binlogflam_edge_anchor_fourier_1e3_1e5_fm64_drop002_multi"
        ),
        "run_tag": (
            "fluxcont_v18_binlogflam_balanced_loglambda_weighted_huber_bias_edge_anchor_"
            "fourier_1e3_1e5_fm64_drop0p02_13par_multi"
        ),
    },
}

#: Settings that were module-level constants inside each multi-experiment file,
#: i.e. shared by every letter of that group.
_GROUP_COMMON = {
    "v11": dict(
        target_scaling="global",
        lambda_min_filter=200.0,
        region_weights=REGION_WEIGHTS_UV200,
        wave_samples_per_model=2500,
        full_sed_sampling=True,
        curve_bias_alpha=0.10,
        track_train_eval=True,
    ),
    "v12": dict(
        target_scaling="binned",
        lambda_min_filter=200.0,
        region_weights=REGION_WEIGHTS_UV125,
        wave_samples_per_model=1024,
        curve_bias_alpha=0.10,
        track_train_eval=True,
    ),
    "v13": dict(
        target_scaling="global",
        lambda_min_filter=200.0,
        region_weights=REGION_WEIGHTS_UV200,
        wave_samples_per_model=2500,
        full_sed_sampling=True,
        curve_bias_alpha=0.10,
        track_train_eval=True,
    ),
    "v14": dict(
        target_scaling="binned",
        lambda_min_filter=1000.0,
        lambda_max_filter=100000.0,
        region_weights=REGION_WEIGHTS_UV125,
        wave_samples_per_model=1024,
        curve_bias_alpha=0.10,
        track_train_eval=True,
        npz_loss_arrays=True,
    ),
    "v15": dict(
        target_scaling="global",
        lambda_min_filter=1000.0,
        lambda_max_filter=100000.0,
        region_weights=REGION_WEIGHTS_UV200,
        wave_samples_per_model=2500,
        full_sed_sampling=True,
        curve_bias_alpha=0.10,
        track_train_eval=True,
        npz_loss_arrays=True,
    ),
    "v16": dict(
        target_scaling="binned",
        lambda_min_filter=1000.0,
        lambda_max_filter=100000.0,
        region_weights=REGION_WEIGHTS_UV125,
        wave_samples_per_model=1024,
        curve_bias_alpha=0.10,
        track_train_eval=True,
        train_eval_loss_note=TRAIN_EVAL_NOTE_V16,
        npz_loss_arrays=True,
    ),
    "v17": dict(
        target_scaling="binned",
        lambda_min_filter=1000.0,
        lambda_max_filter=100000.0,
        region_weights=REGION_WEIGHTS_UV125,
        wave_samples_per_model=1024,
        curve_bias_alpha=0.10,
        track_train_eval=True,
        train_eval_loss_note=TRAIN_EVAL_NOTE_V16,
        npz_loss_arrays=True,
    ),
    "v18": dict(
        target_scaling="binned",
        lambda_min_filter=1000.0,
        lambda_max_filter=100000.0,
        region_weights=REGION_WEIGHTS_UV125,
        wave_samples_per_model=1024,
        curve_bias_alpha=0.10,
        track_train_eval=True,
        train_eval_loss_note=TRAIN_EVAL_NOTE_V16,
        npz_loss_arrays=True,
    ),
}

#: The ``EXPERIMENTS`` lists, transcribed entry for entry.
_GROUP_EXPERIMENTS = {
    "v11": [
        dict(
            subversion="v11a",
            endpoint_alpha=0.005,
            loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
            architecture_name=(
                "fluxcont_sed_v11a_global_logsed_weighted_huber_bias_endpoint0005_"
                "cut200_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v11a_global_logsed_weighted_huber_bias_endpoint0005_"
                "cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v11a_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p005_cut200_fm64_drop0p02_all_bn_13par"
            ),
        ),
        dict(
            subversion="v11b",
            endpoint_alpha=0.010,
            loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
            architecture_name=(
                "fluxcont_sed_v11b_global_logsed_weighted_huber_bias_endpoint001_"
                "cut200_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v11b_global_logsed_weighted_huber_bias_endpoint001_"
                "cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v11b_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_cut200_fm64_drop0p02_all_bn_13par"
            ),
        ),
        dict(
            subversion="v11c",
            endpoint_alpha=0.020,
            loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
            architecture_name=(
                "fluxcont_sed_v11c_global_logsed_weighted_huber_bias_endpoint002_"
                "cut200_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v11c_global_logsed_weighted_huber_bias_endpoint002_"
                "cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v11c_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p02_cut200_fm64_drop0p02_all_bn_13par"
            ),
        ),
    ],
    # v12 was the only file whose EXPERIMENTS entries carried no "loss_name";
    # the code fell back to DEFAULT_LOSS_NAME, which is what is written here.
    "v12": [
        dict(
            subversion="v12a",
            endpoint_alpha=0.005,
            loss_name=DEFAULT_LOSS_NAME,
            architecture_name=(
                "fluxcont_sed_v12a_weighted_huber_bias_endpoint0005_bin_norm_"
                "cut200_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v12a_binlogflam_weighted_huber_bias_endpoint0005_"
                "cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v12a_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p005_cut200_fm64_drop0p02_bn_13par"
            ),
        ),
        dict(
            subversion="v12b",
            endpoint_alpha=0.010,
            loss_name=DEFAULT_LOSS_NAME,
            architecture_name=(
                "fluxcont_sed_v12b_weighted_huber_bias_endpoint001_bin_norm_"
                "cut200_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v12b_binlogflam_weighted_huber_bias_endpoint001_"
                "cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v12b_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_cut200_fm64_drop0p02_bn_13par"
            ),
        ),
    ],
    "v13": [
        dict(
            subversion="v13a",
            endpoint_alpha=0.010,
            derivative_alpha=0.001,
            amplitude_alpha=0.010,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "tiny_derivative_plus_amplitude"
            ),
            architecture_name=(
                "fluxcont_sed_v13a_global_logsed_weighted_huber_bias_endpoint001_"
                "deriv0001_amp001_cut200_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v13a_global_logsed_weighted_huber_bias_endpoint001_"
                "deriv0001_amp001_cut200_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v13a_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_deriv0p001_amp0p01_cut200_fm64_drop0p02_all_bn_13par"
            ),
        ),
    ],
    "v14": [
        dict(
            subversion="v14a",
            endpoint_alpha=0.0,
            loss_name="region_weighted_huber_plus_curve_bias",
            architecture_name=(
                "fluxcont_sed_v14a_weighted_huber_bias_noendpoint_bin_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v14a_binlogflam_weighted_huber_bias_noendpoint_"
                "1e3_1e5_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v14a_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "noendpoint_1e3_1e5_fm64_drop0p02_bn_13par"
            ),
        ),
        dict(
            subversion="v14b",
            endpoint_alpha=0.010,
            loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
            architecture_name=(
                "fluxcont_sed_v14b_weighted_huber_bias_endpoint001_bin_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v14b_binlogflam_weighted_huber_bias_endpoint001_"
                "1e3_1e5_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v14b_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_1e3_1e5_fm64_drop0p02_bn_13par"
            ),
        ),
    ],
    "v15": [
        dict(
            subversion="v15a",
            endpoint_alpha=0.0,
            loss_name="region_weighted_huber_plus_curve_bias",
            architecture_name=(
                "fluxcont_sed_v15a_global_logsed_weighted_huber_bias_noendpoint_"
                "1e3_1e5_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v15a_global_logsed_weighted_huber_bias_noendpoint_"
                "1e3_1e5_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v15a_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "noendpoint_1e3_1e5_fm64_drop0p02_all_bn_13par"
            ),
        ),
        dict(
            subversion="v15b",
            endpoint_alpha=0.010,
            loss_name="region_weighted_huber_plus_curve_bias_plus_endpoint_anchor",
            architecture_name=(
                "fluxcont_sed_v15b_global_logsed_weighted_huber_bias_endpoint001_"
                "1e3_1e5_fm64_drop002_bn_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v15b_global_logsed_weighted_huber_bias_endpoint001_"
                "1e3_1e5_fm64_drop002_bn"
            ),
            run_tag=(
                "fluxcont_v15b_global_logsed_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_1e3_1e5_fm64_drop0p02_all_bn_13par"
            ),
        ),
    ],
    "v16": [
        dict(
            subversion="v16a",
            endpoint_alpha=0.010,
            use_batch_norm=False,
            norm_type="none",
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v16a_binlogflam_endpoint001_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v16a_binlogflam_endpoint001_no_norm_1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v16a_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v16b",
            endpoint_alpha=0.010,
            use_batch_norm=False,
            norm_type="layer_norm",
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_layer_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v16b_binlogflam_endpoint001_layer_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v16b_binlogflam_endpoint001_layer_norm_1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v16b_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_layernorm_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v16c",
            endpoint_alpha=0.010,
            use_batch_norm=True,
            norm_type="batch_norm",
            learning_rate=1.0e-4,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_"
                "batch_norm_lr1e4"
            ),
            architecture_name=(
                "fluxcont_sed_v16c_binlogflam_endpoint001_batch_norm_lr1e4_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v16c_binlogflam_endpoint001_batch_norm_lr1e4_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v16c_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_bn_lr1e4_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v16d",
            endpoint_alpha=0.005,
            use_batch_norm=True,
            norm_type="batch_norm",
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_small_endpoint_anchor"
            ),
            architecture_name=(
                "fluxcont_sed_v16d_binlogflam_endpoint0005_batch_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v16d_binlogflam_endpoint0005_batch_norm_1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v16d_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p005_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
    ],
    "v17": [
        dict(
            subversion="v17a",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0,
            tail_anchor_lambdas=[],
            tail_anchor_coords_norm=[],
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_"
                "no_batch_norm_control"
            ),
            architecture_name=(
                "fluxcont_sed_v17a_binlogflam_endpoint001_no_norm_control_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v17a_binlogflam_endpoint001_no_norm_control_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v17a_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_no_bn_control_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v17b",
            endpoint_alpha=0.005,
            tail_anchor_alpha=0.0,
            tail_anchor_lambdas=[],
            tail_anchor_coords_norm=[],
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint0005_anchor_"
                "no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v17b_binlogflam_endpoint0005_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v17b_binlogflam_endpoint0005_no_norm_1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v17b_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p005_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v17c",
            endpoint_alpha=0.020,
            tail_anchor_alpha=0.0,
            tail_anchor_lambdas=[],
            tail_anchor_coords_norm=[],
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint002_anchor_"
                "no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v17c_binlogflam_endpoint002_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v17c_binlogflam_endpoint002_no_norm_1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v17c_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p02_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v17d",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.010,
            tail_anchor_lambdas=list(TAIL_ANCHOR_LAMBDAS_TAIL),
            # NOTE: v17d wrote 0.889075625191822, v18a-d wrote the extra digit.
            tail_anchor_coords_norm=list(TAIL_ANCHORS_V17D),
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "fixed_tail_anchors_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v17d_binlogflam_endpoint001_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v17d_binlogflam_endpoint001_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v17d_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_tailanchors_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v17e",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0,
            tail_anchor_lambdas=[],
            tail_anchor_coords_norm=[],
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=True,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_residual_"
                "no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v17e_binlogflam_endpoint001_residual_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v17e_binlogflam_endpoint001_residual_no_norm_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v17e_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_residual_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
    ],
    "v18": [
        dict(
            subversion="v18a",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0025,
            tail_anchor_lambdas=list(TAIL_ANCHOR_LAMBDAS_TAIL),
            tail_anchor_coords_norm=list(TAIL_ANCHORS_V18),
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            fourier_modes=64,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "weak_tail_anchors_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v18a_binlogflam_endpoint001_weak_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v18a_binlogflam_endpoint001_weak_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v18a_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_tailanchors0p0025_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v18b",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.005,
            tail_anchor_lambdas=list(TAIL_ANCHOR_LAMBDAS_TAIL),
            tail_anchor_coords_norm=list(TAIL_ANCHORS_V18),
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            fourier_modes=64,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "mild_tail_anchors_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v18b_binlogflam_endpoint001_mild_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v18b_binlogflam_endpoint001_mild_tailanchors_no_norm_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v18b_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_tailanchors0p005_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v18c",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0025,
            tail_anchor_lambdas=list(TAIL_ANCHOR_LAMBDAS_BOTH_EDGES),
            tail_anchor_coords_norm=list(BOTH_EDGE_ANCHORS_V18),
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            fourier_modes=64,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "weak_both_edge_anchors_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v18c_binlogflam_endpoint001_weak_both_edgeanchors_no_norm_"
                "1e3_1e5_fm64_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v18c_binlogflam_endpoint001_weak_both_edgeanchors_no_norm_"
                "1e3_1e5_fm64_drop002"
            ),
            run_tag=(
                "fluxcont_v18c_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_both_edgeanchors0p0025_no_bn_1e3_1e5_fm64_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v18d",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0025,
            tail_anchor_lambdas=list(TAIL_ANCHOR_LAMBDAS_BOTH_EDGES),
            tail_anchor_coords_norm=list(BOTH_EDGE_ANCHORS_V18),
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            fourier_modes=96,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_plus_"
                "weak_both_edge_anchors_fm96_no_batch_norm"
            ),
            architecture_name=(
                "fluxcont_sed_v18d_binlogflam_endpoint001_weak_both_edgeanchors_no_norm_"
                "1e3_1e5_fm96_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v18d_binlogflam_endpoint001_weak_both_edgeanchors_no_norm_"
                "1e3_1e5_fm96_drop002"
            ),
            run_tag=(
                "fluxcont_v18d_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_both_edgeanchors0p0025_no_bn_1e3_1e5_fm96_drop0p02_13par"
            ),
        ),
        dict(
            subversion="v18e",
            endpoint_alpha=0.010,
            tail_anchor_alpha=0.0,
            tail_anchor_lambdas=[],
            tail_anchor_coords_norm=[],
            use_batch_norm=False,
            norm_type="none",
            residual_blocks=False,
            learning_rate=DEFAULT_LEARNING_RATE,
            fourier_modes=96,
            loss_name=(
                "region_weighted_huber_plus_curve_bias_plus_endpoint_anchor_fm96_"
                "no_batch_norm_control"
            ),
            architecture_name=(
                "fluxcont_sed_v18e_binlogflam_endpoint001_no_norm_control_"
                "1e3_1e5_fm96_drop002_deeponet"
            ),
            architecture_tag=(
                "fluxcont_v18e_binlogflam_endpoint001_no_norm_control_"
                "1e3_1e5_fm96_drop002"
            ),
            run_tag=(
                "fluxcont_v18e_binlogflam_balanced_loglambda_weighted_huber_bias_"
                "endpoint0p01_no_bn_control_1e3_1e5_fm96_drop0p02_13par"
            ),
        ),
    ],
}


#: Group key -> ordered list of its member configuration keys.
GROUPS = {}

for _group_key, _entries in _GROUP_EXPERIMENTS.items():
    _meta = GROUP_META[_group_key]
    GROUPS[_group_key] = []
    for _entry in _entries:
        _settings = dict(_GROUP_COMMON[_group_key])
        _settings.update(_entry)
        _settings["group"] = _group_key
        _settings["group_run_tag"] = _meta["run_tag"]
        _settings["group_architecture_name"] = _meta["architecture_name"]
        _settings["group_architecture_tag"] = _meta["architecture_tag"]
        CONFIGS[_entry["subversion"]] = _config(**_settings)
        GROUPS[_group_key].append(_entry["subversion"])

del _group_key, _entries, _meta, _entry, _settings

#: The configuration the thesis adopted.
DEFAULT_CONFIG_KEY = "v17a"


#: Environment variable -> (config key, converter).  Applied *after* the chosen
#: CONFIGS entry, so an unset environment reproduces the published run exactly.
ENV_CONFIG_OVERRIDES = {
    "ML13_SED_LEARNING_RATE": ("learning_rate", float),
    "ML13_SED_WAVE_SAMPLES_PER_MODEL": ("wave_samples_per_model", int),
    "ML13_SED_TARGET_NORM_BINS": ("target_norm_bins", int),
    "ML13_SED_SAMPLING_BINS": ("sampling_bins", int),
    "ML13_SED_HUBER_DELTA": ("huber_delta", float),
    "ML13_SED_CURVE_BIAS_ALPHA": ("curve_bias_alpha", float),
    "ML13_SED_ENDPOINT_ALPHA": ("endpoint_alpha", float),
    "ML13_SED_DERIVATIVE_ALPHA": ("edge_derivative_alpha", float),
    "ML13_SED_DERIVATIVE_EDGE_UV_MAX": ("derivative_edge_uv_max", float),
    "ML13_SED_DERIVATIVE_EDGE_TAIL_START": ("derivative_edge_tail_start", float),
    "ML13_SED_RUN_TAG": ("run_tag", str),
}

#: Environment variable -> region-weight key.  The originals exposed each
#: threshold and each weight separately; the same names are honoured here.
ENV_REGION_WEIGHT_OVERRIDES = {
    "ML13_SED_UV_EXTREME_MAX": "uv_extreme_max",
    "ML13_SED_UV_MAX": "uv_max",
    "ML13_SED_TAIL_START": "tail_start",
    "ML13_SED_FAR_TAIL_START": "far_tail_start",
    "ML13_SED_UV_EXTREME_WEIGHT": "uv_extreme_weight",
    "ML13_SED_UV_WEIGHT": "uv_weight",
    "ML13_SED_MID_WEIGHT": "mid_weight",
    "ML13_SED_TAIL_WEIGHT": "tail_weight",
    "ML13_SED_FAR_TAIL_WEIGHT": "far_tail_weight",
}


def resolve_config(key, env=None):
    """Return a fresh copy of ``CONFIGS[key]`` with environment overrides applied.

    ``ML13_SED_LAMBDA_MIN`` / ``ML13_SED_LAMBDA_MAX`` are treated the way the
    sources treated them: an empty string means "no filter", anything else is a
    float.  Note that in the sources these were *defaults* of ``os.environ.get``,
    so setting the variable overrode the per-version cut; that is preserved.
    """
    if env is None:
        env = os.environ
    if key not in CONFIGS:
        raise KeyError(f"Unknown configuration '{key}'. Known: {sorted(CONFIGS)}")

    cfg = copy.deepcopy(CONFIGS[key])
    cfg["config_key"] = key

    for var, (cfg_key, caster) in ENV_CONFIG_OVERRIDES.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        cfg[cfg_key] = caster(raw)

    for var in ("ML13_SED_LAMBDA_MIN", "ML13_SED_LAMBDA_MAX"):
        if var not in env:
            continue
        raw = env[var]
        target = "lambda_min_filter" if var.endswith("MIN") else "lambda_max_filter"
        cfg[target] = None if raw == "" else float(raw)

    if cfg["region_weights"] is not None:
        weights = dict(cfg["region_weights"])
        for var, weight_key in ENV_REGION_WEIGHT_OVERRIDES.items():
            raw = env.get(var)
            if raw is None or raw == "":
                continue
            weights[weight_key] = float(raw)
        cfg["region_weights"] = weights

    return cfg


def expand_config_keys(keys):
    """Expand group keys into their members, preserving order and uniqueness."""
    expanded = []
    for key in keys:
        members = GROUPS.get(key, [key])
        for member in members:
            if member not in expanded:
                expanded.append(member)
    return expanded


# =============================================================================
# Walltime budget (SLURM-aware), identical logic in all 18 source files
# =============================================================================

RUN_START_TIME = time.time()


class DeadlineReached(RuntimeError):
    """Raised when too little walltime remains to safely continue."""


def _slurm_time_limit_seconds():
    """Parse ``SLURM_TIMELIMIT`` ("D-HH:MM:SS", "HH:MM:SS", "MM" ...) to seconds."""
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


#: Absolute timestamp after which the run must stop; ``None`` disables the check.
DEADLINE_TS = None


def configure_deadline(stop_before_timeout_seconds, max_runtime_seconds):
    """Set the module-level deadline and return (slurm_limit, deadline_ts)."""
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
# Run bookkeeping
# =============================================================================

ARTIFACT_STEM = "emulator_FLUXCONT"


def artifact_paths(output_dir):
    stem = ARTIFACT_STEM
    return {
        "model": os.path.join(output_dir, f"{stem}.pth"),
        "loss": os.path.join(output_dir, f"{stem}_loss.json"),
        "split": os.path.join(output_dir, f"{stem}_split.json"),
        "pred": os.path.join(output_dir, f"{stem}_test_outputs.npz"),
        "meta": os.path.join(output_dir, f"{stem}_meta.json"),
    }


def is_run_complete(output_dir):
    """A run counts as finished once all five primary artefacts exist."""
    paths = artifact_paths(output_dir)
    return all(os.path.exists(p) for p in paths.values())


def load_existing_summary(summary_path):
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[summary] Could not read existing summary at {summary_path}: {e}", flush=True)
        return {}


def save_summary(summary_path, record):
    with open(summary_path, "w") as f:
        json.dump(record, f, indent=2)


# =============================================================================
# FLUXCONT file access
# =============================================================================

def _is_zstd_file(path):
    if path.lower().endswith(".zst"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x28\xb5\x2f\xfd"
    except OSError:
        return False


def read_text_lines_maybe_zst(path, zstd_bin):
    """Return the lines of ``path``, transparently decompressing zstd archives."""
    if _is_zstd_file(path):
        proc = subprocess.run(
            [zstd_bin, "-dc", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        text = proc.stdout.decode("utf-8", errors="replace")
        return text.splitlines(True)

    with open(path, "r", errors="replace") as f:
        return f.readlines()


def find_fluxcont_path(model_dir):
    """Locate the FLUXCONT file of one model directory, ignoring NTFS streams."""
    exact_candidates = [
        os.path.join(model_dir, "FLUXCONT"),
        os.path.join(model_dir, "FLUXCONT.zst"),
        os.path.join(model_dir, "FLUXCONT.ZST"),
    ]
    for path in exact_candidates:
        if os.path.exists(path):
            return path

    candidates = []
    for path in glob.glob(os.path.join(model_dir, "FLUXCONT*")):
        name = os.path.basename(path)
        if name.endswith(":Zone.Identifier") or name.endswith(":mshield"):
            continue
        candidates.append(path)

    return sorted(candidates)[0] if candidates else None


def read_fluxcont_converted(path_fluxcont, rstar, rmax_fw, cfg, runtime):
    """Read one FLUXCONT file and return sorted ``(lambda, F_lambda)`` arrays.

    Column 2 of the file is the wavelength in Angstrom and column 3 is
    log10(F_nu).  Two readers existed:

    ``loadtxt``  (v1)  collected the candidate lines and handed them to
                       ``numpy.loadtxt``; a single unparsable row aborted the
                       whole model.
    ``columns``  (v2+) parsed only the two columns it needs and silently
                       skipped rows that failed ``float()``.

    Both are kept because they can accept slightly different sets of models.
    """
    lines = read_text_lines_maybe_zst(path_fluxcont, runtime["zstd_bin"])
    if len(lines) < 2:
        return None

    min_points = runtime["min_fluxcont_points"]

    if cfg["reader"] == "loadtxt":
        useful_lines = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 1:
                break
            if len(parts) >= 3:
                useful_lines.append(line)

        if len(useful_lines) < min_points:
            return None

        arr = np.loadtxt(io.StringIO("".join(useful_lines)))
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 3:
            return None

        lam = arr[:, 1].astype(np.float64)
        log_fnu = arr[:, 2].astype(np.float64)
    else:
        lam_values = []
        log_fnu_values = []
        skipped_bad_rows = 0
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 1:
                break
            if len(parts) >= 3:
                try:
                    lam_values.append(float(parts[1]))
                    log_fnu_values.append(float(parts[2]))
                except ValueError:
                    skipped_bad_rows += 1
                    continue

        if len(lam_values) < min_points:
            return None

        lam = np.asarray(lam_values, dtype=np.float64)
        log_fnu = np.asarray(log_fnu_values, dtype=np.float64)

    rstar = float(rstar)
    rmax_fw = float(rmax_fw)
    rsun = RSUN_CM
    stellar_surface = 4.0 * np.pi * (rsun * rstar) ** 2

    # F_nu -> F_lambda, then to the outer boundary of the model atmosphere.
    fnu = np.power(10.0, log_fnu)
    flam = C_ANGSTROM_PER_S * fnu / (lam ** 2)
    flam = flam * stellar_surface
    flam = flam * (rmax_fw ** 2)

    finite = np.isfinite(lam) & np.isfinite(flam) & (lam > 0.0) & (flam > 0.0)
    if cfg["lambda_min_filter"] is not None:
        finite &= lam >= cfg["lambda_min_filter"]
    if cfg["lambda_max_filter"] is not None:
        finite &= lam <= cfg["lambda_max_filter"]

    lam = lam[finite]
    flam = flam[finite]
    if lam.size < min_points:
        return None

    order = np.argsort(lam)
    return lam[order], flam[order]


# =============================================================================
# Grid, quality flags and the 13-parameter min-max scaling
# =============================================================================

def load_parameter_table(runtime):
    """Read the grid and quality JSONs and return the accepted-model table."""
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

        required_cols = PARAM_COLS + [RMAX_COL]
        missing = [col for col in required_cols if col not in values]
        if missing:
            print(f"[WARN] Model {model_id} missing required values: {missing}", flush=True)
            continue

        row = {"model": model_id}
        for col in PARAM_COLS:
            row[col] = values[col]
        row[RMAX_COL] = values[RMAX_COL]
        grid_rows.append(row)

    params_df = pd.DataFrame(grid_rows).sort_values("model").reset_index(drop=True)
    print(f"Models kept after parameter/rmax filtering: {len(params_df)}", flush=True)

    if len(params_df) == 0:
        raise RuntimeError("No usable models found after filtering.")

    return params_df


def normalize_parameters(params_df):
    """log10 the mass-loss rate, then min-max scale all 13 inputs to [0, 1]."""
    params_arr = params_df[PARAM_COLS].values.astype(np.float32)
    params_norm = params_arr.copy()

    mdot_idx = PARAM_COLS.index("mdot")
    log_mdot = np.log10(params_arr[:, mdot_idx])
    params_norm[:, mdot_idx] = log_mdot

    param_mins = params_norm.min(axis=0)
    param_maxs = params_norm.max(axis=0)
    param_range = np.where(param_maxs - param_mins == 0.0, 1.0, param_maxs - param_mins)
    params_norm = (params_norm - param_mins) / param_range

    model_param_norm = {
        int(params_df.loc[i, "model"]): params_norm[i]
        for i in range(len(params_df))
    }
    model_radius = {
        int(params_df.loc[i, "model"]): float(params_df.loc[i, RADIUS_COL])
        for i in range(len(params_df))
    }
    model_rmax = {
        int(params_df.loc[i, "model"]): float(params_df.loc[i, RMAX_COL])
        for i in range(len(params_df))
    }

    norm_metadata = {
        "param_cols": PARAM_COLS,
        "param_mins_after_mdot_log": param_mins.tolist(),
        "param_maxs_after_mdot_log": param_maxs.tolist(),
        "param_range_after_mdot_log": param_range.tolist(),
        "mdot_index": mdot_idx,
        "mdot_log10_applied": True,
        "rmax_col": RMAX_COL,
        "rmax_is_ml_input": False,
    }

    return {
        "model_param_norm": model_param_norm,
        "model_radius": model_radius,
        "model_rmax": model_rmax,
        "param_mins": param_mins,
        "param_maxs": param_maxs,
        "param_range": param_range,
        "norm_metadata": norm_metadata,
    }


def build_fluxcont_records(cfg, runtime, param_info):
    """Read every accepted model's FLUXCONT file and build the record list.

    Also computes the dataset-level log-wavelength range and attaches the
    normalised trunk coordinate ``wavelengths_log_norm`` to every record, so
    that a given coordinate always means the same wavelength.
    """
    print("[DATA] Building FLUXCONT dataset", flush=True)

    model_param_norm = param_info["model_param_norm"]
    model_radius = param_info["model_radius"]
    model_rmax = param_info["model_rmax"]

    records = []
    skipped_missing = 0
    skipped_read = 0

    for i, model_id in enumerate(sorted(model_param_norm.keys()), start=1):
        ensure_time_budget("building FLUXCONT dataset", minimum_seconds=120)

        model_dir = os.path.join(runtime["models_root"], str(model_id))
        fluxcont_path = find_fluxcont_path(model_dir)
        if fluxcont_path is None:
            skipped_missing += 1
            continue

        try:
            out = read_fluxcont_converted(
                fluxcont_path,
                rstar=model_radius[model_id],
                rmax_fw=model_rmax[model_id],
                cfg=cfg,
                runtime=runtime,
            )
        except Exception as e:
            print(f"[WARN] Failed to read {fluxcont_path}: {e}", flush=True)
            skipped_read += 1
            continue

        if out is None:
            skipped_read += 1
            continue

        lam, flam = out
        log_lam = np.log10(lam)
        log_flam = np.log10(flam)

        records.append(
            {
                "model_id": model_id,
                "params": model_param_norm[model_id].astype(np.float32),
                "wavelengths_phys": lam.astype(np.float32),
                "log_lambda": log_lam.astype(np.float32),
                "log_flam": log_flam.astype(np.float32),
                "rstar": model_radius[model_id],
                "rmax": model_rmax[model_id],
                "n_points": int(lam.size),
                "source_path": fluxcont_path,
            }
        )

        if i % 500 == 0:
            print(f"[DATA] scanned {i} models, kept {len(records)}", flush=True)

    if not records:
        raise RuntimeError("No usable FLUXCONT records found.")

    all_counts = np.array([rec["n_points"] for rec in records], dtype=int)
    all_log_lambda = np.concatenate([rec["log_lambda"] for rec in records])

    lambda_log_min = float(all_log_lambda.min())
    lambda_log_max = float(all_log_lambda.max())
    if lambda_log_max == lambda_log_min:
        raise RuntimeError("Degenerate FLUXCONT wavelength range.")

    for rec in records:
        rec["wavelengths_log_norm"] = (
            (rec["log_lambda"] - lambda_log_min) / (lambda_log_max - lambda_log_min)
        ).astype(np.float32)

    print(f"[DATA] Kept FLUXCONT records: {len(records)}", flush=True)
    print(f"[DATA] Missing FLUXCONT files: {skipped_missing}", flush=True)
    print(f"[DATA] Failed/empty FLUXCONT files: {skipped_read}", flush=True)
    print(
        "[DATA] wavelength points per model: "
        f"min={all_counts.min()} median={np.median(all_counts):.0f} max={all_counts.max()}",
        flush=True,
    )
    print(
        "[DATA] wavelength range: "
        f"{10.0 ** lambda_log_min:.6e} - {10.0 ** lambda_log_max:.6e} Angstrom",
        flush=True,
    )

    return records, {
        "lambda_log_min": lambda_log_min,
        "lambda_log_max": lambda_log_max,
        "lambda_min_angstrom": float(10.0 ** lambda_log_min),
        "lambda_max_angstrom": float(10.0 ** lambda_log_max),
        "n_records": len(records),
        "n_points_min": int(all_counts.min()),
        "n_points_median": float(np.median(all_counts)),
        "n_points_max": int(all_counts.max()),
        "skipped_missing_fluxcont": int(skipped_missing),
        "skipped_failed_fluxcont": int(skipped_read),
    }


# =============================================================================
# Region weights and target standardisation
# =============================================================================

def compute_region_weights(wavelengths_phys, region_weights):
    """Per-point Huber weight as a function of the *physical* wavelength.

    ``region_weights`` of ``None`` (v1) means unit weight everywhere.
    """
    if region_weights is None:
        return np.ones(wavelengths_phys.shape, dtype=np.float32)

    uv_extreme_max = region_weights["uv_extreme_max"]
    uv_max = region_weights["uv_max"]
    tail_start = region_weights["tail_start"]
    far_tail_start = region_weights["far_tail_start"]

    weights = np.full(wavelengths_phys.shape, region_weights["mid_weight"], dtype=np.float32)
    weights[wavelengths_phys < uv_extreme_max] = region_weights["uv_extreme_weight"]
    weights[
        (wavelengths_phys >= uv_extreme_max) & (wavelengths_phys < uv_max)
    ] = region_weights["uv_weight"]
    weights[
        (wavelengths_phys >= tail_start) & (wavelengths_phys < far_tail_start)
    ] = region_weights["tail_weight"]
    weights[wavelengths_phys >= far_tail_start] = region_weights["far_tail_weight"]
    return weights


def compute_wavelength_bin_scaler(records, n_bins):
    """Accumulate the per-bin mean and standard deviation of log10(F_lambda).

    Empty bins are filled by linear interpolation of the neighbouring bin
    centres and the standard deviations are floored at 1e-4, exactly as in the
    sources.  The variance floor of 1e-8 guards against catastrophic
    cancellation in the ``E[x^2] - E[x]^2`` form used here.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    sums = np.zeros(n_bins, dtype=np.float64)
    sums2 = np.zeros(n_bins, dtype=np.float64)

    for rec in records:
        coords = rec["wavelengths_log_norm"].astype(np.float64)
        values = rec["log_flam"].astype(np.float64)
        bins = np.clip(np.digitize(coords, bin_edges) - 1, 0, n_bins - 1)
        counts += np.bincount(bins, minlength=n_bins)
        sums += np.bincount(bins, weights=values, minlength=n_bins)
        sums2 += np.bincount(bins, weights=values * values, minlength=n_bins)

    good = counts > 0
    if not np.any(good):
        raise RuntimeError("Could not compute wavelength-bin target scaler: all bins are empty.")

    means = np.zeros(n_bins, dtype=np.float64)
    stds = np.ones(n_bins, dtype=np.float64)
    means[good] = sums[good] / counts[good]
    variances = np.maximum(sums2[good] / counts[good] - means[good] ** 2, 1e-8)
    stds[good] = np.sqrt(variances)

    if not np.all(good):
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        means[~good] = np.interp(centers[~good], centers[good], means[good])
        stds[~good] = np.interp(centers[~good], centers[good], stds[good])

    stds = np.maximum(stds, 1e-4)
    return (
        bin_edges.astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
        counts.astype(np.int64),
    )


def lookup_bin_stats(coords_norm, bin_edges, bin_means, bin_stds):
    bins = np.clip(np.digitize(coords_norm, bin_edges) - 1, 0, len(bin_means) - 1)
    return bin_means[bins], bin_stds[bins], bins


class TargetScaler:
    """Standardisation of q = log10(F_lambda), either global or per-bin.

    Both variants expose the same interface -- given trunk coordinates, return
    the mean and standard deviation to apply -- so the rest of the code has a
    single path.  ``kind`` is ``"global"`` or ``"binned"``.
    """

    def __init__(self, kind, mean=None, std=None,
                 bin_edges=None, bin_means=None, bin_stds=None, bin_counts=None):
        self.kind = kind
        self.mean = None if mean is None else float(mean)
        self.std = None if std is None else float(std)
        self.bin_edges = bin_edges
        self.bin_means = bin_means
        self.bin_stds = bin_stds
        self.bin_counts = bin_counts

    @classmethod
    def fit(cls, train_records, cfg):
        if cfg["target_scaling"] == "global":
            train_log_targets = np.concatenate([rec["log_flam"] for rec in train_records])
            target_mean = float(train_log_targets.mean())
            target_std = float(train_log_targets.std())
            if target_std == 0.0:
                target_std = 1.0
            return cls("global", mean=target_mean, std=target_std)

        if cfg["target_scaling"] != "binned":
            raise ValueError(f"Unknown target_scaling: {cfg['target_scaling']!r}")

        bin_edges, bin_means, bin_stds, bin_counts = compute_wavelength_bin_scaler(
            train_records,
            cfg["target_norm_bins"],
        )
        return cls(
            "binned",
            bin_edges=bin_edges,
            bin_means=bin_means,
            bin_stds=bin_stds,
            bin_counts=bin_counts,
        )

    def stats_for(self, coords_norm):
        """Return ``(means, stds)`` arrays shaped like ``coords_norm``."""
        coords_norm = np.asarray(coords_norm)
        if self.kind == "global":
            means = np.full(coords_norm.shape, self.mean, dtype=np.float32)
            stds = np.full(coords_norm.shape, self.std, dtype=np.float32)
            return means, stds
        means, stds, _ = lookup_bin_stats(
            coords_norm, self.bin_edges, self.bin_means, self.bin_stds
        )
        return means, stds

    def scaler_metadata(self):
        """The scaler keys the sources wrote into their JSON outputs."""
        if self.kind == "global":
            return {
                "target_scaling": "global_standardization",
                "target_mean_train": self.mean,
                "target_std_train": self.std,
            }
        return {
            "target_scaling": "wavelength_bin_normalization",
            "target_norm_bins": int(len(self.bin_means)),
            "bin_edges_wavelength_log_norm": self.bin_edges.tolist(),
            "bin_mean_log_flam_train": self.bin_means.tolist(),
            "bin_std_log_flam_train": self.bin_stds.tolist(),
        }

    def npz_arrays(self):
        if self.kind == "global":
            return {
                "target_mean_train": np.array(self.mean),
                "target_std_train": np.array(self.std),
            }
        return {
            "bin_edges_wavelength_log_norm": self.bin_edges.astype(np.float32),
            "bin_mean_log_flam_train": self.bin_means.astype(np.float32),
            "bin_std_log_flam_train": self.bin_stds.astype(np.float32),
        }


# =============================================================================
# Dataset: one item is one SED, from which entries are drawn
# =============================================================================

class FluxcontCurveDataset(Dataset):
    """Draw ``wave_samples_per_model`` entries from each SED.

    Three sampling modes exist, selected by configuration:

    ``uniform_random``  (v1) draw without replacement when the SED has at
        least as many points as requested, otherwise draw uniformly *with*
        replacement.
    ``balanced``        (v2+) split [0, 1] into ``sampling_bins`` equal
        intervals and take ``ceil(p / n_active_bins)`` points from every
        occupied interval, then pad or trim to exactly ``p``.
    ``balanced`` + ``full_sed_sampling``  (v9, v10, v11, v13, v15) when the
        request is at least the number of native points, take *every* native
        point and pad the remainder.

    Padding is resampling with replacement of existing FASTWIND pairs.  No
    value is ever interpolated or moved onto a common grid.
    """

    def __init__(
        self,
        records,
        scaler,
        cfg,
        random_sample,
        seed,
    ):
        self.records = records
        self.scaler = scaler
        self.cfg = cfg
        self.wave_samples_per_model = int(cfg["wave_samples_per_model"])
        self.sampling = cfg["sampling"]
        self.sampling_bins = int(cfg["sampling_bins"])
        self.full_sed_sampling = bool(cfg["full_sed_sampling"])
        self.region_weights = cfg["region_weights"]
        self.random_sample = bool(random_sample)
        self.seed = int(seed)
        self.sampling_edges = np.linspace(0.0, 1.0, self.sampling_bins + 1, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def _deterministic_indices(self, n, p):
        if n == p:
            return np.arange(n)
        return np.linspace(0, n - 1, p).round().astype(int)

    def _uniform_sample_indices(self, coords):
        """v1: uniform random draw over the native points."""
        n = coords.shape[0]
        p = self.wave_samples_per_model
        if self.random_sample:
            if n < p:
                return np.random.randint(0, n, size=p)
            return np.random.choice(n, size=p, replace=False)
        return self._deterministic_indices(n, p)

    def _balanced_sample_indices(self, coords):
        """v2+: balanced draw over the log-lambda intervals."""
        n = coords.shape[0]
        p = self.wave_samples_per_model

        if self.full_sed_sampling:
            if p <= 0:
                return np.arange(n)

            if p >= n:
                base = np.arange(n)
                if p == n:
                    return base
                if self.random_sample:
                    fill = np.random.randint(0, n, size=p - n)
                else:
                    fill = np.linspace(0, n - 1, p - n).round().astype(int)
                return np.concatenate([base, fill]).astype(int)

        if not self.random_sample:
            return self._deterministic_indices(n, p)

        sampling_bins = np.clip(
            np.digitize(coords, self.sampling_edges) - 1, 0, self.sampling_bins - 1
        )
        active_bins = np.unique(sampling_bins)
        per_bin = max(1, int(np.ceil(p / max(len(active_bins), 1))))

        chosen_parts = []
        for b in active_bins:
            candidates = np.flatnonzero(sampling_bins == b)
            if candidates.size == 0:
                continue
            take = min(per_bin, candidates.size)
            chosen_parts.append(np.random.choice(candidates, size=take, replace=False))

        if chosen_parts:
            chosen = np.concatenate(chosen_parts)
        else:
            chosen = np.array([], dtype=int)

        if chosen.size < p:
            fill = np.random.randint(0, n, size=p - chosen.size)
            chosen = np.concatenate([chosen, fill])
        elif chosen.size > p:
            chosen = np.random.choice(chosen, size=p, replace=False)

        return chosen.astype(int)

    def sample_indices(self, coords):
        if self.sampling == "uniform_random":
            return self._uniform_sample_indices(coords)
        return self._balanced_sample_indices(coords)

    def __getitem__(self, idx):
        rec = self.records[idx]
        coords_all = rec["wavelengths_log_norm"]
        chosen = self.sample_indices(coords_all)

        coords = rec["wavelengths_log_norm"][chosen]
        target_log = rec["log_flam"][chosen]
        wavelengths_phys = rec["wavelengths_phys"][chosen]

        means, stds = self.scaler.stats_for(coords)
        target = (target_log - means) / stds
        weights = compute_region_weights(wavelengths_phys, self.region_weights)

        # The last three tensors are only consulted by the v6 edge-derivative
        # term, which needs the physical wavelength and the de-standardisation
        # constants of every sampled point.  They are always returned so that
        # the collate signature does not depend on the configuration.
        return (
            torch.tensor(rec["params"], dtype=torch.float32),
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(wavelengths_phys, dtype=torch.float32),
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32),
        )


# =============================================================================
# The DeepONet
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
    norm_type="batch_norm",
):
    """Plain feed-forward stack: [norm] -> Linear -> activation -> [dropout]."""
    layers = []
    in_dim = input_dim
    norm_key = str(norm_type or "none").lower()

    for idx, out_dim in enumerate(hidden_widths):
        if use_batch_norm and norm_key == "batch_norm":
            layers.append(nn.BatchNorm1d(in_dim))
        elif norm_key == "layer_norm":
            layers.append(nn.LayerNorm(in_dim))

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


class ResidualMLPBlock(nn.Module):
    """Pre-norm two-layer block with a projected shortcut (v17e only)."""

    def __init__(self, in_dim, out_dim, activation_name, dropout, use_batch_norm, norm_type):
        super().__init__()
        norm_key = str(norm_type or "none").lower()
        if use_batch_norm and norm_key == "batch_norm":
            self.norm = nn.BatchNorm1d(in_dim)
        elif norm_key == "layer_norm":
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.Identity()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act1 = activation_layer(activation_name)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.out_act = activation_layer(activation_name)

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act1(y)
        y = self.drop(y)
        y = self.fc2(y)
        return self.out_act(y + residual)


def build_residual_mlp(
    input_dim,
    output_dim,
    hidden_widths,
    activation_sequence,
    dropout,
    dropout_after_layer_indices,
    use_batch_norm,
    norm_type="none",
):
    layers = []
    in_dim = input_dim
    for idx, out_dim in enumerate(hidden_widths):
        act_name = (
            activation_sequence[idx]
            if idx < len(activation_sequence)
            else activation_sequence[-1]
        )
        block_dropout = dropout if idx in dropout_after_layer_indices else 0.0
        layers.append(
            ResidualMLPBlock(in_dim, out_dim, act_name, block_dropout, use_batch_norm, norm_type)
        )
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """Maps the 13 normalised stellar parameters to a latent vector."""

    def __init__(
        self,
        input_dim=13,
        output_dim=LATENT_DIM,
        hidden_widths=None,
        activation_sequence=None,
        dropout=0.02,
        dropout_after_layer_indices=None,
        use_batch_norm=True,
        norm_type="batch_norm",
        residual_blocks=False,
    ):
        super().__init__()
        hidden_widths = BRANCH_WIDTHS if hidden_widths is None else hidden_widths
        activation_sequence = (
            ACTIVATION_SEQUENCE if activation_sequence is None else activation_sequence
        )
        if dropout_after_layer_indices is None:
            dropout_after_layer_indices = list(range(len(hidden_widths)))
        builder = build_residual_mlp if residual_blocks else build_mlp
        self.net = builder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_widths=hidden_widths,
            activation_sequence=activation_sequence,
            dropout=dropout,
            dropout_after_layer_indices=dropout_after_layer_indices,
            use_batch_norm=use_batch_norm,
            norm_type=norm_type,
        )

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """Maps the normalised log-wavelength coordinate to a latent vector.

    The coordinate is expanded into ``[1, sin(2*pi*k*x), cos(2*pi*k*x)]`` for
    ``k = 1 .. fourier_modes`` before entering the MLP, which removes the
    spectral bias of a plain network fed a raw scalar.
    """

    def __init__(
        self,
        output_dim=LATENT_DIM,
        fourier_modes=64,
        hidden_widths=None,
        activation_sequence=None,
        dropout=0.02,
        dropout_after_layer_indices=None,
        use_batch_norm=True,
        norm_type="batch_norm",
        residual_blocks=False,
    ):
        super().__init__()
        hidden_widths = TRUNK_WIDTHS if hidden_widths is None else hidden_widths
        activation_sequence = (
            ACTIVATION_SEQUENCE if activation_sequence is None else activation_sequence
        )
        if dropout_after_layer_indices is None:
            dropout_after_layer_indices = list(range(len(hidden_widths)))
        self.fourier_modes = fourier_modes
        in_dim = 2 * fourier_modes + 1
        builder = build_residual_mlp if residual_blocks else build_mlp
        self.net = builder(
            input_dim=in_dim,
            output_dim=output_dim,
            hidden_widths=hidden_widths,
            activation_sequence=activation_sequence,
            dropout=dropout,
            dropout_after_layer_indices=dropout_after_layer_indices,
            use_batch_norm=use_batch_norm,
            norm_type=norm_type,
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
    """Prediction is the dot product of the branch and trunk latent vectors."""

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net

    def forward(self, params, coords):
        b = self.branch(params)
        t = self.trunk(coords)
        return (t * b.unsqueeze(1)).sum(-1)


def build_model(cfg):
    """Instantiate the DeepONet described by ``cfg``."""
    branch_net = BranchNet(
        input_dim=len(PARAM_COLS),
        output_dim=LATENT_DIM,
        hidden_widths=BRANCH_WIDTHS,
        activation_sequence=ACTIVATION_SEQUENCE,
        dropout=cfg["dropout"],
        dropout_after_layer_indices=cfg["dropout_after_layer_indices"],
        use_batch_norm=cfg["use_batch_norm"],
        norm_type=cfg["norm_type"],
        residual_blocks=cfg["residual_blocks"],
    )
    trunk_net = TrunkNet(
        output_dim=LATENT_DIM,
        fourier_modes=cfg["fourier_modes"],
        hidden_widths=TRUNK_WIDTHS,
        activation_sequence=ACTIVATION_SEQUENCE,
        dropout=cfg["dropout"],
        dropout_after_layer_indices=cfg["dropout_after_layer_indices"],
        use_batch_norm=cfg["use_batch_norm"],
        norm_type=cfg["norm_type"],
        residual_blocks=cfg["residual_blocks"],
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


# =============================================================================
# Loss terms
# =============================================================================

def _huber(err, delta):
    """rho_delta(e), written the way the sources wrote it."""
    abs_err = err.abs()
    quadratic = torch.minimum(
        abs_err, torch.tensor(delta, device=err.device, dtype=err.dtype)
    )
    linear = abs_err - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def weighted_huber_loss(preds, targets, weights, delta):
    """Region-weighted Huber penalty on the standardised target."""
    loss = _huber(preds - targets, delta)
    return (weights * loss).sum() / torch.clamp(weights.sum(), min=1.0)


def weighted_curve_bias_loss(preds, targets, weights):
    """Squared weighted mean signed residual of each sampled SED."""
    weight_sum = torch.clamp(weights.sum(dim=1), min=1.0)
    curve_bias = (weights * (preds - targets)).sum(dim=1) / weight_sum
    return torch.mean(curve_bias ** 2)


def endpoint_anchor_loss(preds, targets, coords, delta, endpoint_alpha):
    """Huber penalty repeated at the two extreme *sampled* coordinates."""
    if endpoint_alpha <= 0.0 or preds.shape[1] < 1:
        return preds.new_tensor(0.0)
    order = torch.argsort(coords, dim=1)
    first_idx = order[:, 0]
    last_idx = order[:, -1]
    row_idx = torch.arange(preds.shape[0], device=preds.device)
    pred_endpoints = torch.stack((preds[row_idx, first_idx], preds[row_idx, last_idx]), dim=1)
    target_endpoints = torch.stack(
        (targets[row_idx, first_idx], targets[row_idx, last_idx]), dim=1
    )
    return _huber(pred_endpoints - target_endpoints, delta).mean()


def fixed_tail_anchor_loss(preds, targets, coords, delta, anchor_alpha, anchor_coords):
    """Huber penalty at the sampled points nearest a list of fixed coordinates."""
    if anchor_alpha <= 0.0 or not anchor_coords or preds.shape[1] < 1:
        return preds.new_tensor(0.0)
    anchor_values = torch.tensor(anchor_coords, device=coords.device, dtype=coords.dtype)
    distances = torch.abs(coords.unsqueeze(-1) - anchor_values.view(1, 1, -1))
    anchor_idx = torch.argmin(distances, dim=1)
    row_idx = torch.arange(preds.shape[0], device=preds.device).unsqueeze(1)
    pred_anchor = preds[row_idx, anchor_idx]
    target_anchor = targets[row_idx, anchor_idx]
    return _huber(pred_anchor - target_anchor, delta).mean()


def derivative_smoothness_loss(preds, targets, coords, delta, derivative_alpha):
    """v13: Huber penalty on the standardised point-to-point slope."""
    if derivative_alpha <= 0.0 or preds.shape[1] < 2:
        return preds.new_tensor(0.0)
    order = torch.argsort(coords, dim=1)
    row_idx = torch.arange(preds.shape[0], device=preds.device).unsqueeze(1)
    pred_sorted = preds[row_idx, order]
    target_sorted = targets[row_idx, order]
    coord_sorted = coords[row_idx, order]
    dx = torch.clamp(coord_sorted[:, 1:] - coord_sorted[:, :-1], min=1.0e-6)
    pred_slope = (pred_sorted[:, 1:] - pred_sorted[:, :-1]) / dx
    target_slope = (target_sorted[:, 1:] - target_sorted[:, :-1]) / dx
    return _huber(pred_slope - target_slope, delta).mean()


def curve_amplitude_loss(preds, targets):
    """v13: squared difference of the mean level of predicted and target curve."""
    pred_level = preds.mean(dim=1)
    target_level = targets.mean(dim=1)
    return torch.mean((pred_level - target_level) ** 2)


def edge_derivative_huber_loss(
    preds, targets, coords, wavelengths_phys, means, stds, delta, cfg
):
    """v6: Huber penalty on the *de-standardised* slope near the domain edges.

    Restricted to adjacent pairs whose midpoint lies below
    ``derivative_edge_uv_max`` or at/above ``derivative_edge_tail_start``.
    Kept only for reproducibility: because consecutive sampled coordinates can
    be arbitrarily close, ``dx`` can be tiny and the gradient explodes, which is
    why the v6 validation loss came out three orders of magnitude worse than
    anything else in the search.
    """
    edge_alpha = cfg["edge_derivative_alpha"]
    if edge_alpha <= 0.0 or preds.shape[1] < 2:
        return preds.new_tensor(0.0)

    pred_log = preds * stds + means
    target_log = targets * stds + means

    order = torch.argsort(coords, dim=1)
    coords_s = torch.gather(coords, 1, order)
    preds_s = torch.gather(pred_log, 1, order)
    targets_s = torch.gather(target_log, 1, order)
    wave_s = torch.gather(wavelengths_phys, 1, order)

    dx = coords_s[:, 1:] - coords_s[:, :-1]
    valid = dx.abs() > 1.0e-8
    pred_slope = (preds_s[:, 1:] - preds_s[:, :-1]) / torch.clamp(dx, min=1.0e-8)
    target_slope = (targets_s[:, 1:] - targets_s[:, :-1]) / torch.clamp(dx, min=1.0e-8)

    wave_mid = 0.5 * (wave_s[:, 1:] + wave_s[:, :-1])
    edge = (wave_mid < cfg["derivative_edge_uv_max"]) | (
        wave_mid >= cfg["derivative_edge_tail_start"]
    )
    mask = valid & edge
    if not torch.any(mask):
        return preds.new_tensor(0.0)

    loss = _huber(pred_slope - target_slope, delta)
    return loss[mask].mean()


def fluxcont_loss(preds, targets, weights, coords, wave_phys, means, stds, cfg):
    """The composite loss.  ``cfg`` selects which terms are present.

    v1 used a plain, unweighted mean-squared error and nothing else; every
    later version used the region-weighted Huber penalty plus whichever of the
    bias / endpoint / tail / derivative / amplitude terms its configuration
    switches on.  A coefficient of 0.0 removes a term exactly.
    """
    if cfg["loss_kind"] == "mse":
        return ((preds - targets) ** 2).mean()

    delta = cfg["huber_delta"]
    total = weighted_huber_loss(preds, targets, weights, delta)

    if cfg["curve_bias_alpha"] > 0.0:
        total = total + cfg["curve_bias_alpha"] * weighted_curve_bias_loss(
            preds, targets, weights
        )

    if cfg["endpoint_alpha"] > 0.0:
        total = total + cfg["endpoint_alpha"] * endpoint_anchor_loss(
            preds, targets, coords, delta, cfg["endpoint_alpha"]
        )

    if cfg["tail_anchor_alpha"] > 0.0:
        total = total + cfg["tail_anchor_alpha"] * fixed_tail_anchor_loss(
            preds,
            targets,
            coords,
            delta,
            cfg["tail_anchor_alpha"],
            cfg["tail_anchor_coords_norm"],
        )

    if cfg["derivative_alpha"] > 0.0:
        total = total + cfg["derivative_alpha"] * derivative_smoothness_loss(
            preds, targets, coords, delta, cfg["derivative_alpha"]
        )

    if cfg["amplitude_alpha"] > 0.0:
        total = total + cfg["amplitude_alpha"] * curve_amplitude_loss(preds, targets)

    if cfg["edge_derivative_alpha"] > 0.0:
        total = total + cfg["edge_derivative_alpha"] * edge_derivative_huber_loss(
            preds, targets, coords, wave_phys, means, stds, delta, cfg
        )

    return total


# =============================================================================
# Device
# =============================================================================

def resolve_device():
    """Return ``(device, cuda_available, n_gpus)``, as every source computed it."""
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    device = torch.device("cuda:0" if cuda_available else "cpu")
    return device, cuda_available, n_gpus


# =============================================================================
# Shared data preparation (one pass per group, as the EXPERIMENTS runners did)
# =============================================================================

def prepare_common_data(cfg, runtime, param_info):
    """Build records, split them 70/15/15 and fit the target scaler.

    The split is the only seeded part of the pipeline: ``SPLIT_SEED`` is the
    ``random_state`` of both ``train_test_split`` calls, first 70/30 and then
    50/50 on the 30% remainder.
    """
    records, data_meta = build_fluxcont_records(cfg, runtime, param_info)

    indices = np.arange(len(records))
    if len(indices) < 3:
        raise RuntimeError(f"Not enough FLUXCONT models for split: {len(indices)}")

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=TEST_SIZE_FIRST_SPLIT,
        random_state=runtime["split_seed"],
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_SIZE_SECOND_SPLIT,
        random_state=runtime["split_seed"],
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    scaler = TargetScaler.fit(train_records, cfg)

    train_dataset = FluxcontCurveDataset(
        train_records, scaler, cfg, random_sample=True, seed=runtime["split_seed"]
    )
    val_dataset = FluxcontCurveDataset(
        val_records, scaler, cfg, random_sample=False, seed=runtime["split_seed"]
    )

    num_workers = runtime["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=runtime["pin_memory"],
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=runtime["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=runtime["pin_memory"],
        persistent_workers=(num_workers > 0),
    )

    return {
        "records": records,
        "data_meta": data_meta,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# =============================================================================
# Full-grid prediction and the resolved run configuration
# =============================================================================

def predict_full_record(model, rec, scaler, runtime):
    """Predict one model's whole native FLUXCONT grid, in coordinate chunks."""
    device = runtime["device"]
    params = torch.tensor(rec["params"][None, :], dtype=torch.float32, device=device)
    coords_all = rec["wavelengths_log_norm"]
    chunk = runtime["eval_wave_chunk_size"]
    preds_scaled = []

    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], chunk):
            stop = min(start + chunk, coords_all.shape[0])
            coords = torch.tensor(
                coords_all[start:stop][None, :],
                dtype=torch.float32,
                device=device,
            )
            pred = model(params, coords).detach().cpu().numpy()[0]
            preds_scaled.append(pred.astype(np.float32))

    pred_scaled = np.concatenate(preds_scaled)
    means, stds = scaler.stats_for(coords_all)
    pred_log_flam = pred_scaled * stds + means
    return pred_scaled.astype(np.float32), pred_log_flam.astype(np.float32)


def make_run_config(cfg, runtime, output_dir):
    """The dictionary written to ``run_config.json``."""
    config = {
        "architecture_name": cfg["architecture_name"],
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": cfg["run_tag"],
        "config_key": cfg.get("config_key"),
        "quantity": "FLUXCONT_SED",
        "param_cols": PARAM_COLS,
        "rmax_col": RMAX_COL,
        "rmax_is_ml_input": False,
        "target": (
            "standardized_log10_converted_F_lambda"
            if cfg["target_scaling"] == "global"
            else "wavelength_bin_normalized_log10_converted_F_lambda"
        ),
        "wavelength_input": "normalized_log10_lambda_angstrom",
        "reader": (
            "numpy_loadtxt"
            if cfg["reader"] == "loadtxt"
            else "parse_only_lambda_and_log_f_nue"
        ),
        "loss": cfg["loss_name"] or ("mse" if cfg["loss_kind"] == "mse" else DEFAULT_LOSS_NAME),
        "loss_kind": cfg["loss_kind"],
        "huber_delta": cfg["huber_delta"],
        "curve_bias_alpha": cfg["curve_bias_alpha"],
        "endpoint_alpha": cfg["endpoint_alpha"],
        "tail_anchor_alpha": cfg["tail_anchor_alpha"],
        "tail_anchor_lambdas": cfg["tail_anchor_lambdas"],
        "tail_anchor_coords_norm": cfg["tail_anchor_coords_norm"],
        "derivative_alpha": cfg["derivative_alpha"],
        "amplitude_alpha": cfg["amplitude_alpha"],
        "edge_derivative_alpha": cfg["edge_derivative_alpha"],
        "derivative_edge_uv_max": cfg["derivative_edge_uv_max"],
        "derivative_edge_tail_start": cfg["derivative_edge_tail_start"],
        "target_scaling": (
            "global_standardization"
            if cfg["target_scaling"] == "global"
            else "wavelength_bin_normalization"
        ),
        "sampling": cfg["sampling"],
        "sampling_bins": cfg["sampling_bins"],
        "full_sed_sampling": cfg["full_sed_sampling"],
        "region_weights": cfg["region_weights"],
        "branch_widths": BRANCH_WIDTHS,
        "trunk_widths": TRUNK_WIDTHS,
        "latent_dim": LATENT_DIM,
        "activation": "mixed_leaky_relu_silu",
        "activation_sequence": ACTIVATION_SEQUENCE,
        "dropout": cfg["dropout"],
        "dropout_after_layer_indices": cfg["dropout_after_layer_indices"],
        "use_batch_norm": cfg["use_batch_norm"],
        "norm_type": cfg["norm_type"],
        "residual_blocks": cfg["residual_blocks"],
        "fourier_modes": cfg["fourier_modes"],
        "learning_rate": cfg["learning_rate"],
        "optimizer": OPTIMIZER,
        "scheduler": SCHEDULER,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "batch_size": runtime["batch_size"],
        "wave_samples_per_model": cfg["wave_samples_per_model"],
        "eval_wave_chunk_size": runtime["eval_wave_chunk_size"],
        "max_epochs": runtime["max_epochs"],
        "early_stop_patience": runtime["early_stop_patience"],
        "weight_decay": runtime["weight_decay"],
        "split_seed": runtime["split_seed"],
        "grid_json_path": runtime["grid_json_path"],
        "quality_json_path": runtime["quality_json_path"],
        "models_root": runtime["models_root"],
        "output_dir": output_dir,
        "cuda_available": runtime["cuda_available"],
        "n_gpus_visible": runtime["n_gpus"],
        "data_parallel_used": runtime["n_gpus"] >= 2,
        "num_workers": runtime["num_workers"],
        "zstd_bin": runtime["zstd_bin"],
        "lambda_min_filter": cfg["lambda_min_filter"],
        "lambda_max_filter": cfg["lambda_max_filter"],
        "min_fluxcont_points": runtime["min_fluxcont_points"],
        "stop_before_timeout_seconds": runtime["stop_before_timeout_seconds"],
        "max_runtime_seconds": runtime["max_runtime_seconds"],
        "time_limit_seconds": runtime["time_limit_seconds"],
    }
    if cfg["subversion"] is not None:
        config["subversion"] = cfg["subversion"]
    if cfg["target_scaling"] == "binned":
        config["target_norm_bins"] = cfg["target_norm_bins"]
    return config


def _loss_descriptor(cfg):
    """The loss-related keys the sources repeated in every metadata file."""
    keys = {
        "loss": cfg["loss_name"] or ("mse" if cfg["loss_kind"] == "mse" else DEFAULT_LOSS_NAME),
        "huber_delta": cfg["huber_delta"],
        "curve_bias_alpha": cfg["curve_bias_alpha"],
        "endpoint_alpha": cfg["endpoint_alpha"],
        "sampling_bins": cfg["sampling_bins"],
        "region_weights": cfg["region_weights"],
    }
    if cfg["tail_anchor_alpha"] > 0.0 or cfg["tail_anchor_coords_norm"]:
        keys["tail_anchor_alpha"] = cfg["tail_anchor_alpha"]
        keys["tail_anchor_lambdas"] = cfg["tail_anchor_lambdas"]
        keys["tail_anchor_coords_norm"] = cfg["tail_anchor_coords_norm"]
    if cfg["derivative_alpha"] > 0.0:
        keys["derivative_alpha"] = cfg["derivative_alpha"]
    if cfg["amplitude_alpha"] > 0.0:
        keys["amplitude_alpha"] = cfg["amplitude_alpha"]
    if cfg["edge_derivative_alpha"] > 0.0:
        keys["derivative_alpha"] = cfg["edge_derivative_alpha"]
        keys["derivative_edge_uv_max"] = cfg["derivative_edge_uv_max"]
        keys["derivative_edge_tail_start"] = cfg["derivative_edge_tail_start"]
    return keys


# =============================================================================
# Training
# =============================================================================

def _batch_to_device(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]


def _epoch_pass(model, loader, cfg, runtime, optimizer=None):
    """One pass over ``loader``; trains when ``optimizer`` is given."""
    device = runtime["device"]
    running = 0.0
    for batch in loader:
        (
            batch_params,
            batch_coords,
            batch_targets,
            batch_weights,
            batch_wave_phys,
            batch_means,
            batch_stds,
        ) = _batch_to_device(batch, device)

        if optimizer is not None:
            optimizer.zero_grad()

        preds = model(batch_params, batch_coords)
        loss = fluxcont_loss(
            preds,
            batch_targets,
            batch_weights,
            batch_coords,
            batch_wave_phys,
            batch_means,
            batch_stds,
            cfg,
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running += loss.item()

    return running / max(len(loader), 1)


def train_one_config(cfg, common, runtime, param_info):
    """Train, evaluate and write out one configuration.  Returns its summary."""
    label = cfg["subversion"] or cfg.get("config_key") or cfg["run_tag"]
    output_dir = os.path.join(runtime["output_root"], cfg["run_tag"])
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "training_summary.json")
    training_summary = load_existing_summary(summary_path)

    if is_run_complete(output_dir):
        print(f"[resume] Completed {label} run detected. Skipping training.", flush=True)
        save_summary(summary_path, training_summary)
        return training_summary

    config = make_run_config(cfg, runtime, output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "parameter_normalization.json"), "w") as f:
        json.dump(param_info["norm_metadata"], f, indent=2)

    data_meta = common["data_meta"]
    train_records = common["train_records"]
    val_records = common["val_records"]
    test_records = common["test_records"]
    scaler = common["scaler"]
    train_loader = common["train_loader"]
    val_loader = common["val_loader"]

    scaler_metadata = {
        "target_name": "log10_converted_F_lambda",
        "target_units_before_log": "erg/s/Angstrom",
        "conversion": CONVERSION_NOTE,
        "sampling": (
            "uniform_random_per_model"
            if cfg["sampling"] == "uniform_random"
            else "balanced_log_lambda_bins_per_model"
        ),
        **scaler.scaler_metadata(),
        **_loss_descriptor(cfg),
        **data_meta,
    }
    if scaler.kind == "binned":
        scaler_metadata["bin_counts_train"] = scaler.bin_counts.tolist()
    with open(os.path.join(output_dir, "sed_target_and_wavelength_scaler.json"), "w") as f:
        json.dump(scaler_metadata, f, indent=2)

    print(f"[RUN] Starting {label} -> {output_dir}", flush=True)

    device = runtime["device"]
    base_model = build_model(cfg).to(device)

    using_data_parallel = False
    if runtime["cuda_available"] and runtime["n_gpus"] >= 2:
        print(f"[GPU] Wrapping model with DataParallel over {runtime['n_gpus']} GPUs", flush=True)
        model = nn.DataParallel(base_model)
        using_data_parallel = True
    else:
        print("[GPU] Using single GPU / CPU execution", flush=True)
        model = base_model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=runtime["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    train_losses = []
    train_eval_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_early = False
    start_time = time.time()

    for epoch in range(1, runtime["max_epochs"] + 1):
        ensure_time_budget(f"{label} epoch loop", minimum_seconds=600)

        model.train()
        avg_train_loss = _epoch_pass(model, train_loader, cfg, runtime, optimizer=optimizer)
        train_losses.append(avg_train_loss)

        model.eval()
        avg_train_eval_loss = None
        with torch.no_grad():
            if cfg["track_train_eval"]:
                avg_train_eval_loss = _epoch_pass(model, train_loader, cfg, runtime)
                train_eval_losses.append(avg_train_eval_loss)
            avg_val_loss = _epoch_pass(model, val_loader, cfg, runtime)

        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(get_base_model_state_dict(model))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg["track_train_eval"]:
            print(
                f"[{label}] epoch {epoch:03d} "
                f"train={avg_train_loss:.6e} "
                f"train_eval={avg_train_eval_loss:.6e} "
                f"val={avg_val_loss:.6e}",
                flush=True,
            )
        else:
            print(
                f"[{label}] epoch {epoch:03d} "
                f"train={avg_train_loss:.6e} val={avg_val_loss:.6e}",
                flush=True,
            )

        if epochs_no_improve >= runtime["early_stop_patience"]:
            stopped_early = True
            print(f"[{label}] Early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        load_base_model_state_dict(model, best_state)

    paths = artifact_paths(output_dir)

    print(f"[TEST] Evaluating full native FLUXCONT grids for {label}", flush=True)
    model_offsets = [0]
    test_model_ids = []
    model_ids_flat = []
    wave_phys_flat = []
    wave_log_norm_flat = []
    target_log_flam_flat = []
    pred_log_flam_flat = []
    target_scaled_flat = []
    pred_scaled_flat = []
    per_model_metrics = []

    for rec in test_records:
        ensure_time_budget(f"{label} full test-set FLUXCONT evaluation", minimum_seconds=300)

        pred_scaled, pred_log_flam = predict_full_record(model, rec, scaler, runtime)
        target_log_flam = rec["log_flam"].astype(np.float32)
        target_means, target_stds = scaler.stats_for(rec["wavelengths_log_norm"])
        target_scaled = ((target_log_flam - target_means) / target_stds).astype(np.float32)

        target_flux = np.power(10.0, target_log_flam.astype(np.float64))
        pred_flux = np.power(10.0, pred_log_flam.astype(np.float64))
        rel_err = np.abs(pred_flux - target_flux) / np.maximum(np.abs(target_flux), 1e-300)

        npts = rec["wavelengths_phys"].shape[0]
        test_model_ids.append(rec["model_id"])
        model_ids_flat.append(np.full(npts, rec["model_id"], dtype=np.int32))
        wave_phys_flat.append(rec["wavelengths_phys"].astype(np.float32))
        wave_log_norm_flat.append(rec["wavelengths_log_norm"].astype(np.float32))
        target_log_flam_flat.append(target_log_flam)
        pred_log_flam_flat.append(pred_log_flam)
        target_scaled_flat.append(target_scaled)
        pred_scaled_flat.append(pred_scaled)
        model_offsets.append(model_offsets[-1] + npts)

        per_model_metrics.append(
            {
                "model_id": int(rec["model_id"]),
                "n_points": int(npts),
                "mse_scaled_log_flam": float(np.mean((pred_scaled - target_scaled) ** 2)),
                "mse_log_flam": float(np.mean((pred_log_flam - target_log_flam) ** 2)),
                "mean_relative_flux_error": float(np.mean(rel_err)),
                "median_relative_flux_error": float(np.median(rel_err)),
            }
        )

    model_ids_flat = np.concatenate(model_ids_flat)
    wave_phys_flat = np.concatenate(wave_phys_flat)
    wave_log_norm_flat = np.concatenate(wave_log_norm_flat)
    target_log_flam_flat = np.concatenate(target_log_flam_flat)
    pred_log_flam_flat = np.concatenate(pred_log_flam_flat)
    target_scaled_flat = np.concatenate(target_scaled_flat)
    pred_scaled_flat = np.concatenate(pred_scaled_flat)

    target_flux_flat = np.power(10.0, target_log_flam_flat.astype(np.float64))
    pred_flux_flat = np.power(10.0, pred_log_flam_flat.astype(np.float64))

    architecture_block = {
        "architecture_name": cfg["architecture_name"],
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": cfg["run_tag"],
        "quantity": "FLUXCONT_SED",
        "branch_widths": BRANCH_WIDTHS,
        "trunk_widths": TRUNK_WIDTHS,
        "latent_dim": LATENT_DIM,
        "activation": "mixed_leaky_relu_silu",
        "activation_sequence": ACTIVATION_SEQUENCE,
        "dropout": cfg["dropout"],
        "dropout_after_layer_indices": cfg["dropout_after_layer_indices"],
        "use_batch_norm": cfg["use_batch_norm"],
        "norm_type": cfg["norm_type"],
        "residual_blocks": cfg["residual_blocks"],
        "fourier_modes": cfg["fourier_modes"],
        "learning_rate": cfg["learning_rate"],
        "batch_size": runtime["batch_size"],
        "wave_samples_per_model": cfg["wave_samples_per_model"],
    }
    if cfg["subversion"] is not None:
        architecture_block["subversion"] = cfg["subversion"]

    checkpoint = {
        "model_state_dict": get_base_model_state_dict(model),
        "param_cols": PARAM_COLS,
        **architecture_block,
        **scaler.scaler_metadata(),
        **_loss_descriptor(cfg),
        "lambda_log_min": data_meta["lambda_log_min"],
        "lambda_log_max": data_meta["lambda_log_max"],
        "param_mins_after_mdot_log": param_info["param_mins"].tolist(),
        "param_maxs_after_mdot_log": param_info["param_maxs"].tolist(),
        "param_range_after_mdot_log": param_info["param_range"].tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "data_parallel_used": using_data_parallel,
        "n_gpus_visible": runtime["n_gpus"],
        "zstd_bin": runtime["zstd_bin"],
    }
    if cfg["track_train_eval"]:
        checkpoint["train_eval_losses"] = train_eval_losses
    torch.save(checkpoint, paths["model"])

    loss_record = {
        "quantity": "FLUXCONT_SED",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": len(train_losses),
        "stopped_early": stopped_early,
    }
    if cfg["track_train_eval"]:
        loss_record["train_eval_losses"] = train_eval_losses
        loss_record["final_train_loss"] = train_losses[-1] if train_losses else None
        loss_record["final_train_eval_loss"] = (
            train_eval_losses[-1] if train_eval_losses else None
        )
        loss_record["final_val_loss"] = val_losses[-1] if val_losses else None
        loss_record["train_eval_loss_note"] = cfg["train_eval_loss_note"]
        loss_record["use_batch_norm"] = cfg["use_batch_norm"]
        loss_record["norm_type"] = cfg["norm_type"]
        loss_record["learning_rate"] = cfg["learning_rate"]
        loss_record["residual_blocks"] = cfg["residual_blocks"]
        loss_record["tail_anchor_alpha"] = cfg["tail_anchor_alpha"]
        loss_record["tail_anchor_lambdas"] = cfg["tail_anchor_lambdas"]
        loss_record["tail_anchor_coords_norm"] = cfg["tail_anchor_coords_norm"]
    with open(paths["loss"], "w") as f:
        json.dump(loss_record, f, indent=2)

    with open(paths["split"], "w") as f:
        json.dump(
            {
                "quantity": "FLUXCONT_SED",
                "train_model_ids": [int(rec["model_id"]) for rec in train_records],
                "val_model_ids": [int(rec["model_id"]) for rec in val_records],
                "test_model_ids": [int(rec["model_id"]) for rec in test_records],
            },
            f,
            indent=2,
        )

    npz_payload = dict(
        quantity="FLUXCONT_SED",
        test_model_ids=np.array(test_model_ids, dtype=np.int32),
        model_offsets=np.array(model_offsets, dtype=np.int64),
        model_ids_flat=model_ids_flat.astype(np.int32),
        wavelengths_phys_flat=wave_phys_flat.astype(np.float32),
        wavelengths_log_norm_flat=wave_log_norm_flat.astype(np.float32),
        target_scaled_log_flam_flat=target_scaled_flat.astype(np.float32),
        pred_scaled_log_flam_flat=pred_scaled_flat.astype(np.float32),
        target_log_flam_flat=target_log_flam_flat.astype(np.float32),
        pred_log_flam_flat=pred_log_flam_flat.astype(np.float32),
        target_flux_flat=target_flux_flat,
        pred_flux_flat=pred_flux_flat,
        param_cols=np.array(PARAM_COLS),
        lambda_log_min=np.array(data_meta["lambda_log_min"]),
        lambda_log_max=np.array(data_meta["lambda_log_max"]),
    )
    npz_payload.update(scaler.npz_arrays())
    if cfg["npz_loss_arrays"]:
        npz_payload.update(
            train_losses=np.asarray(train_losses, dtype=np.float32),
            train_eval_losses=np.asarray(train_eval_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
            best_epoch=np.array(best_epoch, dtype=np.int32),
            best_val_loss=np.array(best_val_loss, dtype=np.float32),
        )
    np.savez_compressed(paths["pred"], **npz_payload)

    metrics_df = pd.DataFrame(per_model_metrics)
    metrics_path = os.path.join(output_dir, "test_metrics_per_model.csv")
    metrics_df.to_csv(metrics_path, index=False)

    counts_block = {
        "n_total_models": int(len(common["records"])),
        "n_train": int(len(train_records)),
        "n_val": int(len(val_records)),
        "n_test": int(len(test_records)),
    }

    meta_record = {
        "quantity": "FLUXCONT_SED",
        "architecture_tag": cfg["architecture_tag"],
        "run_tag": cfg["run_tag"],
        **counts_block,
        "training_seconds": time.time() - start_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": len(train_losses),
        "stopped_early": stopped_early,
        "data_parallel_used": using_data_parallel,
        "n_gpus_visible": runtime["n_gpus"],
        "use_batch_norm": cfg["use_batch_norm"],
        "norm_type": cfg["norm_type"],
        "learning_rate": cfg["learning_rate"],
        "residual_blocks": cfg["residual_blocks"],
        **scaler.scaler_metadata(),
        **_loss_descriptor(cfg),
        "test_mse_scaled_log_flam": float(metrics_df["mse_scaled_log_flam"].mean()),
        "test_mse_log_flam": float(metrics_df["mse_log_flam"].mean()),
        "test_mean_relative_flux_error": float(metrics_df["mean_relative_flux_error"].mean()),
        **data_meta,
    }
    if cfg["subversion"] is not None:
        meta_record["subversion"] = cfg["subversion"]
    # The binned scaler arrays are long; the sources kept them out of the meta
    # file and only reported the bin count there.
    meta_record.pop("bin_edges_wavelength_log_norm", None)
    meta_record.pop("bin_mean_log_flam_train", None)
    meta_record.pop("bin_std_log_flam_train", None)
    with open(paths["meta"], "w") as f:
        json.dump(meta_record, f, indent=2)

    training_summary = {
        "quantity": "FLUXCONT_SED",
        "status": "completed",
        "config_key": cfg.get("config_key"),
        **architecture_block,
        **scaler.scaler_metadata(),
        **_loss_descriptor(cfg),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": len(train_losses),
        "stopped_early": stopped_early,
        **counts_block,
        "data_parallel_used": using_data_parallel,
        "n_gpus_visible": runtime["n_gpus"],
        "test_metrics_path": metrics_path,
        **data_meta,
    }
    if cfg["track_train_eval"]:
        training_summary["final_train_loss"] = train_losses[-1] if train_losses else None
        training_summary["final_train_eval_loss"] = (
            train_eval_losses[-1] if train_eval_losses else None
        )
        training_summary["final_val_loss"] = val_losses[-1] if val_losses else None
        training_summary["train_eval_loss_note"] = cfg["train_eval_loss_note"]
    training_summary.pop("bin_edges_wavelength_log_norm", None)
    training_summary.pop("bin_mean_log_flam_train", None)
    training_summary.pop("bin_std_log_flam_train", None)
    save_summary(summary_path, training_summary)

    print(f"[DONE] {label} finished in {time.time() - start_time:.1f} s", flush=True)
    return training_summary


def run_configs(config_keys, runtime, param_info):
    """Run one or more configurations, sharing the data-preparation pass.

    All members of a group necessarily agree on every data-level setting, so a
    single ``prepare_common_data`` call serves them all, exactly as the
    ``EXPERIMENTS`` runners did.  Keys that differ in a data-level setting are
    prepared separately.
    """
    configs = [resolve_config(key, runtime["env"]) for key in config_keys]

    # Group the configurations by everything that affects data preparation.
    def data_signature(cfg):
        return (
            cfg["reader"],
            cfg["lambda_min_filter"],
            cfg["lambda_max_filter"],
            cfg["target_scaling"],
            cfg["target_norm_bins"],
            cfg["sampling"],
            cfg["sampling_bins"],
            cfg["full_sed_sampling"],
            cfg["wave_samples_per_model"],
            None if cfg["region_weights"] is None else tuple(sorted(cfg["region_weights"].items())),
        )

    batches = []
    for cfg in configs:
        signature = data_signature(cfg)
        if batches and batches[-1][0] == signature:
            batches[-1][1].append(cfg)
        else:
            batches.append((signature, [cfg]))

    results = []
    for _signature, batch in batches:
        common = prepare_common_data(batch[0], runtime, param_info)
        for cfg in batch:
            results.append(train_one_config(cfg, common, runtime, param_info))
    return results


# =============================================================================
# Command line interface
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="fluxcont_training.py",
        description=(
            "Train the FASTWIND FLUXCONT (continuum / SED) DeepONet emulator. "
            "Every configuration of the architecture search is reproducible "
            "through --config; the adopted model is the default, v17a."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fluxcont_training.py\n"
            "  python fluxcont_training.py --config v17a\n"
            "  python fluxcont_training.py --config v18a,v18b\n"
            "  python fluxcont_training.py --group v17\n"
            "  python fluxcont_training.py --list\n"
            "  python fluxcont_training.py --config v17a --print-config\n"
            "  python fluxcont_training.py --config v17a --dry-run\n"
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Configuration key, or a comma-separated list of keys.  Group keys "
            f"(v11 .. v18) expand to all their members.  Default: {DEFAULT_CONFIG_KEY}."
        ),
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Run every member of a group (v11 .. v18) in the original order.",
    )
    parser.add_argument("--list", action="store_true", help="List the configurations and exit.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration(s) as JSON and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything and report what would run, but do not train.",
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    paths.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    paths.add_argument(
        "--models-root",
        default=None,
        help="Default: <storage-dir>/all_models_more_info",
    )
    paths.add_argument(
        "--grid-json",
        default=None,
        help="Default: <storage-dir>/grid_large_log_LHC.json",
    )
    paths.add_argument(
        "--quality-json",
        default=None,
        help="Default: <storage-dir>/grid_quality_large_lhc.json",
    )
    paths.add_argument(
        "--output-root",
        default=None,
        help="Default: <base-dir>/fluxcont_runs",
    )

    knobs = parser.add_argument_group("run-level overrides")
    knobs.add_argument("--batch-size", type=int, default=None)
    knobs.add_argument("--max-epochs", type=int, default=None)
    knobs.add_argument("--early-stop-patience", type=int, default=None)
    knobs.add_argument("--weight-decay", type=float, default=None)
    knobs.add_argument("--split-seed", type=int, default=None)
    knobs.add_argument("--min-fluxcont-points", type=int, default=None)
    knobs.add_argument("--eval-wave-chunk-size", type=int, default=None)
    knobs.add_argument("--num-workers", type=int, default=None)
    knobs.add_argument("--zstd-bin", default=None)
    knobs.add_argument("--stop-before-timeout-seconds", type=int, default=None)
    knobs.add_argument("--max-runtime-seconds", type=int, default=None)

    return parser


def _env_int(env, name, fallback):
    raw = env.get(name)
    return fallback if raw in (None, "") else int(raw)


def _env_float(env, name, fallback):
    raw = env.get(name)
    return fallback if raw in (None, "") else float(raw)


def resolve_runtime(args, env=None):
    """Build the run-level settings dictionary.

    Precedence is: explicit command-line flag > ``ML13_*`` environment variable
    > the module constant copied from the sources.
    """
    if env is None:
        env = os.environ

    base_dir = args.base_dir
    storage_dir = args.storage_dir

    models_root = args.models_root or os.path.join(storage_dir, "all_models_more_info")
    grid_json_path = args.grid_json or os.path.join(storage_dir, "grid_large_log_LHC.json")
    quality_json_path = args.quality_json or os.path.join(
        storage_dir, "grid_quality_large_lhc.json"
    )
    output_root = args.output_root or os.path.join(base_dir, "fluxcont_runs")

    def pick(flag_value, env_value):
        return env_value if flag_value is None else flag_value

    runtime = {
        "env": env,
        "base_dir": base_dir,
        "storage_dir": storage_dir,
        "models_root": models_root,
        "grid_json_path": grid_json_path,
        "quality_json_path": quality_json_path,
        "output_root": output_root,
        "batch_size": pick(args.batch_size, _env_int(env, "ML13_SED_BATCH_SIZE", BATCH_SIZE)),
        "max_epochs": pick(args.max_epochs, _env_int(env, "ML13_SED_MAX_EPOCHS", MAX_EPOCHS)),
        "early_stop_patience": pick(
            args.early_stop_patience,
            _env_int(env, "ML13_SED_EARLY_STOP_PATIENCE", EARLY_STOP_PATIENCE),
        ),
        "weight_decay": pick(
            args.weight_decay, _env_float(env, "ML13_SED_WEIGHT_DECAY", WEIGHT_DECAY)
        ),
        "split_seed": pick(args.split_seed, _env_int(env, "ML13_SED_SPLIT_SEED", SPLIT_SEED)),
        "min_fluxcont_points": pick(
            args.min_fluxcont_points,
            _env_int(env, "ML13_SED_MIN_FLUXCONT_POINTS", MIN_FLUXCONT_POINTS),
        ),
        "eval_wave_chunk_size": pick(
            args.eval_wave_chunk_size,
            _env_int(env, "ML13_SED_EVAL_WAVE_CHUNK_SIZE", EVAL_WAVE_CHUNK_SIZE),
        ),
        "num_workers": pick(args.num_workers, DEFAULT_NUM_WORKERS),
        "zstd_bin": pick(args.zstd_bin, DEFAULT_ZSTD_BIN),
        "stop_before_timeout_seconds": pick(
            args.stop_before_timeout_seconds, DEFAULT_STOP_BEFORE_TIMEOUT_SECONDS
        ),
        "max_runtime_seconds": pick(args.max_runtime_seconds, DEFAULT_MAX_RUNTIME_SECONDS),
    }

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


def print_config_listing():
    print("Available FLUXCONT configurations:")
    print()
    for key, cfg in CONFIGS.items():
        group = cfg["group"] or "-"
        marker = "  <-- ADOPTED" if key == DEFAULT_CONFIG_KEY else ""
        print(
            f"  {key:<6} group={group:<5} "
            f"scaling={cfg['target_scaling']:<6} "
            f"fm={cfg['fourier_modes']:<3} "
            f"drop={cfg['dropout']:<5} "
            f"norm={cfg['norm_type']:<11} "
            f"endpoint={cfg['endpoint_alpha']:<6} "
            f"tail={cfg['tail_anchor_alpha']}"
            f"{marker}"
        )
    print()
    print("Groups (run every member, sharing one data-preparation pass):")
    for group_key, members in GROUPS.items():
        print(f"  {group_key:<6} -> {', '.join(members)}")


def print_runtime_banner(runtime, config_keys):
    print("Using device:", runtime["device"], flush=True)
    print("CUDA available:", runtime["cuda_available"], flush=True)
    print("Detected GPUs:", runtime["n_gpus"], flush=True)
    print("BASE_DIR:", runtime["base_dir"], flush=True)
    print("STORAGE_DIR:", runtime["storage_dir"], flush=True)
    print("MODELS_ROOT:", runtime["models_root"], flush=True)
    print("GRID_JSON_PATH:", runtime["grid_json_path"], flush=True)
    print("QUALITY_JSON_PATH:", runtime["quality_json_path"], flush=True)
    print("OUTPUT_ROOT:", runtime["output_root"], flush=True)
    print("NUM_WORKERS:", runtime["num_workers"], flush=True)
    print("BATCH_SIZE:", runtime["batch_size"], flush=True)
    print("EVAL_WAVE_CHUNK_SIZE:", runtime["eval_wave_chunk_size"], flush=True)
    print("SPLIT_SEED:", runtime["split_seed"], flush=True)
    print("STOP_BEFORE_TIMEOUT_SECONDS:", runtime["stop_before_timeout_seconds"], flush=True)
    print("MAX_RUNTIME_SECONDS:", runtime["max_runtime_seconds"], flush=True)
    print("TIME_LIMIT_SECONDS:", runtime["time_limit_seconds"], flush=True)
    print("DEADLINE_TS:", runtime["deadline_ts"], flush=True)
    print("ZSTD_BIN:", runtime["zstd_bin"], flush=True)
    print("CONFIGS TO RUN:", ", ".join(config_keys), flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_config_listing()
        return 0

    if args.group is not None and args.config is not None:
        parser.error("--group and --config are mutually exclusive")

    if args.group is not None:
        if args.group not in GROUPS:
            parser.error(f"Unknown group '{args.group}'. Known groups: {sorted(GROUPS)}")
        requested = [args.group]
    elif args.config is not None:
        requested = [k.strip() for k in args.config.split(",") if k.strip()]
    else:
        requested = [DEFAULT_CONFIG_KEY]

    config_keys = expand_config_keys(requested)
    unknown = [k for k in config_keys if k not in CONFIGS]
    if unknown:
        parser.error(f"Unknown configuration key(s): {unknown}. Try --list.")

    runtime = resolve_runtime(args)

    if args.print_config:
        resolved = {key: resolve_config(key, runtime["env"]) for key in config_keys}
        print(json.dumps(resolved, indent=2, default=str))
        return 0

    print_runtime_banner(runtime, config_keys)

    if args.dry_run:
        print("\n[dry-run] Nothing was trained.  Resolved plan:", flush=True)
        for key in config_keys:
            cfg = resolve_config(key, runtime["env"])
            print(
                f"  {key:<6} -> {os.path.join(runtime['output_root'], cfg['run_tag'])}",
                flush=True,
            )
        return 0

    # Group-level summary file, written by the multi-experiment runners.
    group_keys = []
    for key in config_keys:
        group = CONFIGS[key]["group"]
        if group is not None and group not in group_keys:
            group_keys.append(group)

    group_summary_paths = {}
    for group in group_keys:
        group_dir = os.path.join(runtime["output_root"], GROUP_META[group]["run_tag"])
        os.makedirs(group_dir, exist_ok=True)
        group_summary_paths[group] = os.path.join(group_dir, "training_summary.json")

    members_of = {
        group: [k for k in config_keys if CONFIGS[k]["group"] == group] for group in group_keys
    }
    group_summaries = {
        group: {
            "status": "started",
            "run_tag": GROUP_META[group]["run_tag"],
            "subversions": members_of[group],
        }
        for group in group_keys
    }
    for group, path in group_summary_paths.items():
        save_summary(path, group_summaries[group])

    param_info = None
    exit_code = 0
    try:
        params_df = load_parameter_table(runtime)
        param_info = normalize_parameters(params_df)
        results = run_configs(config_keys, runtime, param_info)

        for group in group_keys:
            group_summaries[group] = {
                "status": "completed",
                "run_tag": GROUP_META[group]["run_tag"],
                "subversions": members_of[group],
                "results": [
                    result
                    for key, result in zip(config_keys, results)
                    if CONFIGS[key]["group"] == group
                ],
            }
    except DeadlineReached as e:
        print(f"[deadline] {e}", flush=True)
        print("[deadline] No partial FLUXCONT checkpoint saved.", flush=True)
        for group in group_keys:
            group_summaries[group]["status"] = "deadline_reached"
            group_summaries[group]["error"] = str(e)
        exit_code = 0
    finally:
        for group, path in group_summary_paths.items():
            save_summary(path, group_summaries[group])
            print(f"Final multi-run summary written to {path}", flush=True)

    print("\nFLUXCONT training run completed.", flush=True)
    print("\nDone.", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
