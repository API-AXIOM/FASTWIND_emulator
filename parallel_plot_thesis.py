#!/usr/bin/env python3
"""Create thesis-ready parallel-coordinate plots from exported 5-par metadata."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


DISPLAY = [
    ("activation_code", "Activation"),
    ("latent_dim", "Latent\ndimension"),
    ("fourier_modes", "Fourier\nmodes"),
    ("dropout", "Dropout"),
    ("learning_rate_log", "Learning rate"),
    ("batch_size", "Batch size"),
    ("max_epochs", "Maximum\nepochs"),
    ("early_stop_count", "Early-stopped\nlines"),
    ("call_time_log_ms", "Call time\n(ms, log scale)"),
    ("branch_depth", "Branch\ndepth"),
    ("trunk_depth", "Trunk\ndepth"),
    ("branch_width_max", "Maximum\nbranch width"),
    ("trunk_width_max", "Maximum\ntrunk width"),
    ("val_loss_log", "Validation loss\n(log scale)"),
    ("mse_log", "Test MSE\n(log scale)"),
    ("mare_log", "Test MARE\n(log scale)"),
]

AXIS_PAD = 0.04


def load_data(path):
    rows = json.loads(Path(path).read_text())["architecture_level"]
    df = pd.DataFrame(rows)
    df = df[df["activation"].isin(["relu", "gelu"])].copy()
    numeric = [x[0] for x in DISPLAY if x[0] not in {"activation_code", "learning_rate_log", "call_time_log_ms", "val_loss_log", "mse_log", "mare_log"}]
    numeric += ["learning_rate", "total_emulator_call_time_sec", "val_loss", "mse", "mare"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["activation_code"] = df["activation"].map({"relu": 0.0, "gelu": 1.0})
    df["learning_rate_log"] = np.log10(df["learning_rate"])
    df["call_time_log_ms"] = np.log10(1000.0 * df["total_emulator_call_time_sec"])
    for metric in ("val_loss", "mse", "mare"):
        df[f"{metric}_log"] = np.log10(df[metric])
    df = df.dropna(subset=["val_loss", "mse", "mare"]).reset_index(drop=True)
    return df


def tick_spec(key, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    if key == "activation_code":
        return [0, 1], ["ReLU", "GELU"]
    if key == "learning_rate_log":
        return unique, [f"{10**v:.0e}" for v in unique]
    if key == "call_time_log_ms":
        raw = np.array([5, 10, 20, 50, 100, 200], dtype=float)
        keep = raw[(raw >= 10**values.min() * 0.85) & (raw <= 10**values.max() * 1.15)]
        return np.log10(keep), [f"{v:g}" for v in keep]
    if key.endswith("_log"):
        raw = np.geomspace(10**values.min(), 10**values.max(), 4)
        return np.log10(raw), [f"{v:.1e}" for v in raw]
    if len(unique) <= 8:
        return unique, [f"{v:g}" for v in unique]
    ticks = np.linspace(values.min(), values.max(), 5)
    return ticks, [f"{v:g}" for v in ticks]


def normalized(df):
    vals = df[[x[0] for x in DISPLAY]].to_numpy(float)
    lows = np.nanmin(vals, axis=0)
    highs = np.nanmax(vals, axis=0)
    spans = np.where(highs > lows, highs - lows, 1.0)
    return (vals - lows) / spans, lows, spans


def draw_static(df, output, highlighted=False):
    norm, lows, spans = normalized(df)
    best = int(df["mare"].idxmin())
    cmap = mpl.colormaps["viridis"]
    metric_norm = mpl.colors.Normalize(df["mare_log"].min(), df["mare_log"].max())
    fig, host = plt.subplots(figsize=(18.5, 8.2), constrained_layout=False)
    x = np.arange(len(DISPLAY))
    if highlighted:
        for row in norm:
            host.plot(x, row, color="#c9c9c9", lw=0.55, alpha=0.32, zorder=1)
        host.plot(x, norm[best], color=cmap(metric_norm(df.loc[best, "mare_log"])), lw=3.2, alpha=1, zorder=4)
    else:
        order = np.argsort(df["mare"].to_numpy())[::-1]
        for idx in order:
            host.plot(x, norm[idx], color=cmap(metric_norm(df.loc[idx, "mare_log"])), lw=0.8, alpha=0.48, zorder=2)
    host.set_xlim(0, len(DISPLAY) - 1)
    host.set_ylim(-AXIS_PAD, 1 + AXIS_PAD)
    host.axis("off")
    for j, (key, label) in enumerate(DISPLAY):
        ax = host.twinx()
        ax.spines["right"].set_position(("axes", j / (len(DISPLAY) - 1)))
        ax.spines["right"].set_color("#555555")
        ax.spines["right"].set_linewidth(0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-AXIS_PAD, 1 + AXIS_PAD)
        ticks, texts = tick_spec(key, df[key].to_numpy(float))
        tick_positions = (np.asarray(ticks) - lows[j]) / spans[j]
        ax.set_yticks(tick_positions, texts, fontsize=8.5, color="#123b5d")
        ax.tick_params(axis="y", length=3.5, width=0.8, pad=3, colors="#123b5d")
        for tick_label in ax.get_yticklabels():
            tick_label.set_zorder(10)
            tick_label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.7))
        ax.set_ylabel(label, fontsize=9, color="#111111", rotation=0, labelpad=10, va="bottom")
        ax.yaxis.set_label_coords(j / (len(DISPLAY) - 1), 1.045, transform=host.transAxes)
        ax.patch.set_visible(False)
    title = "Hyperparameter exploration: best-MARE architecture highlighted" if highlighted else f"Hyperparameter exploration ({len(df)} architectures)"
    host.set_title(title, fontsize=15, color="#111111", pad=54, weight="semibold")
    sm = mpl.cm.ScalarMappable(norm=metric_norm, cmap=cmap)
    cax = fig.add_axes([0.925, 0.20, 0.012, 0.56])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("log10(Test MARE)", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.patch.set_facecolor("white")
    host.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.80, bottom=0.07)
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plotly_dimensions(df):
    dimensions = []
    for key, label in DISPLAY:
        vals = df[key].to_numpy(float)
        ticks, texts = tick_spec(key, vals)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        span = hi - lo if hi > lo else 1.0
        dimensions.append(dict(label=label.replace("\n", " "), values=vals, range=[lo - AXIS_PAD * span, hi + AXIS_PAD * span], tickvals=ticks, ticktext=texts))
    return dimensions


def draw_html(df, output, highlighted=False):
    dims = plotly_dimensions(df)
    best = int(df["mare"].idxmin())
    traces = []
    if highlighted:
        traces.append(go.Parcoords(dimensions=dims, line=dict(color="#c9c9c9"), labelfont=dict(color="#111111"), tickfont=dict(color="#222222")))
        one = df.iloc[[best]]
        traces.append(go.Parcoords(dimensions=plotly_dimensions(one), line=dict(color=[one.iloc[0]["mare_log"]], colorscale="Viridis", cmin=df["mare_log"].min(), cmax=df["mare_log"].max(), showscale=True, colorbar=dict(title="log10(Test MARE)")), labelfont=dict(color="#111111"), tickfont=dict(color="#222222")))
    else:
        traces.append(go.Parcoords(dimensions=dims, line=dict(color=df["mare_log"], colorscale="Viridis", cmin=df["mare_log"].min(), cmax=df["mare_log"].max(), showscale=True, colorbar=dict(title="log10(Test MARE)")), labelfont=dict(color="#111111"), tickfont=dict(color="#222222")))
    title = "Hyperparameter exploration: best-MARE architecture highlighted" if highlighted else f"Hyperparameter exploration ({len(df)} architectures)"
    fig = go.Figure(traces)
    fig.update_layout(title=dict(text=title, x=0.5), width=1850, height=820, margin=dict(t=120, l=60, r=140, b=40), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#111111", size=12))
    fig.write_html(output.with_suffix(".html"), include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.metadata)
    for stem, highlighted in (("parallel_hyperparams_thesis_all", False), ("parallel_hyperparams_thesis_best", True)):
        output = args.output_dir / stem
        draw_static(df, output, highlighted)
        draw_html(df, output, highlighted)
    best = df.loc[df["mare"].idxmin()]
    print(f"rows={len(df)} best={best['architecture']} MARE={best['mare']:.12g}")


if __name__ == "__main__":
    main()
