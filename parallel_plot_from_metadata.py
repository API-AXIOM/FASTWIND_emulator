import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError as e:
    raise ImportError(
        "Plotly is required for the parallel-coordinates plot. Install with: %pip install plotly"
    ) from e


def _load_metadata(metadata_path="parallel_plot_metadata.json"):
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path.resolve()}")
    with open(path, "r") as f:
        return json.load(f), path


def _prepare_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _set_axis_from_values(dim, values, formatter=str):
    values = [v for v in values if pd.notna(v)]
    if not values:
        return
    unique_vals = sorted(set(values))
    dim["tickvals"] = unique_vals
    dim["ticktext"] = [formatter(v) for v in unique_vals]
    dim["range"] = [min(unique_vals), max(unique_vals)]


def _apply_common_dimension_formatting(fig, df, act_map, early_stop_label):
    for dim in fig.data[0]["dimensions"]:
        label = dim["label"]
        if label == "activation":
            dim["tickvals"] = list(act_map.values())
            dim["ticktext"] = list(act_map.keys())

        elif label == "fourier_modes":
            _set_axis_from_values(dim, [int(v) for v in df["fourier_modes"].dropna().tolist()], str)

        elif label == "latent_dim":
            _set_axis_from_values(dim, [int(v) for v in df["latent_dim"].dropna().tolist()], str)

        elif label == "dropout":
            _set_axis_from_values(dim, [float(v) for v in df["dropout"].dropna().tolist()], str)

        elif label == "learning_rate":
            _set_axis_from_values(dim, [float(v) for v in df["learning_rate"].dropna().tolist()], lambda v: f"{v:.0e}")

        elif label in ["branch_depth", "trunk_depth"]:
            _set_axis_from_values(dim, [int(v) for v in df[label].dropna().tolist()], str)

        elif label in ["branch_width_max", "trunk_width_max"]:
            _set_axis_from_values(dim, [int(v) for v in df[label].dropna().tolist()], str)

        elif label == "batch_size":
            _set_axis_from_values(
                dim,
                [float(v) for v in df["batch_size"].dropna().tolist()],
                lambda v: str(int(v)) if float(v).is_integer() else str(v),
            )

        elif label == "max_epochs":
            _set_axis_from_values(dim, [int(v) for v in df["max_epochs"].dropna().tolist()], str)

        elif label == "call time":
            if "log10_total_emulator_call_time_sec" in df.columns and df["log10_total_emulator_call_time_sec"].notna().any():
                time_values = [float(v) for v in df["log10_total_emulator_call_time_sec"].dropna().tolist()]
            elif "log10_emulator_call_time_sec" in df.columns and df["log10_emulator_call_time_sec"].notna().any():
                time_values = [float(v) for v in df["log10_emulator_call_time_sec"].dropna().tolist()]
            else:
                time_values = []
            _set_axis_from_values(
                dim,
                time_values,
                lambda v: f"{10 ** v:.3e}" if pd.notna(v) else "",
            )

        elif label == early_stop_label:
            if early_stop_label == "stopped_early":
                dim["tickvals"] = [0, 1]
                dim["ticktext"] = ["full_epochs", "early_stop"]
                dim["range"] = [0, 1]
            else:
                _set_axis_from_values(dim, [int(v) for v in df[early_stop_label].dropna().tolist()], str)


def plot_architecture_level(metadata_path="parallel_plot_metadata.json"):
    metadata, _ = _load_metadata(metadata_path)
    plot_df = pd.DataFrame(metadata.get("architecture_level", []))
    if plot_df.empty:
        raise ValueError("No architecture-level rows found in parallel_plot_metadata.json")

    plot_df = plot_df[plot_df["activation"].isin(["relu", "gelu"])].copy()
    if plot_df.empty:
        raise ValueError("No rows left after filtering activation to relu/gelu.")

    act_order = [a for a in ["relu", "gelu"] if a in set(plot_df["activation"])]
    act_map = {v: i for i, v in enumerate(act_order)}
    plot_df["activation_code"] = plot_df["activation"].map(act_map).astype(float)

    plot_df = _prepare_numeric(
        plot_df,
        [
            "latent_dim",
            "fourier_modes",
            "dropout",
            "learning_rate",
            "branch_depth",
            "trunk_depth",
            "branch_width_max",
            "trunk_width_max",
            "batch_size",
            "max_epochs",
            "early_stop_count",
            "total_emulator_call_time_sec",
            "val_loss",
            "mse",
            "mare",
        ],
    )

    for col in ["mse", "mare", "val_loss", "total_emulator_call_time_sec"]:
        if col in plot_df.columns:
            plot_df[f"log10_{col}"] = np.where(plot_df[col] > 0, np.log10(plot_df[col]), np.nan)

    if "log10_mare" in plot_df.columns and plot_df["log10_mare"].notna().any():
        color_col = "log10_mare"
    elif "log10_mse" in plot_df.columns and plot_df["log10_mse"].notna().any():
        color_col = "log10_mse"
    else:
        color_col = "log10_val_loss" if "log10_val_loss" in plot_df.columns else "val_loss"

    metric_cols = [c for c in ["val_loss", "mse", "mare"] if c in plot_df.columns]
    plot_df = plot_df.dropna(subset=metric_cols, how="all").copy()
    if plot_df.empty:
        raise ValueError("No architecture rows with usable metrics were found.")

    if "fourier_modes" in plot_df.columns:
        plot_df = plot_df.sort_values("fourier_modes", kind="stable")

    dimensions = [
        "activation_code",
        "latent_dim",
        "fourier_modes",
        "dropout",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "early_stop_count",
        "log10_total_emulator_call_time_sec",
        "branch_depth",
        "trunk_depth",
        "branch_width_max",
        "trunk_width_max",
    ]
    if "val_loss" in plot_df.columns:
        dimensions.append("val_loss")
    if "log10_mse" in plot_df.columns:
        dimensions.append("log10_mse")
    elif "mse" in plot_df.columns:
        dimensions.append("mse")
    if "log10_mare" in plot_df.columns:
        dimensions.append("log10_mare")
    elif "mare" in plot_df.columns:
        dimensions.append("mare")

    dim_final = [d for d in dimensions if d in plot_df.columns and not plot_df[d].isna().all()]

    fig = px.parallel_coordinates(
        plot_df,
        dimensions=dim_final,
        color=color_col,
        color_continuous_scale=[
            [0.0, "#9d4edd"],
            [0.5, "#ff4d6d"],
            [1.0, "#ff9f1c"],
        ],
        labels={
            "activation_code": "activation",
            "latent_dim": "latent_dim",
            "fourier_modes": "fourier_modes",
            "dropout": "dropout",
            "learning_rate": "learning_rate",
            "branch_depth": "branch_depth",
            "trunk_depth": "trunk_depth",
            "branch_width_max": "branch_width_max",
            "trunk_width_max": "trunk_width_max",
            "batch_size": "batch_size",
            "max_epochs": "max_epochs",
            "early_stop_count": "early_stop_count",
            "log10_total_emulator_call_time_sec": "call time",
            "val_loss": "val_loss",
            "mse": "test_mse",
            "mare": "test_mare",
            "log10_mse": "log10(test_mse)",
            "log10_mare": "log10(test_mare)",
            "log10_val_loss": "log10(val_loss)",
        },
    )

    _apply_common_dimension_formatting(fig, plot_df, act_map, "early_stop_count")

    fig.data[0]["line"]["reversescale"] = True
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Parallel Coordinates: Hyperparameters vs Performance ({len(plot_df)} architectures)",
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            pad=dict(b=34),
        ),
        width=1850,
        height=760,
        margin=dict(t=140, l=90, r=140, b=50),
        coloraxis_colorbar_title=color_col,
        paper_bgcolor="#101217",
        plot_bgcolor="#101217",
        font=dict(color="#e5e7eb", size=12),
    )

    html_path = Path("parallel_hyperparams_interactive.html").resolve()
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True, auto_open=False)
    return html_path, len(plot_df)


def plot_line_level(metadata_path="parallel_plot_metadata.json", verbose=False):
    metadata, _ = _load_metadata(metadata_path)
    line_df = pd.DataFrame(metadata.get("line_level", []))
    if line_df.empty:
        raise ValueError("No line-level rows found in parallel_plot_metadata.json")

    line_df = line_df[line_df["activation"].isin(["relu", "gelu"])].copy()
    if line_df.empty:
        raise ValueError("No rows left after activation filter (relu/gelu).")

    act_order = [a for a in ["relu", "gelu"] if a in set(line_df["activation"])]
    act_map = {v: i for i, v in enumerate(act_order)}
    line_df["activation_code"] = line_df["activation"].map(act_map).astype(float)

    line_df = _prepare_numeric(
        line_df,
        [
            "latent_dim",
            "fourier_modes",
            "dropout",
            "learning_rate",
            "branch_depth",
            "trunk_depth",
            "branch_width_max",
            "trunk_width_max",
            "batch_size",
            "max_epochs",
            "epochs_run",
            "stopped_early",
            "emulator_call_time_sec",
            "val_loss",
            "mse",
            "mare",
        ],
    )

    for col in ["mse", "mare", "val_loss", "emulator_call_time_sec"]:
        if col in line_df.columns:
            line_df[f"log10_{col}"] = np.where(line_df[col] > 0, np.log10(line_df[col]), np.nan)

    lines = sorted(line_df["line"].dropna().astype(str).unique().tolist())
    saved_html_paths = []

    for line_name in lines:
        df_line = line_df[line_df["line"].astype(str) == line_name].copy()
        if df_line.empty:
            continue

        metric_cols = [c for c in ["val_loss", "mse", "mare"] if c in df_line.columns]
        df_line = df_line.dropna(subset=metric_cols, how="all")
        if df_line.empty:
            if verbose:
                print(f"[parallel-per-line] skip {line_name}: no usable metrics")
            continue

        if "fourier_modes" in df_line.columns:
            df_line = df_line.sort_values("fourier_modes", kind="stable")

        if "log10_mare" in df_line.columns and df_line["log10_mare"].notna().any():
            color_col = "log10_mare"
        elif "log10_mse" in df_line.columns and df_line["log10_mse"].notna().any():
            color_col = "log10_mse"
        else:
            color_col = "log10_val_loss" if "log10_val_loss" in df_line.columns else "val_loss"

        dimensions = [
            "activation_code",
            "latent_dim",
            "fourier_modes",
            "dropout",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "stopped_early",
            "log10_emulator_call_time_sec",
            "branch_depth",
            "trunk_depth",
            "branch_width_max",
            "trunk_width_max",
        ]
        if "val_loss" in df_line.columns:
            dimensions.append("val_loss")
        if "log10_mse" in df_line.columns:
            dimensions.append("log10_mse")
        elif "mse" in df_line.columns:
            dimensions.append("mse")
        if "log10_mare" in df_line.columns:
            dimensions.append("log10_mare")
        elif "mare" in df_line.columns:
            dimensions.append("mare")

        dim_final = [d for d in dimensions if d in df_line.columns and not df_line[d].isna().all()]

        fig = px.parallel_coordinates(
            df_line,
            dimensions=dim_final,
            color=color_col,
            color_continuous_scale=[
                [0.0, "#9d4edd"],
                [0.5, "#ff4d6d"],
                [1.0, "#ff9f1c"],
            ],
            labels={
                "activation_code": "activation",
                "latent_dim": "latent_dim",
                "fourier_modes": "fourier_modes",
                "dropout": "dropout",
                "learning_rate": "learning_rate",
                "branch_depth": "branch_depth",
                "trunk_depth": "trunk_depth",
                "branch_width_max": "branch_width_max",
                "trunk_width_max": "trunk_width_max",
                "batch_size": "batch_size",
                "max_epochs": "max_epochs",
                "stopped_early": "stopped_early",
                "log10_emulator_call_time_sec": "call time",
                "val_loss": "val_loss",
                "mse": "test_mse",
                "mare": "test_mare",
                "log10_mse": "log10(test_mse)",
                "log10_mare": "log10(test_mare)",
                "log10_val_loss": "log10(val_loss)",
            },
        )

        _apply_common_dimension_formatting(fig, df_line, act_map, "stopped_early")

        fig.data[0]["line"]["reversescale"] = True
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"Parallel Coordinates Per Line: {line_name} ({len(df_line)} architectures)",
                x=0.5,
                xanchor="center",
                y=0.99,
                yanchor="top",
                pad=dict(b=34),
            ),
            width=1850,
            height=760,
            margin=dict(t=140, l=90, r=140, b=50),
            coloraxis_colorbar_title=color_col,
            paper_bgcolor="#101217",
            plot_bgcolor="#101217",
            font=dict(color="#e5e7eb", size=12),
        )

        safe_line = re.sub(r"[^A-Za-z0-9_.-]+", "_", line_name)
        html_path = Path(f"parallel_hyperparams_per_line_{safe_line}.html").resolve()
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True, auto_open=False)
        saved_html_paths.append(str(html_path))

        if verbose:
            print(f"[parallel-per-line] saved {line_name}: {len(df_line)} rows | html={html_path}")

    saved_html_manifest = Path("parallel_hyperparams_per_line_manifest.txt").resolve()
    with open(saved_html_manifest, "w") as f:
        for h in saved_html_paths:
            f.write(h + "\n")

    return lines, saved_html_paths, saved_html_manifest
