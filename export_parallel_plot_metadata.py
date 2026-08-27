import json
import math
import os
import re
import time
from glob import glob
from pathlib import Path


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
FIVE_PARAMETER_DATA = Path(
    os.environ.get("FASTWIND_5PAR_DATA", DATA_ROOT / "five_parameter")
).expanduser()
FIVE_PARAMETER_OUTPUT = Path(
    os.environ.get("FASTWIND_5PAR_OUTPUT", OUTPUT_ROOT / "five_parameter")
).expanduser()

BASE_PATH = os.environ.get(
    "FASTWIND_5PAR_SEARCH_ROOT",
    str(FIVE_PARAMETER_OUTPUT / "fine_tuning_emulator_exploration"),
)
CACHE_PATH = os.environ.get(
    "FASTWIND_5PAR_METRICS_CACHE",
    str(FIVE_PARAMETER_OUTPUT / "emulator_metrics_cache.json"),
)
OUTPUT_PATH = os.environ.get(
    "FASTWIND_5PAR_METADATA",
    str(FIVE_PARAMETER_OUTPUT / "parallel_plot_metadata.json"),
)
TIMING_CACHE_PATH = os.environ.get(
    "FASTWIND_5PAR_TIMING_CACHE",
    str(FIVE_PARAMETER_OUTPUT / "emulator_call_time_cache.json"),
)
REFERENCE_MODEL_PARAMS_PATH = os.environ.get(
    "FASTWIND_5PAR_REFERENCE_PARAMS",
    str(FIVE_PARAMETER_DATA / "all_parameters.txt"),
)
HG_RUN_NAME = "fw_emulator_per_line_comparison_hg"
HG_ARCHITECTURE_NAME = "original"
HG_DIR = os.environ.get(
    "FASTWIND_5PAR_CHECKPOINTS",
    str(FIVE_PARAMETER_OUTPUT / "emulators_per_line_hg"),
)
TIMEOUT_RUNS = {"don_finetune_run_timeout", "don_finetune_run_timeout_extra"}
TIMING_DEVICE = os.environ.get("EMULATOR_TIMING_DEVICE", "cpu")
TIMING_WARMUP = 2
TIMING_REPEATS = 5
TIMING_NUM_POINTS = 161


def normalize_line_name(value):
    s = str(value)
    s = s.replace("emulator_", "")
    s = s.replace("_loss.json", "")
    s = s.replace(".pth", "")
    return s


def parse_arch_hparams(arch):
    s = str(arch)
    out = {
        "latent_dim": None,
        "fourier_modes": None,
        "dropout": None,
        "learning_rate": None,
        "activation": None,
        "branch_depth": None,
        "trunk_depth": None,
        "branch_width_max": None,
        "trunk_width_max": None,
    }

    special_named = {
        "original": dict(
            latent_dim=128.0,
            fourier_modes=32.0,
            dropout=0.0,
            learning_rate=None,
            activation="relu",
            branch_depth=3.0,
            trunk_depth=3.0,
            branch_width_max=512.0,
            trunk_width_max=512.0,
        ),
        "emulators_per_line_hg": dict(
            latent_dim=128.0,
            fourier_modes=32.0,
            dropout=0.0,
            learning_rate=None,
            activation="relu",
            branch_depth=3.0,
            trunk_depth=3.0,
            branch_width_max=512.0,
            trunk_width_max=512.0,
        ),
    }
    if s in special_named:
        out.update(special_named[s])
        return out

    m = re.search(r"(?:^|_)lat(\d+)(?:_|$)", s)
    if m:
        out["latent_dim"] = float(m.group(1))

    m = re.search(r"_fm(\d+)", s)
    if m:
        out["fourier_modes"] = float(m.group(1))

    m = re.search(r"_drop([0-9]+p[0-9]+)", s)
    if m:
        out["dropout"] = float(m.group(1).replace("p", "."))

    m = re.search(r"_lr([0-9]+e[-+][0-9]+)", s)
    if m:
        try:
            out["learning_rate"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"_(gelu|relu)(?:_|$)", s)
    if m:
        out["activation"] = m.group(1)

    m = re.match(r"^eq(\d+)x(\d+)_w(\d+)(?:_|$)", s)
    if m:
        out["branch_depth"] = float(m.group(1))
        out["trunk_depth"] = float(m.group(2))
        out["branch_width_max"] = float(m.group(3))
        out["trunk_width_max"] = float(m.group(3))

    m = re.match(r"^inc(\d+)x(\d+)_([0-9_]+)_lat\d+(?:_|$)", s)
    if m:
        widths = [float(x) for x in m.group(3).split("_") if x]
        out["branch_depth"] = float(m.group(1))
        out["trunk_depth"] = float(m.group(2))
        if widths:
            out["branch_width_max"] = float(max(widths))
            out["trunk_width_max"] = float(max(widths))

    m = re.search(r"_d(\d+)x(\d+)_", s)
    if m:
        out["branch_depth"] = float(m.group(1))
        out["trunk_depth"] = float(m.group(2))

    m = re.search(r"_w(\d+)_", s)
    if m:
        out["branch_width_max"] = float(m.group(1))
        out["trunk_width_max"] = float(m.group(1))

    mb = re.search(r"_b([0-9\-]+)_", s)
    if mb:
        b_list = [float(x) for x in mb.group(1).split("-") if x]
        if b_list:
            out["branch_depth"] = float(len(b_list))
            out["branch_width_max"] = float(max(b_list))

    mt = re.search(r"_t([0-9\-]+)_", s)
    if mt:
        t_list = [float(x) for x in mt.group(1).split("-") if x]
        if t_list:
            out["trunk_depth"] = float(len(t_list))
            out["trunk_width_max"] = float(max(t_list))

    if (s.startswith("original") or s.startswith("latent")) and out["branch_depth"] is None:
        out["branch_depth"] = 3.0
        out["trunk_depth"] = 3.0
        out["branch_width_max"] = 512.0
        out["trunk_width_max"] = 512.0

    return out


def load_reference_model_params(path=REFERENCE_MODEL_PARAMS_PATH):
    with open(path, "r") as f:
        header = f.readline()
        first_row = f.readline()
    if not first_row:
        raise ValueError(f"No model rows found in {path}")
    values = [x.strip() for x in first_row.split(",")]
    if len(values) < 8:
        raise ValueError(f"Unexpected parameter row in {path}: {first_row}")
    return {
        "Teff": float(values[1]),
        "logg": float(values[2]),
        "R": float(values[3]),
        "Mdot": float(values[4]),
        "v_inf": float(values[5]),
        "Y_He": float(values[6]),
        "v_turb": float(values[7]),
    }


def normalize_reference_params(reference_params, param_mins, param_maxs):
    params_vec = [
        float(reference_params["Teff"]),
        float(reference_params["logg"]),
        float(reference_params["R"]),
        float(reference_params["Mdot"]),
        float(reference_params["v_inf"]),
        float(reference_params["Y_He"]),
        float(reference_params["v_turb"]),
    ]
    param_mins = [float(v) for v in param_mins]
    param_maxs = [float(v) for v in param_maxs]
    params_norm = []
    for idx, (value, vmin, vmax) in enumerate(zip(params_vec, param_mins, param_maxs)):
        if idx == 3:
            value = math.log10(value)
            vmin = math.log10(vmin)
            vmax = math.log10(vmax)
        vrange = 1.0 if vmax == vmin else (vmax - vmin)
        params_norm.append((value - vmin) / vrange)
    return params_norm


def infer_run_max_epochs(base_path, run, fallback=300):
    run_id = str(run).split("_")[-1]
    log_path = os.path.join(base_path, run, "logs", f"don_finetune-{run_id}.out")
    if not os.path.isfile(log_path):
        return int(fallback)

    vals = []
    pat = re.compile(r"Epoch\s+\d+/(\d+)")
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    vals.append(int(m.group(1)))
    except Exception:
        return int(fallback)

    if not vals:
        return int(fallback)

    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return int(max(counts.items(), key=lambda item: item[1])[0])


def extract_epochs_from_loss_json(loss_json_path):
    try:
        with open(loss_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None, None

    line_name = normalize_line_name(data.get("line_file", os.path.basename(loss_json_path)))
    for k in ["val_losses", "val_loss_history", "val"]:
        v = data.get(k)
        if isinstance(v, list) and v:
            return line_name, int(len(v)), float(v[-1])
    return line_name, None, None


def extract_epochs_from_checkpoint(checkpoint_path):
    try:
        import torch
    except Exception:
        return None

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None
    except Exception:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None

    if isinstance(state, dict):
        for k in ["val_loss_history", "val_losses", "val", "train_loss_history"]:
            v = state.get(k)
            if isinstance(v, list) and v:
                return int(len(v))
    return None


def extract_final_val_loss_from_checkpoint(checkpoint_path):
    try:
        import torch
    except Exception:
        return None

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None
    except Exception:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None

    if isinstance(state, dict):
        for k in ["val_loss_history", "val_losses", "val"]:
            v = state.get(k)
            if isinstance(v, list) and v:
                try:
                    return float(v[-1])
                except Exception:
                    return None
    return None


def load_timeout_summary_map(base_path, run_folders):
    mapping = {}
    for run in run_folders:
        if run not in TIMEOUT_RUNS:
            continue
        summary_path = os.path.join(base_path, run, "fine_tuning_emulator_exploration", "training_summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)
        for item in data:
            if not isinstance(item, dict):
                continue
            arch = item.get("architecture_tag") or item.get("architecture_name")
            if not arch:
                continue
            branch_widths = item.get("branch_widths") if isinstance(item.get("branch_widths"), list) else []
            trunk_widths = item.get("trunk_widths") if isinstance(item.get("trunk_widths"), list) else []
            mapping[(str(run), str(arch))] = {
                "latent_dim": to_float(item.get("latent_dim")),
                "fourier_modes": to_float(item.get("fourier_modes")),
                "dropout": to_float(item.get("dropout")),
                "learning_rate": to_float(item.get("learning_rate")),
                "activation": item.get("activation"),
                "branch_depth": float(len(branch_widths)) if branch_widths else None,
                "trunk_depth": float(len(trunk_widths)) if trunk_widths else None,
                "branch_width_max": float(max(branch_widths)) if branch_widths else None,
                "trunk_width_max": float(max(trunk_widths)) if trunk_widths else None,
                "batch_size": to_float(item.get("batch_size")),
            }
    return mapping


def to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def load_metrics_cache(path):
    with open(path, "r") as f:
        data = json.load(f)

    by_arch = {}
    by_line = {}
    for metric_name in ["mse", "mare"]:
        for row in data.get(metric_name, []):
            run = str(row.get("run"))
            arch = str(row.get("architecture"))
            line = normalize_line_name(row.get("line"))
            val = to_float(row.get(metric_name))

            arch_key = (run, arch)
            by_arch.setdefault(arch_key, {}).setdefault(metric_name, []).append(val)

            line_key = (run, arch, line)
            by_line.setdefault(line_key, {})[metric_name] = val

    arch_avg = {}
    for key, metric_map in by_arch.items():
        arch_avg[key] = {}
        for metric_name, values in metric_map.items():
            vals = [v for v in values if v is not None]
            arch_avg[key][metric_name] = (sum(vals) / len(vals)) if vals else None

    return arch_avg, by_line


def load_timing_cache(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_timing_cache(path, cache):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _checkpoint_config_value(state, *keys, default=None):
    config = state.get("config", {}) if isinstance(state, dict) else {}
    for key in keys:
        if isinstance(config, dict) and config.get(key) is not None:
            return config.get(key)
        if isinstance(state, dict) and state.get(key) is not None:
            return state.get(key)
    return default


def _default_widths_for_architecture(architecture_name):
    name = str(architecture_name or "").lower()
    if name == "original":
        return [128, 256, 512], [256, 512, 512]
    if name == "latent128":
        return [64, 128, 128], [128, 256, 256]
    if name == "latent64":
        return [64, 128, 128], [128, 256, 256]
    return None, None


def parse_arch_width_lists(arch, branch_depth=None, trunk_depth=None, branch_width_max=None, trunk_width_max=None):
    s = str(arch)

    m = re.match(r"^eq(\d+)x(\d+)_w(\d+)(?:_|$)", s)
    if m:
        return [int(m.group(3))] * int(m.group(1)), [int(m.group(3))] * int(m.group(2))

    m = re.match(r"^inc(\d+)x(\d+)_([0-9_]+)_lat\d+(?:_|$)", s)
    if m:
        widths = [int(x) for x in m.group(3).split("_") if x]
        return widths[: int(m.group(1))], widths[: int(m.group(2))]

    mb = re.search(r"_b([0-9\-]+)_", s)
    mt = re.search(r"_t([0-9\-]+)_", s)
    if mb and mt:
        branch_widths = [int(x) for x in mb.group(1).split("-") if x]
        trunk_widths = [int(x) for x in mt.group(1).split("-") if x]
        if branch_widths and trunk_widths:
            return branch_widths, trunk_widths

    m = re.search(r"_d(\d+)x(\d+)_", s)
    mw = re.search(r"_w(\d+)(?:_|$)", s)
    if m and mw:
        return [int(mw.group(1))] * int(m.group(1)), [int(mw.group(1))] * int(m.group(2))

    if branch_depth and branch_width_max and trunk_depth and trunk_width_max:
        return [int(branch_width_max)] * int(branch_depth), [int(trunk_width_max)] * int(trunk_depth)

    return None, None


def build_timing_signature(
    architecture_name,
    latent_dim,
    fourier_modes,
    activation_name,
    dropout,
    branch_widths,
    trunk_widths,
    checkpoint_path=None,
    line_name=None,
    device=TIMING_DEVICE,
    num_points=TIMING_NUM_POINTS,
):
    payload = {
        "cache_schema": "per_checkpoint_v1",
        "architecture_name": str(architecture_name),
        "latent_dim": int(latent_dim or 128),
        "fourier_modes": int(fourier_modes or 32),
        "activation": str(activation_name or "relu").lower(),
        "dropout": float(dropout or 0.0),
        "branch_widths": [int(v) for v in branch_widths],
        "trunk_widths": [int(v) for v in trunk_widths],
        "num_points": int(num_points),
        "device": str(device),
    }
    if line_name is not None:
        payload["line_name"] = str(line_name)
    if checkpoint_path is not None:
        abs_checkpoint_path = os.path.abspath(checkpoint_path)
        payload["checkpoint_path"] = abs_checkpoint_path
        try:
            stat = os.stat(abs_checkpoint_path)
            payload["checkpoint_size"] = int(stat.st_size)
            payload["checkpoint_mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            payload["checkpoint_size"] = None
            payload["checkpoint_mtime_ns"] = None
    return json.dumps(payload, sort_keys=True)


def benchmark_checkpoint_call_time(
    checkpoint_path,
    timing_cache,
    reference_params,
    precomputed_signature=None,
    fallback_architecture_name=None,
    fallback_latent_dim=None,
    fallback_fourier_modes=None,
    fallback_activation_name=None,
    fallback_dropout=None,
    fallback_branch_widths=None,
    fallback_trunk_widths=None,
    timing_stats=None,
    device=TIMING_DEVICE,
    warmup=TIMING_WARMUP,
    repeats=TIMING_REPEATS,
    num_points=TIMING_NUM_POINTS,
):
    def fail(reason):
        if timing_stats is not None:
            failures = timing_stats.setdefault("failures", {})
            failures[reason] = failures.get(reason, 0) + 1
        return None

    if precomputed_signature is not None and precomputed_signature in timing_cache:
        if timing_stats is not None:
            timing_stats["cache_hits"] = timing_stats.get("cache_hits", 0) + 1
        return timing_cache[precomputed_signature]

    try:
        import torch
        import torch.nn as nn
    except Exception:
        return fail("import_torch")

    class BranchNet(nn.Module):
        def __init__(self, input_dim, latent_dim, hidden_widths, activation_fn, dropout):
            super().__init__()
            layers = []
            in_dim = input_dim
            for out_dim in hidden_widths:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, latent_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class TrunkNet(nn.Module):
        def __init__(self, latent_dim, hidden_widths, activation_fn, dropout, fourier_modes):
            super().__init__()
            self.fourier_modes = fourier_modes
            in_dim = 2 * fourier_modes + 1
            layers = []
            for out_dim in hidden_widths:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, latent_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, coords):
            if coords.dim() == 2:
                coords = coords.unsqueeze(-1)
            batch_size, num_eval_points, _ = coords.shape
            freqs = 2.0 * math.pi * torch.arange(
                1, self.fourier_modes + 1, device=coords.device, dtype=coords.dtype
            ).view(1, 1, self.fourier_modes)
            sin_feats = torch.sin(freqs * coords)
            cos_feats = torch.cos(freqs * coords)
            features = torch.cat([coords, sin_feats, cos_feats], dim=-1)
            features = features.view(-1, features.shape[-1])
            out = self.net(features)
            return out.view(batch_size, num_eval_points, -1)

    class DeepONetModel(nn.Module):
        def __init__(self, branch_net, trunk_net):
            super().__init__()
            self.branch = branch_net
            self.trunk = trunk_net

        def forward(self, params, coords):
            branch_out = self.branch(params)
            trunk_out = self.trunk(coords)
            return (trunk_out * branch_out.unsqueeze(1)).sum(dim=-1)

    try:
        runtime_device = torch.device(device)
        state = torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)
    except Exception:
        return fail("load_checkpoint")

    latent_dim = int(_checkpoint_config_value(state, "latent_dim", default=fallback_latent_dim or 128) or 128)
    fourier_modes = int(_checkpoint_config_value(state, "fourier_modes", default=fallback_fourier_modes or 32) or 32)
    activation_name = str(
        _checkpoint_config_value(state, "activation_name", "activation", default=fallback_activation_name or "relu") or "relu"
    ).lower()
    dropout = float(_checkpoint_config_value(state, "dropout", default=fallback_dropout or 0.0) or 0.0)
    architecture_name = str(
        _checkpoint_config_value(state, "architecture_name", default=fallback_architecture_name or "original") or "original"
    )
    branch_widths = _checkpoint_config_value(state, "branch_widths", default=None)
    trunk_widths = _checkpoint_config_value(state, "trunk_widths", default=None)

    if not branch_widths and fallback_branch_widths:
        branch_widths = fallback_branch_widths
    if not trunk_widths and fallback_trunk_widths:
        trunk_widths = fallback_trunk_widths
    if not branch_widths or not trunk_widths:
        branch_widths, trunk_widths = _default_widths_for_architecture(architecture_name)
    if not branch_widths or not trunk_widths:
        return fail("missing_widths")

    signature = precomputed_signature or build_timing_signature(
        architecture_name=architecture_name,
        latent_dim=latent_dim,
        fourier_modes=fourier_modes,
        activation_name=activation_name,
        dropout=dropout,
        branch_widths=branch_widths,
        trunk_widths=trunk_widths,
        checkpoint_path=checkpoint_path,
        device=runtime_device,
        num_points=num_points,
    )
    if signature in timing_cache:
        if timing_stats is not None:
            timing_stats["cache_hits"] = timing_stats.get("cache_hits", 0) + 1
        return timing_cache[signature]

    activation_map = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    activation_fn = activation_map.get(activation_name)
    if activation_fn is None:
        return fail("unknown_activation")

    state_dict = state.get("model_state") or state.get("model_state_dict")
    if state_dict is None:
        return fail("missing_state_dict")

    try:
        model = DeepONetModel(
            BranchNet(7, latent_dim, list(branch_widths), activation_fn, dropout),
            TrunkNet(latent_dim, list(trunk_widths), activation_fn, dropout, fourier_modes),
        ).to(runtime_device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception:
        return fail("build_or_load_model")

    try:
        params_norm = normalize_reference_params(reference_params, state["param_mins"], state["param_maxs"])
    except Exception:
        return fail("normalize_reference_params")

    params_tensor = torch.tensor([params_norm], dtype=torch.float32, device=runtime_device)
    waves_tensor = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32, device=runtime_device).view(1, -1)

    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            _ = model(params_tensor, waves_tensor)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)

        timings = []
        for _ in range(max(1, int(repeats))):
            start = time.perf_counter()
            _ = model(params_tensor, waves_tensor)
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)
            timings.append(time.perf_counter() - start)

    call_time = float(sorted(timings)[len(timings) // 2]) if timings else None
    timing_cache[signature] = call_time
    if timing_stats is not None:
        timing_stats["new_measurements"] = timing_stats.get("new_measurements", 0) + 1
    return call_time


def collect_run_folders(base_path):
    runs = []
    if not os.path.isdir(base_path):
        return runs
    for run in sorted(os.listdir(base_path)):
        run_path = os.path.join(base_path, run, "fine_tuning_emulator_exploration")
        if os.path.isdir(run_path):
            runs.append(run)
    return runs


def main():
    run_folders = collect_run_folders(BASE_PATH)
    timeout_summary_map = load_timeout_summary_map(BASE_PATH, run_folders)
    arch_metric_map, line_metric_map = load_metrics_cache(CACHE_PATH)
    timing_cache = load_timing_cache(TIMING_CACHE_PATH)
    reference_params = load_reference_model_params()
    timing_stats = {"cache_hits": 0, "new_measurements": 0}

    architecture_rows = []
    line_rows = []
    arch_row_lookup = {}

    for run in run_folders:
        run_path = os.path.join(BASE_PATH, run, "fine_tuning_emulator_exploration")
        run_max_epochs = infer_run_max_epochs(BASE_PATH, run, fallback=300)

        for arch in sorted(os.listdir(run_path)):
            arch_path = os.path.join(run_path, arch)
            if not os.path.isdir(arch_path):
                continue

            hparams = parse_arch_hparams(arch)
            hparams.update({k: v for k, v in timeout_summary_map.get((run, arch), {}).items() if v is not None})
            branch_widths, trunk_widths = parse_arch_width_lists(
                arch,
                branch_depth=hparams.get("branch_depth"),
                trunk_depth=hparams.get("trunk_depth"),
                branch_width_max=hparams.get("branch_width_max"),
                trunk_width_max=hparams.get("trunk_width_max"),
            )
            if (not branch_widths or not trunk_widths) and str(arch).lower() == "original":
                branch_widths, trunk_widths = _default_widths_for_architecture("original")

            if arch != "original" and hparams.get("learning_rate") is None:
                hparams["learning_rate"] = 1e-3

            batch_size = timeout_summary_map.get((run, arch), {}).get("batch_size")
            if batch_size is None:
                batch_size = 2048.0

            line_info = {}
            json_lines_present = set()
            for loss_json in glob(os.path.join(arch_path, "*_loss.json")):
                line_name, epochs, val_loss = extract_epochs_from_loss_json(loss_json)
                if line_name is None:
                    continue
                json_lines_present.add(str(line_name))
                line_info[line_name] = {
                    "epochs_run": epochs,
                    "stopped_early": float(epochs < run_max_epochs) if epochs is not None else None,
                    "val_loss": val_loss,
                    "loss_json_path": os.path.abspath(loss_json),
                }

            for ckpt in glob(os.path.join(arch_path, "emulator_*.pth")):
                line_name = normalize_line_name(os.path.basename(ckpt))
                line_info.setdefault(line_name, {})
                if line_name not in json_lines_present:
                    epochs = extract_epochs_from_checkpoint(ckpt)
                    line_info[line_name].update(
                        {
                            "epochs_run": epochs,
                            "stopped_early": float(epochs < run_max_epochs) if epochs is not None else None,
                        }
                    )
                line_info[line_name]["checkpoint_path"] = os.path.abspath(ckpt)
                timing_signature = None
                if branch_widths and trunk_widths:
                    timing_signature = build_timing_signature(
                        architecture_name=arch,
                        latent_dim=hparams.get("latent_dim"),
                        fourier_modes=hparams.get("fourier_modes"),
                        activation_name=hparams.get("activation"),
                        dropout=hparams.get("dropout"),
                        branch_widths=branch_widths,
                        trunk_widths=trunk_widths,
                        checkpoint_path=ckpt,
                        line_name=line_name,
                    )
                emulator_call_time = benchmark_checkpoint_call_time(
                    ckpt,
                    timing_cache=timing_cache,
                    reference_params=reference_params,
                    precomputed_signature=timing_signature,
                    fallback_architecture_name=arch,
                    fallback_latent_dim=hparams.get("latent_dim"),
                    fallback_fourier_modes=hparams.get("fourier_modes"),
                    fallback_activation_name=hparams.get("activation"),
                    fallback_dropout=hparams.get("dropout"),
                    fallback_branch_widths=branch_widths,
                    fallback_trunk_widths=trunk_widths,
                    timing_stats=timing_stats,
                )
                line_info[line_name]["emulator_call_time_sec"] = emulator_call_time

            val_losses = [v["val_loss"] for v in line_info.values() if v.get("val_loss") is not None]
            early_stop_count = sum(1 for v in line_info.values() if v.get("stopped_early") == 1.0)
            full_epoch_count = sum(1 for v in line_info.values() if v.get("stopped_early") == 0.0)
            lines_count = len(line_info)
            line_call_times = [
                v["emulator_call_time_sec"] for v in line_info.values() if v.get("emulator_call_time_sec") is not None
            ]
            total_emulator_call_time = sum(line_call_times) if line_call_times else None
            arch_metrics = arch_metric_map.get((run, arch), {})

            architecture_rows.append(
                {
                    "run": run,
                    "architecture": arch,
                    "source_folder": os.path.abspath(arch_path),
                    "latent_dim": hparams.get("latent_dim"),
                    "fourier_modes": hparams.get("fourier_modes"),
                    "dropout": hparams.get("dropout"),
                    "learning_rate": hparams.get("learning_rate"),
                    "activation": hparams.get("activation"),
                    "branch_depth": hparams.get("branch_depth"),
                    "trunk_depth": hparams.get("trunk_depth"),
                    "branch_width_max": hparams.get("branch_width_max"),
                    "trunk_width_max": hparams.get("trunk_width_max"),
                    "batch_size": batch_size,
                    "max_epochs": float(run_max_epochs),
                    "early_stop_count": float(early_stop_count) if lines_count else None,
                    "full_epoch_count": float(full_epoch_count) if lines_count else None,
                    "lines_count": float(lines_count) if lines_count else None,
                    "early_stop_fraction": (float(early_stop_count) / float(lines_count)) if lines_count else None,
                    "total_emulator_call_time_sec": total_emulator_call_time,
                    "val_loss": (sum(val_losses) / len(val_losses)) if val_losses else None,
                    "mse": arch_metrics.get("mse"),
                    "mare": arch_metrics.get("mare"),
                }
            )
            arch_row_lookup[(run, arch)] = architecture_rows[-1]

            for line_name, info in sorted(line_info.items()):
                metrics = line_metric_map.get((run, arch, line_name), {})
                line_rows.append(
                    {
                        "run": run,
                        "architecture": arch,
                        "line": line_name,
                        "source_folder": os.path.abspath(arch_path),
                        "loss_json_path": info.get("loss_json_path"),
                        "checkpoint_path": info.get("checkpoint_path"),
                        "latent_dim": hparams.get("latent_dim"),
                        "fourier_modes": hparams.get("fourier_modes"),
                        "dropout": hparams.get("dropout"),
                        "learning_rate": hparams.get("learning_rate"),
                        "activation": hparams.get("activation"),
                        "branch_depth": hparams.get("branch_depth"),
                        "trunk_depth": hparams.get("trunk_depth"),
                        "branch_width_max": hparams.get("branch_width_max"),
                        "trunk_width_max": hparams.get("trunk_width_max"),
                        "batch_size": batch_size,
                        "max_epochs": float(run_max_epochs),
                        "epochs_run": info.get("epochs_run"),
                        "stopped_early": info.get("stopped_early"),
                        "emulator_call_time_sec": info.get("emulator_call_time_sec"),
                        "val_loss": info.get("val_loss"),
                        "mse": metrics.get("mse"),
                        "mare": metrics.get("mare"),
                    }
                )

    hg_line_rows = []
    hg_epoch_counts = []
    if os.path.isdir(HG_DIR):
        hparams = parse_arch_hparams(HG_ARCHITECTURE_NAME)
        hg_branch_widths, hg_trunk_widths = _default_widths_for_architecture(HG_ARCHITECTURE_NAME)
        hg_val_losses = []
        for ckpt in glob(os.path.join(HG_DIR, "emulator_*.pth")):
            line_name = normalize_line_name(os.path.basename(ckpt))
            epochs = extract_epochs_from_checkpoint(ckpt)
            val_loss = extract_final_val_loss_from_checkpoint(ckpt)
            stopped_early = float(epochs < 200) if epochs is not None else None
            timing_signature = None
            if hg_branch_widths and hg_trunk_widths:
                timing_signature = build_timing_signature(
                    architecture_name=HG_ARCHITECTURE_NAME,
                    latent_dim=hparams.get("latent_dim"),
                    fourier_modes=hparams.get("fourier_modes"),
                    activation_name=hparams.get("activation"),
                    dropout=hparams.get("dropout"),
                    branch_widths=hg_branch_widths,
                    trunk_widths=hg_trunk_widths,
                    checkpoint_path=ckpt,
                    line_name=line_name,
                )
            emulator_call_time = benchmark_checkpoint_call_time(
                ckpt,
                timing_cache=timing_cache,
                reference_params=reference_params,
                precomputed_signature=timing_signature,
                fallback_architecture_name=HG_ARCHITECTURE_NAME,
                fallback_latent_dim=hparams.get("latent_dim"),
                fallback_fourier_modes=hparams.get("fourier_modes"),
                fallback_activation_name=hparams.get("activation"),
                fallback_dropout=hparams.get("dropout"),
                fallback_branch_widths=hg_branch_widths,
                fallback_trunk_widths=hg_trunk_widths,
                timing_stats=timing_stats,
            )
            if epochs is not None:
                hg_epoch_counts.append(epochs)
            if val_loss is not None:
                hg_val_losses.append(val_loss)
            metrics = line_metric_map.get((HG_RUN_NAME, HG_ARCHITECTURE_NAME, line_name), {})
            hg_line_rows.append(
                {
                    "run": HG_RUN_NAME,
                    "architecture": HG_ARCHITECTURE_NAME,
                    "line": line_name,
                    "source_folder": os.path.abspath(HG_DIR),
                    "loss_json_path": None,
                    "checkpoint_path": os.path.abspath(ckpt),
                    "latent_dim": hparams.get("latent_dim"),
                    "fourier_modes": hparams.get("fourier_modes"),
                    "dropout": hparams.get("dropout"),
                    "learning_rate": hparams.get("learning_rate"),
                    "activation": hparams.get("activation"),
                    "branch_depth": hparams.get("branch_depth"),
                    "trunk_depth": hparams.get("trunk_depth"),
                    "branch_width_max": hparams.get("branch_width_max"),
                    "trunk_width_max": hparams.get("trunk_width_max"),
                    "batch_size": 1024.0,
                    "max_epochs": 200.0,
                    "epochs_run": epochs,
                    "stopped_early": stopped_early,
                    "emulator_call_time_sec": emulator_call_time,
                    "val_loss": val_loss,
                    "mse": metrics.get("mse"),
                    "mare": metrics.get("mare"),
                }
            )

        arch_metrics = arch_metric_map.get((HG_RUN_NAME, HG_ARCHITECTURE_NAME), {})
        hg_call_times = [row["emulator_call_time_sec"] for row in hg_line_rows if row.get("emulator_call_time_sec") is not None]
        architecture_rows.append(
            {
                "run": HG_RUN_NAME,
                "architecture": HG_ARCHITECTURE_NAME,
                "source_folder": os.path.abspath(HG_DIR),
                "latent_dim": hparams.get("latent_dim"),
                "fourier_modes": hparams.get("fourier_modes"),
                "dropout": hparams.get("dropout"),
                "learning_rate": hparams.get("learning_rate"),
                "activation": hparams.get("activation"),
                "branch_depth": hparams.get("branch_depth"),
                "trunk_depth": hparams.get("trunk_depth"),
                "branch_width_max": hparams.get("branch_width_max"),
                "trunk_width_max": hparams.get("trunk_width_max"),
                "batch_size": 1024.0,
                "max_epochs": 200.0,
                "early_stop_count": float(sum(1 for ep in hg_epoch_counts if ep < 200)) if hg_epoch_counts else None,
                "full_epoch_count": float(sum(1 for ep in hg_epoch_counts if ep >= 200)) if hg_epoch_counts else None,
                "lines_count": float(len(hg_epoch_counts)) if hg_epoch_counts else None,
                "early_stop_fraction": (float(sum(1 for ep in hg_epoch_counts if ep < 200)) / float(len(hg_epoch_counts))) if hg_epoch_counts else None,
                "total_emulator_call_time_sec": sum(hg_call_times) if hg_call_times else None,
                "val_loss": (sum(hg_val_losses) / len(hg_val_losses)) if hg_val_losses else None,
                "mse": arch_metrics.get("mse"),
                "mare": arch_metrics.get("mare"),
            }
        )
        arch_row_lookup[(HG_RUN_NAME, HG_ARCHITECTURE_NAME)] = architecture_rows[-1]
        line_rows.extend(hg_line_rows)

    source_key = ("don_finetune_run_192979", "original_lat128_relu_drop0p0_fm32")
    target_key = (HG_RUN_NAME, HG_ARCHITECTURE_NAME)
    if source_key in arch_row_lookup and target_key in arch_row_lookup:
        source_row = arch_row_lookup[source_key]
        target_row = arch_row_lookup[target_key]
        for field in [
            "learning_rate",
            "early_stop_count",
            "full_epoch_count",
            "lines_count",
            "early_stop_fraction",
        ]:
            if target_row.get(field) is None:
                target_row[field] = source_row.get(field)
        for row in line_rows:
            if row.get("run") == HG_RUN_NAME and row.get("architecture") == HG_ARCHITECTURE_NAME:
                if row.get("learning_rate") is None:
                    row["learning_rate"] = source_row.get("learning_rate")

    output = {
        "base_path": os.path.abspath(BASE_PATH),
        "metrics_cache": os.path.abspath(CACHE_PATH),
        "architecture_level": architecture_rows,
        "line_level": line_rows,
    }

    expected_line_rows = len(architecture_rows) * 17
    incomplete_arch_line_counts = [row for row in architecture_rows if row.get("lines_count") != 17.0]
    missing_line_timing = [row for row in line_rows if row.get("emulator_call_time_sec") is None]
    missing_arch_timing = [row for row in architecture_rows if row.get("total_emulator_call_time_sec") is None]
    if len(line_rows) != expected_line_rows or incomplete_arch_line_counts or missing_line_timing or missing_arch_timing:
        save_timing_cache(TIMING_CACHE_PATH, timing_cache)
        print(f"wrote partial timing cache progress to {TIMING_CACHE_PATH}")
        print(f"architecture_rows={len(architecture_rows)}")
        print(f"line_rows={len(line_rows)} expected_line_rows={expected_line_rows}")
        print(f"incomplete_arch_line_count_rows={len(incomplete_arch_line_counts)}/{len(architecture_rows)}")
        print(f"missing_arch_call_time_rows={len(missing_arch_timing)}/{len(architecture_rows)}")
        print(f"missing_line_call_time_rows={len(missing_line_timing)}/{len(line_rows)}")
        if timing_stats.get("failures"):
            for reason, count in sorted(timing_stats["failures"].items()):
                print(f"timing_failure_{reason}={count}")
        for row in incomplete_arch_line_counts[:10]:
            print(
                "[incomplete arch lines] "
                f"run={row.get('run')} architecture={row.get('architecture')} "
                f"lines_count={row.get('lines_count')}"
            )
        for row in missing_arch_timing[:10]:
            print(f"[missing arch call time] run={row.get('run')} architecture={row.get('architecture')}")
        for row in missing_line_timing[:10]:
            print(
                "[missing line call time] "
                f"run={row.get('run')} architecture={row.get('architecture')} "
                f"line={row.get('line')} checkpoint={row.get('checkpoint_path')}"
            )
        raise RuntimeError(
            "Call-time metadata is incomplete. The parallel plot metadata was not written; "
            "fix the listed checkpoints or timing errors and rerun with FORCE_RECOMPUTE_CALL_TIME_METADATA=True."
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    save_timing_cache(TIMING_CACHE_PATH, timing_cache)

    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {TIMING_CACHE_PATH}")
    print(f"architecture_level_rows={len(architecture_rows)}")
    print(f"line_level_rows={len(line_rows)}")
    print(f"timing_cache_hits={timing_stats['cache_hits']}")
    print(f"timing_new_measurements={timing_stats['new_measurements']}")
    if timing_stats.get("failures"):
        for reason, count in sorted(timing_stats["failures"].items()):
            print(f"timing_failure_{reason}={count}")


if __name__ == "__main__":
    main()
