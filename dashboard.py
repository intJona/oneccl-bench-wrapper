#!/usr/bin/env python3
"""xpu-bench dashboard — compare all-reduce runs side by side.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent


def resolve_results_root() -> Path:
    """Results dir: $XPU_BENCH_RESULTS_DIR, then .storage-config, then ./results."""
    env_override = os.environ.get("XPU_BENCH_RESULTS_DIR")
    if env_override:
        return Path(env_override).expanduser()
    config_file = HERE / ".storage-config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return Path(line).expanduser()
    return HERE / "results"


RESULTS = resolve_results_root()

# Okabe-Ito colorblind-safe palette
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def parse_run(path: Path) -> dict:
    """Parse one official-benchmark CSV into {size_bytes: (algbw, busbw)}."""
    rows = {}
    with path.open() as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            try:
                size = int(row[6])
                algbw = float(row[14])
                busbw = float(row[15])
            except (ValueError, IndexError):
                continue
            rows[size] = (algbw, busbw)
    return rows


def run_id(path: Path) -> str:
    # filename: <timestamp>_<config>.csv
    return path.stem


def config_name(path: Path) -> str:
    # filename: <timestamp>_<config>.csv
    m = re.match(r"\d{8}-\d{6}_(.+)\.csv$", path.name)
    return m.group(1) if m else path.stem


@st.cache_data(show_spinner=False)
def discover_runs() -> list[dict]:
    """Discover all runs, grouped by config name."""
    if not RESULTS.exists():
        return []
    runs = []
    for p in sorted(RESULTS.glob("*.csv")):
        try:
            data = parse_run(p)
            if not data:
                continue
            runs.append({
                "id": run_id(p),
                "config": config_name(p),
                "data": data,
                "file": p,
            })
        except Exception:
            continue
    return runs


def aggregate_runs(runs: list[dict]) -> dict[str, dict[float, list[float]]]:
    """config -> {size: [busbw values]} -> median."""
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for size, (algbw, busbw) in run["data"].items():
            grouped[run["config"]][size].append(busbw)
    return {
        config: {size: median(values) for size, values in sorted(by_size.items())}
        for config, by_size in grouped.items()
    }


def build_figure(runs: list[dict], aggregated: dict[str, dict[float, list[float]]]) -> go.Figure:
    figure = go.Figure()
    # Unique style per individual run
    run_styles = {
        run["id"]: {
            "color": PALETTE[index % len(PALETTE)],
            "dash": DASHES[index % len(DASHES)],
        }
        for index, run in enumerate(runs)
    }
    # One style per config for median lines
    config_keys = sorted(aggregated.keys())
    config_styles = {
        key: {
            "color": PALETTE[index % len(PALETTE)],
            "dash": DASHES[index % len(DASHES)],
        }
        for index, key in enumerate(config_keys)
    }

    # Thin dotted lines for individual runs
    for run in runs:
        sizes = sorted(run["data"])
        bw = [run["data"][s][1] for s in sizes]
        style = run_styles[run["id"]]
        figure.add_trace(go.Scatter(
            x=[human_size(s) for s in sizes],
            y=bw,
            mode="lines",
            line={"color": style["color"], "width": 1, "dash": "dot"},
            opacity=0.28,
            hovertemplate=f"{run['config']} | {run['id']}<br>%{{x}}<br>%{{y:.2f}} GB/s<extra></extra>",
            showlegend=False,
        ))

    # Solid bold lines for medians per config
    for config, by_size in aggregated.items():
        sizes = sorted(by_size)
        bw = [by_size[s] for s in sizes]
        style = config_styles[config]
        figure.add_trace(go.Scatter(
            x=[human_size(s) for s in sizes],
            y=bw,
            mode="lines+markers",
            name=f"{config} median",
            line={"color": style["color"], "width": 3, "dash": style["dash"]},
            marker={"size": 8, "line": {"width": 1, "color": "white"}},
            hovertemplate=f"{config} median<br>%{{x}}<br>%{{y:.2f}} GB/s<extra></extra>",
        ))

    figure.update_layout(
        title="All-Reduce Bandwidth (busbw)",
        xaxis_title="message size",
        yaxis_title="busbw (GB/s)",
        xaxis_type="category",
        height=520,
        legend=dict(orientation="h", y=1.12),
        template="plotly_white",
    )
    return figure


st.set_page_config(page_title="xpu-bench", layout="wide")
st.title("XPU All-Reduce Bench")
st.caption("Official oneCCL benchmark · 2x Intel Arc B70 · float32 allreduce")

all_runs = discover_runs()
if not all_runs:
    st.info("No runs yet. Run `./bench.sh <config-name>` first.")
    st.stop()

configs = sorted({run["config"] for run in all_runs})
selected_configs = st.sidebar.multiselect("Configurations", configs, default=configs)
selected_runs = [run for run in all_runs if run["config"] in selected_configs]
if not selected_runs:
    st.warning("Select at least one configuration.")
    st.stop()

aggregated = aggregate_runs(selected_runs)
st.caption(f"{len(selected_runs)} run(s) across {len(selected_configs)} configuration(s). Thin lines are individual runs; solid lines are per-size medians.")

fig = build_figure(selected_runs, aggregated)
st.plotly_chart(fig, use_container_width=True)

# peak table
st.subheader("Peak bandwidth per run")
table = []
for run in selected_runs:
    peak_bus = max(v[1] for v in run["data"].values())
    table.append({"config": run["config"], "id": run["id"], "peak busbw (GB/s)": round(peak_bus, 2)})
st.dataframe(table, use_container_width=True, hide_index=True)

# summary table: key sizes per config (like ZePeer format)
st.subheader("Summary")
summary = []
for config in selected_configs:
    config_runs = [r for r in selected_runs if r["config"] == config]
    # Use median values for summary
    if config in aggregated:
        sizes = sorted(aggregated[config])
        entry = {"config": config, "runs": len(config_runs)}
        for size in sizes:
            entry[human_size(size)] = round(aggregated[config][size], 3)
        summary.append(entry)
st.dataframe(summary, use_container_width=True, hide_index=True)
