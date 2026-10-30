#!/usr/bin/env python3
"""xpu-bench dashboard — compare all-reduce runs side by side.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

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


def config_name(path: Path) -> str:
    # filename: <timestamp>_<config>.csv
    m = re.match(r"\d{8}-\d{6}_(.+)\.csv$", path.name)
    return m.group(1) if m else path.stem


def load_runs() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    runs = {}
    for p in sorted(RESULTS.glob("*.csv")):
        try:
            runs[config_name(p)] = parse_run(p)
        except Exception:
            continue
    return runs


st.set_page_config(page_title="xpu-bench", layout="wide")
st.title("XPU All-Reduce Bench")
st.caption("Official oneCCL benchmark · 2x Intel Arc B70 · float32 allreduce")

runs = load_runs()
if not runs:
    st.info("No runs yet. Run `./bench.sh <config-name>` first.")
    st.stop()

# newest-first, dedupe by config name (keep latest)
seen: dict[str, dict] = {}
for name, data in runs.items():
    seen[name] = data  # sorted() => later (newer) overwrites
names = list(seen.keys())

st.sidebar.header("Runs")
selected = st.sidebar.multiselect("Configs to show", names, default=names)

fig = go.Figure()
for i, name in enumerate(selected):
    data = seen[name]
    sizes = sorted(data)
    bw = [data[s][1] for s in sizes]  # busbw
    fig.add_trace(go.Scatter(
        x=[human_size(s) for s in sizes],
        y=bw,
        mode="lines+markers",
        name=name,
        line=dict(color=PALETTE[i % len(PALETTE)], dash=DASHES[i % len(DASHES)]),
    ))

fig.update_layout(
    xaxis_title="message size",
    yaxis_title="busbw (GB/s)",
    xaxis_type="category",
    height=520,
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True)

# peak table
st.subheader("Peak bandwidth per run")
table = []
for name in selected:
    data = seen[name]
    peak_alg = max(v[0] for v in data.values())
    peak_bus = max(v[1] for v in data.values())
    table.append({"config": name, "peak algbw (GB/s)": round(peak_alg, 2), "peak busbw (GB/s)": round(peak_bus, 2)})
st.dataframe(table, use_container_width=True, hide_index=True)

# full sweep table
st.subheader("Sweep Results")
sweep_data = []
for name in selected:
    data = seen[name]
    for size in sorted(data):
        algbw, busbw = data[size]
        sweep_data.append({
            "Config": name,
            "Size (B)": size,
            "algbw (GB/s)": round(algbw, 3),
            "busbw (GB/s)": round(busbw, 3),
        })
st.dataframe(sweep_data, use_container_width=True, hide_index=True)
