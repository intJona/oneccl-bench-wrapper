# xpu-bench

oneCCL all-reduce sweep across multiple Intel GPUs, for comparing PCIe configurations
(topology, lane width, ACS, switch vs. direct attach, ...). Wraps the official oneCCL
benchmark (`-j off` to avoid its `iter_policy` bug).

## Why a wrapper?

The oneCCL benchmark needs a specific environment and launch line. Typing it by hand every
time is error-prone, and one forgotten variable changes the result:

```bash
env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/local/bin" \
  LD_LIBRARY_PATH="<conda-env>/lib:<dpcpp>/lib" \
  CCL_ATL_SHM=1 CCL_ATL_TRANSPORT=ofi \
  ONEAPI_DEVICE_SELECTOR="level_zero:*" \
  mpirun -n 2 -x LD_LIBRARY_PATH -x CCL_ATL_SHM -x CCL_ATL_TRANSPORT -x ONEAPI_DEVICE_SELECTOR \
  <path>/oneccl_benchmark -l allreduce -d float32 -f 8 -t 134217728 -i 20 -w 1000 -j off \
  -o results/$(date +%Y%m%d-%H%M%S)_config.csv
```

`bench.sh` bakes all of that in (clean environment via `env -i`, the variables above, `-j off`)
and adds:

- a timestamped CSV + log per run under `results/`, named after your config label
- the kernel command line and PCIe ACS state saved next to each result (`.env.txt`)
- a second, untimed pass that records which all-reduce algorithm each size used (`.algo.txt`)
- a timeout, so a hung run does not block forever

## Requirements

- Linux, 2+ Intel GPUs with a Level Zero driver, Open MPI (`mpirun`)
- The oneCCL benchmark binary (`oneccl_benchmark`, built from oneCCL's `examples/benchmark`)
- A conda env (or any prefix) containing the oneCCL / Level Zero runtime libraries
- The DPC++ runtime libraries

`acs_state.sh` only looks at Intel GPUs with PCI ID `8086:e223` (Arc B70). Edit the ID in that
script for other GPUs.

## Setup

The script's built-in default paths are machine-specific. Set these for your system:

| Variable | Meaning |
|---|---|
| `XPU_BENCH_BIN` | Path to `oneccl_benchmark` |
| `XPU_BENCH_CONDA_LIB` | `lib/` directory with the oneCCL / Level Zero runtime |
| `XPU_BENCH_SYCL_LIB` | `lib/` directory of the DPC++ runtime |
| `XPU_BENCH_NPROC` | Number of ranks / GPUs (default: `2`) |
| `XPU_BENCH_RESULTS_DIR` | Where results go (default: `./results`, or the path in `.storage-config`) |

```bash
export XPU_BENCH_BIN=/path/to/oneccl_benchmark
export XPU_BENCH_CONDA_LIB=/path/to/env/lib
export XPU_BENCH_SYCL_LIB=/path/to/dpcpp/lib
```

## Run a benchmark

```bash
./bench.sh <config-name>          # full sweep, 32 B .. 512 MB
./bench.sh <config-name> <size>   # single size: bytes, or e.g. 256KB / 32MB / 1GB
```

`<config-name>` is a free-form label for the physical setup under test
(e.g. `direct_x16`, `switch_x8`). A sweep saves `results/<timestamp>_<config-name>.csv` and
prints a summary table. A single-size run prints its result but does not save it.

Options (environment variables):

| Variable | Default | Effect |
|---|---|---|
| `XPU_BENCH_DESCENDING` | `0` | `1` = sweep from largest to smallest size |
| `XPU_BENCH_WARMUP` | `1000` | Warm-up iterations per size |
| `XPU_BENCH_ITERS` | `20` | Timed iterations per size |
| `XPU_BENCH_TIMEOUT` | `600` | Seconds before a hung run is killed (`0` = off); partial CSV is kept as `.csv.partial` |
| `XPU_BENCH_TRACE` | `1` | `0` = skip the extra pass that writes `<run>.algo.txt` |

```bash
XPU_BENCH_DESCENDING=1 XPU_BENCH_ITERS=50 ./bench.sh <config-name>
```

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Open http://localhost:8501 and pick which runs to compare.
