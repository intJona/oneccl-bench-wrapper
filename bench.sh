#!/usr/bin/env bash
# xpu-bench — one-liner all-reduce benchmark (official oneCCL tool, validated).
#
# Usage:
#   ./bench.sh <config-name>              full sweep (32B .. 256MB)
#   ./bench.sh <config-name> <size-bytes> single size
#
# Examples:
#   ./bench.sh direct
#   ./bench.sh swx16
#   ./bench.sh mixed 67108864
#
# Each run saves results/<timestamp>_<config>.csv (+ .log) and prints a summary.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve results dir: $XPU_BENCH_RESULTS_DIR > .storage-config > ./results
if [[ -n "${XPU_BENCH_RESULTS_DIR:-}" ]]; then
    RESULTS="$XPU_BENCH_RESULTS_DIR"
elif [[ -f "$HERE/.storage-config" ]]; then
    RESULTS="$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' "$HERE/.storage-config")"
else
    RESULTS="$HERE/results"
fi
mkdir -p "$RESULTS"

# --- layout (override via env if your paths differ) ---
BENCH_BIN="${XPU_BENCH_BIN:-$HERE/../_archive/PCIe XPU Benchmarks/oneccl_official_bench/oneccl_benchmark}"
CONDA_LIB="${XPU_BENCH_CONDA_LIB:-/home/intel/.conda/envs/vllm-xpu/lib}"
SYCL_LIB="${XPU_BENCH_SYCL_LIB:-$HOME/dpcpp_nightly/lib}"
NPROC="${XPU_BENCH_NPROC:-2}"

CONFIG="${1:?usage: ./bench.sh <config-name> [size-bytes]}"
SIZE="${2:-}"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "error: benchmark binary not found at $BENCH_BIN" >&2
  echo "       (set XPU_BENCH_BIN to point at oneccl_benchmark)" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT_CSV="$RESULTS/${TS}_${CONFIG}.csv"
OUT_LOG="$RESULTS/${TS}_${CONFIG}.log"

# size args: full sweep by default; single size if given (bytes -> float32 elems)
if [[ -n "$SIZE" ]]; then
  SIZE_ARGS=(-y $(( SIZE / 4 )))
  DESC="single size ${SIZE} bytes"
else
  SIZE_ARGS=(-f 8 -t 536870912)
  DESC="sweep 32B..512MB"
fi

echo "xpu-bench: config=$CONFIG  ($DESC)"
echo "  csv: $OUT_CSV"

cd "$(dirname "$BENCH_BIN")"
set +e
env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/local/bin" \
  LD_LIBRARY_PATH="$CONDA_LIB:$SYCL_LIB" \
  CCL_ATL_SHM=1 CCL_ATL_TRANSPORT=ofi \
  ONEAPI_DEVICE_SELECTOR="level_zero:*" \
  ZE_ENABLE_PCIOM_DEVICE="${ZE_ENABLE_PCIOM_DEVICE:-1}" \
  /usr/bin/mpirun -n "$NPROC" \
    -x LD_LIBRARY_PATH -x CCL_ATL_SHM -x CCL_ATL_TRANSPORT -x ONEAPI_DEVICE_SELECTOR -x ZE_ENABLE_PCIOM_DEVICE \
    "$BENCH_BIN" -l allreduce -d float32 "${SIZE_ARGS[@]}" -i 20 -w 5 -j off \
    -o "$OUT_CSV" >"$OUT_LOG" 2>&1
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "  FAILED (exit $rc) — last lines of $OUT_LOG:" >&2
  tail -n 20 "$OUT_LOG" >&2
  exit "$rc"
fi

echo
echo "  message_size(bytes)   busbw(GB/s)"
awk -F, 'NR>1 {printf "  %18d   %11.2f\n", $7, $16}' "$OUT_CSV"
echo
echo "  peak busbw: $(awk -F, 'NR>1 && $16>m {m=$16} END{printf "%.2f GB/s", m}' "$OUT_CSV")"
echo "  done -> $OUT_CSV"
