#!/usr/bin/env bash
# xpu-bench — one-liner all-reduce benchmark (official oneCCL tool, validated).
#
# Usage:
#   ./bench.sh <config-name>              full sweep (32B .. 256MB)
#   ./bench.sh <config-name> <size>       single size, bytes or 256KB/32MB/1GB (terminal only, not saved)
#   XPU_BENCH_DESCENDING=1 ./bench.sh <config-name>   sweep from largest to smallest
#   XPU_BENCH_WARMUP=N / XPU_BENCH_ITERS=N             warm-up / timed iterations per size (1000 / 20)
#   XPU_BENCH_TIMEOUT=secs                             kill a hung run (default 600, 0 = off)
#   XPU_BENCH_TRACE=0                                  skip the algorithm-trace pass
#
# Examples:
#   ./bench.sh direct
#   ./bench.sh swx16
#   ./bench.sh mixed 67108864
#
# Each sweep saves results/<timestamp>_<config>.csv (+ .log, .env.txt with cmdline and ACS state) and prints a summary.
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

# --- layout (override via env if your paths differ) ---
BENCH_BIN="${XPU_BENCH_BIN:-$HERE/../_archive/PCIe XPU Benchmarks/oneccl_official_bench/oneccl_benchmark}"
CONDA_LIB="${XPU_BENCH_CONDA_LIB:-/home/intel/.conda/envs/vllm-xpu/lib}"
SYCL_LIB="${XPU_BENCH_SYCL_LIB:-$HOME/dpcpp_nightly/lib}"
NPROC="${XPU_BENCH_NPROC:-2}"
WARMUP="${XPU_BENCH_WARMUP:-1000}"
ITERS="${XPU_BENCH_ITERS:-20}"
RUN_TIMEOUT="${XPU_BENCH_TIMEOUT:-600}"   # seconds per mpirun, 0 disables

CONFIG="${1:?usage: ./bench.sh <config-name> [size]}"
SIZE="${2:-}"

# 1024-based: 32MB = 33554432 bytes
parse_size() {
  local s="${1^^}"
  [[ $s =~ ^([0-9]+)(B|KB?|MB?|GB?)?$ ]] || { echo "error: bad size '$1' (use bytes, or e.g. 256KB, 32MB, 1GB)" >&2; exit 1; }
  local n=${BASH_REMATCH[1]}
  case "${BASH_REMATCH[2]}" in
    K|KB) n=$(( n * 1024 )) ;;
    M|MB) n=$(( n * 1024 * 1024 )) ;;
    G|GB) n=$(( n * 1024 * 1024 * 1024 )) ;;
  esac
  echo "$n"
}
if [[ -n "$SIZE" ]]; then SIZE="$(parse_size "$SIZE")"; fi

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "error: benchmark binary not found at $BENCH_BIN" >&2
  echo "       (set XPU_BENCH_BIN to point at oneccl_benchmark)" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
if [[ -n "$SIZE" ]]; then
  OUT_DIR="$(mktemp -d)"
  trap 'rm -rf "$OUT_DIR"' EXIT
else
  OUT_DIR="$RESULTS"
  mkdir -p "$OUT_DIR"
fi
OUT_CSV="$OUT_DIR/${TS}_${CONFIG}.csv"
OUT_LOG="$OUT_DIR/${TS}_${CONFIG}.log"

# -t/-y take float32 element counts, not bytes: 134217728 elems = 512 MB
MAX_ELEMS=134217728

# size args: full sweep by default; single size if given (bytes -> float32 elems)
if [[ -n "$SIZE" ]]; then
  SIZE_ARGS=(-y $(( SIZE / 4 )))
  DESC="single size ${SIZE} bytes"
elif [[ "${XPU_BENCH_DESCENDING:-0}" == 1 ]]; then
  ELEM_LIST=""
  for ((e = MAX_ELEMS; e >= 8; e /= 2)); do ELEM_LIST+="${ELEM_LIST:+,}$e"; done
  SIZE_ARGS=(-y "$ELEM_LIST")
  DESC="sweep 512MB..32B, descending"
else
  SIZE_ARGS=(-f 8 -t "$MAX_ELEMS")
  DESC="sweep 32B..512MB"
fi

echo "xpu-bench: config=$CONFIG  ($DESC, warmup=$WARMUP iters=$ITERS)"
[[ -z "$SIZE" ]] && echo "  csv: $OUT_CSV"

# Record the kernel command line and ACS state of the ports above the GPUs (ACS changes results ~2.6x on the switch).
ENV_INFO="$("$HERE/acs_state.sh" 2>&1 || true)"
echo "  acs: $(grep '^acs ' <<<"$ENV_INFO" | sed 's/^acs //;s/0000://' | grep -v '=none' | sort -u | tr '\n' ' ')"
[[ -z "$SIZE" ]] && printf '%s\n' "$ENV_INFO" >"${OUT_CSV%.csv}.env.txt"

cd "$(dirname "$BENCH_BIN")"

# run_mpi <iters> <warmup> <csv> <log> [VAR=value ...]: extra vars are exported to the ranks
run_mpi() {
  local iters=$1 warmup=$2 csv=$3 log=$4; shift 4
  local xs=() kv tmo=()
  for kv in "$@"; do xs+=(-x "${kv%%=*}"); done
  (( RUN_TIMEOUT > 0 )) && tmo=(timeout --kill-after=10 "$RUN_TIMEOUT")
  "${tmo[@]}" env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/local/bin" \
    LD_LIBRARY_PATH="$CONDA_LIB:$SYCL_LIB" \
    CCL_ATL_SHM=1 CCL_ATL_TRANSPORT=ofi \
    ONEAPI_DEVICE_SELECTOR="level_zero:*" \
    ZE_ENABLE_PCIOM_DEVICE="${ZE_ENABLE_PCIOM_DEVICE:-1}" "$@" \
    /usr/bin/mpirun -n "$NPROC" \
      -x LD_LIBRARY_PATH -x CCL_ATL_SHM -x CCL_ATL_TRANSPORT -x ONEAPI_DEVICE_SELECTOR -x ZE_ENABLE_PCIOM_DEVICE "${xs[@]}" \
      "$BENCH_BIN" -l allreduce -d float32 "${SIZE_ARGS[@]}" -i "$iters" -w "$warmup" -j off \
      -o "$csv" >"$log" 2>&1
}

set +e
t0=$SECONDS
run_mpi "$ITERS" "$WARMUP" "$OUT_CSV" "$OUT_LOG"
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  # uutils timeout returns 125 (not 124) with --kill-after, so judge by elapsed time
  if (( RUN_TIMEOUT > 0 && SECONDS - t0 >= RUN_TIMEOUT )); then
    echo "  TIMED OUT after ${RUN_TIMEOUT}s (likely a hang) — last lines of $OUT_LOG:" >&2
  else
    echo "  FAILED (exit $rc) — last lines of $OUT_LOG:" >&2
  fi
  tail -n 20 "$OUT_LOG" >&2
  # keep a partial CSV out of the dashboard's *.csv scan
  if [[ -z "$SIZE" && -f "$OUT_CSV" ]]; then
    mv "$OUT_CSV" "$OUT_CSV.partial"
    echo "  partial results kept as $OUT_CSV.partial" >&2
  fi
  exit "$rc"
fi

echo
echo "  message_size(bytes)   busbw(GB/s)"
awk -F, 'NR>1 {printf "  %18d   %11.2f\n", $7, $16}' "$OUT_CSV"
echo
echo "  peak busbw: $(awk -F, 'NR>1 && $16>m {m=$16} END{printf "%.2f GB/s", m}' "$OUT_CSV")"

# Second, short pass with oneCCL debug logging (kept out of the timed run) to record which
# allreduce algorithm each size dispatches to. XPU_BENCH_TRACE=0 skips it.
if [[ "${XPU_BENCH_TRACE:-1}" == 1 ]]; then
  TRACE_DIR="$(mktemp -d)"
  set +e
  run_mpi 1 1 "$TRACE_DIR/t.csv" "$TRACE_DIR/t.log" CCL_LOG_LEVEL=debug
  set -e
  # keep only sizes present in the timed CSV (interleaved rank output can garble a count)
  ALGO="$(grep -ohE 'invoking [A-Za-z0-9_ ]*, ?count:[0-9]+' "$TRACE_DIR/t.log" \
    | sed -E 's/^invoking (.*), ?count:([0-9]+)$/\2|\1/' \
    | awk -F'|' -v csv="$OUT_CSV" '
        BEGIN { while ((getline line < csv) > 0) { split(line, f, ","); if (f[7] ~ /^[0-9]+$/) ok[f[7]] = 1 } }
        { c = $1 * 4; l = $2; if (!(c in ok)) next
          a = (l ~ /LL25/) ? "LL256" : (l ~ /large/) ? "large" : l   # LL25: line truncated by rank interleave
          if (!((c, a) in seen)) { seen[c, a] = 1; set[c] = (set[c] == "") ? a : set[c] "+" a } }
        END { for (c in set) print c "," set[c] }' \
    | sort -t, -n -k1,1)"
  rm -rf "$TRACE_DIR"
  if [[ -n "$ALGO" ]]; then
    echo "  algorithm: $(awk -F, '$2 != p { printf "%sfrom %d B: %s", (NR>1 ? " | " : ""), $1, $2; p = $2 }' <<<"$ALGO")"
    if [[ -z "$SIZE" ]]; then
      { echo "size_bytes,algorithm"; echo "$ALGO"; } >"${OUT_CSV%.csv}.algo.txt"
    fi
  else
    echo "  algorithm: not detected (no dispatch lines in debug trace)"
  fi
fi
if [[ -n "$SIZE" ]]; then
  echo "  done (single payload: shown here only, not saved to results)"
else
  echo "  done -> $OUT_CSV"
fi
