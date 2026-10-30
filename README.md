# xpu-bench

All-reduce benchmark for 2x Intel Arc B70. Uses the official oneCCL benchmark
(validated, `-j off` to avoid the iter_policy bug).

## Why a wrapper?

Without it, you'd need to type this every time:

```bash
env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/local/bin" \
  LD_LIBRARY_PATH="/home/intel/.conda/envs/vllm-xpu/lib:$HOME/dpcpp_nightly/lib" \
  CCL_ATL_SHM=1 CCL_ATL_TRANSPORT=ofi \
  ONEAPI_DEVICE_SELECTOR="level_zero:*" \
  /usr/bin/mpirun -n 2 -x LD_LIBRARY_PATH -x CCL_ATL_SHM -x CCL_ATL_TRANSPORT -x ONEAPI_DEVICE_SELECTOR \
  $HOME/PCIe\ GNR/_archive/PCIe\ XPU\ Benchmarks/oneccl_official_bench/oneccl_benchmark -l allreduce -d float32 -f 8 -t 67108864 -i 20 -w 5 -j off -o results/$(date +%Y%m%d-%H%M%S)_config.csv
```

The wrapper bakes in the critical env vars (`ONEAPI_DEVICE_SELECTOR=level_zero:*`,
`CCL_ATL_SHM=1`, `CCL_ATL_TRANSPORT=ofi`) and the `-j off` flag so you can't
accidentally forget them.

## Run a benchmark

```bash
./bench.sh <config-name>              # full sweep, 32B..256MB
./bench.sh <config-name> <size-bytes> # single size
```

Examples:

```bash
./bench.sh dix16
```

```bash
./bench.sh swx16
```


```bash
./bench.sh sw+di
```

Each run saves `results/<timestamp>_<config>.csv` and prints a summary table.

## Dashboard

```bash
pip install -r requirements.txt
```

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 — pick which runs to compare.

## Notes

- Config name is free-form — use whatever describes the wiring
  (e.g. `direct`, `swx16`, `mixed`, `swx16_acsdis`).
- The script bakes in the required env (`ONEAPI_DEVICE_SELECTOR=level_zero:*`,
  `CCL_ATL_SHM=1`, `CCL_ATL_TRANSPORT=ofi`, `-j off`). Don't remove them.
- Paths are overridable via `XPU_BENCH_BIN`, `XPU_BENCH_CONDA_LIB`,
  `XPU_BENCH_SYCL_LIB`, `XPU_BENCH_NPROC` if your layout differs.
