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

## Miscellaneous

### Run without P2P (staged / host-mediated)

oneCCL auto-detects PCIe P2P capability. To force a staged run (GPU→host→GPU instead of direct GPU→GPU):

```bash
CCL_P2P_DISABLE=1 ./bench.sh <config-name>
```

Or disable topology recognition entirely:

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 ./bench.sh <config-name>
```

This is useful for comparing P2P vs non-P2P performance, similar to the `p2p` vs `no_p2p` traces in CommBench.
