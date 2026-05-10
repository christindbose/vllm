# cascade_bench — vLLM cascade attention benchmark harness

Experiment harness for measuring cascade attention paths in vLLM, both
**offline** (`LLM.generate(...)`) and **online** (`vllm serve` + an async
HTTP client driving Poisson-distributed request rate).

Used to evaluate the `VLLM_TREE_WALK` tree-walk cascade path against the
stock cascade and FlashAttention backends, plus to reproduce Fig 12-style
serving plots from the PAT paper (arXiv 2511.22333).

## File layout

```
backend_request_func.py     Async HTTP client per-backend request funcs
benchmark_serving.py        Driver that loads dataset + sends Poisson load
benchmark_utils.py          Helpers (calculate_metrics, etc.)
loader.py                   Dataset loaders (Mooncake conversation/toolagent
                            traces, BurstGPT, Qwen traceA/B). Synthesizes
                            prompts from each row's hash_ids using a
                            non-overlapping block_id -> token-range map so
                            same hash_id deterministically yields same tokens.
plot_sweep.py               Renders Fig 12-style multi-panel plot from
                            saved benchmark JSONs.
run_sweep.sh                Convenience wrapper: sweeps a list of request
                            rates against an already-running vllm serve
                            and saves one JSON per rate.

smoke_test_cascade.py       OFFLINE: hits the cascade path through
                            LLM.generate() with a short hand-written shared
                            prefix. Used to confirm VLLM_TREE_WALK plumbing
                            works end-to-end.
bench_treewalk_vs_stock.py  OFFLINE A/B microbench: same workload twice
                            (env=0 stock cascade vs env=1 tree-walk), reports
                            median TTFT/TPOT.

results/                    Saved JSONs + plots from the latest sweep
  FlashAttn/                  Per-rate JSONs (FA backend, 1000 reqs)
  FlashInfer/                 Per-rate JSONs (stock FlashInfer 0.2.5, 1000 reqs)
  fig12_fa_vs_fi.png          Light-load sweep (8 rates × 100 reqs)
  fig12_sat_with_itl.png      Saturation sweep (6 rates × 1000 reqs) with TTFT/TPOT/ITL/P99 panels
```

## Provenance

The bench client (`backend_request_func.py`, `benchmark_serving.py`,
`benchmark_utils.py`, `loader.py`) is vendored from the PAT artifact
(github.com/flashserve/PAT, MIT license) with three local edits:

1. **`benchmark_serving.py:331`**: replaced hard-coded `PORT_TO_BACKEND[port]`
   lookup with `.get(port, f"port-{port}")` so non-PAT ports don't crash the
   report-printing step.
2. **`benchmark_serving.py:362-367`**: uncommented `"itl"` from
   `metric_definitions` so mean/median/p99 ITL (a.k.a. TBT) are saved into the
   result JSONs alongside TTFT and TPOT.
3. **`loader.py:generate_block_token_ids`**: changed from PAT's overlapping
   `range(block_id, block_id+block_size)` to a non-overlapping
   `range(base, base+block_size)` where `base = (block_id*block_size) % cycle`.
   Same-hash_id determinism preserved; different hash_ids produce disjoint
   token ranges. PAT's overlapping version was suspected (incorrectly, in the
   end) of triggering an FI engine wedge during early debugging, so the fix
   stayed in for cleanliness.

## Datasets

The loader expects these files at `dataset/`:

```
dataset/toolagent_trace.jsonl     Mooncake FAST25 toolagent trace
dataset/conversation_trace.jsonl  Mooncake FAST25 conversation trace
```

Source: github.com/kvcache-ai/Mooncake, `FAST25-release/traces/`. Symlink them
into `dataset/` before running.

## How the benchmark works

1. `MooncakeLoader` parses the trace's `(timestamp, input_length,
   output_length, hash_ids)` rows. Filters by `--max-context-len`.
2. `CustomDataset.sample(N)` synthesizes `prompt_token_ids` per row by
   concatenating per-hash_id token blocks (`generate_block_token_ids`).
3. `benchmark` issues `N` requests to `/v1/completions` over an aiohttp client,
   inter-arrival times drawn from `Exp(rate)` for Poisson load.
4. Each request streams its response; the client records first-chunk-time
   (TTFT) and per-chunk gaps (ITL) per request.
5. `calculate_metrics` aggregates: mean/median/std/percentiles for TTFT, TPOT,
   ITL, plus throughput.
6. Result JSON saved per (backend, rate).

## Reproducing the saturation plot

```bash
# 1. Start vllm serve under the backend you want
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching --port 8000 --disable-log-requests

# 2. Sweep rates
BACKEND=FlashAttn RESULT_DIR=results/FlashAttn PORT=8000 \
    NUM_PROMPTS=1000 RATES="8 12 16 20 25 30" \
    bash run_sweep.sh

# 3. Repeat for VLLM_ATTENTION_BACKEND=FLASHINFER (and others) into separate dirs

# 4. Plot
python plot_sweep.py results/FlashAttn results/FlashInfer \
    -o results/fig12_sat_with_itl.png \
    --title "Toolagent saturation - FA vs FI"
```

Plot has 4 panels: Mean TTFT, Mean TPOT, Mean TBT/ITL, P99 TPOT.

## Tree-walk cascade path

To exercise the user's tree-walk SM90 prefill kernel through this harness,
swap `flashinfer-python` for the local fork at `~/myflashinfer_old/flashinfer/`
(editable install) and set `VLLM_TREE_WALK=1` on the server. Default vLLM
behavior is unchanged when the env var is unset.

The offline microbench at `bench_treewalk_vs_stock.py` uses the same env-gate
to A/B test the kernel without the serving stack.
