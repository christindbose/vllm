"""
A/B benchmark: vLLM V1 cascade attention with VLLM_TREE_WALK=1 (Track A.1
fork's tree-walk SM90 prefill kernel via num_levels=1 wrapper) vs default
(stock cascade through the fork's baseline=True non-fused merge path).

Run TWICE — once per env setting — because the env is read at import time:
  VLLM_TREE_WALK=0 python tests/bench_treewalk_vs_stock.py
  VLLM_TREE_WALK=1 python tests/bench_treewalk_vs_stock.py

Workload: 64 reqs sharing a long prefix, max_tokens=64, temperature=0.
Same SHARED_PREFIX as the smoke test so cascade fires reliably during decode.

Outputs are written as a single JSON line so a wrapper script can collect
both runs and diff them. Stdout is otherwise suppressed via flushed prints
of a few INFO lines.
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import torch
from vllm import LLM, SamplingParams


SHARED_PREFIX = (
    "You are an expert software engineer. The following is a long technical document "
    "describing the architecture of a large-scale distributed system. The system has "
    "many components including a load balancer, an authentication service, a primary "
    "database, a read replica pool, an in-memory cache, an asynchronous message bus, "
    "a worker pool for background jobs, an observability stack with metrics and "
    "tracing, a deployment pipeline with continuous integration, an infrastructure-"
    "as-code layer, and a feature-flag service that controls progressive rollouts. "
    "Each component is independently scaled and monitored. The load balancer routes "
    "requests based on weighted round-robin with health checks. The authentication "
    "service issues short-lived JWT tokens that are validated at the API gateway. "
    "The database is partitioned by tenant id and replicated across three zones. "
    "Reads are served from replicas with bounded staleness. Writes go to the primary "
    "and are propagated asynchronously. The cache uses a least-recently-used policy "
    "with per-tenant eviction. The message bus guarantees at-least-once delivery and "
    "uses idempotency keys to deduplicate retries. Workers are autoscaled based on "
    "queue depth and processing latency. Observability captures p50, p95, and p99 "
    "latency for every endpoint, plus error rates broken down by tenant and version. "
    "Now answer the following question concisely: "
)
QUESTIONS = [
    "What is the load balancer's routing policy?",
    "How are JWT tokens validated?",
    "How is the database partitioned?",
    "What is the cache eviction policy?",
    "What does the message bus guarantee?",
    "How are workers autoscaled?",
    "Which percentiles are tracked?",
    "How is staleness bounded?",
    "Where do writes go?",
    "How are reads served?",
    "What does the auth service issue?",
    "How are retries deduplicated?",
    "How many zones is the database replicated across?",
    "What controls progressive rollouts?",
    "What is captured per endpoint?",
    "What scales autonomously?",
]

NUM_REQS = 64                # 16 unique × 4 replicas → 64 reqs (cascade-vs-FlashDecoding heuristic prefers cascade)
MAX_TOKENS = 64
WARMUP_ITERS = 2
TIMED_ITERS = 5

prompts = ([SHARED_PREFIX + q for q in QUESTIONS]) * (NUM_REQS // len(QUESTIONS))
assert len(prompts) == NUM_REQS, len(prompts)

env_value = os.environ.get("VLLM_TREE_WALK", "0")
print(f"[BENCH] VLLM_TREE_WALK={env_value}", flush=True)
print(f"[BENCH] num_reqs={NUM_REQS} max_tokens={MAX_TOKENS} "
      f"warmup={WARMUP_ITERS} timed={TIMED_ITERS}", flush=True)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.55,
    max_model_len=2048,
)


def _time_iters(sampling, n_iters):
    """Run `n_iters` timed iterations of llm.generate(prompts, sampling).
    Returns a list of wall-clock seconds per iter and the last RequestOutput list."""
    times = []
    outs = None
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, outs


# --- Warmup --------------------------------------------------------------
# Lets FlashInfer JIT compile cascade kernels and the KV-cache layout settle.
sampling_full = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)
for _ in range(WARMUP_ITERS):
    llm.generate(prompts, sampling_full, use_tqdm=False)
torch.cuda.synchronize()

# --- Pass 1: max_tokens=1, isolates prefill + first-decode ---------------
# Wall time here is the batch's TTFT (when the first output token is sampled
# for the entire batch, which arrives together since all reqs were submitted
# at the same instant by llm.generate()).
sampling_1 = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
ttft_times_s, _ = _time_iters(sampling_1, TIMED_ITERS)
for i, t in enumerate(ttft_times_s):
    print(f"[BENCH] ttft iter {i+1}/{TIMED_ITERS}: {t*1000:.1f} ms", flush=True)

# --- Pass 2: max_tokens=N, full wall -------------------------------------
full_times_s, full_outputs = _time_iters(sampling_full, TIMED_ITERS)
for i, t in enumerate(full_times_s):
    print(f"[BENCH] full iter {i+1}/{TIMED_ITERS}: {t*1000:.1f} ms", flush=True)

# --- Derive TTFT / TPOT --------------------------------------------------
# TTFT (batch-level): median of pass-1 wall times.
# TPOT (per-token): (full - prefill) / (N - 1), per pair of medians.
median_ttft_s = statistics.median(ttft_times_s)
median_full_s = statistics.median(full_times_s)
decode_only_s = max(median_full_s - median_ttft_s, 0.0)
median_tpot_s = decode_only_s / max(MAX_TOKENS - 1, 1)

total_output_tokens = sum(len(o.outputs[0].token_ids) for o in full_outputs)
median_out_tps = total_output_tokens / median_full_s

result = {
    "vllm_tree_walk": env_value,
    "num_reqs": NUM_REQS,
    "max_tokens": MAX_TOKENS,
    "warmup_iters": WARMUP_ITERS,
    "timed_iters": TIMED_ITERS,
    # raw timings
    "ttft_iters_ms":  [round(t * 1000, 2) for t in ttft_times_s],
    "full_iters_ms":  [round(t * 1000, 2) for t in full_times_s],
    # derived
    "median_full_ms":  round(median_full_s * 1000, 2),
    "min_full_ms":     round(min(full_times_s) * 1000, 2),
    "max_full_ms":     round(max(full_times_s) * 1000, 2),
    "stdev_full_ms":   round(statistics.stdev(full_times_s) * 1000, 2) if len(full_times_s) > 1 else 0.0,
    "median_ttft_ms":  round(median_ttft_s * 1000, 2),
    "median_tpot_ms":  round(median_tpot_s * 1000, 3),
    "total_output_tokens_per_iter": total_output_tokens,
    "median_output_tps":            round(median_out_tps, 1),
}
print(f"[BENCH] RESULT_JSON {json.dumps(result)}", flush=True)
