"""
Stock cascade attention smoke test.

Goal: confirm vLLM's V1 FlashInfer backend triggers cascade attention when a batch
of requests share a long common prefix. We print [SMOKE_TEST_CASCADE] markers from
inside the backend's cascade plan/run paths.

Cascade gating (from vllm/v1/attention/backends/flash_attn.py:use_cascade_attention):
- common_prefix_len >= 256
- num_reqs >= 8
- no ALiBi, no sliding window
"""
import os

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
# Silence OTel "ConnectionRefused" spam at localhost:4318 (no collector running)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from vllm import LLM, SamplingParams

# Long shared prefix: ~600 tokens of identical text across requests so common prefix
# detection blows past the 256-token threshold.
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

prompts = [SHARED_PREFIX + q for q in QUESTIONS]
# Replicate to push past vLLM's cascade-vs-FlashDecoding heuristic threshold for
# Qwen2.5-7B (num_qo=28 / num_kv=4): at ~64 reqs the perf model picks cascade
# over FlashDecoding on decode steps. With 16 reqs it picks FlashDecoding and
# we never exercise the cascade path during decode.
prompts = prompts * 4  # 16 -> 64 distinct (q is different per req but prefix shared)
# Larger max_tokens produces more decode-only cascade steps after prefill is done.
sampling = SamplingParams(temperature=0.0, max_tokens=64)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,        # avoids cudagraph capture path during smoke test
    enable_prefix_caching=True,  # required for common-prefix detection across requests
    gpu_memory_utilization=0.55,
    max_model_len=2048,
)

print(f"\n[SMOKE_TEST_CASCADE] launching {len(prompts)} requests, shared prefix len ≈ {len(SHARED_PREFIX.split())} words", flush=True)
outputs = llm.generate(prompts, sampling)
print("[SMOKE_TEST_CASCADE] generation complete", flush=True)
for i, o in enumerate(outputs[:3]):
    print(f"  req[{i}]: {o.outputs[0].text!r}")
