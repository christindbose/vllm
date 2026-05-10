#!/bin/bash
# Rate sweep against an already-running vllm serve on $PORT.
# Output: one JSON per (backend_label, rate) under $RESULT_DIR.
#
# Usage: BACKEND=FA RESULT_DIR=results/fa PORT=8000 bash run_sweep.sh

set -u
BACKEND="${BACKEND:-FA}"
RESULT_DIR="${RESULT_DIR:-results/$BACKEND}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
RATES=(${RATES:-4 6 8 10})
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_CTX="${MAX_CTX:-7800}"

mkdir -p "$RESULT_DIR"
echo "[SWEEP] backend=$BACKEND rates=${RATES[@]} num_prompts=$NUM_PROMPTS port=$PORT"

for rate in "${RATES[@]}"; do
    echo "[SWEEP] === rate=$rate ==="
    OTEL_SDK_DISABLED=true PYTHONUNBUFFERED=1 \
    PATH=/home/chris241/myflashinfer_old/vllm_workspace/.venv/bin:$PATH \
    /home/chris241/myflashinfer_old/vllm_workspace/.venv/bin/python -u \
        /home/chris241/myflashinfer_old/vllm_workspace/tests/serving/benchmark_serving.py \
        --backend vllm --model "$MODEL" --tokenizer "$MODEL" \
        --port "$PORT" --host 127.0.0.1 --endpoint /v1/completions \
        --dataset-type toolagent --block-size 512 \
        --num-prompts "$NUM_PROMPTS" --request-rate "$rate" \
        --max-context-len "$MAX_CTX" --disable-tqdm \
        --save-result --result-dir "$RESULT_DIR" \
        --result-filename "${BACKEND}-${rate}qps.json" \
        2>&1 | grep -E "(rate|Mean|Median|P99|Successful|Throughput|Backend|Failed|Error|Traceback)" \
        | grep -v "Transient\|Failed to export"
    echo
done

echo "[SWEEP] done. JSON files in $RESULT_DIR/"
ls "$RESULT_DIR"/
