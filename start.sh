#!/bin/bash
set -e

# Start llama-server in the background
echo "Starting llama-server..."
/app/llama-server \
    -m /app/models/jina-reranker-v3-Q8_0.gguf \
    --port 8081 \
    --reranking \
    -ngl 99 \
    --ctx-size 131072 \
    --ubatch-size 2048 \
    --batch-size 2048 \
    --host 127.0.0.1 \
    > /var/log/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "llama-server started with PID $LLAMA_PID"

# Wait for llama-server to be ready
echo "Waiting for llama-server to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8081/health > /dev/null 2>&1; then
        echo "llama-server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: llama-server failed to start"
        cat /var/log/llama-server.log
        exit 1
    fi
    sleep 1
done

# Start uvicorn in the foreground
echo "Starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 80
