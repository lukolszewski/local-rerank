#!/bin/bash
# Debug commands for jina-reranker container

echo "=== 1. Check if llama-embedding exists and is executable ==="
docker exec jina-reranker ls -la /app/llama-embedding

echo ""
echo "=== 2. Test llama-embedding with --help ==="
docker exec jina-reranker /app/llama-embedding --help 2>&1 | head -30

echo ""
echo "=== 3. Check model file exists ==="
docker exec jina-reranker ls -lh /app/models/jina-reranker-v3-Q8_0.gguf

echo ""
echo "=== 4. Create a test prompt file and try embedding ==="
docker exec jina-reranker bash -c 'echo "test query" > /tmp/test.txt && /app/llama-embedding -m /app/models/jina-reranker-v3-Q8_0.gguf -f /tmp/test.txt --pooling none --embd-output-format json 2>&1 | head -50'

echo ""
echo "=== 5. Check Python and dependencies ==="
docker exec jina-reranker python3 -c "import numpy, safetensors; print('Python deps OK')"

echo ""
echo "=== 6. Test a simple rerank via Python ==="
docker exec jina-reranker python3 -c "
from rerank import GGUFReranker
reranker = GGUFReranker(
    model_path='/app/models/jina-reranker-v3-Q8_0.gguf',
    projector_path='/app/models/projector.safetensors',
    llama_embedding_path='/app/llama-embedding'
)
print('Reranker initialized')
results = reranker.rerank('test query', ['document 1', 'document 2'])
print('Results:', results)
"

echo ""
echo "=== 7. Check stderr from llama-embedding ==="
docker exec jina-reranker bash -c 'echo "test" > /tmp/test2.txt && /app/llama-embedding -m /app/models/jina-reranker-v3-Q8_0.gguf -f /tmp/test2.txt 2>&1'
