# Jina Reranker v3 GGUF Migration Plan

## Objective
Migrate from fastembed-based jina-reranker-v2 to GGUF-quantized jina-reranker-v3 using llama.cpp for improved performance.

## Current State
- Using: `jinaai/jina-reranker-v2-base-multilingual` via fastembed (1.1GB ONNX)
- Performance: Slow, mind-bogglingly slow
- API format: Compatible with Jina AI API (model, object, usage, results)

## Target State
- Model: `jina-reranker-v3-Q8_0.gguf` (8-bit quantized, ~650MB)
- Base image: `ghcr.io/ggml-org/llama.cpp:full-cuda`
- Backend: llama.cpp binaries (llama-embedding, llama-tokenize)
- API: FastAPI with uvicorn on port 80
- Future: Enable NVIDIA GPU acceleration

## Implementation Steps

### 1. Research Base Image
- Investigate `ghcr.io/ggml-org/llama.cpp:full-cuda`
- Determine base distro (likely Ubuntu/Debian)
- Check if llama.cpp binaries are pre-installed
- Verify CUDA libraries presence

### 2. File Structure
```
/home/luk/dev/claude/talk1/jina-pl/
├── plan.md                          # This file
├── Dockerfile                       # To be created
├── main.py                          # FastAPI application
├── rerank.py                        # GGUFReranker implementation
├── requirements.txt                 # Python dependencies
└── models/
    ├── jina-reranker-v3-Q8_0.gguf  # Model weights (user will download)
    └── projector.safetensors        # MLP projector (need to obtain)
```

### 3. Python Dependencies
- fastapi
- uvicorn[standard]
- numpy
- safetensors
- pydantic (for request/response models)

### 4. Create rerank.py
Implement `GGUFReranker` class based on model card:
- Initialize with model_path, projector_path, llama_embedding_path
- Implement `rerank(query, documents, top_n, return_embeddings, instruction)` method
- Call llama.cpp binaries via subprocess
- Return results in format: `[{"index": int, "relevance_score": float, "document": str}]`

### 5. Create main.py
Port existing FastAPI application to use GGUFReranker:
- Keep existing Pydantic models: RerankRequest, RerankResult, RerankResponse, InfoResponse
- Replace TextCrossEncoder with GGUFReranker
- Maintain API compatibility (query, documents, batch_size, top_n, return_documents, model)
- Keep /rerank and /info endpoints

### 6. Create Dockerfile
```dockerfile
FROM ghcr.io/ggml-org/llama.cpp:full-cuda

# Install Python dependencies
# Determine distro and use appropriate package manager
# Install: python3-pip, python3-venv (or equivalent)

# Install Python packages
# Copy requirements.txt and run pip install

# Copy application files
# COPY main.py rerank.py /app/
# COPY models/ /app/models/

# Expose port 80
# CMD: uvicorn main:app --host 0.0.0.0 --port 80
```

### 7. API Compatibility
Ensure the new implementation maintains the same API format:
```json
{
  "model": "jina-reranker-v3",
  "object": "list",
  "usage": {"total_tokens": 123},
  "results": [
    {"index": 3, "relevance_score": 0.95, "document": "..."},
    {"index": 0, "relevance_score": 0.82, "document": "..."}
  ]
}
```

### 8. Key Questions - RESOLVED
1. ✅ llama.cpp binaries: `llama-cli` is available at `/app/llama-cli`
2. ✅ Base image: Ubuntu 22.04 (apt package manager)
3. ✅ projector.safetensors: Downloaded to models/ folder by user
4. ✅ Model file: jina-reranker-v3-Q8_0.gguf downloaded by user
5. ⚠️ Note: `llama-tokenize` binary not found, but rerank.py uses it. May need workaround or different approach.

### 9. Testing Strategy
1. Build image without CUDA first (CPU-only for development)
2. Test /rerank endpoint with sample query
3. Compare response format with current implementation
4. Verify performance improvement
5. Add NVIDIA runtime support
6. Test with GPU acceleration

### 10. Deployment Steps (After Development)
1. Stop current jina-reranker container
2. Build new image: `docker build -t jina-reranker-v3:latest .`
3. Run with NVIDIA runtime: `docker run --gpus all -p 8000:80 jina-reranker-v3:latest`
4. Test integration with LibreChat
5. Commit working container if needed

## Notes
- Start with CPU-only implementation to validate logic
- GPU acceleration can be added after basic functionality works
- Keep API 100% compatible to avoid changes in LibreChat agents code
- Consider adding batch processing optimization later
