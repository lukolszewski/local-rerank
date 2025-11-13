# Local Reranker API

Jina AI compatible reranking API using llama.cpp with Jina Reranker v3 (GGUF quantized). Runs locally with GPU acceleration for fast, private document reranking.

Please note this service is completely unsecured. Only deploy in isolated network segments and do not expose to the outside world.

There is no warranty whatsoever for this code.

## Quick Start

### Using Docker Compose

1. **Download the model:**
   ```bash
   mkdir -p models
   # Download from Hugging Face (requires HF token)
   curl -L -H "Authorization: Bearer YOUR_HF_TOKEN" \
     https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/jina-reranker-v3-Q8_0.gguf \
     -o models/jina-reranker-v3-Q8_0.gguf
   ```

2. **Build the image:**
   ```bash
   docker compose build
   ```

3. **Start the service:**
   ```bash
   docker compose up -d
   ```

4. **Test the API:**
   ```bash
   curl -X POST http://localhost:8000/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the capital of France?",
       "documents": [
         "Paris is the capital of France.",
         "Berlin is the capital of Germany.",
         "Rome is the capital of Italy."
       ]
     }'
   ```

### Using Docker Hub Image

```yaml
version: "3.9"

services:
  local-rerank:
    image: lukolszewski/local-rerank:latest
    container_name: local-rerank
    restart: unless-stopped
    ports:
      - "8000:80"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_PATH` | `/app/models/jina-reranker-v3-Q8_0.gguf` | Path to GGUF model file |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | llama-server endpoint |
| `PORT` | `80` | Internal port |
| `CUDA_VISIBLE_DEVICES` | - | GPU devices to use |

### Model Details

- **Model**: Jina Reranker v3 (Q8_0 quantized)
- **Size**: ~680MB
- **Context**: 131,072 tokens
- **Batch Size**: 2048
- **Source**: [jinaai/jina-reranker-v3-GGUF](https://huggingface.co/jinaai/jina-reranker-v3-GGUF)

## API Endpoints

### `POST /rerank`

Rerank documents by relevance to query.

**Request:**
```json
{
  "query": "string",
  "documents": ["doc1", "doc2"],
  "top_n": 10,
  "return_documents": true
}
```

**Response:**
```json
{
  "model": "local-rerank",
  "object": "list",
  "usage": {"total_tokens": 123},
  "results": [
    {
      "index": 0,
      "relevance_score": 7.74,
      "document": "doc1"
    }
  ]
}
```

### `GET /info`

Service information including model details and version.

## Features

- ✅ **Jina AI Compatible** - Drop-in replacement for Jina reranker API
- ✅ **GPU Accelerated** - Offloads computation to GPU for speed
- ✅ **CPU Fallback** - Automatically uses CPU if GPU unavailable
- ✅ **Large Context** - Supports up to 131K tokens
- ✅ **Persistent Model** - Model stays loaded in memory for fast responses
- ✅ **Multi-platform** - Supports amd64 and arm64

## GPU Requirements

- NVIDIA GPU with CUDA support recommended
- Runs on CPU if GPU unavailable (slower)
- Configure GPU visibility with `CUDA_VISIBLE_DEVICES`

## Get Hugging Face Token

1. Sign up at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create new access token

## Building from Source

```bash
# Build the Docker image
docker build -t local-rerank:latest .

# Run with docker compose
docker compose up -d
```

## GitHub Actions Setup

To enable automated builds to Docker Hub:

1. **Create Docker Hub repository**: `your-username/local-rerank`

2. **Add GitHub Secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add the following secrets:
     - `DOCKERHUB_USERNAME`: Your Docker Hub username
     - `DOCKERHUB_TOKEN`: Docker Hub access token
     - `HF_TOKEN`: Hugging Face access token

3. **Update workflow file**:
   - Edit `.github/workflows/docker-publish.yml`
   - Change `DOCKER_IMAGE: lukolszewski/local-rerank` to your repository

4. **Push changes**:
   - Workflow triggers on push to `main`/`master` or version tags (`v*.*.*`)
   - Only builds when relevant files change (main.py, Dockerfile, etc.)

## License

MIT
