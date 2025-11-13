import os
import subprocess
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, conlist

from rerank import GGUFReranker

##
# Load the config
##
MODEL_NAME = 'jina-reranker-v3'
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/jina-reranker-v3-Q8_0.gguf")
PROJECTOR_PATH = os.getenv("PROJECTOR_PATH", "/app/models/projector.safetensors")
LLAMA_EMBEDDING_PATH = os.getenv("LLAMA_EMBEDDING_PATH", "/app/llama-embedding")
VERSION = os.getenv("VERSION", "v3-gguf")
BUILD_ID = os.getenv("BUILD_ID", "unknown")
COMMIT_SHA = os.getenv("COMMIT_SHA", "unknown")
PORT = int(os.getenv("PORT", "80"))


##
# Models
##
class RerankRequest(BaseModel):
    query: str = Field(..., description="The search query")
    documents: conlist(str, min_length=1) = Field(..., description="List of documents to rerank")
    batch_size: int = Field(32, description="Batch size for the model (not used in GGUF implementation)")
    top_n: Optional[int] = Field(None, description="Number of top results to return")
    return_documents: bool = Field(True, description="Whether to return document text in results")
    model: Optional[str] = Field(None, description="Model name (for API compatibility)")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    model: str
    object: str = "list"
    usage: dict
    results: List[RerankResult]


class InfoResponse(BaseModel):
    model_name: str = MODEL_NAME
    version: str = VERSION
    build_id: str = BUILD_ID
    commit_sha: str = COMMIT_SHA
    model_path: str = MODEL_PATH
    projector_path: str = PROJECTOR_PATH


##
# Create the FastAPI app
##
app = FastAPI(
    title="Jina Reranker v3 GGUF API",
    description=f"API for reranking documents based on query relevance using {MODEL_NAME} (GGUF quantized)",
    version=VERSION,
)

##
# Load the model
##
try:
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Projector path: {PROJECTOR_PATH}")
    print(f"llama-cli path: {LLAMA_EMBEDDING_PATH}")

    reranker = GGUFReranker(
        model_path=MODEL_PATH,
        projector_path=PROJECTOR_PATH,
        llama_embedding_path=LLAMA_EMBEDDING_PATH
    )
    print(f"Model {MODEL_NAME} loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


##
# Routes
##
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/info", response_model=InfoResponse)
async def info():
    return InfoResponse()


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest = Body(...)):
    try:
        # Call the GGUF reranker
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            return_embeddings=False
        )

        # Convert to response format
        # The reranker already returns results sorted by relevance and filtered by top_n
        formatted_results = []
        for result in results:
            formatted_result = {
                "index": result["index"],
                "relevance_score": result["relevance_score"]
            }
            if request.return_documents:
                formatted_result["document"] = result["document"]
            formatted_results.append(formatted_result)

        # Calculate total tokens (approximation based on text length)
        total_tokens = len(request.query.split()) + sum(len(doc.split()) for doc in request.documents)

        # Build response matching Jina AI API format
        response = RerankResponse(
            model=request.model or MODEL_NAME,
            object="list",
            usage={"total_tokens": total_tokens},
            results=[RerankResult(**r) for r in formatted_results]
        )

        return response

    except subprocess.CalledProcessError as e_:
        stderr_output = e_.stderr if hasattr(e_, 'stderr') and e_.stderr else 'No stderr available'
        raise HTTPException(
            status_code=500,
            detail=f"Error during reranking: Command failed with exit code {e_.returncode}. Stderr: {stderr_output}"
        ) from e_
    except Exception as e_:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error during reranking: {str(e_)}. Traceback: {traceback.format_exc()}"
        ) from e_


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
