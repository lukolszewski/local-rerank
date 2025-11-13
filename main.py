import os
import urllib.request
import urllib.error
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, conlist

##
# Load the config
##
MODEL_NAME = 'jina-reranker-v3'
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/jina-reranker-v3-Q8_0.gguf")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
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


##
# Create the FastAPI app
##
app = FastAPI(
    title="Jina Reranker v3 GGUF API",
    description=f"API for reranking documents based on query relevance using {MODEL_NAME} (GGUF quantized)",
    version=VERSION,
)


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
        # Call llama-server's /rerank endpoint
        payload = {
            "query": request.query,
            "documents": request.documents
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{LLAMA_SERVER_URL}/rerank",
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))

            # llama-server returns: {"results": [{"index": 0, "relevance_score": 7.74}, ...]}
            if 'results' not in result:
                raise ValueError(f"Unexpected response format from llama-server: {result}")

            results = result['results']

            # Sort by relevance_score descending
            results.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Apply top_n filter if specified
            if request.top_n is not None:
                results = results[:request.top_n]

            # Add document text if requested
            formatted_results = []
            for res in results:
                formatted_result = {
                    "index": res["index"],
                    "relevance_score": res["relevance_score"]
                }
                if request.return_documents:
                    formatted_result["document"] = request.documents[res["index"]]
                formatted_results.append(formatted_result)

            # Calculate total tokens (approximation)
            total_tokens = len(request.query.split()) + sum(len(doc.split()) for doc in request.documents)

            # Build response matching Jina AI API format
            response = RerankResponse(
                model=request.model or MODEL_NAME,
                object="list",
                usage={"total_tokens": total_tokens},
                results=[RerankResult(**r) for r in formatted_results]
            )

            return response

    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to llama-server at {LLAMA_SERVER_URL}: {e}"
        )
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error during reranking: {str(e)}. Traceback: {traceback.format_exc()}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
