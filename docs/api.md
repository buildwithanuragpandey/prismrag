# Production REST API Documentation

- **Service**: Adaptive Agentic RAG Inference Service
- **Framework**: FastAPI + Uvicorn + Pydantic v2
- **Default Address**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Playground**: `http://127.0.0.1:8000/docs`
- **Specification Version**: OpenAPI 3.1.0

---

## 1. Overview & Lifespan Architecture

The API layer wraps the canonical **Adaptive Agentic RAG** pipeline (`AdaptiveRAGGraph`) into a production-grade asynchronous web service under [`src/adaptive_agentic_rag/api/`](../src/adaptive_agentic_rag/api/).

### Lifespan Model Management
- **Startup**: On ASGI application startup (`lifespan` context in [`app.py`](../src/adaptive_agentic_rag/api/app.py)), `RAGService` loads all transformer models (`Qwen3-Embedding-0.6B`, `BAAI/bge-reranker-base`, `Qwen2.5-1.5B-Instruct`, `cross-encoder/nli-deberta-v3-small`) and initializes the local Qdrant vector store (`multihop_chunks_v2`).
- **Shutdown**: Gracefully closes vector database connections and releases GPU memory.

### Concurrency & Threadpool Model
- **GPU Inference Guard**: PyTorch model inference is guarded by `asyncio.Semaphore(1)` within each worker process, serializing GPU execution to prevent Out-Of-Memory (OOM) errors during concurrent bursts.
- **Worker Thread Offloading**: CPU-intensive vector calculations and model calls run in background worker threads via `anyio.to_thread.run_sync`, keeping the FastAPI async event loop unblocked.

---

## 2. API Endpoints

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Lightweight liveness probe for orchestrators. | `200 OK` |
| `GET` | `/ready` | Readiness probe confirming pipeline initialization and model warmth. | `200 OK`, `503 Service Unavailable` |
| `GET` | `/v1/system` | System metadata, model versions, device info, and corpus statistics. | `200 OK` |
| `POST` | `/v1/query` | Executes full multi-hop RAG query with citations and timing telemetry. | `200 OK`, `422`, `500`, `503` |

---

## 3. Detailed Endpoint Specifications

### 3.1 GET `/health`
Returns process health status without invoking models.

**Response `200 OK`:**
```json
{
  "status": "ok"
}
```

---

### 3.2 GET `/ready`
Confirms all models, vector database connections, and tokenizers are initialized in memory.

**Response `200 OK`:**
```json
{
  "status": "ready",
  "pipeline_loaded": true,
  "qdrant_collection": "multihop_chunks_v2",
  "models_initialized": true,
  "device": "cuda:NVIDIA GeForce RTX 4070 Laptop GPU"
}
```

**Response `503 Service Unavailable`:**
```json
{
  "detail": {
    "error_type": "PipelineUnavailableError",
    "message": "RAG pipeline service is not yet initialized.",
    "status_code": 503
  }
}
```

---

### 3.3 GET `/v1/system`
Returns architectural metadata, frozen component specifications, and hardware status.

**Response `200 OK`:**
```json
{
  "project": "Adaptive Agentic RAG with Hybrid Retrieval, Reranking & Self-Correction",
  "api_version": "1.0.0",
  "architecture_version": "V2-A (Frozen Canonical)",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "reranker_model": "BAAI/bge-reranker-base",
  "generator_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "grounding_model": "cross-encoder/nli-deberta-v3-small",
  "qdrant_collection": "multihop_chunks_v2",
  "corpus_file": "data/processed/processed_corpus_v2.json",
  "corpus_chunks": 8173,
  "device": "NVIDIA GeForce RTX 4070 Laptop GPU",
  "cuda_available": true,
  "python_version": "3.12.11",
  "torch_version": "2.6.0+cu124"
}
```

---

### 3.4 POST `/v1/query`
Executes end-to-end multi-hop RAG with evidence grading, adaptive retrieval, single-pass generation, and NLI grounding.

**Request Body Schema (`QueryRequest`):**
```json
{
  "query": "Did Sam Altman return as CEO of OpenAI according to news reports?",
  "request_id": "req_optional_custom_id",
  "include_trace": true,
  "max_retries": 1
}
```

| Field | Type | Required | Default | Description |
| :--- | :--- | :---: | :---: | :--- |
| `query` | `string` | **Yes** | — | The user question to answer (1 to 2,000 characters). |
| `request_id` | `string` | No | Auto UUID | Custom tracking identifier. |
| `include_trace`| `boolean`| No | `false` | When `true`, includes execution strategy, candidate count, and citation validity telemetry. |
| `max_retries` | `integer`| No | `1` | Maximum adaptive retry budget ($0 \le N \le 3$). |

**Response Body Schema (`QueryResponse`):**
```json
{
  "request_id": "req_8a7f29b4e1",
  "query": "Did Sam Altman return as CEO of OpenAI according to news reports?",
  "answer": "Yes, Sam Altman returned as CEO of OpenAI with a new initial board [1].",
  "direct_answer": "Yes",
  "abstained": false,
  "citations": [
    {
      "citation_id": 1,
      "document_id": "doc_0042_chunk_0",
      "source": "The Verge",
      "title": "Sam Altman returns to OpenAI as CEO",
      "url": "https://www.theverge.com/2023/11/22/sam-altman-returns-openai-ceo",
      "supporting_text": "Sam Altman will return as CEO of OpenAI with a new initial board.",
      "entailment_score": 0.9412
    }
  ],
  "sources": [
    {
      "document_id": "doc_0042_chunk_0",
      "source": "The Verge",
      "title": "Sam Altman returns to OpenAI as CEO"
    }
  ],
  "route": "inference_query",
  "retry": {
    "attempted": false,
    "rescued": false,
    "target_sources": []
  },
  "latency_ms": 589.67,
  "timing": {
    "total_ms": 589.67,
    "route_ms": 8.20,
    "retrieval_ms": 140.50,
    "generation_ms": 440.97
  },
  "trace": {
    "retrieval_strategy": "hybrid",
    "retrieved_candidate_count": 10,
    "citation_valid": true
  }
}
```

---

## 4. Abstention Contract & Safe Fail-Closed Behavior

When a query cannot be safely verified by the evidence gates or NLI grounding, the API returns `200 OK` with:
- `direct_answer: "UNKNOWN"`
- `abstained: true`
- `citations: []`

**Example Safe Abstention Response (`200 OK`):**
```json
{
  "request_id": "req_99b1c2d3e4",
  "query": "Which country won the 2038 FIFA World Cup according to news reports?",
  "answer": "UNKNOWN",
  "direct_answer": "UNKNOWN",
  "abstained": true,
  "citations": [],
  "sources": [],
  "route": "simple",
  "retry": {
    "attempted": false,
    "rescued": false,
    "target_sources": []
  },
  "latency_ms": 169.35,
  "timing": {
    "total_ms": 169.35,
    "route_ms": 4.10,
    "retrieval_ms": 165.25,
    "generation_ms": 0.00
  },
  "trace": {
    "retrieval_strategy": "dense",
    "retrieved_candidate_count": 10,
    "citation_valid": true
  }
}
```

---

## 5. Client Integration Examples

### PowerShell Example
```powershell
$body = @{
    query = "Did Sam Altman return as CEO of OpenAI according to news reports?"
    include_trace = $true
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/v1/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$response | Format-List
```

### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Did Sam Altman return as CEO of OpenAI according to news reports?",
       "include_trace": true
     }'
```

### Python `httpx` Example
```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
    response = client.post(
        "/v1/query",
        json={"query": "Did Sam Altman return as CEO of OpenAI?", "include_trace": True}
    )
    data = response.json()
    print(f"Answer: {data['answer']}")
    print(f"Citations: {len(data['citations'])} items")
```

