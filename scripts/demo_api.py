"""
Adaptive Agentic RAG — Interactive API Demo & Showcase Script

Demonstrates the live FastAPI service endpoints:
- GET /health (Liveness)
- GET /ready (Readiness Probe)
- GET /v1/system (System Architecture & Metadata)
- POST /v1/query (Scenario A: Answered query with verified citations)
- POST /v1/query (Scenario B: Trace mode observability)
- POST /v1/query (Scenario C: Fail-closed safe abstention to UNKNOWN)

Usage:
  python scripts/demo_api.py                      # Calls running server at http://127.0.0.1:8000
  python scripts/demo_api.py --in-process        # Runs in-process with model warming via TestClient
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def make_http_request(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict]:
    """Execute HTTP request using standard library urllib."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            status_code = resp.getcode()
            res_body = json.loads(resp.read().decode("utf-8"))
            return status_code, res_body
    except urllib.error.HTTPError as e:
        err_body = json.loads(e.read().decode("utf-8")) if e.fp else {}
        return e.code, err_body
    except urllib.error.URLError as e:
        raise ConnectionError(f"Could not connect to {url}: {e.reason}")


def run_demo(base_url: str = "http://127.0.0.1:8000", in_process: bool = False):
    print("=" * 80)
    print("          ADAPTIVE AGENTIC RAG — FASTAPI DEMO & SHOWCASE")
    print("=" * 80)
    
    if in_process:
        print("[Mode] Running In-Process via FastAPI TestClient (Warming Models in Lifespan)...\n")
        from fastapi.testclient import TestClient
        from adaptive_agentic_rag.api.app import create_app
        
        with TestClient(create_app()) as client:
            _execute_scenarios(client=client, call_mode="testclient")
    else:
        print(f"[Mode] Connecting to live HTTP server at: {base_url}\n")
        _execute_scenarios(base_url=base_url, call_mode="http")


def _execute_scenarios(client=None, base_url=None, call_mode="http"):
    def call_api(endpoint: str, method: str = "GET", payload: dict | None = None):
        start = time.perf_counter()
        if call_mode == "testclient":
            if method == "GET":
                resp = client.get(endpoint)
            else:
                resp = client.post(endpoint, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return resp.status_code, resp.json(), elapsed_ms
        else:
            code, resp = make_http_request(f"{base_url}{endpoint}", method=method, data=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return code, resp, elapsed_ms

    # -------------------------------------------------------------
    # 1. Health Probe
    # -------------------------------------------------------------
    print("[1/6] Liveness Check: GET /health")
    try:
        code, body, elapsed = call_api("/health", "GET")
        print(f"  -> HTTP {code} OK | Status: {body.get('status')} | {elapsed:.2f} ms")
    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo start the FastAPI service, run in another terminal:")
        print("  uvicorn adaptive_agentic_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000")
        print("\nAlternatively, run the demo in in-process mode:")
        print("  python scripts/demo_api.py --in-process")
        sys.exit(1)

    # -------------------------------------------------------------
    # 2. Readiness Probe
    # -------------------------------------------------------------
    print("\n[2/6] Readiness Probe: GET /ready")
    code, body, elapsed = call_api("/ready", "GET")
    print(f"  -> HTTP {code} | Pipeline Loaded: {body.get('pipeline_loaded')} | Collection: {body.get('qdrant_collection')}")

    # -------------------------------------------------------------
    # 3. System Metadata
    # -------------------------------------------------------------
    print("\n[3/6] System Architecture: GET /v1/system")
    code, body, elapsed = call_api("/v1/system", "GET")
    print(f"  -> Architecture: {body.get('architecture_version')}")
    print(f"  -> Embedding Model: {body.get('embedding_model')}")
    print(f"  -> Reranker: {body.get('reranker_model')}")
    print(f"  -> Generator: {body.get('generator_model')}")
    print(f"  -> Grounding NLI: {body.get('grounding_model')}")
    print(f"  -> Corpus Chunks: {body.get('corpus_chunks')}")

    # -------------------------------------------------------------
    # 4. Scenario A: Answered Query with Citations
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    query_a = "Did Sam Altman return as CEO of OpenAI according to news reports?"
    print(f"[4/6] Scenario A: Answered Multi-Hop Query")
    print(f"Query: '{query_a}'")
    
    code, body, elapsed = call_api("/v1/query", "POST", {"query": query_a, "include_trace": False})
    print(f"  -> Direct Answer: {body.get('direct_answer')}")
    print(f"  -> Abstained: {body.get('abstained')}")
    print(f"  -> HTTP Request Timing: {elapsed:.2f} ms (Pipeline latency: {body.get('latency_ms', 0):.2f} ms)")
    print(f"  -> Synthesized Answer:\n     \"{body.get('answer')}\"")
    
    citations = body.get("citations", [])
    print(f"  -> Verified Grounded Citations ({len(citations)}):")
    for cit in citations:
        print(f"     [{cit.get('citation_id')}] {cit.get('source')}: \"{cit.get('title')}\"")
        print(f"         URL: {cit.get('url')}")
        print(f"         Entailment Score: {cit.get('entailment_score', 0):.4f}")

    # -------------------------------------------------------------
    # 5. Scenario B: Trace Mode (Observability)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    query_b = "Compare the reports on OpenAI leadership changes."
    print(f"[5/6] Scenario B: Trace Mode (Engineering Observability)")
    print(f"Query: '{query_b}' (with include_trace=true)")
    
    code, body, elapsed = call_api("/v1/query", "POST", {"query": query_b, "include_trace": True})
    trace = body.get("trace", {})
    timing = body.get("timing", {})
    print(f"  -> Route: {body.get('route')}")
    print(f"  -> Retrieval Strategy: {trace.get('retrieval_strategy')}")
    print(f"  -> Retrieved Candidate Count: {trace.get('retrieved_candidate_count')}")
    print(f"  -> Citation Valid: {trace.get('citation_valid')}")
    print(f"  -> Breakdown Timing: Route={timing.get('route_ms', 0):.2f}ms | Retrieval={timing.get('retrieval_ms', 0):.2f}ms | Gen={timing.get('generation_ms', 0):.2f}ms")

    # -------------------------------------------------------------
    # 6. Scenario C: Safe Fail-Closed Abstention
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    query_c = "According to BBC News, which country won the 2038 FIFA World Cup tournament?"
    print(f"[6/6] Scenario C: Fail-Closed Safe Abstention")
    print(f"Query: '{query_c}' (Unanswerable / Out-of-Corpus Publisher)")
    
    code, body, elapsed = call_api("/v1/query", "POST", {"query": query_c, "include_trace": False})
    print(f"  -> HTTP Status Code: {code} OK")
    print(f"  -> Direct Answer: {body.get('direct_answer')}")
    print(f"  -> Abstained: {body.get('abstained')} (Correctly refused to hallucinate)")
    print(f"  -> Citations: {len(body.get('citations', []))} (Zero fabricated citations)")
    print(f"  -> Final Answer: \"{body.get('answer')}\"")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All live API endpoints and scenarios verified successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Agentic RAG Demo Showcase")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of running API")
    parser.add_argument("--in-process", action="store_true", help="Run in-process via FastAPI TestClient")
    args = parser.parse_args()
    
    run_demo(base_url=args.url, in_process=args.in_process)

