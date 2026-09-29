import time
from fastapi.testclient import TestClient
from adaptive_agentic_rag.api.app import create_app
from adaptive_agentic_rag.api.service import RAGService


def main():
    print("=" * 80)
    print("LIVE REAL API ENDPOINT VERIFICATION")
    print("=" * 80)

    print("\n1. Initializing real RAGService with live models...")
    t0 = time.perf_counter()
    service = RAGService()
    service.initialize()
    init_time = time.perf_counter() - t0
    print(f"   RAGService initialized in {init_time:.2f}s")

    app = create_app(service=service)

    with TestClient(app) as client:
        # 1. Health
        print("\n2. Testing GET /health...")
        r_health = client.get("/health")
        print(f"   Status: {r_health.status_code} | Body: {r_health.json()}")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}

        # 2. Readiness
        print("\n3. Testing GET /ready...")
        r_ready = client.get("/ready")
        print(f"   Status: {r_ready.status_code} | Body: {r_ready.json()}")
        assert r_ready.status_code == 200
        ready_data = r_ready.json()
        assert ready_data["status"] == "ready"
        assert ready_data["pipeline_loaded"] is True
        assert ready_data["qdrant_collection"] == "multihop_chunks_v2"

        # 3. System Info
        print("\n4. Testing GET /v1/system...")
        r_sys = client.get("/v1/system")
        print(f"   Status: {r_sys.status_code} | Body: {r_sys.json()}")
        assert r_sys.status_code == 200
        sys_data = r_sys.json()
        assert sys_data["architecture_version"] == "V2-A (Frozen Canonical)"
        assert sys_data["qdrant_collection"] == "multihop_chunks_v2"
        assert sys_data["corpus_chunks"] == 8173

        # 4. Real Query 1: Multi-Hop Query with trace
        print("\n5. Testing POST /v1/query (Real Multi-Hop Inference Query)...")
        q1 = "Did Sam Altman return as CEO of OpenAI according to news reports?"
        t_req_0 = time.perf_counter()
        r_q1 = client.post(
            "/v1/query",
            json={
                "query": q1,
                "request_id": "live_smoke_001",
                "include_trace": True,
            },
        )
        req_latency_ms = (time.perf_counter() - t_req_0) * 1000.0
        print(f"   Status: {r_q1.status_code}")
        assert r_q1.status_code == 200
        data1 = r_q1.json()
        print(f"   Request ID: {data1['request_id']}")
        print(f"   Direct Answer: {data1['direct_answer']}")
        print(f"   Abstained: {data1['abstained']}")
        print(f"   Citations: {len(data1['citations'])} items")
        print(f"   Route: {data1['route']}")
        print(f"   Pipeline Latency: {data1['latency_ms']:.2f}ms (HTTP Roundtrip: {req_latency_ms:.2f}ms)")
        if data1["trace"]:
            print(f"   Trace Strategy: {data1['trace']['retrieval_strategy']} | Candidates: {data1['trace']['retrieved_candidate_count']}")

        # 5. Real Query 2: Unanswerable Null Query
        print("\n6. Testing POST /v1/query (Real Unanswerable Query - Safe Abstention)...")
        q2 = "What was the exact price of quantum teleporters on Mars in 1842?"
        t_req_1 = time.perf_counter()
        r_q2 = client.post(
            "/v1/query",
            json={
                "query": q2,
                "request_id": "live_smoke_002",
                "include_trace": True,
            },
        )
        req_latency_null_ms = (time.perf_counter() - t_req_1) * 1000.0
        print(f"   Status: {r_q2.status_code}")
        assert r_q2.status_code == 200
        data2 = r_q2.json()
        print(f"   Direct Answer: {data2['direct_answer']}")
        print(f"   Abstained: {data2['abstained']}")
        print(f"   Pipeline Latency: {data2['latency_ms']:.2f}ms (HTTP Roundtrip: {req_latency_null_ms:.2f}ms)")
        assert data2["abstained"] is True
        assert data2["direct_answer"] == "UNKNOWN"

    service.close()
    print("\nALL LIVE REAL API ENDPOINT CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    main()
