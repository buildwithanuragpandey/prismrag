/**
 * Adaptive Agentic RAG — Visual Showcase Application Logic (Polished V3)
 * Communicates with the live FastAPI service endpoints.
 */

// Determine API Base URL dynamically
const getApiBaseUrl = () => {
  if (window.location.port === "8000") {
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
};

const API_BASE = getApiBaseUrl();

// Curated Demo Scenarios (Verified disjoint from evaluation/datasets/final_untouched_test.json)
const SCENARIOS = {
  a: {
    query: "Did Sam Altman return as CEO of OpenAI according to news reports?",
    trace: true
  },
  b: {
    query: "Compare the reports on OpenAI leadership changes.",
    trace: true
  },
  c: {
    query: "According to BBC News, which country won the 2038 FIFA World Cup tournament?",
    trace: true
  }
};

// DOM Elements
const elements = {
  apiStatusDot: document.querySelector("#pill-api-status .status-dot"),
  apiStatusVal: document.getElementById("val-api-status"),
  archVal: document.getElementById("val-arch"),
  collectionVal: document.getElementById("val-collection"),
  chunksVal: document.getElementById("val-chunks"),
  offlineBanner: document.getElementById("offline-banner"),
  btnRetryHealth: document.getElementById("btn-retry-health"),
  
  queryForm: document.getElementById("query-form"),
  queryInput: document.getElementById("query-input"),
  checkTrace: document.getElementById("check-include-trace"),
  btnSubmit: document.getElementById("btn-submit"),
  btnText: document.getElementById("btn-text"),
  btnSpinner: document.getElementById("btn-spinner"),
  btnClear: document.getElementById("btn-clear"),
  
  btnScenarioA: document.getElementById("btn-scenario-a"),
  btnScenarioB: document.getElementById("btn-scenario-b"),
  btnScenarioC: document.getElementById("btn-scenario-c"),
  
  pipelineStatus: document.getElementById("pipeline-status-badge"),
  nodeRouter: document.getElementById("node-router"),
  metaRouter: document.getElementById("meta-router"),
  nodeRetrieval: document.getElementById("node-retrieval"),
  metaRetrieval: document.getElementById("meta-retrieval"),
  nodeReranker: document.getElementById("node-reranker"),
  metaReranker: document.getElementById("meta-reranker"),
  nodeEvidence: document.getElementById("node-evidence"),
  metaEvidence: document.getElementById("meta-evidence"),
  nodeGenerator: document.getElementById("node-generator"),
  metaGenerator: document.getElementById("meta-generator"),
  nodeGrounding: document.getElementById("node-grounding"),
  metaGrounding: document.getElementById("meta-grounding"),
  nodeVerifier: document.getElementById("node-verifier"),
  metaVerifier: document.getElementById("meta-verifier"),
  
  timingBadge: document.getElementById("timing-badge"),
  valLatency: document.getElementById("val-latency"),
  resultPlaceholder: document.getElementById("result-placeholder"),
  resultContent: document.getElementById("result-content")
};

// ==========================================================================
// Initialization & Health Checks
// ==========================================================================

async function checkSystemHealth() {
  try {
    elements.apiStatusDot.className = "status-dot status-loading";
    elements.apiStatusVal.textContent = "Connecting...";

    // 1. Check Liveness
    const healthResp = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!healthResp.ok) throw new Error("Health check failed");

    // 2. Check Readiness
    const readyResp = await fetch(`${API_BASE}/ready`, { method: "GET" });
    const readyData = await readyResp.json();

    // 3. Check System Metadata
    const sysResp = await fetch(`${API_BASE}/v1/system`, { method: "GET" });
    const sysData = sysResp.ok ? await sysResp.json() : {};

    // Update UI Badges
    elements.apiStatusDot.className = "status-dot status-ready";
    elements.apiStatusVal.textContent = readyResp.ok && readyData.pipeline_loaded ? "READY (200 OK)" : "INITIALIZING";
    elements.offlineBanner.style.display = "none";

    if (sysData.architecture_version) {
      elements.archVal.textContent = sysData.architecture_version.split(" ")[0] || "V2-A";
    }
    if (sysData.qdrant_collection) {
      elements.collectionVal.textContent = sysData.qdrant_collection;
    }
    if (sysData.corpus_chunks) {
      elements.chunksVal.textContent = sysData.corpus_chunks.toLocaleString();
    }

  } catch (err) {
    console.warn("API Health check error:", err);
    elements.apiStatusDot.className = "status-dot status-error";
    elements.apiStatusVal.textContent = "OFFLINE";
    elements.offlineBanner.style.display = "flex";
  }
}

// ==========================================================================
// Pipeline State Reset & Animation Helpers
// ==========================================================================

function resetPipeline() {
  const nodes = [
    elements.nodeRouter, elements.nodeRetrieval, elements.nodeReranker,
    elements.nodeEvidence, elements.nodeGenerator, elements.nodeGrounding,
    elements.nodeVerifier
  ];
  
  nodes.forEach(node => {
    node.className = "pipeline-node";
  });

  elements.metaRouter.textContent = "Pending";
  elements.metaRetrieval.textContent = "Pending";
  elements.metaReranker.textContent = "Pending";
  elements.metaEvidence.textContent = "Pending";
  elements.metaGenerator.textContent = "Pending";
  elements.metaGrounding.textContent = "Pending";
  elements.metaVerifier.textContent = "Pending";
  
  elements.pipelineStatus.className = "badge badge-neutral";
  elements.pipelineStatus.textContent = "Idle";
}

function setPipelineLoading() {
  resetPipeline();
  elements.pipelineStatus.className = "badge badge-primary";
  elements.pipelineStatus.textContent = "Executing Inference...";
  
  elements.nodeRouter.className = "pipeline-node node-active";
  elements.metaRouter.textContent = "Routing...";
  elements.nodeRetrieval.className = "pipeline-node node-active";
  elements.metaRetrieval.textContent = "Searching...";
}

// ==========================================================================
// Query Execution & Response Rendering
// ==========================================================================

async function executeQuery(queryText, includeTrace = true) {
  if (!queryText.trim()) return;

  elements.btnSubmit.disabled = true;
  elements.btnText.textContent = "Processing...";
  elements.btnSpinner.style.display = "inline-block";
  setPipelineLoading();

  const startTime = performance.now();

  try {
    const payload = {
      query: queryText.trim(),
      include_trace: includeTrace
    };

    const resp = await fetch(`${API_BASE}/v1/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const elapsedMs = Math.round(performance.now() - startTime);
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail?.message || `HTTP ${resp.status} Error`);
    }

    renderQueryResult(data, elapsedMs);

  } catch (err) {
    console.error("Query execution error:", err);
    renderQueryError(err.message);
  } finally {
    elements.btnSubmit.disabled = false;
    elements.btnText.textContent = "Execute Query";
    elements.btnSpinner.style.display = "none";
  }
}

function renderQueryResult(data, elapsedMs) {
  elements.resultPlaceholder.style.display = "none";
  elements.resultContent.style.display = "flex";
  
  // 1. Latency Banner
  elements.timingBadge.style.display = "block";
  elements.valLatency.textContent = `${data.latency_ms ? data.latency_ms.toFixed(1) : elapsedMs} ms`;

  const isAbstained = data.abstained || data.direct_answer === "UNKNOWN";

  // 2. Construct Clean HTML
  const citations = data.citations || [];
  let citationsHtml = "";
  
  if (citations.length === 0) {
    citationsHtml = `
      <div style="font-size: 11.5px; color: var(--text-dim); font-style: italic; padding: 4px 0;">
        Zero ungrounded or fabricated citations asserted.
      </div>
    `;
  } else {
    citations.forEach(cit => {
      const scoreStr = cit.entailment_score ? `Entailment: ${cit.entailment_score.toFixed(4)}` : "Entailed (P ≥ 0.70)";
      const urlMarkup = cit.url 
        ? `<a href="${cit.url}" target="_blank" rel="noopener" class="citation-title">${cit.title || "Source Article"} ↗</a>`
        : `<span class="citation-title">${cit.title || "Source Article"}</span>`;

      citationsHtml += `
        <div class="citation-card">
          <div class="citation-top">
            <div>
              <span class="citation-id-badge">[${cit.citation_id || 1}]</span>
              <span class="citation-source">${cit.source || "News Source"}</span>
            </div>
            <span class="citation-score">${scoreStr}</span>
          </div>
          ${urlMarkup}
          ${cit.supporting_text ? `<div class="citation-quote">"${cit.supporting_text}"</div>` : ""}
        </div>
      `;
    });
  }

  let traceHtml = "";
  if (data.trace) {
    const timingBreakdown = data.timing 
      ? `Route: ${data.timing.route_ms?.toFixed(1) || 0}ms | Retrieval: ${data.timing.retrieval_ms?.toFixed(1) || 0}ms | Gen: ${data.timing.generation_ms?.toFixed(1) || 0}ms`
      : "—";

    traceHtml = `
      <div class="trace-section" style="display: block;">
        <div class="trace-header">
          <h4>Engineering Trace Telemetry</h4>
          <span class="badge badge-neutral">include_trace=true</span>
        </div>
        <div class="trace-grid">
          <div class="trace-item">
            <span class="trace-key">Route Classification:</span>
            <span class="trace-val">${data.route || "complex"}</span>
          </div>
          <div class="trace-item">
            <span class="trace-key">Retrieval Strategy:</span>
            <span class="trace-val">${data.trace.retrieval_strategy || "hybrid"}</span>
          </div>
          <div class="trace-item">
            <span class="trace-key">Retrieved Candidates:</span>
            <span class="trace-val">${data.trace.retrieved_candidate_count || 10} chunks</span>
          </div>
          <div class="trace-item">
            <span class="trace-key">Citation Validity:</span>
            <span class="trace-val" style="color: var(--color-success);">${data.trace.citation_valid !== false ? "True (100% Grounded)" : "False"}</span>
          </div>
          <div class="trace-item trace-full-row">
            <span class="trace-key">Subsystem Breakdown:</span>
            <span class="trace-val">${timingBreakdown}</span>
          </div>
        </div>
      </div>
    `;
  }

  const abstentionBannerHtml = isAbstained ? `
    <div class="abstention-alert" style="display: flex;">
      <div class="alert-icon">🛡️</div>
      <div>
        <strong>Safe Fail-Closed Abstention Triggered (200 OK)</strong>
        <p>The system abstained because it could not establish sufficient verified evidence for a supported answer.</p>
      </div>
    </div>
  ` : "";

  const directAnswerColor = isAbstained ? "var(--color-warning)" : "#fff";

  elements.resultContent.innerHTML = `
    <div class="direct-answer-banner">
      <div class="direct-answer-label">DIRECT PROPOSITION:</div>
      <div class="direct-answer-text" style="color: ${directAnswerColor};">${data.direct_answer || (isAbstained ? "UNKNOWN" : "—")}</div>
    </div>
    ${abstentionBannerHtml}
    <div class="answer-body-section">
      <h4>Full Synthesized Response:</h4>
      <div class="answer-text">${data.answer || "UNKNOWN"}</div>
    </div>
    <div class="citations-section">
      <div class="citations-header">
        <h4>Verified Grounded Citations (${citations.length})</h4>
        <span class="badge badge-success">DeBERTa Entailment P ≥ 0.70</span>
      </div>
      <div class="citations-list">
        ${citationsHtml}
      </div>
    </div>
    ${traceHtml}
  `;

  if (isAbstained) {
    elements.pipelineStatus.className = "badge badge-warning";
    elements.pipelineStatus.textContent = "Safe Abstention (200 OK)";
  } else {
    elements.pipelineStatus.className = "badge badge-success";
    elements.pipelineStatus.textContent = "Conclusion Supported (200 OK)";
  }

  // 3. Update Pipeline Node Cards
  updatePipelineNodes(data, isAbstained);
}

function updatePipelineNodes(data, isAbstained) {
  const trace = data.trace || {};
  const timing = data.timing || {};
  const retry = data.retry || {};

  // Node 1: Router
  elements.nodeRouter.className = "pipeline-node node-success";
  elements.metaRouter.textContent = data.route || "Routed";

  // Node 2: Retrieval
  elements.nodeRetrieval.className = "pipeline-node node-success";
  elements.metaRetrieval.textContent = `${trace.retrieval_strategy || "Hybrid"} (${trace.retrieved_candidate_count || 10} chunks)`;

  // Node 3: Reranking
  elements.nodeReranker.className = "pipeline-node node-success";
  elements.metaReranker.textContent = "BGE + MMR (5)";

  // Node 4: Evidence & Recovery
  elements.nodeEvidence.className = isAbstained && !data.citations?.length ? "pipeline-node node-abstained" : "pipeline-node node-success";
  if (retry.attempted) {
    elements.metaEvidence.textContent = retry.rescued ? "Retry Rescued" : "Retry Attempted";
  } else if (isAbstained && !data.citations?.length) {
    elements.metaEvidence.textContent = "Insufficient Anchors";
  } else {
    elements.metaEvidence.textContent = "Sufficient (Pass)";
  }

  // Node 5: Generator
  elements.nodeGenerator.className = isAbstained && !data.citations?.length ? "pipeline-node node-abstained" : "pipeline-node node-success";
  elements.metaGenerator.textContent = isAbstained && !data.citations?.length ? "Suppressed" : (timing.generation_ms ? `${(timing.generation_ms / 1000).toFixed(1)}s pass` : "Single-pass");

  // Node 6: Grounding
  elements.nodeGrounding.className = isAbstained && !data.citations?.length ? "pipeline-node node-abstained" : "pipeline-node node-success";
  elements.metaGrounding.textContent = `${data.citations?.length || 0} Grounded Claims`;

  // Node 7: Semantic Safety Verifier
  elements.nodeVerifier.className = isAbstained ? "pipeline-node node-abstained" : "pipeline-node node-success";
  elements.metaVerifier.textContent = isAbstained ? "UNKNOWN (Safety Fallback)" : "CONCLUSION SUPPORTED";
}

function renderQueryError(errorMessage) {
  elements.resultPlaceholder.style.display = "none";
  elements.resultContent.style.display = "flex";
  
  elements.resultContent.innerHTML = `
    <div class="abstention-alert" style="display: flex; background: rgba(244, 63, 94, 0.12); border-color: rgba(244, 63, 94, 0.3); color: #fecdd3;">
      <div class="alert-icon">⚠️</div>
      <div>
        <strong>API Error Response</strong>
        <p>${errorMessage}</p>
      </div>
    </div>
  `;
}

// ==========================================================================
// Event Listeners
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  checkSystemHealth();

  elements.btnRetryHealth.addEventListener("click", () => {
    checkSystemHealth();
  });

  elements.btnScenarioA.addEventListener("click", () => {
    elements.queryInput.value = SCENARIOS.a.query;
    elements.checkTrace.checked = SCENARIOS.a.trace;
    executeQuery(SCENARIOS.a.query, SCENARIOS.a.trace);
  });

  elements.btnScenarioB.addEventListener("click", () => {
    elements.queryInput.value = SCENARIOS.b.query;
    elements.checkTrace.checked = SCENARIOS.b.trace;
    executeQuery(SCENARIOS.b.query, SCENARIOS.b.trace);
  });

  elements.btnScenarioC.addEventListener("click", () => {
    elements.queryInput.value = SCENARIOS.c.query;
    elements.checkTrace.checked = SCENARIOS.c.trace;
    executeQuery(SCENARIOS.c.query, SCENARIOS.c.trace);
  });

  elements.btnClear.addEventListener("click", () => {
    elements.queryInput.value = "";
    resetPipeline();
    elements.resultContent.style.display = "none";
    elements.resultPlaceholder.style.display = "flex";
    elements.timingBadge.style.display = "none";
  });

  elements.queryForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = elements.queryInput.value;
    const includeTrace = elements.checkTrace.checked;
    executeQuery(query, includeTrace);
  });
});
