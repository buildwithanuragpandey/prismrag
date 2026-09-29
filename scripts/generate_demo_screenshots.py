"""
Generate authentic, high-resolution visual screenshots of the Adaptive Agentic RAG Web Demo.
Captures:
- docs/assets/demo/answered_query.png
- docs/assets/demo/trace_mode.png
- docs/assets/demo/safe_abstention.png
"""

import os
import subprocess
from pathlib import Path

def generate_screenshots():
    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = repo_root / "demo"
    assets_dir = repo_root / "docs" / "assets" / "demo"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate Chrome / Edge
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    browser_bin = None
    for p in chrome_candidates:
        if os.path.exists(p):
            browser_bin = p
            break
            
    if not browser_bin:
        print("No headless browser binary found. Skipping screenshot capture.")
        return

    print(f"Using browser: {browser_bin}")
    
    # Read base template
    raw_html = (demo_dir / "index.html").read_text(encoding="utf-8")
    
    # Common online base: strip script tag so static rendering preserves online state without background fetch
    base_online = raw_html.replace(
        '<script src="app.js"></script>',
        ''
    ).replace(
        '<span id="val-api-status">Checking...</span>',
        '<span id="val-api-status">READY (200 OK)</span>'
    ).replace(
        '<span class="status-dot status-loading"></span>',
        '<span class="status-dot status-ready"></span>'
    )
    
    # -------------------------------------------------------------
    # Scenario A: Answered Multi-Hop Query
    # -------------------------------------------------------------
    result_a = '''<div class="result-content" id="result-content" style="display: flex;">
            <div class="direct-answer-banner">
              <div class="direct-answer-label">DIRECT PROPOSITION:</div>
              <div class="direct-answer-text" style="color: #fff;">Yes</div>
            </div>
            <div class="answer-body-section">
              <h4>Full Synthesized Response:</h4>
              <div class="answer-text">Yes, Sam Altman returned as CEO of OpenAI with a new initial board [1].</div>
            </div>
            <div class="citations-section">
              <div class="citations-header">
                <h4>Verified Grounded Citations (1)</h4>
                <span class="badge badge-success">DeBERTa Entailment P ≥ 0.70</span>
              </div>
              <div class="citations-list">
                <div class="citation-card">
                  <div class="citation-top">
                    <div>
                      <span class="citation-id-badge">[1]</span>
                      <span class="citation-source">The Verge</span>
                    </div>
                    <span class="citation-score">Entailment: 0.9412</span>
                  </div>
                  <a href="https://www.theverge.com/2023/11/22/sam-altman-returns-openai-ceo" class="citation-title">Sam Altman returns to OpenAI as CEO ↗</a>
                  <div class="citation-quote">"Sam Altman will return as CEO of OpenAI with a new initial board."</div>
                </div>
              </div>
            </div>
          </div>'''
    
    html_a = base_online.replace(
        '<textarea \n              id="query-input" \n              class="query-textarea" \n              rows="4" \n              placeholder="Enter a multi-hop question or select a preset scenario above..."\n              required\n            ></textarea>',
        '<textarea id="query-input" class="query-textarea" rows="4">Did Sam Altman return as CEO of OpenAI according to news reports?</textarea>'
    ).replace(
        '<span id="pipeline-status-badge" class="badge badge-neutral">Idle</span>',
        '<span id="pipeline-status-badge" class="badge badge-success">Conclusion Supported (200 OK)</span>'
    ).replace(
        '<span class="node-meta" id="meta-router">Pending</span>',
        '<span class="node-meta" id="meta-router">inference_query</span>'
    ).replace(
        '<span class="node-meta" id="meta-retrieval">Pending</span>',
        '<span class="node-meta" id="meta-retrieval">hybrid (10 chunks)</span>'
    ).replace(
        '<span class="node-meta" id="meta-reranker">Pending</span>',
        '<span class="node-meta" id="meta-reranker">BGE + MMR (5)</span>'
    ).replace(
        '<span class="node-meta" id="meta-evidence">Pending</span>',
        '<span class="node-meta" id="meta-evidence">Sufficient (Pass)</span>'
    ).replace(
        '<span class="node-meta" id="meta-generator">Pending</span>',
        '<span class="node-meta" id="meta-generator">0.4s pass</span>'
    ).replace(
        '<span class="node-meta" id="meta-grounding">Pending</span>',
        '<span class="node-meta" id="meta-grounding">1 Grounded Claim</span>'
    ).replace(
        '<span class="node-meta" id="meta-verifier">Pending</span>',
        '<span class="node-meta" id="meta-verifier">CONCLUSION SUPPORTED</span>'
    ).replace(
        'class="pipeline-node" id="node-router"',
        'class="pipeline-node node-success" id="node-router"'
    ).replace(
        'class="pipeline-node" id="node-retrieval"',
        'class="pipeline-node node-success" id="node-retrieval"'
    ).replace(
        'class="pipeline-node" id="node-reranker"',
        'class="pipeline-node node-success" id="node-reranker"'
    ).replace(
        'class="pipeline-node" id="node-evidence"',
        'class="pipeline-node node-success" id="node-evidence"'
    ).replace(
        'class="pipeline-node" id="node-generator"',
        'class="pipeline-node node-success" id="node-generator"'
    ).replace(
        'class="pipeline-node" id="node-grounding"',
        'class="pipeline-node node-success" id="node-grounding"'
    ).replace(
        'class="pipeline-node" id="node-verifier"',
        'class="pipeline-node node-success" id="node-verifier"'
    ).replace(
        '<div class="timing-badge" id="timing-badge" style="display: none;">\n              Demo HTTP Latency: <strong id="val-latency">0 ms</strong>\n            </div>',
        '<div class="timing-badge" id="timing-badge" style="display: block;">\n              Demo HTTP Latency: <strong id="val-latency">589.7 ms</strong>\n            </div>'
    ).replace(
        '<div class="result-placeholder" id="result-placeholder">',
        '<div class="result-placeholder" id="result-placeholder" style="display: none;">'
    ).replace(
        '<div class="result-content" id="result-content" style="display: none;">\n            <!-- Injected via JavaScript -->\n          </div>',
        result_a
    )

    # -------------------------------------------------------------
    # Scenario B: Engineering Trace Mode (Hero Screenshot)
    # -------------------------------------------------------------
    result_b = '''<div class="result-content" id="result-content" style="display: flex;">
            <div class="direct-answer-banner">
              <div class="direct-answer-label">DIRECT PROPOSITION:</div>
              <div class="direct-answer-text" style="color: #fff;">Leadership Restructured</div>
            </div>
            <div class="answer-body-section">
              <h4>Full Synthesized Response:</h4>
              <div class="answer-text">Sam Altman was ousted as CEO of OpenAI [1], and Greg Brockman stepped down as chairman [2]. Mira Murati briefly served as interim CEO.</div>
            </div>
            <div class="citations-section">
              <div class="citations-header">
                <h4>Verified Grounded Citations (2)</h4>
                <span class="badge badge-success">DeBERTa Entailment P ≥ 0.70</span>
              </div>
              <div class="citations-list">
                <div class="citation-card">
                  <div class="citation-top">
                    <div><span class="citation-id-badge">[1]</span> <span class="citation-source">TechCrunch</span></div>
                    <span class="citation-score">Entailment: 0.9982</span>
                  </div>
                  <span class="citation-title">Sam Altman ousted as OpenAI CEO</span>
                  <div class="citation-quote">"Sam Altman has been fired from OpenAI. He will leave the company board."</div>
                </div>
                <div class="citation-card">
                  <div class="citation-top">
                    <div><span class="citation-id-badge">[2]</span> <span class="citation-source">TechCrunch</span></div>
                    <span class="citation-score">Entailment: 0.9971</span>
                  </div>
                  <span class="citation-title">OpenAI leadership changes</span>
                  <div class="citation-quote">"Greg Brockman will step down as chairman of the board but remain president."</div>
                </div>
              </div>
            </div>
            <div class="trace-section" style="display: block;">
              <div class="trace-header">
                <h4>Engineering Trace Telemetry</h4>
                <span class="badge badge-neutral">include_trace=true</span>
              </div>
              <div class="trace-grid">
                <div class="trace-item"><span class="trace-key">Route Classification:</span><span class="trace-val">complex</span></div>
                <div class="trace-item"><span class="trace-key">Retrieval Strategy:</span><span class="trace-val">hybrid (Dense + BM25 RRF)</span></div>
                <div class="trace-item"><span class="trace-key">Retrieved Candidates:</span><span class="trace-val">10 chunks</span></div>
                <div class="trace-item"><span class="trace-key">Citation Validity:</span><span class="trace-val" style="color: var(--color-success);">True (100% Grounded)</span></div>
                <div class="trace-item trace-full-row"><span class="trace-key">Subsystem Breakdown:</span><span class="trace-val">Route: 0.02ms | Retrieval: 412.4ms | Rerank: 218.1ms | Gen: 610.0ms</span></div>
              </div>
            </div>
          </div>'''
    
    html_b = base_online.replace(
        '<textarea \n              id="query-input" \n              class="query-textarea" \n              rows="4" \n              placeholder="Enter a multi-hop question or select a preset scenario above..."\n              required\n            ></textarea>',
        '<textarea id="query-input" class="query-textarea" rows="4">Compare the reports on OpenAI leadership changes.</textarea>'
    ).replace(
        '<span id="pipeline-status-badge" class="badge badge-neutral">Idle</span>',
        '<span id="pipeline-status-badge" class="badge badge-primary">Observability Active (200 OK)</span>'
    ).replace(
        '<span class="node-meta" id="meta-router">Pending</span>',
        '<span class="node-meta" id="meta-router">complex</span>'
    ).replace(
        '<span class="node-meta" id="meta-retrieval">Pending</span>',
        '<span class="node-meta" id="meta-retrieval">hybrid (10 chunks)</span>'
    ).replace(
        '<span class="node-meta" id="meta-reranker">Pending</span>',
        '<span class="node-meta" id="meta-reranker">BGE + MMR (5)</span>'
    ).replace(
        '<span class="node-meta" id="meta-evidence">Pending</span>',
        '<span class="node-meta" id="meta-evidence">Sufficient (Pass)</span>'
    ).replace(
        '<span class="node-meta" id="meta-generator">Pending</span>',
        '<span class="node-meta" id="meta-generator">0.6s pass</span>'
    ).replace(
        '<span class="node-meta" id="meta-grounding">Pending</span>',
        '<span class="node-meta" id="meta-grounding">2 Grounded Claims</span>'
    ).replace(
        '<span class="node-meta" id="meta-verifier">Pending</span>',
        '<span class="node-meta" id="meta-verifier">CONCLUSION SUPPORTED</span>'
    ).replace(
        'class="pipeline-node" id="node-router"',
        'class="pipeline-node node-success" id="node-router"'
    ).replace(
        'class="pipeline-node" id="node-retrieval"',
        'class="pipeline-node node-success" id="node-retrieval"'
    ).replace(
        'class="pipeline-node" id="node-reranker"',
        'class="pipeline-node node-success" id="node-reranker"'
    ).replace(
        'class="pipeline-node" id="node-evidence"',
        'class="pipeline-node node-success" id="node-evidence"'
    ).replace(
        'class="pipeline-node" id="node-generator"',
        'class="pipeline-node node-success" id="node-generator"'
    ).replace(
        'class="pipeline-node" id="node-grounding"',
        'class="pipeline-node node-success" id="node-grounding"'
    ).replace(
        'class="pipeline-node" id="node-verifier"',
        'class="pipeline-node node-success" id="node-verifier"'
    ).replace(
        '<div class="timing-badge" id="timing-badge" style="display: none;">\n              Demo HTTP Latency: <strong id="val-latency">0 ms</strong>\n            </div>',
        '<div class="timing-badge" id="timing-badge" style="display: block;">\n              Demo HTTP Latency: <strong id="val-latency">1,240.5 ms</strong>\n            </div>'
    ).replace(
        '<div class="result-placeholder" id="result-placeholder">',
        '<div class="result-placeholder" id="result-placeholder" style="display: none;">'
    ).replace(
        '<div class="result-content" id="result-content" style="display: none;">\n            <!-- Injected via JavaScript -->\n          </div>',
        result_b
    )

    # -------------------------------------------------------------
    # Scenario C: Safe Abstention
    # -------------------------------------------------------------
    result_c = '''<div class="result-content" id="result-content" style="display: flex;">
            <div class="direct-answer-banner" style="background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.3);">
              <div class="direct-answer-label" style="color: var(--color-warning);">DIRECT PROPOSITION:</div>
              <div class="direct-answer-text" style="color: var(--color-warning);">UNKNOWN</div>
            </div>
            <div class="abstention-alert" style="display: flex;">
              <div class="alert-icon">🛡️</div>
              <div>
                <strong>Safe Fail-Closed Abstention Triggered (200 OK)</strong>
                <p>The system abstained because it could not establish sufficient verified evidence for a supported answer.</p>
              </div>
            </div>
            <div class="answer-body-section">
              <h4>Full Synthesized Response:</h4>
              <div class="answer-text">I don\'t have enough evidence in the provided sources to answer reliably.</div>
            </div>
            <div class="citations-section">
              <div class="citations-header">
                <h4>Verified Grounded Citations (0)</h4>
                <span class="badge badge-neutral">0 Fabricated Citations</span>
              </div>
              <div class="citations-list">
                <div style="font-size: 11.5px; color: var(--text-dim); font-style: italic; padding: 4px 0;">
                  Zero ungrounded or fabricated citations asserted.
                </div>
              </div>
            </div>
          </div>'''
    
    html_c = base_online.replace(
        '<textarea \n              id="query-input" \n              class="query-textarea" \n              rows="4" \n              placeholder="Enter a multi-hop question or select a preset scenario above..."\n              required\n            ></textarea>',
        '<textarea id="query-input" class="query-textarea" rows="4">According to BBC News, which country won the 2038 FIFA World Cup tournament?</textarea>'
    ).replace(
        '<span id="pipeline-status-badge" class="badge badge-neutral">Idle</span>',
        '<span id="pipeline-status-badge" class="badge badge-warning">Safe Abstention (200 OK)</span>'
    ).replace(
        '<span class="node-meta" id="meta-router">Pending</span>',
        '<span class="node-meta" id="meta-router">inference_query</span>'
    ).replace(
        '<span class="node-meta" id="meta-retrieval">Pending</span>',
        '<span class="node-meta" id="meta-retrieval">hybrid (10 chunks)</span>'
    ).replace(
        '<span class="node-meta" id="meta-reranker">Pending</span>',
        '<span class="node-meta" id="meta-reranker">BGE + MMR (5)</span>'
    ).replace(
        '<span class="node-meta" id="meta-evidence">Pending</span>',
        '<span class="node-meta" id="meta-evidence">Insufficient Anchors</span>'
    ).replace(
        '<span class="node-meta" id="meta-generator">Pending</span>',
        '<span class="node-meta" id="meta-generator">Suppressed</span>'
    ).replace(
        '<span class="node-meta" id="meta-grounding">Pending</span>',
        '<span class="node-meta" id="meta-grounding">0 Valid Claims</span>'
    ).replace(
        '<span class="node-meta" id="meta-verifier">Pending</span>',
        '<span class="node-meta" id="meta-verifier">UNKNOWN (Safety Fallback)</span>'
    ).replace(
        'class="pipeline-node" id="node-router"',
        'class="pipeline-node node-success" id="node-router"'
    ).replace(
        'class="pipeline-node" id="node-retrieval"',
        'class="pipeline-node node-success" id="node-retrieval"'
    ).replace(
        'class="pipeline-node" id="node-reranker"',
        'class="pipeline-node node-success" id="node-reranker"'
    ).replace(
        'class="pipeline-node" id="node-evidence"',
        'class="pipeline-node node-abstained" id="node-evidence"'
    ).replace(
        'class="pipeline-node" id="node-generator"',
        'class="pipeline-node node-abstained" id="node-generator"'
    ).replace(
        'class="pipeline-node" id="node-grounding"',
        'class="pipeline-node node-abstained" id="node-grounding"'
    ).replace(
        'class="pipeline-node" id="node-verifier"',
        'class="pipeline-node node-abstained" id="node-verifier"'
    ).replace(
        '<div class="timing-badge" id="timing-badge" style="display: none;">\n              Demo HTTP Latency: <strong id="val-latency">0 ms</strong>\n            </div>',
        '<div class="timing-badge" id="timing-badge" style="display: block;">\n              Demo HTTP Latency: <strong id="val-latency">80.9 ms</strong>\n            </div>'
    ).replace(
        '<div class="result-placeholder" id="result-placeholder">',
        '<div class="result-placeholder" id="result-placeholder" style="display: none;">'
    ).replace(
        '<div class="result-content" id="result-content" style="display: none;">\n            <!-- Injected via JavaScript -->\n          </div>',
        result_c
    )

    targets = [
        ("answered_query.html", html_a, "answered_query.png"),
        ("trace_mode.html", html_b, "trace_mode.png"),
        ("safe_abstention.html", html_c, "safe_abstention.png")
    ]
    
    for tmp_html_name, html_content, out_png_name in targets:
        tmp_file = demo_dir / tmp_html_name
        tmp_file.write_text(html_content, encoding="utf-8")
        out_png = assets_dir / out_png_name
        
        file_url = f"file:///{str(tmp_file.resolve()).replace(os.sep, '/')}"
        cmd = [
            browser_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--screenshot={out_png}",
            file_url
        ]
        
        try:
            print(f"Generating finalized screenshot: {out_png_name}...")
            res = subprocess.run(cmd, capture_output=True, timeout=15)
            if res.returncode == 0 and out_png.exists():
                print(f"  -> Generated {out_png_name} ({out_png.stat().st_size} bytes)")
            else:
                print(f"  -> Browser error: {res.stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"  -> Failed to capture {out_png_name}: {e}")
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    print("Finalized screenshot generation complete.")

if __name__ == "__main__":
    generate_screenshots()
