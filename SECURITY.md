# Security Policy

## Supported Versions

| Version | Supported | Notes |
| :--- | :---: | :--- |
| **1.0.x** | ✅ | Stable release series |
| **< 1.0** | ❌ | Pre-release iterations (unsupported) |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability in this project, please **do not open a public issue**. Public disclosure before mitigation can expose users and systems to risk.

Instead, report security concerns through GitHub's private security reporting:
- Go to the repository's **Security** tab.
- Click **Report a vulnerability** to open an advisory draft.
- Alternatively, use GitHub's private advisory mechanism if enabled on your fork.

---

## What to Include in a Report

To help assess and resolve the issue efficiently, please provide:
1. **Description**: Clear summary of the vulnerability and its potential impact.
2. **Affected Component**: Specific module, API route, or dependency version (e.g., `src/adaptive_agentic_rag/api/routes/query.py`).
3. **Environment Details**: Python version, operating system, and installed package versions (`uv.lock`).
4. **Reproduction Steps**: Step-by-step instructions or a minimal proof of concept (PoC).
5. **Suggested Fix / Mitigation**: If known, recommended remediations or patch proposals.

---

## Security Scope

Security reports are actively evaluated for:
- **API Security**: Input validation flaws, deserialization vulnerabilities, or unhandled exceptions leading to denial of service.
- **Resource Exhaustion**: Denial-of-service (DoS) vectors targeting vector search or inference concurrency.
- **Dependency Vulnerabilities**: Critical CVEs in upstream libraries (`fastapi`, `pydantic`, `qdrant-client`, `transformers`).
- **Data & Credential Exposure**: Accidental credential leaks, insecure default configurations, or path traversal during data ingestion.
- **Adversarial Injection**: Prompt injection patterns that achieve unauthorized system execution or bypass core safety invariants.

### Out of Scope (Quality & Research Issues)
The following are treated as modeling, benchmark, or engineering trade-offs rather than security vulnerabilities:
- Incorrect or incomplete model answers on multi-hop questions.
- Retrieval omissions or low candidate recall on ambiguous search queries.
- Conservative false abstentions where the system safely returns `UNKNOWN`.
- Differences in benchmark scores compared to external datasets.

---

## Responsible Disclosure

We appreciate the efforts of security researchers in improving the security of open-source software. Verified reports will be investigated and addressed in future patch releases.

