# Problems Log

This file tracks notable problems encountered during the project, along with observed symptoms and fixes or mitigations.

## 2026-05-16

### QA showed `Internal Server Error` even when backend finished successfully

- Symptom:
  - browser logged `[qa] request_failed { error: 'Internal Server Error', ... }`
  - backend logs for the same `request_id` later showed `qa_request_completed`
  - affected requests were long-running QA calls around the 30s mark
- Observed cause:
  - frontend was sending requests through the Next.js `/api` rewrite proxy
  - the proxy/client layer could fail before the FastAPI backend completed, creating a false frontend error
- Mitigation:
  - in local development, frontend now defaults to calling FastAPI directly at `http://localhost:8000`
  - production still uses `/api` unless `NEXT_PUBLIC_API_BASE_URL` is set explicitly

### QA chunking can separate formulas, claims, and qualifiers from their explanations

- Symptom:
  - formula/equation questions may retrieve the equation line but miss the nearby explanation or variable definitions
  - result/method questions can surface a core claim without its qualifier, scope condition, or follow-up sentence
- Observed cause:
  - chunking is section-aware and sentence-packed, but still mostly heuristic
  - important adjacent evidence can land in neighboring chunks and not be passed forward together
- Risk:
  - answers sound incomplete even when the paper contains the needed detail nearby
  - formula extraction questions may answer "yes" or show bare expressions without enough context
- Planned fix:
  - bundle equations with local explanatory context where possible
  - add lightweight claim-aware sentence grouping
  - pass neighboring chunks heuristically during QA context assembly for equation/result-heavy evidence

### Short high-signal technical chunks may be filtered out too aggressively

- Symptom:
  - concise but important spans like objective definitions, theorem statements, formula lines, or short metric statements may be underrepresented
- Observed cause:
  - chunk sanitization and quality filtering prefer fuller sentence-like chunks with higher textual coherence
- Risk:
  - retrieval misses precisely the compact technical evidence users often ask for
- Planned fix:
  - loosen filtering for high-signal short technical chunks, especially formulas, objectives, theorem-like statements, and compact metric lines

### Table-heavy evidence is still under-modeled for retrieval

- Symptom:
  - evaluation/result questions can miss useful evidence from tables or table-adjacent text
  - numeric comparison evidence is often weaker than method/equation evidence
- Observed cause:
  - current quality scoring treats many table-like patterns as noise
  - there is no dedicated compact result/table snippet extraction path yet
- Risk:
  - retrieval and QA underperform on benchmark, ablation, and metric-focused questions
- Planned fix:
  - improve table-aware preservation/extraction so metric rows, captions, and compact comparison snippets remain usable evidence

### Gemini analysis pipeline hit `429 Too Many Requests`

- Symptom:
  - paper analysis triggered Gemini `429` responses
  - fallback to `gemini-2.5-flash-lite` occurred
  - section-agent runs became slow and noisy
- Observed cause:
  - section synthesis pipeline made many Gemini calls per analysis run
  - concurrent section workers increased burst pressure on Gemini quotas
- Mitigation:
  - reduce default `SECTION_AGENT_MAX_WORKERS` from `2` to `1`
  - replace pre-synthesis LLM evidence assessment with deterministic scoring
- Follow-up:
  - if needed later, make worker count configurable by environment only
  - keep monitoring whether review/rewrite loops still generate excessive burst load

### Backend startup looked broken while Supabase was slow/down

- Symptom:
  - backend health checks initially failed
  - startup appeared stalled
- Observed cause:
  - remote database availability/startup latency affected app boot timing
- Mitigation:
  - rechecked with a longer startup wait window
  - confirmed backend served `/health` once DB recovered

### `conda run -n research-agent` failed on Windows path handling

- Symptom:
  - `conda run` failed with path/activation errors referencing `C:\\Users\\Shashwat`
- Observed cause:
  - Windows path handling with spaces in the user directory
- Mitigation:
  - use PowerShell conda hook + `conda activate research-agent` explicitly instead of `conda run`

### Research QA grounding dilemma: retrieved evidence vs model prior knowledge

- Symptom:
  - retrieval can be only moderate while the generated answer still looks strong
  - there is tension between strict grounding and allowing the model to fill gaps from prior knowledge
- Observed cause:
  - the system is optimized for research-paper QA, where paper-specific precision matters
  - LLM prior knowledge can improve fluency and completeness, but may silently introduce unsupported details from similar papers or generic ML knowledge
- Risk:
  - mixing source-grounded claims with model priors without labeling them can create subtle hallucinations
  - relying only on retrieved evidence can make answers incomplete when retrieval is weak
- Follow-up:
  - define an explicit 3-tier answer policy:
    - evidence-backed
    - inferred-from-evidence
    - general background knowledge
  - ensure future QA outputs can distinguish these categories instead of blending them invisibly
