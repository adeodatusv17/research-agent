# Problems Log

This file tracks notable problems encountered during the project, along with observed symptoms and fixes or mitigations.

## 2026-05-16

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
