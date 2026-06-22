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

## 2026-05-31

### QA reliability investigation: conversational follow-ups, equation retrieval, and math rendering

- Scope:
  - audit the current QA stack before making further behavioral changes
  - preserve the existing hierarchical retrieval strengths
  - prefer small, reversible fixes over retrieval redesign

### Current retrieval and QA architecture

- Retrieval is not a flat generic RAG stack.
- The active QA path is:
  - frontend `QAChat` -> `POST /papers/{paper_id}/qa`
  - `rag_service.answer_question(...)`
  - LangGraph QA workflow in `research_qa_graph.py`
  - `query_analysis -> retrieve_sections -> retrieve_subsections -> retrieve_context -> rerank_chunks -> evidence_diagnostics -> optional retry -> generate_answer -> optional verify/critique/revise -> evaluate`
- Retrieval is a dense pgvector pipeline hybridized with structural and heuristic signals:
  - query embedding with `BAAI/bge-base-en-v1.5`
  - section retrieval
  - subsection retrieval inside selected sections
  - chunk retrieval inside selected section/subsection scope
  - quality filtering
  - role/importance annotation
  - heuristic reranking
  - section-balanced final selection
- Chunk reranking currently uses:
  - `0.5 * cosine_similarity`
  - `0.3 * role_match_weight`
  - `0.2 * importance`
- Evidence diagnostics and adaptive retry already exist, but retry rewriting is retrieval-broadening logic, not conversational follow-up resolution.
- The current "hybrid" behavior comes from combining dense similarity with:
  - section priors
  - section weights
  - chunk role inference
  - chunk importance
  - quality filtering
  - balanced selection

### Issue 1: missing short-term conversational memory for follow-up QA

- Observed failure:
  - follow-up questions like `What's the mathematical expression?`, `How does it work?`, `Where does the paper say that?`, and `What's the equation?` can lose the referent from the previous turn
- Findings:
  - frontend chat history is stored only in browser `sessionStorage`
  - frontend sends only `{ query }` to the backend QA route
  - backend `QARequest` contains only `query: str`
  - `QAState` has `query` and `active_query`, but no chat history, previous turn, or session memory field
  - retrieval nodes embed `state["active_query"]` or `state["query"]` only
  - current adaptive rewrite only happens after weak evidence and simply broadens scope by appending missing evidence hints like `broader evidence`
- Root cause:
  - the system is currently single-turn for retrieval
  - short-term conversation context never reaches the retrieval pipeline
- Low-risk proposed fix:
  - add bounded short-term conversation context scoped to the current paper and current chat session only
  - send the last 1-3 user/assistant turns to the backend
  - add a pre-retrieval standalone-query rewrite step that activates only on follow-up or deictic questions
  - keep the user-facing answer tied to the original question, while retrieval uses the rewritten standalone query
  - examples:
    - `What's the mathematical expression?` -> `What is the mathematical expression for LoRA?`
    - `How does it work?` -> `How does LoRA work?`
    - `Where does the paper say that?` -> preserve the prior claim target in the rewritten retrieval query
- Existing mechanisms that can be extended safely:
  - frontend already has recent chat messages in `sessionStorage`
  - backend already has `active_query`, which is the right place to store a rewritten standalone retrieval query
- Risks:
  - over-aggressive rewriting could distort already self-contained questions
  - leaking prior assistant wording into retrieval can amplify earlier mistakes if not bounded
- Acceptance criteria:
  - follow-up questions resolve to the prior topic within the same paper session
  - standalone queries remain unchanged
  - no long-term user memory or cross-paper memory is introduced

### Issue 2: formula / equation retrieval quality

- Observed failure:
  - formula-oriented questions can retrieve hyperparameters, settings, symbol fragments, or table entries instead of actual equations or mathematical definitions
- Findings:
  - formula awareness currently exists more strongly in answer generation than in retrieval
  - `_is_formula_query(...)` and `response_mode = equation_extraction` affect prompting, not section selection or chunk intent classification
  - retrieval intent classification has no dedicated `equation` or `formula` query type
  - a query like `What's the mathematical expression?` will often default to `method`
  - for `method` intent, rerank role matches currently favor `method` and `algorithm`, not `equation` or `formula`
  - chunk storage uses sanitized content, not raw parsed text
  - equation-heavy chunks can still be removed during indexing because `sanitize_chunk_content(...)` drops chunks judged too symbol-heavy
  - `EQUATION:` lines are preserved during parsing and chunk splitting, but indexing sanitization can later strip or drop them
  - PDF equation detection currently includes mojibake symbol markers like `âˆ‘` and `Î£`, which indicates symbol normalization is already imperfect in the parse path
  - the structured paper analysis already contains `methods.equations.items`, but the QA shortcut path does not use those extracted equations for formula queries
- Root causes:
  - formula intent is not first-class in retrieval routing
  - equation/formula chunks are under-boosted during reranking
  - some math-heavy evidence is lost or weakened before it becomes searchable chunk content
- Low-risk proposed fixes:
  - add a formula/equation retrieval flag separate from the existing high-level query type
  - when that flag is on:
    - rewrite the retrieval query toward standalone mathematical intent
    - mildly boost section priors for `method`, `theory`, and math-dense subsections
    - mildly boost chunk roles `equation`, `formula`, and `theory` during reranking
    - allow a slightly broader candidate pool before final rerank
  - keep the change gated to formula-mode queries only
  - add a fallback that uses extracted `paper_analysis.methods.equations.items` when the source is `extracted`, not `llm_generated`
  - add evaluation examples specifically for formula queries before and after the change
- Medium-to-high risk ideas that may affect retrieval quality globally:
  - changing global rerank weights for all queries
  - disabling or heavily relaxing quality filters across the board
  - replacing hierarchical retrieval with flat retrieval, self-query retrieval, or a new retrieval framework
  - changing global chunking boundaries without a targeted formula-specific evaluation set
- What should not be changed:
  - the section -> subsection -> chunk retrieval shape
  - the current dense embedding stack and vector schema as part of this QA-only reliability work
  - the evidence diagnostics / bounded retry loop
- Acceptance criteria:
  - formula queries return full equations or mathematical definitions when they exist in the paper
  - symbol fragments like isolated `r = 4`, `W_q`, `W_v` are not returned as the main answer unless they are part of a full retrieved expression
  - non-math queries preserve current retrieval quality
- Implementation status after fix:
  - implemented a gated `formula_mode` retrieval path with conservative formula query expansion
  - added formula-query logging for `retrieved_top_20`, `reranked_top_20`, and `selected_context`
  - added ranked section-level equation extraction with quality scoring and formula-mode answer preference
  - validated on the LoRA paper:
    - chunk-level retrieval top-20 is still noisy and remains dominated by hyperparameter/table fragments
    - the final answer is now rescued by high-confidence section-extracted equations such as:
      - `W0 + ΔW = W0 + BA`
      - `W = W0 + BA`
      - `h = W0x + ΔWx = W0x + BAx`
  - validated a non-formula LoRA comparison query and observed no obvious regression in the answer quality
- Residual risk:
  - chunk-level math retrieval is still weak for some papers, so the current formula fix relies on the extracted-equation path more than the chunk path
  - this is acceptable as a low-risk targeted fix, but chunking/indexing quality for equations may still need a future dedicated pass

### Issue 3: mathematical rendering and QA UX

- Observed failure:
  - even when math evidence exists, the QA answer is hard to read and often appears as plain text bullets
- Findings across the full path:
  - PDF parsing marks some short lines as `EQUATION: ...`
  - chunking tries to keep equation lines with nearby explanation
  - indexing sanitization can replace `EQUATION:` markers and may drop equation-heavy content altogether
  - QA response formatting currently returns only markdown text plus `answer_tiers`
  - QA does not return structured equation blocks or LaTeX payloads
  - the paper analysis UI already has working KaTeX rendering for extracted equations
  - the QA chat UI uses `ReactMarkdown` only and does not enable `remark-math` / `rehype-katex`
- Root causes:
  - mathematical content is not preserved as a first-class structured QA response
  - the QA frontend cannot render math markup even if the backend returns it
- Low-risk proposed fixes:
  - preserve structured equations for formula-mode QA responses when available
  - reuse the existing equation rendering pattern from the paper analysis UI in the QA UI
  - enable math-aware markdown rendering in QA chat if inline/block LaTeX is returned
  - keep citations/source cards separate from equation rendering
- Risks:
  - partial or malformed LaTeX could render poorly if sent straight to KaTeX without fallback handling
  - mixing plain-text equation fragments and KaTeX output in the same message could look inconsistent unless structured cleanly
- Acceptance criteria:
  - formula-mode answers render equations as readable block math or clearly formatted expressions
  - the QA UI remains readable for normal prose answers
  - source citations remain intact and do not break math layout

## 2026-06-01

### Priority track: chunk-level math retrieval, first-class table artifacts, and structured QA rendering

- Goal:
  - preserve the current hierarchical retrieval stack
  - improve weak formula/table behavior incrementally
  - add evaluation coverage before larger retrieval changes

### Issue 4: chunk-level mathematical retrieval is still weaker than formula-mode answer rescue

- Observed failure:
  - formula answers for papers like LoRA are now correct at the user-facing level, but raw chunk retrieval still frequently surfaces:
    - hyperparameter rows
    - appendix-style math fragments
    - table-adjacent symbol snippets
- Findings:
  - the canonical LoRA equations are recoverable from section-level extracted equation text
  - raw and reranked chunk top-20 still do not reliably elevate the canonical formula chunk itself
  - formula-mode currently succeeds because the answer path prefers cleaned extracted equations, not because chunk retrieval is fully fixed
- Root cause:
  - chunk-level searchable math evidence is still noisy
  - chunk content and chunk ranking are not yet strong enough to guarantee canonical formula retrieval on their own
- Low-risk next fixes:
  - add equation/math metadata during ingestion for chunks:
    - `contains_equation`
    - `contains_latex_like_math`
    - `math_density`
    - `equation_type` when inferable (`objective`, `definition`, `derivation`, `constraint`)
  - use those metadata signals only in formula-mode retrieval and reranking
  - validate that canonical formulations are present in the final selected evidence for papers like LoRA
- Risks:
  - broadening these boosts beyond formula-mode could distort normal retrieval
  - treating all symbol-heavy chunks as valuable would reintroduce hyperparameter/table noise
- Acceptance criteria:
  - canonical mathematical formulations are present in final evidence for formula queries when the paper contains them
  - non-formula retrieval remains unchanged in quality

### Issue 5: tables are not first-class retrievable artifacts

- Current behavior:
  - table-like blocks are detected in PDF parsing and rewritten as `TABLE:` text
  - those blocks are chunked and sanitized into ordinary `PaperChunk` rows
  - no separate table artifact, table metadata store, or direct table retrieval path exists
- Observed failures:
  - questions like:
    - `Are there any tables in the paper?`
    - `Show me the results table.`
    - `What does Table 3 contain?`
    can still return weak evidence or `Insufficient grounded evidence` even when the document contains tables
  - captions, headers, and metric rows can be split across multiple neighboring chunks
  - table-heavy chunks are often partially penalized as noisy evidence
- Root causes:
  - tables are treated as special text, not as first-class content objects
  - there is no persistent table metadata such as:
    - caption
    - table label/number
    - page number
    - section/subsection
    - detected table type (`results`, `ablation`, `hyperparameter`, `architecture`)
    - detected metric names / dataset names / model names
  - table questions currently depend on chunk retrieval plus sanitization heuristics
- Chosen direction:
  - prioritize first-class table storage with metadata over summary-only table handling
- Low-risk proposed implementation:
  - parse and store table artifacts separately from normal chunks
  - minimally store:
    - `paper_id`
    - `table_id`
    - `table_label`
    - `caption`
    - `section_name`
    - `subsection_name`
    - `page_number`
    - `raw_table_text`
    - `normalized_table_text`
    - `table_type`
    - `metric_names`
    - `dataset_names`
    - `model_names`
  - embed normalized table text separately
  - add a parallel table retrieval path for table/result-oriented QA
  - keep linkages to nearby chunk ids so caption + explanation can be rejoined in QA
- What should not be done first:
  - do not start with table summaries alone
  - do not replace chunk retrieval with table-only retrieval
  - do not redesign the whole ingestion pipeline around a document parser swap in this pass
- Acceptance criteria:
  - LoRA and similar papers expose actual table artifacts during retrieval
  - `Table 3` / `results table` / `are there any tables` questions return table-backed evidence instead of generic chunk fallback
  - table retrieval is additive and does not harm non-table QA

### Issue 6: table and formula rendering should be structured, not flattened

- Observed failure:
  - even when the right equation or table evidence exists, the UI can flatten it into text that is harder to scan
- Root causes:
  - formula rendering is improved, but still partly text-oriented
  - tables do not yet have a dedicated presentation model
- Planned low-risk UI changes:
  - render formula answers as structured math blocks where possible
  - render retrieved tables as explicit table cards with:
    - caption
    - table label
    - rows/columns or compact normalized text
    - optional nearby explanatory text
- Acceptance criteria:
  - equations appear as clearly separated math content
  - tables appear as structured result artifacts instead of flattened prose blobs

### Small generalized evaluation set

- We need a reusable cross-paper QA evaluation set covering:
  - general paper understanding
  - conversational follow-ups
  - formula queries
  - table queries
  - result/evidence lookup
- Planned artifact:
  - maintain a compact hand-authored evaluation file under `docs/` so future retrieval changes can be checked before and after implementation

### Table extraction needed a stricter shape filter

- Symptom:
  - after first wiring first-class table storage, some caption-adjacent prose blocks were being mistaken for table artifacts
  - other PDFs produced duplicate table artifacts for the same label/page when the same table text was repeated by the parser
- Observed cause:
  - PDF text extraction can flatten table rows into short line fragments that look different from normal prose
  - a naive "caption nearby" check is not enough on its own
- Mitigation:
  - gate table extraction on row-like block shape, not just nearby captions
  - deduplicate exact repeated `(table_label, page_number, normalized_table_text)` signatures before storing
- Follow-up:
  - keep using LoRA and Conformer as the concrete sanity-check papers for table retrieval

### Implementation plan

- Phase 1: document and instrument
  - add a compact follow-up query evaluation set
  - add a compact formula query evaluation set
  - capture before/after retrieval outputs and final answers
- Phase 2: low-risk conversational follow-up support
  - extend the QA request shape with recent turns
  - add a follow-up detection + standalone query rewrite step before retrieval
  - write the rewritten query into `active_query`
- Phase 3: low-risk formula retrieval improvements
  - add a formula-mode retrieval flag
  - gate section and role boosts to formula-mode only
  - use extracted equation items as supplementary evidence when available
- Phase 4: math rendering
  - return structured equation payloads for formula-mode answers
  - render them in the QA UI using the existing KaTeX-capable pattern

### What can be improved with low risk

- bounded short-term conversation context for the current paper session only
- standalone-query rewriting for follow-up questions
- formula-mode-only retrieval boosts
- supplementary use of extracted equation items from paper analysis
- QA frontend math rendering improvements

### What may affect retrieval quality globally

- changing global rerank weights
- changing global section priors
- relaxing quality filters for all queries
- changing chunk sanitization or indexing behavior for all chunk types without a focused evaluation

### What should not be changed in this pass

- do not replace hierarchical retrieval
- do not swap retrieval frameworks
- do not add long-term memory
- do not redesign the LangGraph QA workflow
- do not broaden equation-friendly retrieval heuristics to all query types by default
