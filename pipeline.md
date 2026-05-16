# Repository Pipeline Map

This file documents the pipeline that is actually implemented in the repository as of the current codebase state.

## 1. System entrypoints

- API server: `apps/api/main.py`
- Worker bootstrap: `apps/worker/main.py`
- Worker graph runner stub: `apps/worker/graph_runner.py`
- Main Python package: `packages/research_agent`
- Frontend shell: `frontend`
- Database: PostgreSQL with `pgvector` via `docker-compose.yml`

### What is really active

- The active runtime path is mostly API route -> service function -> DB / embedding / LLM calls.
- LangGraph is actively used for:
  - QA flow
  - Experiment generation flow
- LangGraph scaffolding exists but is not meaningfully implemented for:
  - paper analysis graph
  - reproducibility graph
- The worker process is currently just a bootstrap print for the stub paper-analysis graph, not an async execution engine.

## 2. Core persisted data

The pipeline persists the following major entities:

- `papers`
  - uploaded paper metadata
  - source type / path
  - detected domain
- `paper_sections`
  - section-level text + embeddings
- `paper_subsections`
  - subsection-level text + embeddings
- `paper_chunks`
  - retrieval chunks + embeddings
- `paper_analysis`
  - structured analysis output
  - inferred structure
  - synthesis output
- `paper_repositories`
  - discovered candidate repos
- `reproducibility_scores`
  - computed reproducibility metrics + evidence
- `generated_experiments`
  - generated artifact metadata and code/config snapshots

## 3. End-to-end pipeline overview

The implemented repo has four main execution pipelines:

1. Ingestion and indexing pipeline
2. Analysis pipeline
3. RAG / QA pipeline
4. Experiment generation pipeline

There is also a direct semantic search endpoint and a read-only reproducibility fetch endpoint.

## 4. Ingestion and indexing pipeline

Main entrypoint:

- `POST /papers/upload`

Code path:

- `apps/api/routers/ingestion.py`
- `research_agent.tools.pdf_parser`
- `research_agent.tools.pdf_text_extractor`
- `research_agent.services.paper_indexing_service`
- `research_agent.services.section_parser_service`
- `research_agent.services.chunking_service`
- `research_agent.services.chunk_structure`
- `research_agent.tools.embedder`
- `research_agent.tools.vector_store`

### Step-by-step flow

1. The API validates that the upload is a PDF.
2. A new `paper_id` is created and the PDF is saved under `storage/papers/<uuid>_<filename>.pdf`.
3. The PDF is parsed with PyMuPDF (`fitz`) into page/block structures.
4. A title is heuristically extracted from the first page blocks.
5. A `papers` row is inserted before indexing.
6. Full text is re-extracted from the PDF.
7. The indexing service runs:
   - parse document into canonical sections
   - build merged section entries
   - build merged subsection entries
   - split section content into chunk-sized units
   - normalize chunk content for embedding token limits
   - infer chunk structure / quality metadata
   - detect paper domain from top chunks
   - embed sections, subsections, and chunks
   - store all embeddings in Postgres via `pgvector`
8. The transaction commits.

### Parsing details

- PDF parsing preserves page order and block order.
- Equation-looking lines are labeled as `EQUATION: ...`.
- Table-like blocks are normalized into `TABLE:` text blocks.
- Section parsing maps headings into canonical buckets like:
  - `abstract`
  - `introduction`
  - `method`
  - `experiments`
  - `results`
  - `discussion`
  - `conclusion`
  - `references`
- If no headings are detected, the document falls back to a single `front_matter` section.

### Chunking details

- Chunking is section-aware.
- The effective chunk token cap is the embedder max token cap, currently `400`.
- Overlap between chunks is `50` tokens.
- Long single units are hard-trimmed through embedding normalization.

### Embedding details

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Embeddings are normalized before storage.
- Search uses cosine distance in `pgvector`.

### Output of ingestion

After upload, the system has enough state to support:

- semantic search
- graph QA
- analysis
- experiment generation later

## 5. Analysis pipeline

Main entrypoint:

- `POST /papers/{paper_id}/analyze`

Code path:

- `apps/api/routers/analysis.py`
- `research_agent.services.paper_analysis_service.analyze_paper`
- `research_agent.services.domain_detector`
- `research_agent.services.domain_adapters`
- `research_agent.services.repository_discovery_service`
- `research_agent.services.reproducibility_service`
- `research_agent.tools.gemini_client`

### What this pipeline does

This is the richest pipeline in the repo. It turns indexed chunks into:

- structured paper analysis fields
- inferred section structure
- section synthesis output
- repository candidates
- reproducibility score

### Step-by-step flow

1. Load the `papers`, `paper_chunks`, and `paper_sections` rows.
2. Select the most relevant / useful chunks for analysis.
3. Build chunk payloads with extra structure such as:
   - role
   - importance
   - confidence
   - summary
4. Build an `inferred_structure` from selected chunks.
   - `key_ideas`
   - `methods`
   - `results`
   - `discussion`
5. Build a `synthesis_structure`, including discussion fallback chunks if needed.
6. Confirm or infer paper domain if not already set.
7. Run section-level synthesis agents for:
   - `key_ideas`
   - `methods`
   - `results`
   - `discussion`
8. Merge section outputs into `synthesis_output`.
9. Run equation fallback logic to enrich method equations if synthesis evidence is weak.
10. Build the base analysis payload.
11. Run domain adapters to derive domain-specific fields.
12. Discover likely code repositories.
13. Compute reproducibility score and evidence.
14. Persist:
   - `paper_analysis`
   - `paper_repositories`
   - `reproducibility_scores`
15. Return a combined API payload.

### Section-agent behavior

This is the closest thing to a multi-agent pass in the current repo.

- The code creates one agent task per synthesis section.
- The sections are processed concurrently with `ThreadPoolExecutor`.
- `SECTION_AGENT_MAX_WORKERS = 2`, so at most 2 section agents run at the same time.
- Each section agent can:
  - synthesize evidence
  - retrieve one extra round of chunks if needed
  - do one rewrite round if review flags issues
- This is not a multi-process or distributed agent system.
- It is an in-process parallel fan-out/fan-in pattern around LLM-backed section synthesis.

### LLM use inside analysis

The analysis pipeline uses Gemini wrappers for:

- structured JSON generation
- synthesis / reasoning fallback
- evidence assessment and review-style passes

Model behavior:

- primary model: `gemini-2.5-flash`
- fallback reasoning candidates:
  - `gemini-2.5-flash-lite`
  - `gemini-3-flash`
  - `gemini-3.1-flash-lite`
  - `gemma-3-27b-it`

### Repository discovery pass

The analysis pipeline also runs a repository discovery sub-pipeline:

1. Extract GitHub URLs directly mentioned in paper text.
2. Query Papers With Code for repo links.
3. Query GitHub search for likely repos.
4. Score candidates using title similarity, architecture similarity, and source bonus.
5. Sort and persist candidates.

This is heuristic retrieval, not a graph agent.

### Reproducibility scoring pass

The analysis pipeline immediately computes reproducibility after analysis.

Signals include:

- dataset availability
- code availability
- methodology completeness
- result reproducibility signals
- ML-specific hyperparameter completeness
- ML-specific training detail score
- ML-specific evaluation protocol score

If ML scores are weak, an extra LLM reasoning pass may adjust confidence upward when evidence supports it.

### Analysis output shape

The stored `paper_analysis` payload may include:

- `model_architecture`
- `architectures`
- `dataset`
- `loss_function`
- `losses`
- `training_objective`
- `optimizer`
- `optimizers`
- `training_details`
- `evaluation_metrics`
- `contributions`
- `domain`
- `inferred_structure`
- `synthesis_output`

## 6. RAG / QA pipeline

Main entrypoint:

- `POST /papers/{paper_id}/qa`

Code path:

- `apps/api/routers/qa.py`
- `research_agent.services.rag_service.answer_question`
- `research_agent.agents.graphs.research_qa_graph`
- `research_agent.agents.nodes.*`
- `research_agent.services.retrieval_service`
- `research_agent.tools.vector_store`
- `research_agent.tools.gemini_client`

### High-level behavior

The QA path has two branches:

1. Structured shortcut branch
2. Full RAG graph branch

### Branch A: structured shortcut

Before running retrieval, the system checks whether the query can be answered from stored structured analysis.

It can directly answer questions about things like:

- dataset
- training objective
- optimizer / optimizers
- loss / losses
- architecture / architectures
- reproducibility

If a match is found, it returns an answer immediately without vector retrieval.

Important exception:

- If the query looks code-oriented, the shortcut is skipped on purpose so the system can use the full RAG context.

### Branch B: full RAG graph

If the shortcut does not apply, the system invokes a LangGraph pipeline:

1. `query_analysis`
   - classify query intent such as `definition`, `method`, `evaluation`, `theory`, `comparison`
   - tokenize crude keywords
2. `retrieve_sections`
   - embed query
   - semantic search over section embeddings
   - apply section priors by query intent
3. `retrieve_subsections`
   - semantic search over subsection embeddings within selected sections
4. `retrieve_context`
   - semantic search over chunk embeddings
   - constrain by chosen sections / subsections
5. `rerank_chunks`
   - filter low-quality chunks
   - rerank using semantic score + role match + importance
   - balance final chunk selection across sections
6. `generate_answer`
   - build context window from final chunks
   - call Gemini to answer
   - return answer + source snippets

### RAG retrieval design

This is a hierarchical retrieval pipeline:

1. section-level retrieval
2. subsection-level retrieval
3. chunk-level retrieval
4. reranking / balancing
5. answer generation

That gives the repo a real RAG pipeline, not just flat embedding search.

### Retrieval features

- Query intent classification changes section priors and rerank behavior.
- Reference-style queries can include `references` sections.
- Chunk quality filters penalize:
  - citation-heavy text
  - table-heavy noise
  - heading fragments
  - very short fragments
- Chunk rerank score combines:
  - cosine similarity
  - role match to query intent
  - inferred importance
- Final selection is balanced by section, not just global top-k.

### Generation behavior

- The answer prompt explicitly allows code generation if the user asks for implementation help.
- If context is incomplete, the model is allowed to fill in reasonable defaults for code-oriented answers.

## 7. Direct semantic search pipeline

Main entrypoint:

- `GET /papers/search?query=...`

This path is simpler than QA:

1. embed raw query
2. run direct chunk-level `semantic_search`
3. return top matches

This endpoint does not use:

- the QA LangGraph
- section/subsection hierarchy
- chunk reranking
- LLM answer generation

So it is vector search, not full RAG.

## 8. Experiment generation pipeline

Main entrypoint:

- `POST /experiments/{paper_id}/generate`

Code path:

- `apps/api/routers/experiments.py`
- `research_agent.services.experiment_generation_service.generate_experiment`
- `research_agent.agents.graphs.experiment_generation_graph`
- `research_agent.services.repository_verification_service`
- `research_agent.services.experiment_codegen_service`
- `research_agent.tools.gemini_client`

### High-level behavior

This pipeline creates reproducibility artifacts and scaffold code from the latest stored paper analysis.

### LangGraph stages

The experiment graph is linear:

1. `load_analysis`
2. `verify_repositories`
3. `apply_defaults`
4. `infer_missing_fields`
5. `generate_config`
6. `generate_code`
7. `write_files`
8. `validate_artifact`
9. `store_metadata`

### Step-by-step flow

1. Create a `generated_experiments` row with `pending` status.
2. Load the latest `paper_analysis` and `paper`.
3. Resolve domain.
4. For ML papers, backfill missing analysis fields from chunk-derived ML adapters if needed.
5. Verify candidate repositories:
   - load stored repos
   - if missing, rediscover
   - inspect GitHub API / README / contents / repo page
   - compute trust score
   - emit `recommended_action`
6. Build config defaults.
   - ML path: structured experiment config
   - non-ML path: domain-agnostic scaffold config
7. Infer missing config fields with LLM for selected ML fields.
8. Generate final YAML config.
9. Generate code files:
   - `model.py`
   - `dataset.py`
   - `train.py`
   - `utils/config_loader.py`
   - `requirements.txt`
10. Write files under:
    - `artifacts/experiments/<paper_id>/<experiment_id>/`
11. Validate generated artifact.
12. Write `metadata.json`.
13. Update DB row to `completed`.

### ML experiment path

For ML papers, config generation tries to derive:

- model family
- task type
- optimizer
- loss
- scheduler
- dimensions / heads / layers
- batch size
- epochs
- learning rate

Source priority is:

1. explicit paper analysis fields
2. regex extraction from training details
3. conservative defaults
4. controlled LLM inference for remaining missing fields

### Repository verification details

This is a distinct pass after repository discovery.

It scores repos using signals like:

- repo name similar to paper title
- repo name mentions architecture
- README mentions paper title
- README mentions architecture
- training entrypoint present
- environment spec present
- community signal via stars

Based on the top trust score, the graph recommends:

- `clone_existing_repo`
- `review_repo_then_generate`
- `use_generated_scaffold`

### Generated artifact outputs

The experiment pipeline persists both files and metadata:

- filesystem artifacts in `artifacts/experiments/...`
- DB snapshot in `generated_experiments`
- `metadata.json` with:
  - inferred fields
  - defaults used
  - repository recommendation
  - validation results
  - recommended action

## 9. Reproducibility endpoint pipeline

Main entrypoint:

- `GET /reproducibility/{paper_id}`

This endpoint does not run a graph or recompute scores.

It simply:

1. loads latest `reproducibility_scores` row
2. returns the stored values

## 10. Frontend-facing flow

The frontend is a thin UI over the API surfaces. The main backend pipelines exposed to the UI are:

- upload paper
- list papers
- fetch paper
- analyze paper
- fetch analysis
- ask QA question
- generate experiment
- fetch reproducibility
- search papers

The real intelligence lives in backend services, not the frontend.

## 11. Actual agent / multi-agent picture

If the question is "does this repo have multi-agent behavior?", the honest answer is:

- Yes, but only in a limited sense.
- No, not as a large orchestrated multi-agent platform.

### What qualifies as agentic here

- QA uses a LangGraph node pipeline.
- Experiment generation uses a LangGraph node pipeline.
- Analysis uses concurrent section-synthesis workers that behave like a mini multi-agent fan-out pass.

### What does not qualify as a real multi-agent system yet

- No autonomous worker fleet
- No cross-agent negotiation
- No planner/executor split
- No durable graph-runner service
- No implemented background queue worker execution path

## 12. Important current-state caveats

### Active vs stubbed pieces

- `research_qa_graph` is active.
- `experiment_generation_graph` is active.
- `paper_analysis_graph` is effectively a stub.
- `reproducibility_graph` is effectively a stub.
- `apps/worker/graph_runner.py` is a stub.
- `apps/worker/main.py` only builds and prints the stub analysis graph.

### Practical meaning

- Paper analysis is currently service-driven, not graph-driven.
- Reproducibility scoring is currently service-driven, not graph-driven.
- QA and experiment generation are the only fully graph-orchestrated flows.

## 13. Concise pipeline summary

If you want the shortest accurate summary of the repo:

1. Upload PDF.
2. Parse PDF into sections, subsections, and chunks.
3. Embed all retrievable text into `pgvector`.
4. Analyze chunks into structured paper metadata and section syntheses.
5. Discover repos and compute reproducibility.
6. Answer questions through either:
   - structured-analysis shortcut, or
   - hierarchical RAG graph over section/subsection/chunk embeddings.
7. Generate experiment scaffolds through a LangGraph pipeline that loads analysis, verifies repos, fills config defaults, writes code artifacts, validates them, and stores metadata.

## 14. Key files to read first

- `apps/api/main.py`
- `apps/api/routers/ingestion.py`
- `apps/api/routers/analysis.py`
- `apps/api/routers/qa.py`
- `apps/api/routers/search.py`
- `apps/api/routers/experiments.py`
- `packages/research_agent/services/paper_indexing_service.py`
- `packages/research_agent/services/paper_analysis_service.py`
- `packages/research_agent/services/retrieval_service.py`
- `packages/research_agent/services/rag_service.py`
- `packages/research_agent/services/experiment_generation_service.py`
- `packages/research_agent/agents/graphs/research_qa_graph.py`
- `packages/research_agent/agents/graphs/experiment_generation_graph.py`
