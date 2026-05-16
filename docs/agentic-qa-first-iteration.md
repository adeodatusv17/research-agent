# Agentic QA First Iteration

This iteration extends the active QA graph without replacing the existing hierarchical retrieval stack.

## Current orchestration boundary

- Retrieval remains hierarchical:
  - section retrieval
  - subsection retrieval
  - chunk retrieval
  - reranking / balancing
- Structured shortcuts still bypass the graph for direct answerable questions.
- The new logic adds gated orchestration around retrieval and synthesis instead of flattening the stack.

## New graph flow

Level 0:
- structured shortcut

Level 1:
- query analysis
- retrieve sections
- retrieve subsections
- retrieve chunks
- rerank
- evidence diagnostics
- grounded answer
- evaluate

Level 2:
- Level 1
- verifier
- optional retry before synthesis finalization

Level 3:
- query analysis
- planner
- retrieve sections
- retrieve subsections
- retrieve chunks
- rerank
- evidence diagnostics
- bounded adaptive retry
- grounded answer
- verifier
- critic
- bounded revision
- evaluate

## New structured outputs

- `execution_plan`
- `evidence_diagnostics`
- `grounded_claims`
- `verifier_report`
- `critic_report`
- `evaluation_report`
- `retrieval_attempts`
- `execution_trace`
- `final_confidence`

## Confidence derivation

Confidence is now evidence-derived:

- retrieval strength
- chunk support score
- rerank score
- lexical overlap between claim and cited evidence
- section coverage
- contradiction penalty

No claim confidence is taken from model self-report.

## Retry policy

- retries are bounded by orchestration level
- retries expand `top_k` values and disable subsection filtering
- retries append missing evidence hints to the active query
- retries trigger only when diagnostics show weak density, weak coverage, contradictions, or missing evidence

## Verification and critique

- verifier is deterministic first
- optional LLM verification only escalates for complex or problematic cases
- critic is gated by orchestration level or verifier issues
- revision budget is bounded to one pass in this iteration

## Compatibility

- existing `/papers/{paper_id}/qa` response remains backward compatible with `answer` and `sources`
- additional fields are additive
- existing retrieval and indexing behavior is preserved
