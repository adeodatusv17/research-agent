import logging
import uuid

from sqlalchemy.orm import Session

from research_agent.services.paper_analysis_service import get_structured_answer_if_available
from research_agent.tools.embedder import get_tokenizer
from research_agent.tools.gemini_client import generate_answer


logger = logging.getLogger(__name__)
MAX_CONTEXT_TOKENS = 2000

CODE_INTENT_KEYWORDS = [
    "code",
    "implement",
    "write",
    "example",
    "snippet",
    "how to build",
    "python",
]

def _has_code_intent(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in CODE_INTENT_KEYWORDS)


def generate_answer_from_chunks(
    query: str,
    paper_id: uuid.UUID,
    filtered_chunks: list[dict],
) -> dict[str, object]:
    tokenizer = get_tokenizer()
    context_parts: list[str] = []
    current_tokens = 0
    used_chunks: list[dict] = []

    for chunk in filtered_chunks:
        section_name = chunk.get("section_name") or "unknown"
        subsection_name = chunk.get("subsection_name")
        location = f"Section: {section_name}"
        if subsection_name:
            location += f" | Subsection: {subsection_name}"
        part = f"[Paper {chunk['paper_id']} | {location} | Score {chunk['score']:.4f}]\n{chunk['content']}"
        part_tokens = len(
            tokenizer(
                part,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
        )

        if current_tokens + part_tokens > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(part)
        current_tokens += part_tokens
        used_chunks.append(chunk)

    context = "\n\n".join(context_parts)
    prompt = (
        "You are a knowledgeable research assistant.\n\n"
        "Use the provided research paper context to answer the question as thoroughly as possible.\n"
        "If the user asks for code or an implementation, generate complete, working Python code based on "
        "the architecture, equations, and methods described in the context. Do not refuse to generate code.\n"
        "If specific implementation details are not in the context, use your knowledge of the described "
        "architecture to fill in reasonable defaults.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )

    logger.info(
        "rag_generation query=%s paper_id=%s number_of_chunks_used=%s retrieved_chunk_ids=%s similarity_scores=%s",
        query,
        paper_id,
        len(used_chunks),
        [chunk["chunk_id"] for chunk in used_chunks],
        [round(float(chunk["score"]), 4) for chunk in used_chunks],
    )

    answer = generate_answer(prompt)
    return {
        "context": context,
        "answer": answer,
        "sources": [
            {
                "paper_id": chunk["paper_id"],
                "section_name": chunk.get("section_name"),
                "subsection_name": chunk.get("subsection_name"),
                "page_number": chunk.get("page_number"),
                "content": chunk["content"],
                "content_snippet": chunk["content"][:400],
                "score": chunk["score"],
            }
            for chunk in used_chunks
        ],
    }


def answer_question(db: Session, paper_id: uuid.UUID, query: str) -> dict[str, object]:
    # Skip the structured shortcut for code/implementation requests — it only returns
    # a brief summary like "Proposed: Conformer; Baselines: Transformer" which is not
    # useful when the user explicitly asks for code or a detailed explanation.
    if not _has_code_intent(query):
        structured_answer = get_structured_answer_if_available(db, paper_id, query)
        if structured_answer is not None:
            logger.info(
                "structured_analysis_shortcut paper_id=%s query=%s analysis_hits=%s",
                paper_id,
                query,
                list(structured_answer.get("analysis_hits", {}).keys()),
            )
            return {
                "answer": structured_answer["answer"],
                "sources": structured_answer["sources"],
            }

    from research_agent.agents.graphs.research_qa_graph import build_research_qa_graph

    graph = build_research_qa_graph()
    final_state = graph.invoke(
        {
            "db": db,
            "query": query,
            "query_type": "",
            "paper_id": paper_id,
            "query_embedding": [],
            "analysis_hits": {},
            "retrieved_sections": [],
            "selected_sections": [],
            "retrieved_subsections": [],
            "retrieved_chunks": [],
            "filtered_chunks": [],
            "retrieval_confidence": 0.0,
            "context": "",
            "answer": "",
            "sources": [],
        }
    )
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
    }
