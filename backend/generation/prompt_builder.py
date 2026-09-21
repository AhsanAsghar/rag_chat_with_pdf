"""
Assembles the final prompt sent to the LLM: a system instruction
that constrains the model to answer only from the provided context,
plus the retrieved chunks and the user's question.
"""
from typing import List
from ..retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context from the user's uploaded documents.

Rules:
- Answer using only the information in the context below. Do not use outside knowledge.
- If the answer isn't in the context, say clearly that the documents don't contain that information — do not guess or make something up.
- Keep answers concise and direct.
- When helpful, mention which document/page the information came from.
"""


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a numbered context block the
    model can reference, and that we can map back to citations."""
    if not chunks:
        return "No relevant context was found in the uploaded documents."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page_info = f", page {chunk.page}" if chunk.page else ""
        parts.append(
            f"[Source {i}: {chunk.filename}{page_info}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(
    question: str,
    chunks: List[RetrievedChunk],
    history: list | None = None,
) -> list[dict]:
    """Builds the full messages array for the chat completion call."""
    context_block = build_context_block(chunks)

    user_content = (
        f"Context from uploaded documents:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include prior conversation turns (if any) so follow-up
    # questions like "what about chapter 2?" retain context.
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})
    return messages
