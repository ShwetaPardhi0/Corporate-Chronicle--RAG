"""
Prompt templates for the RAG pipeline.
Centralizing prompts here makes it easy to tweak and version them independently.
"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build the full RAG prompt from retrieved context chunks.

    Args:
        query: The user's original question.
        context_chunks: List of result dicts (each has 'chunk' and 'score').

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_text = _format_context(context_chunks)

    return (
        "You are a helpful and precise assistant. Answer the user's question USING ONLY the "
        "provided retrieved context snippets. If the context does not supply enough information to "
        "answer, reply that you don't know based on the documents.\n\n"
        f"--- START RETRIEVED CONTEXT ---\n{context_text}\n--- END RETRIEVED CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def build_mock_response(query: str, context_chunks: list[dict]) -> str:
    """Generate a mock response when no API key is configured."""
    response = (
        "[MOCK GENERATION - NO API KEY SET]\n"
        f"I processed your query: '{query}'\n"
        f"Retrieved {len(context_chunks)} matching context chunk(s):\n"
    )
    for i, res in enumerate(context_chunks):
        chunk = res["chunk"]
        snippet = chunk["text"][:150].replace("\n", " ")
        response += f'- [Source {i+1} / Score: {res["score"]:.4f}]: "{snippet}..."\n'
    response += "\nTo get real answers, please set a valid GEMINI_API_KEY in your .env file."
    return response


def _format_context(context_chunks: list[dict]) -> str:
    """Format retrieved chunks into a structured context block."""
    context_text = ""
    for i, res in enumerate(context_chunks):
        chunk = res["chunk"]
        score = res["score"]
        context_text += f"\n--- Source [{i+1}] (Similarity: {score:.4f}) ---\n"
        context_text += f"File: {chunk['source']}\n"
        context_text += f"Content:\n{chunk['text']}\n"
    return context_text
