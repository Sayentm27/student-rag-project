# workflow.py
# -----------
# This file improves retrieval quality using multi-step AI workflows.
#
# The retrieval quality problem:
# The quality of a RAG answer depends heavily on what gets retrieved.
# And what gets retrieved depends on how similar the query embedding is
# to the document embeddings. If the user's query is vague or uses
# different vocabulary than the documents, retrieval suffers.
#
# Two solutions:
#
# 1. Query rewriting: Use an LLM to rewrite the user's question into a
#    version that will produce a better embedding for semantic search.
#    "tell me about that database thing" → "How do relational databases
#    store and query structured data using SQL?"
#
# 2. Query decomposition: Some questions are actually multiple questions.
#    Split them up and retrieve separately, then combine the results.
#    This is called "multi-hop retrieval."

from google import genai
from google.genai import types
import re
from config import GEMINI_API_KEY, GEMINI_MODEL
from embeddings import embed_text
from vector_store import query_similar

_client = genai.Client(api_key=GEMINI_API_KEY)

_FOLLOWUP_PRONOUNS = (
    " it ", " its ", " that ", " this ", " they ", " them ", " those ", " these ",
)
_PRONOUN_RE = re.compile(
    r"\b(it|its|that|this|they|them|those|these)\b", re.IGNORECASE
)
_KNOWN_TOPICS = (
    ("python", "Python"),
    ("machine learning", "machine learning"),
    ("neural network", "neural networks"),
    ("vector database", "vector databases"),
    ("database", "databases"),
    ("natural language processing", "NLP"),
    ("large language model", "LLMs"),
    ("api", "APIs"),
    ("rag", "RAG"),
    ("git", "Git"),
)


def extract_topic_from_context(conversation_context):
    """Identify the main topic from recent conversation text."""
    combined = conversation_context.lower()
    for needle, label in _KNOWN_TOPICS:
        if needle in combined:
            return label

    for line in reversed(conversation_context.splitlines()):
        if line.startswith("User:"):
            return line[len("User:"):].strip()

    return ""


def _needs_context_resolution(query):
    """Return True if the query still relies on pronouns that need conversation context."""
    padded = f" {query.lower()} "
    return any(pronoun in padded for pronoun in _FOLLOWUP_PRONOUNS)


def contextual_search_fallback(original_query, conversation_context):
    """
    Build a standalone search query from conversation history when a follow-up
    question uses pronouns like 'it' or 'that' that won't embed well on their own.
    """
    if not conversation_context:
        return original_query

    topic = extract_topic_from_context(conversation_context)
    if not topic:
        return original_query

    resolved = _PRONOUN_RE.sub(topic, original_query)
    if resolved.lower() != original_query.lower():
        return resolved

    return f"{topic} {original_query}"


def topic_search_query(conversation_context, original_query=""):
    """Build a topic-focused search query for real-world follow-up questions."""
    topic = extract_topic_from_context(conversation_context)
    if not topic:
        return original_query

    query_lower = original_query.lower()
    if any(word in query_lower for word in ("real world", "use case", "application", "else", "do")):
        return f"{topic} real world applications use cases"

    return f"{topic} {original_query}".strip()


def rewrite_query(original_query, conversation_context=""):
    """
    Use Gemini to rewrite the user's query for better semantic search.

    Args:
        original_query:      The user's original question.
        conversation_context: Recent conversation history (helps resolve
                              pronouns like "it" or "that").

    Returns:
        A rewritten query string, or the original if rewriting fails.
    """
    # TODO (Week 15): Implement query rewriting using Gemini.
    #
    # --- The RAG concept ---
    # Embeddings capture meaning, but they're sensitive to phrasing.
    # A user might type casually ("how does python deal with dbs?") while
    # documents are written formally ("Python database connectivity and ORMs").
    # These two phrasings may not be close in embedding space even though
    # they mean the same thing. Query rewriting bridges that gap.
    #
    # Also important: if the user asks a follow-up like "What else can it do?",
    # the conversation_context lets you resolve "it" to the right topic.
    #
    # Steps:
    #   1. If conversation_context is not empty, include it in the prompt
    #   2. Build a prompt asking Gemini to rewrite the question to be more
    #      specific and technical, suitable for semantic search
    #   3. Call _client.models.generate_content() with temperature=0.1
    #      (low temperature = focused rewriting, not creative)
    #   4. Return response.text.strip() if it's not empty and under 500 chars
    #   5. Wrap in try/except — if anything fails, return original_query unchanged
    #
    try:
        prompt = (
            "Rewrite this question into a clear, self-contained question suitable "
            "for semantic search.\n"
            "- Replace pronouns like 'it', 'that', or 'this' with the actual subject.\n"
            "- Return only the rewritten question, with no explanation.\n\n"
            f"Question: {original_query}"
        )
        if conversation_context:
            prompt = (
                f"{prompt}\n\nPrevious conversation (use this to resolve pronouns "
                f"like 'it' or 'that'):\n{conversation_context}"
            )
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        rewritten = (response.text or "").strip()
        if rewritten and len(rewritten) < 500:
            if conversation_context and _needs_context_resolution(rewritten):
                return contextual_search_fallback(original_query, conversation_context)
            return rewritten
    except Exception:
        pass

    if conversation_context:
        return contextual_search_fallback(original_query, conversation_context)
    return original_query



def decompose_query(query):
    """
    Break a complex multi-part question into simpler sub-questions.

    Args:
        query: A question that may contain multiple distinct topics.

    Returns:
        A list of sub-question strings (up to 3), or [query] if it's
        already simple or if decomposition fails.
    """
    # TODO (Week 15): Implement query decomposition using Gemini.
    #
    # --- The RAG concept ---
    # Some questions have multiple parts, each requiring different documents.
    # "How does Python connect to databases, and what's the difference between
    # SQL and NoSQL?" needs documents about Python AND about SQL/NoSQL separately.
    # By splitting the question and searching for each part independently,
    # we get much better document coverage for complex questions.
    #
    # Steps:
    #   1. Build a prompt asking Gemini: if this question covers multiple topics,
    #      split it into 2-3 simpler sub-questions; otherwise return it as-is
    #   2. Call _client.models.generate_content() with temperature=0.1
    #   3. Split response.text on newlines, strip each line, drop empty/short lines
    #   4. Return at most 3 sub-questions
    #   5. Wrap in try/except — if anything fails, return [query]
    #
    try:
        prompt = f"if this question, {query}, covers multiple topics, split it into 2-3 simpler sub-questions; otherwise return it as-is"
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )

        sub_questions = [
            line.strip()
            for line in response.text.splitlines()
            if line.strip() and len(line.strip()) > 10
        ]
        if sub_questions:
            return sub_questions[:3]
    except Exception:
        pass

    return [query]


def multi_hop_retrieve(query, n_per_hop=2):
    """
    Retrieve documents for each sub-question and combine the results.

    Steps:
      1. Decompose the query into sub-questions
      2. Embed and search for each sub-question independently
      3. Combine results, removing duplicates

    Args:
        query:     The original complex query.
        n_per_hop: Documents to retrieve per sub-question.

    Returns:
        A deduplicated list of relevant document strings.
    """
    sub_queries = decompose_query(query)

    all_documents = []
    seen_documents = set()

    for sub_query in sub_queries:
        embedding = embed_text(sub_query)
        results = query_similar(embedding, n_per_hop)

        for doc in results["documents"][0]:
            if doc not in seen_documents:
                seen_documents.add(doc)
                all_documents.append(doc)

    return all_documents
