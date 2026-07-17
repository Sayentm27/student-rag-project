# RAG Learning App

A Retrieval-Augmented Generation (RAG) application built with Python, ChromaDB, SentenceTransformers, and Google Gemini. You'll build this incrementally over Weeks 10–15.

## What This App Does

You can ask this app questions about Python, machine learning, databases, APIs, and AI concepts. It finds the most relevant documents from its knowledge base and sends them to Gemini as context — so the answers are grounded in real information rather than guesswork.

## System Architecture

```
User Query
    │
    ▼
[security.py]      ← Validate and sanitize input (Week 12)
    │
    ▼
[workflow.py]      ← Rewrite query for better retrieval (Week 15)
    │
    ▼
[embeddings.py]    ← Convert query to a vector
    │
    ▼
[vector_store.py]  ← Find similar document vectors in ChromaDB
    │
    ▼
[filters.py]       ← Remove irrelevant results (Week 14)
    │
    ▼
[rag_pipeline.py]  ← Build prompt with retrieved context
    │
    ▼
  Gemini API       ← Generate answer
    │
    ▼
[monitoring.py]    ← Check for hallucinations (Week 13)
    │
    ▼
[app.py]           ← Display answer, sources, confidence
```



## Setup



### 1. Clone the repository

```bash
git clone <repo-url>
cd student-rag-project
```



### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`



### 3. Install dependencies

```bash
pip install -r requirements.txt
```



### 4. Set up your Gemini API key

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and replace `your-gemini-api-key-here` with your actual key.
Get a free key at: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 5. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---



## File Descriptions


| File              | Purpose                                           |
| ----------------- | ------------------------------------------------- |
| `app.py`          | Streamlit web interface                           |
| `config.py`       | Configuration constants                           |
| `embeddings.py`   | Convert text to vector embeddings                 |
| `vector_store.py` | Store and search vectors with ChromaDB            |
| `data_loader.py`  | Sample tech documents                             |
| `rag_pipeline.py` | Central orchestration — ties everything together  |
| `conversation.py` | Conversation history (Week 11)                    |
| `security.py`     | Input validation and security (Week 12)           |
| `monitoring.py`   | Hallucination detection (Week 13)                 |
| `filters.py`      | Similarity filtering and fallbacks (Week 14)      |
| `workflow.py`     | Query rewriting and multi-hop retrieval (Week 15) |


---



## Weekly Progress

Update this checklist as you complete each week's assignment.

- [x] Week 10 — Ran the starter app and explored the codebase
- [x] Week 11 — Implemented conversation context
- [x] Week 12 — Implemented input security
- [x] Week 13 — Implemented hallucination monitoring
- [ ] Week 14 — Implemented filtering and fallbacks
- [ ] Week 15 — Implemented multi-step AI workflows

---

---

## Assignment: Week 14 — Filtering, Fallbacks, and Graceful Failure

**Learning objective:** Understand similarity thresholds and why production RAG systems need graceful degradation.

### Background

ChromaDB always returns results — even when no document is actually relevant to the query. Ask the app *"What is the best pizza topping?"* and it will still return the 3 "most similar" tech documents and attempt to generate an answer from them. Without filtering, you get hallucinated nonsense.

The solution: **similarity thresholds**. We only keep documents where the vector distance is below a cutoff. Documents that are too far away from the query get dropped. When nothing passes the filter, we **degrade gracefully** — return a helpful message rather than crashing or hallucinating.

### What to implement

**File 1 —** `filters.py`

Implement `filter_by_threshold()` and `get_fallback_response()`. The first is the core filtering logic (read the TODO). The second is just a well-written, helpful message — but think about what a user actually needs to know when their question can't be answered.

**File 2 —** `rag_pipeline.py`

Find the **Week 14 TODO** block in `run_rag()`. Add the filter step after retrieval. If `has_relevant_results()` returns False, return the fallback dict immediately.

Then wrap the `generate_answer()` call in a `try/except Exception as e:` block and call `handle_api_error(e)` in the except branch to return a user-friendly error dict.

### How to test

Ask a completely off-topic question:

- *"Who won the Super Bowl this year?"*

Before your implementation: the app tries to answer from irrelevant docs. After: it returns your fallback message.

### ✅ When done

Check off **Week 14** in the Weekly Progress section above, then delete this entire Week 14 assignment section.

---

---



## Assignment: Week 15 — Multi-Step AI Workflows

**Learning objective:** Understand how query quality affects retrieval, and how to improve it with LLM-powered pre-processing.

### Background

The quality of a RAG answer depends directly on what gets retrieved. And what gets retrieved depends on how similar the **query embedding** is to the **document embeddings**. If the user writes a vague or casual question, the resulting embedding may not match well with the more formal language in our documents.

Two solutions:

1. **Query rewriting**: Use an LLM to rewrite the user's question into a clearer, more specific version before embedding it. Better query → better embedding → better retrieval.
2. **Query decomposition**: Some questions are actually multiple questions. Split them and search separately, then combine results. This is called **multi-hop retrieval**.



### What to implement

**File 1 —** `workflow.py`

Implement `rewrite_query()` and `decompose_query()`. Both use Gemini to process the query before it reaches the vector store. Read the TODO comments — they explain exactly what each function should do and why.

Note: `multi_hop_retrieve()` is already implemented for you — it uses your `decompose_query()` internally.

**File 2 —** `rag_pipeline.py`

Find the **Week 15 TODO** block at the top of `run_rag()`. Add the query rewriting step before retrieval. The rewritten query gets passed to `retrieve_context()` instead of the original.

### How to test

Try a vague follow-up question:

- First ask: *"What is Python?"*
- Then ask: *"What else can it do in the real world?"*

Before your implementation: "it" doesn't get resolved and retrieval is poor. After: the rewriter uses conversation context to turn it into a specific query.

### ✅ When done

Check off **Week 15** in the Weekly Progress section above, then delete this entire Week 15 assignment section. You've built a full, production-patterned RAG system — nice work.

---

