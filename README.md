# HEX Internship — Python AI/ML Projects

A hands-on internship project collection covering LangChain RAG, LangGraph agents, tool-calling, CrewAI multi-agent systems, and more.

---

## Project Structure

```
HEX_INTERN_PYTHON/
├── 01_langchain_rag/          # LangChain + RAG pipelines
├── 02_langgraph/              # LangGraph stateful agent flows
├── 03_tools_and_agents/       # Tool-calling agents & HR pipeline
├── 04_crewai/                 # CrewAI multi-agent crews
├── notebooks/                 # Jupyter notebooks (exploration & learning)
├── data/
│   ├── pdfs/                  # Source PDF documents
│   ├── databases/             # SQLite databases
│   └── vector_stores/         # FAISS indexes and caches
├── assets/                    # Generated graph/workflow images
├── docs/                      # Reference notes
├── jd/                        # Job description files (for resume pipeline)
└── resumes/                   # Candidate resume PDFs (for resume pipeline)
```

---

## Setup

```bash
# Install dependencies (uses uv)
uv sync

# Activate virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

Copy `.env` and fill in your keys:
```
GEMINI_API_KEY=...
GROQ_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=hex_intern
```

---

## 01 — LangChain RAG  (`01_langchain_rag/`)

**Tech stack:** LangChain · FAISS · HuggingFace Embeddings · Google Gemini · Groq (LLaMA)

### `rag_chunking_strategies.py`
Compares three chunking strategies (Recursive, Character, Spacy) on a product manual PDF using FAISS + Groq.

```bash
python 01_langchain_rag/rag_chunking_strategies.py
```

### `self_correcting_rag.py`
Self-corrective RAG pipeline — evaluates retrieval quality and rewrites the query with Gemini when context is insufficient.

```bash
python 01_langchain_rag/self_correcting_rag.py
```

### `memory_chat.py`
Multi-turn conversation with persistent in-memory chat history using `RunnableWithMessageHistory` and Gemini.

```bash
python 01_langchain_rag/memory_chat.py
```

---

## 02 — LangGraph (`02_langgraph/`)

**Tech stack:** LangGraph · Groq (LLaMA) · DuckDuckGo Search · Structured Output (Pydantic)

### `portfolio_state_graph.py`
Introductory LangGraph example — state machine that converts a USD portfolio value to total-with-tax then to INR.

```bash
python 02_langgraph/portfolio_state_graph.py
# Saves graph diagram → assets/graph_output.png
```

### `code_gen_agent.py`
Agentic LangChain code generator with security guards, doc fetching via DuckDuckGo, code execution/validation, and retry loop (up to 5 iterations).

```bash
python 02_langgraph/code_gen_agent.py
# Saves flow diagram → assets/code_gen_flow.png
```

### `telecom_multi_agent.py`
Multi-agent telecom customer support system with 6 agents: Competitor A & B → Evaluator → Adversary → Refiner → Federated consolidation.

```bash
python 02_langgraph/telecom_multi_agent.py
# Saves agent graph → assets/telecom_agents.png
```

---

## 03 — Tools & Agents (`03_tools_and_agents/`)

**Tech stack:** LangGraph · LangChain Tools · SQLite · Groq SDK · LangChain Groq · pdfplumber · LangSmith

### `sqlite_react_agent.py`
LangGraph prebuilt ReAct agent that answers natural-language questions by querying a SQLite employee database using a custom `@tool`.

```bash
python 03_tools_and_agents/sqlite_react_agent.py
# Reads from: data/databases/test.db
```

### `resume_pipeline_groq.py`
Automated resume screening pipeline using raw Groq SDK. Nodes: PDF parsing → resume scoring → interview scheduling → response simulation → hiring decision.

```bash
# Place job description in jd/  and candidate PDFs in resumes/
python 03_tools_and_agents/resume_pipeline_groq.py
```

### `resume_pipeline_langsmith.py`
Same pipeline but uses LangChain `ChatGroq` wrapper for automatic LangSmith token tracking and tracing.

```bash
# Place job description in jd/  and candidate PDFs in resumes/
python 03_tools_and_agents/resume_pipeline_langsmith.py
# Saves workflow diagram → assets/workflow_diagram.png
```

---

## 04 — CrewAI (`04_crewai/`)

**Tech stack:** CrewAI · Google Gemini

### `ai_trends_crew.py`
Two-agent crew: a Research Analyst gathers AI/ML trends (2024-2025), then a Content Writer produces an 800-1000 word blog post.

```bash
python 04_crewai/ai_trends_crew.py
```

---

## Notebooks (`notebooks/`)

```bash
jupyter notebook notebooks/
# or
jupyter lab notebooks/
```

| File | Purpose |
|---|---|
| `hello_chat.ipynb` | First LangChain chat interaction |
| `hello_rag.ipynb` | Basic RAG pipeline intro |
| `hello_pdf.ipynb` | Chat with a PDF document |
| `hello_rag_web.ipynb` | RAG over web content |
| `langchain_tools.ipynb` | LangChain tool calling |
| `graph_tools.ipynb` | LangGraph with tool nodes |
| `learn_langchain.ipynb` | LangChain LCEL learning exercises |
| `pdf_rag.ipynb` | Full PDF RAG pipeline |
| `pydantic_practice.ipynb` | Pydantic structured outputs |
| `practice.ipynb` | General practice exercises |
| `final_practice.ipynb` | End-of-module exercises |

---

## Tech Stack Summary

| Area | Libraries |
|---|---|
| LLMs | `langchain-groq`, `langchain-google-genai`, `crewai` |
| Embeddings | `langchain-huggingface`, `google-generativeai` |
| Vector Store | `faiss-cpu`, `langchain-community` |
| Agent Frameworks | `langgraph`, `crewai` |
| PDF Parsing | `pymupdf`, `pypdf`, `pdfplumber` |
| Web Search | `ddgs` (DuckDuckGo) |
| Observability | `langsmith` |
| Data Validation | `pydantic` |
| Package Manager | `uv` |
