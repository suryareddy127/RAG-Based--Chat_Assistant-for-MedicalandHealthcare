# Intelligent Multimodal RAG-Based Conversational Chat Assistant for Medical and Healthcare Awareness

A context-aware medical study assistant built on a **multimodal Retrieval-Augmented Generation (RAG)** pipeline. Every answer is grounded in a local medical knowledge base, and when a question is visual the assistant also retrieves the relevant figures from the source encyclopedia and explains what is actually shown in them.

The assistant answers in **text + images + an explanation of each image**, remembers the conversation so follow-up questions work naturally, and refuses anything outside medicine.

> **Retrieval is local; generation is not.** The PDFs, embeddings and FAISS indexes never leave the machine. The question, the retrieved context and the retrieved figures are sent to Groq's hosted API to produce the written answer and the figure descriptions.

---

## Key Features

### Retrieval & grounding
* **Multimodal retrieval** — answers combine encyclopedia prose with figures taken from the same source, each shown with its caption, page number and a description.
* **Hybrid figure search** — CLIP scores the question against figure *pixels*, MiniLM scores it against figure *captions*, and the two rankings are merged with reciprocal rank fusion. CLIP alone is weak on fine medical wording; it ranked "corneal abrasion" above "cataract surgery" for a cataract query, and the fusion corrects that.
* **Calibrated relevance gate** — figures appear only when they genuinely match. Measured on this corpus, relevant figures score 0.30+, medical questions with no matching figure about 0.26, and non-medical questions about 0.21; the gate sits at **0.28**. Past the best match a figure must also be supported by the caption index before it is shown.
* **Real image understanding** — figure descriptions come from a vision model that looks at the image rather than restating the caption, and it says so plainly when a retrieved figure does not actually answer the question.
* **Hallucination mitigation** — a strict RAG prompt confines every medical claim to the retrieved context, with a fixed response when the sources are insufficient and an explicit rule for partial coverage.

### Conversation
* **Conversational memory** — follow-ups are rewritten into standalone questions before retrieval, so *"What causes it?"* becomes *"What causes candidiasis?"*. History is also passed to the answer prompt.
* **Server-side sessions** — transcripts live on the server behind a chat id; the browser cookie is a flat **43 bytes**, `HttpOnly` and `SameSite=Lax`. Storing the transcript in the cookie previously grew it ~1 KB per turn and would have silently blown the 4093-byte browser limit around turn five.

### Safety & interface
* **Healthcare safety guardrails** — educational framing only: no diagnosis of a specific person, no prescriptions, no drug dosages, no emergency instructions. Off-topic questions get a single fixed refusal line.
* **Formatted answers** — responses render as real markdown (headings, bold terms, bullets, tables) sanitised against a tag allowlist, so model output cannot inject HTML.
* **Local vector indexing** — three FAISS indexes on disk, no external vector database.

---

## System Architecture & Data Flow

```text
                              ┌─────────────────────────┐
  [User question] ──────────► │   Flask  (mainfile/app) │
                              └────────────┬────────────┘
                                           │  chat id  ──► server-side store
                                           ▼                (transcript, not in cookie)
                            ┌──────────────────────────────┐
                            │  CONDENSE  (chat history)    │
                            │  "What causes it?"           │
                            │     ▼                        │
                            │  "What causes candidiasis?"  │
                            └──────────────┬───────────────┘
                                           │  standalone question
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
      ── TEXT RETRIEVAL ──                              ── FIGURE RETRIEVAL ──
      MiniLM embedding                                   CLIP(question)   MiniLM(question)
             │                                                 │               │
             ▼                                                 ▼               ▼
      FAISS  db_faiss                                    FAISS db_images  FAISS db_captions
             │                                                 └───────┬───────┘
             │ top-k chunks                           reciprocal rank fusion + 0.28 gate
             │                                                         │ 0-2 figures
             ▼                                                         ▼
    ┌───────────────────────────────┐                     ┌──────────────────────────┐
    │  ANSWER PROMPT                │ ◄── figure_note ─── │  (were figures found?)   │
    │  scope · grounding · safety   │                     └────────────┬─────────────┘
    │  conversation · format        │                                  │
    └───────────────┬───────────────┘                                  ▼
                    ▼                                      ┌──────────────────────────┐
         Groq chat model                                   │  Groq VISION model       │
                    |                                      │  sees each figure        │
                    │                                      └────────────┬─────────────┘
                    │  markdown answer                                  │ description
                    └──────────────────────┬────────────────────────────┘
                                           ▼
                           markdown → sanitised HTML + figure grid
                                           ▼
                                   [Answer + images + explanations]
```

**Why figures are searched *before* the answer is written:** the answer prompt is told whether any figure will actually be displayed, so the model never writes "as shown in the figure below" when nothing was retrieved.

---

## Project Structure

```
│
├── mainfile/                     # Core application
│   ├── common/                   # Shared infrastructure
│   │   ├── chat_store.py         # Server-side transcripts (file cache, TTL + size cap)
│   │   ├── custom_exception.py   # Error type carrying file/line context
│   │   └── logger.py             # Daily rotating log setup
│   │
│   ├── components/               # RAG pipeline
│   │   ├── document_loader.py    # PDF text loading/splitting + figure & caption extraction
│   │   ├── embeddings.py         # MiniLM (prose) + CLIP (figures, shared image/text space)
│   │   ├── vectorstore.py        # All three FAISS indexes + hybrid figure search
│   │   ├── dataloader.py         # Ingestion pipeline (builds every index)
│   │   ├── llm.py                # Groq clients: chat model + vision model
│   │   └── retreive.py           # All prompts + retrieval/answer orchestration
│   │
│   ├── config/config.py          # Every tunable setting, read from .env
│   ├── templates/index.html      # Chat UI (markdown bubbles + figure grid)
│   └── app.py                    # Flask routes, sessions, markdown rendering
│
├── data/                         # Source PDFs
│   └── figures/                  # Extracted figures + figures.json   (generated)
├── vectorstore/
│   ├── db_faiss/                 # Text index      (MiniLM)
│   ├── db_images/                # Figure index    (CLIP)             (generated)
│   └── db_captions/              # Caption index   (MiniLM)           (generated)
├── logs/                         # Daily application logs
│
├── requirements.txt
├── setup.py
├── .env                          # API keys and settings (not committed)
└── README.md
```

Items marked *(generated)* are build artifacts and are gitignored — they rebuild with one command.

---

## Tech Stack

- Language: Python

- Pipeline Orchestration: LangChain

- Vector Computation Engine: FAISS (Facebook AI Similarity Search)

- Embedding Architectures: Hugging Face Transformer Models (Text and Images)
- Pdf and Images: PyMuPDF, Pillow

- Backend Framework: Flask,-session cachelib

- Frontend Interface: HTML, Tailwind CSS
---

## Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/suryareddy127/RAG-Based--Chat_Assistant-for-MedicalandHealthcare.git
cd RAG-Based--Chat_Assistant-for-MedicalandHealthcare
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install the project as a package
```bash
pip install -e .
```

### 4. Configure your environment

Create a `.env` file in the project root. `GROQ_API_KEY` is **required** — the app cannot answer without it.

```ini
GROQ_API_KEY="your_groq_api_key"
HF_TOKEN="your_huggingface_token"
SECRET_KEY="a_long_random_string"
```

Generate a secret key with:
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Without `SECRET_KEY` the app still runs, but a random key is generated at every startup and all sessions are dropped on restart.

### 5. Place your source PDFs
Put one or more medical PDFs in `data/`.

### 6. Run the ingestion pipeline
Builds the text index, extracts figures and builds both figure indexes. This downloads the embedding models on first run and takes several minutes.

```bash
python -m mainfile.components.dataloader
```

### 7. Run the web application
```bash
python mainfile/app.py
```

Open **http://127.0.0.1:5000**.

> The first question after a restart takes ~30 seconds while the embedding and CLIP models load into memory. Subsequent questions take roughly 1–3 seconds.

---

This project is built as a **medical education and public-awareness tool**, not a clinical one. Its value lies in what it refuses to do as much as in what it does.

**What it is useful for**

* **Medical and nursing students** revising a topic — answers arrive structured by Overview, Causes, Symptoms, Diagnosis and Management, with the relevant textbook figure alongside, which is how most medical material is actually learned.
* **Visual learning.** Pathology, dermatology and radiology are taught through images. Being able to ask *"what does oral candidiasis look like?"* and receive the actual figure **plus** a description of what is visible in it is closer to how a tutor would explain it than text alone.
* **Studying a reference work conversationally.** A 759-page encyclopedia is hard to browse. Here you can ask a question, then keep asking follow-ups without repeating yourself.
* **General health awareness** for non-specialists, with guardrails that keep the system from drifting into advice it should not give.
* **A worked reference implementation** of a multimodal RAG system: figure extraction, two embedding spaces, hybrid rank fusion, a calibrated relevance gate, history-aware retrieval, and prompt-level safety design.

