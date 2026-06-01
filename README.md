# Intelligent RAG-Based Conversational Chat Assistant for Medical and Healthcare Awareness

An enterprise-grade, privacy-focused, and context-aware medical assistant built using a **Retrieval-Augmented Generation (RAG)** pipeline. The system eliminates common large language model (LLM) hallucinations by grounding all generative responses entirely in verified local medical knowledge bases using high-speed semantic vector retrieval.

---

##  Key Features

* **Hallucination Mitigation:** Employs a strict RAG framework ensuring the LLM relies strictly on the provided context.
* **Local, Secure Vector Indexing:** Utilizes a local Facebook AI Similarity Search (**FAISS**) vector store—eliminating external cloud database latency and subscription dependencies.
* **Semantic Context Awareness:** Converts raw medical knowledge into deep mathematical vectors using **Hugging Face Transformers**, allowing the engine to understand user intent depth beyond literal keyword matches.
* **Built-in Healthcare Safety Filters:** Prompt-engineered validation layers actively block the system from dispensing clinical diagnoses or issuing prescriptive medicine routines, making it safe for general public awareness.
* **Asynchronous Web Interface:** Features a clean, modular UI built with a responsive HTML/CSS configuration interacting dynamically with a Flask backend.

---

##  System Architecture & Data Flow

```text
[User Query] 
     │
     ▼
[UI / Flask Backend] ──► [Query Preprocessing & Cleaning]
                                      │
                                      ▼
[Local FAISS DB] ◄─── [Hugging Face Embedding Conversion]
          │
          ▼
(Top-k Semantic Retrieval)
          │
          ▼
[LangChain RAG Pipeline] ──► [Prompt Engineering Guardrails]
                                      │
                                      ▼
                               [LLM Generation]
                                      │
                                      ▼
                            [Safe User Response]

```
---
## Project Structure
```

│
├── mainfile/                # Core application files
│   ├── common/              # Shared utilities and helper functions
│   ├── components/          # Modular components (UI, retriever, etc.)
│   ├── config/              # Configuration settings and parameters
│   ├── templates/           # UI templates and frontend layouts
│   ├── __init__.py          # Package initializer
│   ├── app.py               # Main application entry point
│   └── vectorstore/         # Vector database integration
│
├── data/                    # Raw medical reference text datasets and source PDFs
├── logs/                    # Log files for debugging and monitoring
├── RAG_Medical_assistant.egg-info/  # Package metadata
│
├── requirements.txt         # Python production dependencies and frameworks
├── setup.py                 # Packages the repository modularly as a distributable package
├── .gitignore               # Ignored files for Git
└── README.md                # Project documentation
```
---
## Tech Stack
- Language: Python

- Pipeline Orchestration: LangChain

- Vector Computation Engine: FAISS (Facebook AI Similarity Search)

- Embedding Architectures: Hugging Face Transformer Models

- Backend Framework: Flask

- Frontend Interface: HTML, CSS
---
## Installation & Quickstart
1. Clone the Repository
   ```
   git clone [https://github.com/suryareddy127/RAG-Based--Chat_Assistant-for-MedicalandHealthcare.git](https://github.com/suryareddy127/RAG-Based--Chat_Assistant-for-MedicalandHealthcare.git)
   cd RAG-Based--Chat_Assistant-for-MedicalandHealthcare
   ```
2. Set Up a Virtual Environment & Install Dependencies
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Build local environment package packages
   ```
   pip install -e .
   ```
4. Run the Data Ingestion Pipeline
   ```
   python mainfile/ingestion.py
   ```
5. Run the Web Application
   ```
   python mainfile/app.py
   ```
