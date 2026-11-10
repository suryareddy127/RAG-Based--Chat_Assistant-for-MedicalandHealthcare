"""Retrieval, prompting and answer assembly.

This module owns every prompt in the project and the orchestration that ties
the pieces together:

    question
      -> condensed against chat history           (CONDENSE_PROMPT)
      -> text chunks retrieved from the text index
      -> written answer                           (ANSWER_PROMPT)
      -> matching figures retrieved from the image indexes
      -> each figure described by the vision model (FIGURE_PROMPT)
"""

import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mainfile.components.llm import ask_about_image, load_llm
from mainfile.components.vectorstore import loader_faiss_vectorstore, search_figures

from mainfile.config.config import (
    FIGURES_DIR,
    HISTORY_TURNS,
    RETRIEVER_K,
)
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger = get_logger(__name__)


# =========================================================================
# 1. ANSWER PROMPT - the written reply
# =========================================================================

ANSWER_PROMPT = """You are MedAssist AI, a multimodal medical and healthcare
assistant built for medical students and healthcare education. You answer from
a medical encyclopedia and, when the question is visual, from figures taken
from that same encyclopedia.

=====================  1. SCOPE  =====================
Answer ONLY questions about medicine, healthcare, biomedical science, anatomy,
physiology, pathology, pharmacology, diagnostics, public health or clinical
education.

For ANY question outside that scope - politics, history, sport, geography,
celebrities, general news, programming, or small talk - do not explain, do not
partially answer, and do not add a disclaimer. Reply with exactly this line and
nothing else:

"I'm designed strictly for medical and healthcare-related questions. Please ask
a question related to medicine, healthcare, or clinical education."

=====================  2. GROUNDING  =====================
- Base every medical claim on the Context below. It is your only source.
- Never invent findings, numbers, studies or guidelines.
- If the Context does not cover the question, reply with exactly:
  "The available medical sources do not provide sufficient information on this
  topic."
- If the Context covers the question only partly, answer what it supports and
  state plainly which part is not covered.

=====================  3. SAFETY  =====================
Educational use only. Do not diagnose a specific person, do not prescribe, do
not give drug dosages, and do not give emergency instructions. Describe
management only in general educational terms.

=====================  4. CONVERSATION  =====================
- "Conversation so far" is earlier turns of this same chat. Use it to resolve
  follow-ups and to avoid repeating yourself.
- The Question has already been rewritten to stand on its own; answer it
  directly rather than asking what the user meant.

=====================  5. FIGURES  =====================
{figure_note}
- Figures are rendered by the interface beneath your answer, each with its own
  caption and description. Do not paste image links or markdown images.
- Never claim a figure exists when none was retrieved, and never describe the
  contents of a figure here - that description is produced separately.

=====================  6. FORMAT (Markdown)  =====================
- Open with a `##` heading naming the topic, e.g. "## Asthma".
- Group the body under `###` sub-headings chosen from: Overview, Key Features,
  Causes, Risk Factors, Symptoms, Diagnosis, Management, Complications,
  Prevention. Use only those that the Context supports.
- Under each sub-heading use `-` bullets, one idea per bullet.
- Begin a bullet with its key term in **bold**, then an en dash, then the
  explanation. Example: "- **Wheezing** - a whistling sound on expiration."
- Keep each bullet to one or two sentences.
- No code blocks, no closing summary, no "In conclusion".
- EXCEPTION: the two fixed sentences in sections 1 and 2 are returned as plain
  text, with no heading and no bullets.

=====================  INPUT  =====================
Conversation so far (may be empty):
{history}

Context:
{context}

Question: {question}

Answer:"""


# =========================================================================
# 2. CONDENSE PROMPT - conversational memory
# =========================================================================

CONDENSE_PROMPT = """Given the conversation below and a follow-up message,
rewrite the follow-up as a standalone question that can be understood on its
own, resolving pronouns such as "it", "that" or "they" against the
conversation.

Rules:
- Return ONLY the rewritten question, nothing else.
- If the follow-up is already standalone, return it unchanged.
- Keep the user's original wording wherever possible.
- Never answer the question; only rewrite it.

Conversation:
{history}

Follow-up: {question}

Standalone question:"""


# =========================================================================
# 3. FIGURE PROMPT - what the vision model is asked about each image
# =========================================================================

FIGURE_PROMPT = """You are MedAssist AI, a medical education assistant.

A medical student asked: "{question}"

The figure below was retrieved from a medical encyclopedia.
Its printed caption is: "{caption}"

Describe what is actually visible in this figure and how it relates to the
question, in 2-4 sentences, for a medical student.

Rules:
- Describe only what you can genuinely see; never invent detail.
- Educational framing only: no diagnosis, no treatment plan, no drug dosages.
- If the figure does not actually relate to the question, say so plainly.
- Plain prose. No headings, no bullet points."""


# =========================================================================
# Prompt helpers
# =========================================================================

FIGURE_NOTE_PRESENT = (
    "- Relevant figures WERE retrieved for this question. You may refer to them\n"
    "  naturally (\"the figure shows a typical presentation\"), but keep the\n"
    "  written answer complete on its own."
)
FIGURE_NOTE_ABSENT = (
    "- No figures were retrieved for this question. Answer in text only and do\n"
    "  not mention figures, images or diagrams at all."
)


def set_custom_prompt():
    """The answer prompt, as a LangChain template."""
    return ChatPromptTemplate.from_template(ANSWER_PROMPT)


def set_condense_prompt():
    """The follow-up rewriting prompt."""
    return ChatPromptTemplate.from_template(CONDENSE_PROMPT)


def format_docs(docs):
    """Format retrieved chunks for the prompt (handles the empty case)."""
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(messages, turns=HISTORY_TURNS):
    """Render recent chat turns for the prompts. Oldest first."""
    if not messages:
        return ""
    recent = messages[-turns * 2:]
    lines = []
    for message in recent:
        who = "User" if message.get("role") == "user" else "Assistant"
        text = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
        if text:
            lines.append(f"{who}: {text[:600]}")
    return "\n".join(lines)


# =========================================================================
# Chains - built once and reused
# =========================================================================

_qa_chain = None
_condense_chain = None


def create_qa_chain():
    """Text chain over {history, context, question, figure_note}."""
    global _qa_chain

    if _qa_chain is not None:
        return _qa_chain

    try:
        logger.info("Loading vector store for context")
        db = loader_faiss_vectorstore()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded")

        _qa_chain = set_custom_prompt() | llm | StrOutputParser()

        logger.info("Successfully created the QA chain")
        return _qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return None


def create_condense_chain():
    """Rewrites a follow-up into a standalone question using chat history."""
    global _condense_chain

    if _condense_chain is not None:
        return _condense_chain

    llm = load_llm()
    if llm is None:
        return None

    _condense_chain = set_condense_prompt() | llm | StrOutputParser()
    return _condense_chain


# =========================================================================
# Orchestration
# =========================================================================

def condense_question(question, messages):
    """Resolve a follow-up against history; falls back to the raw question."""
    history = format_history(messages)
    if not history:
        return question

    chain = create_condense_chain()
    if chain is None:
        return question

    try:
        rewritten = chain.invoke({"history": history, "question": question}).strip()
        rewritten = rewritten.strip('"').strip()
        if not rewritten or len(rewritten) > 400:
            return question
        logger.info(f"Condensed follow-up to: {rewritten}")
        return rewritten
    except Exception as e:
        logger.warning(f"Could not condense question, using it as-is: {e}")
        return question


def explain_figures(figures, question):
    """Attach an `explanation` to each figure; failures degrade to caption only."""
    enriched = []
    for figure in figures:
        item = dict(figure)
        path = os.path.join(FIGURES_DIR, figure["file"])

        if not os.path.exists(path):
            logger.warning(f"Figure file missing: {path}")
            item["explanation"] = ""
        else:
            item["explanation"] = ask_about_image(
                FIGURE_PROMPT.format(
                    question=question,
                    caption=figure.get("caption") or "(no caption printed)",
                ),
                path,
            )
        enriched.append(item)
    return enriched


def answer_question(question, messages=None, with_figures=True):
    """Full multimodal answer: text, plus relevant figures and explanations.

    Returns {"answer": str, "figures": [...], "standalone_question": str}.
    """
    messages = messages or []
    failed = {"answer": "", "figures": [], "standalone_question": question}

    try:
        chain = create_qa_chain()
        if chain is None:
            raise CustomException("QA chain could not be created")

        db = loader_faiss_vectorstore()
        if db is None:
            raise CustomException("Vector store not present or empty")

        # Conversational memory: resolve "it"/"that" before retrieving.
        standalone = condense_question(question, messages)

        docs = db.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(standalone)

        # Figures are searched before the answer is written so the prompt can
        # tell the model whether any will actually be shown.
        found = search_figures(standalone) if with_figures else []

        answer = chain.invoke({
            "history": format_history(messages) or "(no previous messages)",
            "context": format_docs(docs),
            "question": standalone,
            "figure_note": FIGURE_NOTE_PRESENT if found else FIGURE_NOTE_ABSENT,
        })

        figures = explain_figures(found, standalone) if found else []

        return {
            "answer": str(answer),
            "figures": figures,
            "standalone_question": standalone,
        }

    except Exception as e:
        logger.error(str(CustomException("Failed to answer question", e)))
        failed["answer"] = f"Error: {e}"
        return failed
