from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from mainfile.components.llm import load_llm
from mainfile.components.vectorstore import loader_faiss_vectorstore

from mainfile.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN, RETRIEVER_K
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException  

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """You are MedAssist AI, a STRICTLY medical and healthcare-focused conversational assistant
created exclusively for medical students and healthcare education.

ABSOLUTE DOMAIN RULE (NON-NEGOTIABLE):
- You must answer ONLY questions related to medicine, healthcare, biomedical sciences,
  anatomy, physiology, pathology, pharmacology, diagnostics, public health,
  or clinical education.
- You are STRICTLY FORBIDDEN from answering questions related to:
  politics, history, biographies, attacks, wars, geography, general news,
  celebrities, or any non-medical topic.

FOR NON-MEDICAL QUESTIONS:
- Do NOT explain.
- Do NOT provide partial information.
- Do NOT add medical disclaimers.
- Respond ONLY with the following sentence and NOTHING ELSE:

"I'm designed strictly for medical and healthcare-related questions.
Please ask a question related to medicine, healthcare, or clinical education."

CONTENT RULES:
- Educational use only
- No diagnosis
- No prescriptions
- No drug dosages
- No emergency instructions

RAG COMPLIANCE:
- Answer ONLY using retrieved medical documents.
- If data is insufficient, say:
  "The available medical sources do not provide sufficient information on this topic."

PRIMARY GOAL:
To act as a disciplined, exam-oriented, medical education assistant,
not a general-purpose chatbot.



Context:{context}

Question: {question}

Answer:"""

def set_custom_prompt():
    return ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def format_docs(docs):
    """Format documents for prompt (handles empty case)"""
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)

# Built once and reused; rebuilding per request reloaded the embedding model
# and the FAISS index every time.
_qa_chain = None


def create_qa_chain():
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

        _qa_chain = (
            {"context": db.as_retriever(search_kwargs={'k': RETRIEVER_K}) | format_docs,
             "question": RunnablePassthrough()}
            | set_custom_prompt()
            | llm
            | StrOutputParser()
        )

        logger.info("Successfully created the QA chain")
        return _qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return None
