from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from services.vector_store import get_vector_store
from config import settings

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always cite which document/page your answer comes from.

Context:
{context}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def query_rag(question: str) -> dict:

    from google import genai
    from langchain_core.messages import HumanMessage, SystemMessage

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # Retrieve relevant chunks
    retrieved_docs = await retriever.ainvoke(question)
    context = format_docs(retrieved_docs)

    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always cite which document/page your answer comes from.

Context:
{context}

Question: {question}"""


    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt
    )

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })

    return {
        "answer": response.text,
        "sources": sources
    }