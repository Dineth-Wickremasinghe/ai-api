from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from config import settings
from google import genai
from typing import List


client = genai.Client(api_key=settings.GEMINI_API_KEY)

class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=self.model,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return result.embeddings[0].values

embeddings = GeminiEmbeddings(model="models/gemini-embedding-001")

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

# services/vector_store.py

def clear_vector_store() -> None:
    """Delete and recreate the ChromaDB collection."""
    import chromadb
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_client.delete_collection(settings.COLLECTION_NAME)
    # Calling get_vector_store() will recreate it fresh
    get_vector_store()