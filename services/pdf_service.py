import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings
import hashlib

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Extract text from PDF bytes and return LangChain Documents."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue  # skip blank pages

        documents.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": page_num + 1,
                "total_pages": len(doc)
            }
        ))

    doc.close()
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"📄 Split into {len(chunks)} chunks")
    return chunks

def generate_doc_id(filename: str) -> str:
    """Generate a consistent ID for deduplication."""
    return hashlib.md5(filename.encode()).hexdigest()