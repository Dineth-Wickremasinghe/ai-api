# check_chroma.py
from services.vector_store import get_vector_store

vector_store = get_vector_store()
collection = vector_store._collection
count = collection.count()
print(f"Documents in ChromaDB: {count}")

if count > 0:
    results = vector_store.similarity_search("fabric waste", k=3)
    print(f"\nTest search returned {len(results)} results")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"    Preview: {doc.page_content[:100]}...")
else:
    print("⚠️  ChromaDB is EMPTY — no documents ingested yet")