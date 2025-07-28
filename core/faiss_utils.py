"""
core/faiss_utils.py
Wrapper for building/querying a FAISS index with BGE-M3 embeddings.
Now delegates actual logic to vectorstore.py (LangChain-based).
"""
from typing import List
import numpy as np
from vectorstore import BGEEmbeddings as LC_BGEEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Proxy for langchain-style embeddings
class BGEEmbeddings:
    """Wrapper that mimics the original API but uses LangChain's BGE under the hood."""
    def __init__(self):
        self.embeddings = LC_BGEEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

# Mimics old return type: simple namespace with .index and .texts
def load_faiss_index(texts: List[str]):
    if not texts:
        raise ValueError("No texts supplied to build a FAISS index")

    docs = [Document(page_content=t) for t in texts]
    embeddings = LC_BGEEmbeddings()
    store = FAISS.from_documents(docs, embedding=embeddings)

    # Create a fake namespace to match the original structure
    class _Store:
        pass
    wrapper = _Store()
    wrapper.index = store.index  # Raw FAISS index (for .search compatibility)
    wrapper.texts = texts        # Needed for reverse lookup

    return wrapper
