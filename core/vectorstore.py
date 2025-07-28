#vectorstore.py
import json
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from config.constants import MEMORY_FILE

# Load BGE-M3 model (singleton to avoid reloading)
_bge_model = None

def get_bge_model():
    global _bge_model
    if _bge_model is None:
        print("🔄 Loading BGE-M3 model...")
        _bge_model = SentenceTransformer("BAAI/bge-m3")
        print("✅ BGE-M3 model loaded")
    return _bge_model

def _bge_embed(texts: list[str]) -> list[list[float]]:
    model = get_bge_model()
    return model.encode(texts, normalize_embeddings=True)

class BGEEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return _bge_embed([f"passage: {t}" for t in texts])

    def embed_query(self, text):
        return _bge_embed([f"query: {text}"])[0]

def load_docs_from_memory_json():
    docs = []
    if not MEMORY_FILE.exists():
        print("⚠️ Memory file doesn't exist yet, returning empty docs")
        return docs

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    turn = item.get("turn", {})
                    user = turn.get("user", "")
                    assistant = turn.get("assistant", "")
                    if user or assistant:
                        text = f"User: {user}\nAssistant: {assistant}"
                        docs.append(Document(page_content=text))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping bad line in memory.json: {e}")
    except Exception as e:
        print(f"⚠️ Error loading memory file: {e}")
    return docs

def build_store():
    embeddings = BGEEmbeddings()
    memory_docs = load_docs_from_memory_json()

    if not memory_docs:
        print("📝 No memory documents found, creating empty store with dummy doc")
        # Create a dummy document to avoid empty index issues
        dummy_doc = Document(page_content="No conversations yet")
        store = FAISS.from_documents([dummy_doc], embedding=embeddings)
        return store

    print(f"🔍 Building FAISS index from {len(memory_docs)} documents...")
    store = FAISS.from_documents(memory_docs, embedding=embeddings)

    # Optional: log top-4 matches to sanity check
    if memory_docs:
        try:
            query = "Explain LangGraph"
            scores, indices = store.index.search(
                np.array([embeddings.embed_query(query)]), k=min(4, len(memory_docs))
            )
            print("\n🔍 FAISS Sample Search Results:")
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(memory_docs):
                    print(f"{i+1}. {memory_docs[idx].page_content[:80]}... (score: {scores[0][i]:.4f})")
        except Exception as e:
            print(f"⚠️ Error during sample search: {e}")

    return store

__all__ = ["build_store", "BGEEmbeddings"]
