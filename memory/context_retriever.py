#memory/context_retriever.py
from typing import List
import numpy as np

from memory.turn_memory import load_memory
from memory.long_term_memory import load_long_term_memory
from core.faiss_utils import load_faiss_index, BGEEmbeddings


def retrieve_top_memories(
    user_query: str,
    k_short: int = 2,
    k_long: int = 2,
    user_id: str | None = None,
    score_threshold: float = 0.35,
) -> tuple[list[str], list[str]]:
    """Return (short_matches, long_matches)."""
    # -------- short-term -------- #
    turns = load_memory()
    short_texts = [
        f"User: {t['user']}\nAssistant: {t['assistant']}" for t in turns if t.get("user")
    ]
    short_matches: List[str] = []
    if short_texts:
        store = load_faiss_index(short_texts)
        D, I = store.index.search(
            np.array([BGEEmbeddings().embed_query(user_query)]), k=k_short
        )
        for idx, score in zip(I[0], D[0]):
            if idx >= 0 and score >= score_threshold:
                short_matches.append(short_texts[idx])

    # -------- long-term -------- #
    summaries = [e.get("summary", "") for e in load_long_term_memory(user_id)]
    long_matches: List[str] = []
    if summaries:
        store = load_faiss_index(summaries)
        D, I = store.index.search(
            np.array([BGEEmbeddings().embed_query(user_query)]), k=k_long
        )
        for idx, score in zip(I[0], D[0]):
            if idx >= 0 and score >= score_threshold:
                long_matches.append(summaries[idx])

    return short_matches, long_matches
