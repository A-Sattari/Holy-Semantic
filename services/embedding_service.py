import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple

def load_embedded_verses(model) -> faiss.IndexFlatIP:
    """
    Load the pre-embedded and indexed verses from the disk.
    If the file does not exist, it will create a new one.
    """
    index_file_path = os.path.join("data", "quran_embeddings.index")

    if os.path.exists(index_file_path):
        print("🔃 Loading existing index from disk...")
        return faiss.read_index(index_file_path)
    
    print("Embedding and storing Quran on the disk...")
    index = _embed_and_index_verses(model)
    faiss.write_index(index, index_file_path)
    print("✅ Done")
    return index

def _embed_and_index_verses(model) -> faiss.IndexFlatIP:
    verses = _get_quran_verses()

    embeddings = _embed(verses, model)
    index = faiss.IndexFlatIP(embeddings.shape[1]) # Empty FAISS index with the size of embeddings array
    index.add(embeddings)

    return index

def _embed(texts: List[str], model) -> np.ndarray:
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")  # FAISS requires float32
    faiss.normalize_L2(embeddings)
    return embeddings

def _get_quran_verses() -> Tuple[List[str], List[str]]:
    FILE_NAME = os.path.join("data", "quran-en.json")
    with open(FILE_NAME, "r", encoding="utf-8") as wrappedText:
        versesAndId: List[Dict[str, str]] = json.load(wrappedText)
    
    verses = [v["text"] for v in versesAndId]
    return verses