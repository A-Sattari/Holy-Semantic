import json
import os
import faiss
from typing import Dict, List
from models.search_result import SearchResult
from sentence_transformers import SentenceTransformer

def search(model: SentenceTransformer, index: faiss.IndexFlatIP, query: str, result_limit=5):
    """
    Search for the most relevant verses based on the query.
    Args:
        model (SentenceTransformer): The embedding model.
        index (faiss.IndexFlatIP): The FAISS index.
        query (str): The search query.
    Returns:
        List[SearchResult]: List of SearchResult tuples (verse, score, index, id).
    """

    embedded_query = _embed_query(model, query)
    scores, matched_indices = index.search(embedded_query, result_limit)
    verses, ids = _get_verse_id_list()

    return [
        SearchResult(score=float(scores[0][rank]), id=ids[i], verse=verses[i])
        for rank, i in enumerate(matched_indices[0])
    ]

def _embed_query(model: SentenceTransformer, search_query: str):
    embedded_query = model.encode([search_query], convert_to_numpy=True)
    faiss.normalize_L2(embedded_query)
    return embedded_query

def _get_verse_id_list():
    FILE_NAME = os.path.join("data", "quran-en.json")
    with open(FILE_NAME, "r", encoding="utf-8") as wrappedText:
        versesAndId: List[Dict[str, str]] = json.load(wrappedText)
    
    verses = [v["text"] for v in versesAndId]
    ids = [v.get("verse", "") for v in versesAndId]
    return verses, ids