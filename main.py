import os
import sys
from typing import Final
from sentence_transformers import SentenceTransformer

# Prevent creation of __pycache__ and .pyc files globally
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from services import embedding_service, search_service

model: Final = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    print("💗In the Name of God💗")
    search_query = input("Search Quran: ")

    embed_index = embedding_service.load_embedded_verses(model)
    results = search_service.search(model, embed_index, search_query)
    for r in results:
        print(f"Score: {r.score:.3f}\nID: {r.id}\nVerse: {r.verse}\n")