from typing import NamedTuple

class SearchResult(NamedTuple):
    score: float
    id: str
    verse: str
