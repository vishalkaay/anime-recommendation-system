"""Content-based anime recommendations.

Each title is described by its name, genres and type. Those are turned into
TF-IDF vectors, and titles are compared by cosine similarity.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_PATH = Path(__file__).with_name("anime.csv")


def load_anime(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the dataset and tidy the columns the recommender uses."""
    anime = pd.read_csv(path)
    # Titles in the dataset contain HTML entities, e.g. "Gintama&#039;".
    anime["name"] = anime["name"].map(html.unescape)
    anime["genre"] = anime["genre"].fillna("")
    anime["type"] = anime["type"].fillna("")
    return anime.reset_index(drop=True)


class Recommender:
    """Finds the titles most similar to a given one."""

    def __init__(self, anime: pd.DataFrame) -> None:
        self.anime = anime
        features = anime["name"] + " " + anime["genre"] + " " + anime["type"]
        # TF-IDF rows are L2-normalised, so a dot product between two rows is
        # their cosine similarity. That lets one title be compared with all the
        # others on demand, instead of storing every pair up front.
        self.vectors = TfidfVectorizer(stop_words="english").fit_transform(features)
        self._index: dict[str, int] = {}
        for position, name in enumerate(anime["name"]):
            self._index.setdefault(name.casefold(), position)

    @classmethod
    def from_csv(cls, path: str | Path = DATA_PATH) -> Recommender:
        return cls(load_anime(path))

    @property
    def names(self) -> list[str]:
        return self.anime["name"].tolist()

    def similar(self, name: str, k: int = 5) -> pd.DataFrame:
        """Return the ``k`` titles most similar to ``name``, best first.

        The result has the dataset's columns plus ``similarity``. It is empty if
        ``name`` is not in the dataset.
        """
        position = self._index.get(name.casefold())
        if position is None or k <= 0:
            return self.anime.iloc[0:0].assign(similarity=[])

        scores = linear_kernel(self.vectors[position], self.vectors).ravel()
        scores[position] = -1.0  # never recommend the title itself
        k = min(k, len(scores) - 1)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return self.anime.iloc[top].assign(similarity=scores[top])

    def recommend(self, name: str, k: int = 5) -> list[str]:
        """Return the names of the ``k`` titles most similar to ``name``."""
        return self.similar(name, k)["name"].tolist()
