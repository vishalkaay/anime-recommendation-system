"""Measure the recommender over the whole dataset.

Two numbers are reported for the top-k recommendations of every title:

* genre overlap: the mean Jaccard similarity between the genres of a title
  and the genres of each title recommended for it;
* coverage: the share of the catalogue that is recommended at least once.
"""

from __future__ import annotations

import argparse

from recommender import Recommender


def genre_set(genres: str) -> frozenset[str]:
    return frozenset(g.strip() for g in genres.split(",") if g.strip())


def evaluate(recommender: Recommender, k: int = 5) -> tuple[float, float]:
    genres = [genre_set(g) for g in recommender.anime["genre"]]
    overlap_total = 0.0
    overlap_count = 0
    recommended: set[int] = set()

    for position, name in enumerate(recommender.anime["name"]):
        for other in recommender.similar(name, k).index:
            recommended.add(other)
            union = genres[position] | genres[other]
            if union:
                overlap_total += len(genres[position] & genres[other]) / len(union)
                overlap_count += 1

    overlap = overlap_total / overlap_count if overlap_count else 0.0
    coverage = len(recommended) / len(recommender.anime)
    return overlap, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-k", type=int, default=5, help="recommendations per title")
    args = parser.parse_args()

    overlap, coverage = evaluate(Recommender.from_csv(), args.k)
    print(f"Genre overlap@{args.k}: {overlap:.4f}")
    print(f"Coverage@{args.k}:      {coverage:.4f}")


if __name__ == "__main__":
    main()
