import pandas as pd
import pytest

from recommender import Recommender, load_anime


@pytest.fixture
def recommender() -> Recommender:
    anime = pd.DataFrame(
        {
            "name": ["Space Pirates", "Space Pirates 2", "Quiet Garden", "Robot Wars"],
            "genre": ["Action, Sci-Fi", "Action, Sci-Fi", "Slice of Life", "Action, Mecha"],
            "type": ["TV", "Movie", "TV", "TV"],
        }
    )
    return Recommender(anime)


def test_most_similar_title_comes_first(recommender: Recommender) -> None:
    assert recommender.recommend("Space Pirates", k=1) == ["Space Pirates 2"]


def test_never_recommends_the_title_itself(recommender: Recommender) -> None:
    assert "Space Pirates" not in recommender.recommend("Space Pirates", k=10)


def test_lookup_ignores_case(recommender: Recommender) -> None:
    assert recommender.recommend("space pirates", k=1) == ["Space Pirates 2"]


def test_unknown_title_gives_no_recommendations(recommender: Recommender) -> None:
    assert recommender.recommend("Not In The Dataset") == []


def test_k_larger_than_the_catalogue(recommender: Recommender) -> None:
    assert len(recommender.recommend("Quiet Garden", k=50)) == 3


def test_scores_are_sorted(recommender: Recommender) -> None:
    scores = recommender.similar("Space Pirates", k=3)["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_dataset_loads_with_clean_titles() -> None:
    anime = load_anime()
    assert not anime["name"].str.contains("&#039;").any()
    assert not anime[["genre", "type"]].isna().any().any()
