# Suggesteria

A content-based recommender for anime. Pick a title and it suggests the most similar ones from a catalogue of about 12,000, using each title's name, genres and type.

**Try it: [suggesteria-anime.streamlit.app](https://suggesteria-anime.streamlit.app/)**

```console
$ python -c "from recommender import Recommender; print(Recommender.from_csv().recommend('Fullmetal Alchemist: Brotherhood', 3))"
['Fullmetal Alchemist: Brotherhood Specials', 'Fullmetal Alchemist', 'Fullmetal Alchemist: Brotherhood - 4-Koma Theater']
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Choose a title, set how many recommendations you want, and it lists them with their genres, type, rating and similarity score.

## How it works

1. `anime.csv` is loaded and each title's name, genres and type are joined into one piece of text.
2. That text is turned into a TF-IDF vector, so rare words such as a franchise name count for more than common ones such as "TV".
3. To recommend, the chosen title's vector is compared with every other title by cosine similarity, and the closest ones are returned.

The comparison is done on demand. Building the vectors takes about a tenth of a second and a lookup takes a few milliseconds, so nothing needs to be precomputed or stored.

## Files

| File | Purpose |
|:--|:--|
| `recommender.py` | Loads the data and finds similar titles. |
| `app.py` | The Streamlit interface. |
| `evaluate.py` | Scores the recommender over the whole catalogue. |
| `tests/` | Unit tests. Run them with `pip install pytest` and then `pytest`. |
| `anime.csv` | The dataset: name, genres, type, episodes, rating and member count. |

## Evaluation

```console
$ python evaluate.py
Genre overlap@5: 0.4930
Coverage@5:      0.8554
```

- **Genre overlap** is the mean Jaccard similarity between a title's genres and the genres of the titles recommended for it. On average, a recommendation shares about half of its genres with the title it was recommended for.
- **Coverage** is the share of the catalogue that appears in at least one title's top five: about 86%.

Because the title's name is one of the features, sequels and spin-offs rank highly. That is usually what you want, but it also means two unrelated titles that share a common word can be matched.
