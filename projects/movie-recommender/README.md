# Movie Recommender — item-item collaborative filtering

Item-item collaborative filtering on the MovieLens `ml-latest-small` ratings matrix.

**[Live demo →](https://aktyzon.github.io/Akshay-portfolio/demos/movie/)**

## Method
Ratings are mean-centred per user (so a generous rater and a harsh rater contribute
comparably), assembled into a sparse users × items matrix, and scored with cosine
similarity between item columns. Only films with at least 20 ratings are recommendable —
below that, similarity is noise.

## Evaluation
Leave-last-out: for each user, hide their most recent 4★+ film, score candidates from the
rest of their history, and check whether the hidden film returns in the top 10.

| Metric | Model | Popularity baseline |
|---|---|---|
| Hit-rate@10 | **15.5%** | 5.9% |
| Lift | **2.63×** | — |
| Mean rank when hit | 3.78 | — |

594 users evaluated. The popularity baseline matters: a recommender that cannot beat
"show everyone the most-rated films" has not learned anything.

## Run it
```bash
pip install pandas numpy scipy scikit-learn
mkdir -p data && curl -L https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o data/ml.zip && unzip -d data data/ml.zip
python train_movie.py     # writes movie_data.json
```

Similarities are precomputed offline and exported as JSON, so the demo is a static page —
no server, no cold start.
