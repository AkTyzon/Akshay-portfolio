"""
Item-item collaborative filtering on MovieLens (ml-latest-small).

Trains a cosine-similarity recommender over mean-centred ratings, evaluates it
against a popularity baseline with a leave-last-out protocol, and exports the
top-N neighbours per movie as JSON for a static front end.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

MIN_RATINGS = 20      # a movie needs this many ratings to be recommendable
TOP_N       = 12      # neighbours exported per movie
SEED        = 42

rng = np.random.default_rng(SEED)


def load():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    counts = ratings.groupby("movieId").size()
    keep = counts[counts >= MIN_RATINGS].index
    ratings = ratings[ratings.movieId.isin(keep)]
    movies = movies[movies.movieId.isin(keep)]
    return ratings, movies


def build_matrix(ratings):
    """Users x items sparse matrix of mean-centred ratings."""
    uids = {u: i for i, u in enumerate(sorted(ratings.userId.unique()))}
    mids = {m: i for i, m in enumerate(sorted(ratings.movieId.unique()))}
    user_mean = ratings.groupby("userId").rating.transform("mean")
    centred = ratings.rating - user_mean
    mat = csr_matrix(
        (centred.values,
         (ratings.userId.map(uids).values, ratings.movieId.map(mids).values)),
        shape=(len(uids), len(mids)),
    )
    return mat, uids, mids


def evaluate(ratings, mids, sim):
    """Leave-last-out hit-rate@10 vs a popularity baseline.

    For each user with enough history, hide their most recent highly-rated film,
    score candidates from the rest of their history, and check whether the held-out
    film appears in the top 10.
    """
    inv_mids = {v: k for k, v in mids.items()}
    pop = ratings.groupby("movieId").size().sort_values(ascending=False)
    pop_top = list(pop.index[:50])

    hits_cf = hits_pop = trials = 0
    ranks = []

    for uid, grp in ratings.sort_values("timestamp").groupby("userId"):
        liked = grp[grp.rating >= 4.0]
        if len(liked) < 6:
            continue
        held = liked.iloc[-1].movieId
        history = liked.iloc[:-1].movieId.tolist()
        if held not in mids:
            continue

        # aggregate similarity from the user's history
        idx = [mids[m] for m in history if m in mids]
        if not idx:
            continue
        scores = np.asarray(sim[idx].sum(axis=0)).ravel()
        for m in idx:
            scores[m] = -np.inf                      # never recommend seen items
        top = np.argsort(-scores)[:10]
        rec = [inv_mids[i] for i in top]

        trials += 1
        if held in rec:
            hits_cf += 1
            ranks.append(rec.index(held) + 1)
        if held in [p for p in pop_top if p not in history][:10]:
            hits_pop += 1

    return {
        "users_evaluated": trials,
        "hit_rate_at_10_cf": round(hits_cf / trials, 4),
        "hit_rate_at_10_popularity": round(hits_pop / trials, 4),
        "lift_over_popularity": round((hits_cf / trials) / max(hits_pop / trials, 1e-9), 2),
        "mean_rank_when_hit": round(float(np.mean(ranks)), 2) if ranks else None,
    }


def main():
    ratings, movies = load()
    mat, uids, mids = build_matrix(ratings)
    print(f"matrix: {mat.shape[0]} users x {mat.shape[1]} movies, {mat.nnz} ratings")

    sim = cosine_similarity(mat.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(sim, 0.0)

    metrics = evaluate(ratings, mids, sim)
    print("metrics:", json.dumps(metrics, indent=2))

    # export top-N neighbours per movie
    inv_mids = {v: k for k, v in mids.items()}
    meta = movies.set_index("movieId")
    counts = ratings.groupby("movieId").size()
    means = ratings.groupby("movieId").rating.mean()

    items, neighbours = [], {}
    for col in range(sim.shape[1]):
        mid = inv_mids[col]
        order = np.argsort(-sim[col])[:TOP_N]
        neighbours[str(mid)] = [
            {"id": int(inv_mids[j]), "s": round(float(sim[col][j]), 3)}
            for j in order if sim[col][j] > 0
        ]
        title = meta.loc[mid, "title"]
        items.append({
            "id": int(mid),
            "t": title,
            "g": meta.loc[mid, "genres"].split("|")[:3],
            "n": int(counts[mid]),
            "r": round(float(means[mid]), 2),
        })

    items.sort(key=lambda x: -x["n"])
    out = {
        "meta": {
            "dataset": "MovieLens ml-latest-small",
            "ratings": int(len(ratings)),
            "users": int(ratings.userId.nunique()),
            "movies": len(items),
            "min_ratings_per_movie": MIN_RATINGS,
            "method": "item-item cosine similarity on mean-centred ratings",
            "metrics": metrics,
        },
        "items": items,
        "neighbours": neighbours,
    }
    with open("movie_data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote movie_data.json with {len(items)} movies")


if __name__ == "__main__":
    main()
