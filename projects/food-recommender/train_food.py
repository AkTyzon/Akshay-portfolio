"""
Content-based recipe recommender.

Parses free-text ingredient lines into normalised ingredient tokens, builds a
TF-IDF profile per recipe over ingredients + tags + nutrition band, and ranks
neighbours by cosine similarity. Evaluated against a random baseline using
tag agreement, since the dataset carries no user ratings.
"""
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_N = 10
SEED = 42
rng = np.random.default_rng(SEED)

# quantities, units and prep words carry no signal about what a dish *is*
UNITS = {
    "cup", "cups", "teaspoon", "teaspoons", "tablespoon", "tablespoons", "tsp", "tbsp",
    "pound", "pounds", "lb", "lbs", "ounce", "ounces", "oz", "gram", "grams", "g", "kg",
    "ml", "liter", "liters", "l", "pinch", "dash", "clove", "cloves", "can", "cans",
    "package", "packages", "slice", "slices", "stick", "sticks", "bunch", "quart",
    "pint", "small", "large", "medium", "extra", "inch",
}
PREP = {
    "minced", "chopped", "diced", "sliced", "softened", "melted", "grated", "shredded",
    "peeled", "seeded", "crushed", "ground", "fresh", "freshly", "finely", "coarsely",
    "optional", "taste", "room", "temperature", "cut", "into", "pieces", "halved",
    "quartered", "divided", "plus", "more", "for", "and", "or", "the", "of", "to",
    "with", "about", "such", "as", "if", "needed", "well", "very", "thinly",
}
STOP = UNITS | PREP


def norm_ingredients(lines):
    out = []
    for ln in lines:
        if not isinstance(ln, str):
            continue
        ln = ln.replace("<hr>", " ").replace("\r", " ").lower()
        ln = re.sub(r"\([^)]*\)", " ", ln)          # drop parentheticals
        ln = re.sub(r"[\d/\.\-–]+", " ", ln)         # drop quantities
        ln = re.sub(r"[^a-z\s]", " ", ln)
        toks = [t for t in ln.split() if len(t) > 2 and t not in STOP]
        out.extend(toks)
    return out


def nutrition_band(r):
    """Coarse nutrition descriptors — real signal where present, silent where not."""
    bands = []
    cal, srv = r.get("calories"), r.get("servings") or 1
    if cal:
        per = cal / max(srv, 1)
        bands.append("light" if per < 300 else "moderate" if per < 600 else "rich")
    p, c = r.get("protein"), r.get("carbs")
    if p and c:
        bands.append("highprotein" if p > c else "highcarb")
    return bands


def main():
    db = json.load(open("recipes.json"))
    recipes = [r for r in db.values() if r.get("ingredients") and r.get("name")]
    print(f"{len(recipes)} usable recipes")

    docs, items = [], []
    for r in recipes:
        ing = norm_ingredients(r["ingredients"])
        tags = [t.lower() for t in (r.get("tags") or [])]
        # tags weighted x3: they describe the dish, ingredients describe its parts
        docs.append(" ".join(ing + tags * 3 + nutrition_band(r)))
        cal, srv = r.get("calories"), r.get("servings") or 1
        items.append({
            "id": str(r["id"]),
            "t": r["name"],
            "tags": tags[:4],
            "ing": len(set(ing)),
            "kcal": int(cal / max(srv, 1)) if cal else None,
            "mins": (r.get("preptime") or 0) + (r.get("cooktime") or 0) or None,
        })

    vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    X = vec.fit_transform(docs)
    print(f"tf-idf matrix: {X.shape[0]} recipes x {X.shape[1]} features")

    sim = cosine_similarity(X, dense_output=True).astype(np.float32)
    np.fill_diagonal(sim, 0.0)

    # --- evaluation: do neighbours share a tag more often than chance? ---
    tagsets = [set(i["tags"]) for i in items]
    n = len(items)
    model_hits = base_hits = 0
    for i in range(n):
        if not tagsets[i]:
            continue
        top5 = np.argsort(-sim[i])[:5]
        model_hits += sum(1 for j in top5 if tagsets[i] & tagsets[j]) / 5
        rand5 = rng.choice([j for j in range(n) if j != i], 5, replace=False)
        base_hits += sum(1 for j in rand5 if tagsets[i] & tagsets[j]) / 5
    scored = sum(1 for t in tagsets if t)
    metrics = {
        "recipes": n,
        "vocabulary": int(X.shape[1]),
        "tag_agreement_at_5": round(model_hits / scored, 4),
        "tag_agreement_at_5_random": round(base_hits / scored, 4),
        "lift_over_random": round((model_hits / scored) / max(base_hits / scored, 1e-9), 2),
    }
    print("metrics:", json.dumps(metrics, indent=2))

    neighbours = {}
    for i, it in enumerate(items):
        order = np.argsort(-sim[i])[:TOP_N]
        neighbours[it["id"]] = [
            {"id": items[j]["id"], "s": round(float(sim[i][j]), 3)}
            for j in order if sim[i][j] > 0.02
        ]

    items.sort(key=lambda x: x["t"])
    out = {
        "meta": {
            "dataset": "tabatkins/recipe-db (public recipe corpus)",
            "method": "content-based TF-IDF over normalised ingredients, tags and nutrition bands",
            "note": "no user ratings exist in this corpus, so this is content-based, not collaborative",
            "metrics": metrics,
        },
        "items": items,
        "neighbours": neighbours,
    }
    json.dump(out, open("food_data.json", "w"), separators=(",", ":"))
    print(f"wrote food_data.json ({len(items)} recipes)")


if __name__ == "__main__":
    main()
