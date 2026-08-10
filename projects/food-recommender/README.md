# Recipe Recommender — content-based similarity

Content-based recipe similarity over a public recipe corpus (556 recipes).

**[Live demo →](https://aktyzon.github.io/Akshay-portfolio/demos/food/)**

## Method
The real work is parsing. Ingredient lines arrive as free text — `2/3 cup panko` — and have
to become `panko`, or every recipe looks alike because they all contain cups and teaspoons.
Quantities, units and prep words are stripped, then ingredients are combined with dish tags
(weighted ×3, since a tag describes what a dish *is* while an ingredient describes a part of
it) and a coarse nutrition band into a TF-IDF profile per recipe.

## Evaluation
No user ratings exist in this corpus, so collaborative filtering is impossible and claiming a
hybrid model would be fiction. Quality is measured by tag agreement instead:

| Metric | Model | Random baseline |
|---|---|---|
| Tag agreement@5 | **96%** | 38.5% |
| Lift | **2.49×** | — |

Being content-based has a real advantage here: it can recommend a brand-new recipe with zero
interaction history, which a collaborative model cannot.

## Run it
```bash
pip install numpy scikit-learn
curl -LO https://raw.githubusercontent.com/tabatkins/recipe-db/master/db-recipes.json -o recipes.json
python train_food.py      # writes food_data.json
```
