# Cascade Learning on Spark

Two-stage classification cascade on UCI Covertype (581,012 rows, 54 features, 7 classes),
built with Spark MLlib.

**[Results →](https://aktyzon.github.io/Akshay-portfolio/demos/cascade/)**

## The idea
A cheap model handles every case it is confident about; only the residual escalates to an
expensive model. The question is not whether the big model is more accurate — it is — but how
much of its cost can be avoided without giving that accuracy back.

| Stage | Model | Accuracy | Fit | Predict |
|---|---|---|---|---|
| 1 — cheap | LogisticRegression (30 iters) | 72.06% | 4.6s | 0.7s |
| 2 — expensive | RandomForest (60 trees, depth 15) | 79.04% | 25.8s | 1.5s |

## Cost / accuracy tradeoff
Stage-1 confidence is the max class probability; rows below the threshold escalate.

| Threshold | Escalated | Stage-2 work avoided | Cascade accuracy | Accuracy retained |
|---|---|---|---|---|
| 0.50 | 4.1% | 95.9% | 73.03% | 92.4% |
| 0.60 | 22.8% | 77.2% | 75.83% | 95.9% |
| 0.70 | 44.0% | 56.0% | 77.70% | 98.3% |
| **0.75** | 55.7% | **44.3%** | 78.34% | **99.1%** |
| 0.80 | 68.2% | 31.8% | 78.77% | 99.7% |
| 0.90 | 91.0% | 9.0% | 79.04% | 100.0% |

The useful region is the knee, not either end. At 0.70 you skip 56% of the expensive work and keep
98.3% of the accuracy; by 0.90 the accuracy is identical to the full model but there is almost
nothing left to save. Which point to pick is a product decision about what a millisecond of latency
is worth — not something the model decides.

## Run it
```bash
pip install pyspark        # needs a JRE (e.g. openjdk 17)
curl -LO https://archive.ics.uci.edu/static/public/31/covertype.zip
unzip covertype.zip && gunzip covtype.data.gz
python train_cascade.py    # writes cascade_results.json
```
Runs on `local[4]` with 4 GB driver memory; the full sweep takes a few minutes.
