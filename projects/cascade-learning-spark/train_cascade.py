"""
Cascade learning on Spark — Covertype (581,012 rows, 54 features, 7 classes).

A cheap model answers everything it is confident about; only the residual is escalated
to an expensive model. The question the experiment answers is not "is the big model
better" (it is) but "how much of the big model's cost can be avoided without losing
its accuracy".

Run:  JAVA_HOME=... python train_cascade.py
"""
import json
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

SEED = 42
CONF_THRESHOLD = 0.75      # stage-1 confidence required to answer without escalating


def main():
    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("cascade-learning")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    cols = [f"f{i}" for i in range(54)] + ["label_raw"]
    df = spark.read.csv("covtype.data", inferSchema=True).toDF(*cols)
    df = df.withColumn("label", F.col("label_raw") - 1)          # 1..7 -> 0..6
    n = df.count()
    print(f"rows={n:,}  features=54  classes={df.select('label').distinct().count()}")

    df = VectorAssembler(inputCols=[f"f{i}" for i in range(54)], outputCol="features").transform(df)
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train.cache(); test.cache()
    n_train, n_test = train.count(), test.count()
    print(f"train={n_train:,}  test={n_test:,}")

    acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    results = {}

    # ---------- stage 1: cheap ----------
    t0 = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
    lr_model = lr.fit(train)
    t_lr_fit = time.time() - t0

    t0 = time.time()
    lr_test = lr_model.transform(test).cache()
    a_lr = acc.evaluate(lr_test)
    t_lr_pred = time.time() - t0
    print(f"stage 1 (logistic)     acc={a_lr:.4f}  fit={t_lr_fit:.1f}s  predict={t_lr_pred:.1f}s")

    # ---------- stage 2: expensive ----------
    t0 = time.time()
    rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                numTrees=60, maxDepth=15, seed=SEED)
    rf_model = rf.fit(train)
    t_rf_fit = time.time() - t0

    t0 = time.time()
    rf_test = rf_model.transform(test).cache()
    a_rf = acc.evaluate(rf_test)
    t_rf_pred = time.time() - t0
    print(f"stage 2 (random forest) acc={a_rf:.4f}  fit={t_rf_fit:.1f}s  predict={t_rf_pred:.1f}s")

    # ---------- cascade, swept across confidence thresholds ----------
    # stage-1 confidence = max class probability
    conf = lr_test.withColumn("conf", F.array_max(vector_to_array(F.col("probability")))).cache()

    sweep = []
    for th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        confident = conf.filter(F.col("conf") >= th)
        escalate = conf.filter(F.col("conf") < th)
        n_c, n_e = confident.count(), escalate.count()
        a_c = acc.evaluate(confident) if n_c else 0.0
        a_e = acc.evaluate(rf_model.transform(escalate.select("features", "label"))) if n_e else 0.0
        a_casc = ((a_c * n_c) + (a_e * n_e)) / n_test
        sweep.append({
            "threshold": th,
            "escalation_rate": round(n_e / n_test, 4),
            "cascade_accuracy": round(a_casc, 4),
            "accuracy_retained": round(a_casc / a_rf, 4),
            "stage2_work_avoided": round(1 - n_e / n_test, 4),
        })
        print(f"  th={th:.2f}  escalate={n_e/n_test:6.1%}  acc={a_casc:.4f}  "
              f"retained={a_casc/a_rf:.1%}")

    chosen = next(s for s in sweep if s["threshold"] == CONF_THRESHOLD)
    escalation_rate = chosen["escalation_rate"]
    a_cascade = chosen["cascade_accuracy"]
    n_esc = int(round(escalation_rate * n_test))
    n_conf = n_test - n_esc
    a_conf = a_esc = None
    t_casc_pred = t_lr_pred + t_rf_pred * escalation_rate
    print(f"cascade @{CONF_THRESHOLD}       acc={a_cascade:.4f}  escalated={escalation_rate:.1%}")

    results = {
        "dataset": "UCI Covertype",
        "rows": n, "train": n_train, "test": n_test,
        "features": 54, "classes": 7,
        "confidence_threshold": CONF_THRESHOLD,
        "stage1": {"model": "LogisticRegression(maxIter=30)",
                   "accuracy": round(a_lr, 4),
                   "fit_seconds": round(t_lr_fit, 1),
                   "predict_seconds": round(t_lr_pred, 1)},
        "stage2": {"model": "RandomForest(60 trees, depth 15)",
                   "accuracy": round(a_rf, 4),
                   "fit_seconds": round(t_rf_fit, 1),
                   "predict_seconds": round(t_rf_pred, 1)},
        "cascade": {"accuracy": round(a_cascade, 4),
                    "escalation_rate": round(escalation_rate, 4),
                    "rows_answered_by_stage1": n_conf,
                    "rows_escalated": n_esc,
                    "predict_seconds": round(t_casc_pred, 1)},
        "sweep": sweep,
        "summary": {
            "accuracy_retained_vs_stage2": round(a_cascade / a_rf, 4),
            "stage2_work_avoided": round(1 - escalation_rate, 4),
        },
    }
    with open("cascade_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + json.dumps(results["summary"], indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
