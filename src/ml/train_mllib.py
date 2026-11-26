# src/ml/save_best_model.py
import argparse
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_estimator(args: argparse.Namespace, features_col: str, num_features: int, num_classes: int):
    if args.model == "rf":
        return RandomForestClassifier(
            labelCol="label_idx",
            featuresCol=features_col,
            numTrees=args.num_trees,
            maxDepth=args.max_depth,
            seed=args.seed,
        )
    if args.model == "lr":
        return LogisticRegression(
            labelCol="label_idx",
            featuresCol=features_col,
            family="multinomial",
            regParam=args.reg_param,
            elasticNetParam=args.elastic_net,
            maxIter=args.max_iter,
        )
    if args.model == "svm":
        lsvc = LinearSVC(regParam=args.reg_param, maxIter=args.max_iter)
        return OneVsRest(
            classifier=lsvc,
            labelCol="label_idx",
            featuresCol=features_col,
        )
    if args.model == "mlp":
        hidden = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
        layers = [num_features, *hidden, num_classes]
        return MultilayerPerceptronClassifier(
            labelCol="label_idx",
            featuresCol=features_col,
            layers=layers,
            maxIter=args.max_iter,
            stepSize=args.step_size,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Train and save the selected Spark MLlib fault model.")
    parser.add_argument("--model", choices=["rf", "lr", "svm", "mlp"], required=True)
    parser.add_argument(
        "--train-path",
        default="hdfs://localhost:9000/spark/data/kaist/train",
    )
    parser.add_argument(
        "--test-path",
        default="hdfs://localhost:9000/spark/data/kaist/test",
    )
    parser.add_argument(
        "--output-path",
        default="hdfs://localhost:9000/spark/models/kaist_fault_best",
    )
    # RF hyperparameters
    parser.add_argument("--num-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    # Linear / SVM hyperparameters
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--elastic-net", type=float, default=0.0)
    # MLP hyperparameters
    parser.add_argument("--hidden-layers", default="32,16", help="Comma-separated hidden layer sizes, e.g. '32,16'")
    parser.add_argument("--step-size", type=float, default=0.03)
    # Shared training options
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("SaveBestAcousticFaultModel")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    train_df = spark.read.parquet(args.train_path).cache()
    test_df = spark.read.parquet(args.test_path).cache()
    feature_cols = [c for c in train_df.columns if c.startswith("f_")]
    num_features = len(feature_cols)
    num_classes = int(train_df.select("label").distinct().count())

    indexer = StringIndexer(inputCol="label", outputCol="label_idx", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="keep")
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    classifier = build_estimator(
        args,
        features_col="scaled_features",
        num_features=num_features,
        num_classes=num_classes,
    )

    pipeline = Pipeline(stages=[indexer, assembler, scaler, classifier])
    model: PipelineModel = pipeline.fit(train_df)

    indexer_model = model.stages[0]
    labels = list(indexer_model.labels)
    idx_to_label = F.create_map(
        *[item for i, lbl in enumerate(labels) for item in (F.lit(float(i)), F.lit(lbl))]
    )

    preds = (
        model.transform(test_df)
        .withColumn("pred_label", idx_to_label[F.col("prediction")])
        .cache()
    )

    print(f"\n=== Evaluation for best model: {args.model.upper()} ===")
    for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
        ev = MulticlassClassificationEvaluator(
            labelCol="label_idx", predictionCol="prediction", metricName=metric
        )
        print(f"  {metric:18s}: {ev.evaluate(preds):.4f}")

    cm_pdf = (
        preds.groupBy("label", "pred_label")
        .count()
        .toPandas()
        .pivot(index="label", columns="pred_label", values="count")
        .fillna(0)
        .astype(int)
    )
    print("\n=== Confusion Matrix (rows=actual, cols=predicted) ===")
    print(cm_pdf)

    model.write().overwrite().save(args.output_path)
    print(f"\nSaved best PipelineModel to: {args.output_path}")

    preds.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()