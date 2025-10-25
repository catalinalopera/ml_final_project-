import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

def train_and_log(model_name, model, experiment_name, X_train, y_train, X_val, y_val, model_dir):
    """Train a model and log all relevant information to MLflow"""
    # Set or create the experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Evaluate
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save model locally
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ Model {model_name} trained and logged in experiment: {experiment_name}")
        print(f"   Accuracy: {acc:.4f}, F1-score: {f1:.4f}\n")


def train_experiments(train_path, val_path, model_dir):
    # Load data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    # Assume last column is the target
    target_col = train.columns[-1]
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]
    X_val, y_val = val.drop(columns=[target_col]), val[target_col]

    # -------- Experiment 1: Random Forest --------
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    train_and_log(
        model_name="RandomForestClassifier",
        model=rf,
        experiment_name="RandomForest_Experiment",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=model_dir
    )

    # -------- Experiment 2: Logistic Regression --------
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    train_and_log(
        model_name="LogisticRegression",
        model=lr,
        experiment_name="LogisticRegression_Experiment",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=model_dir
    )

    # -------- Experiment 3: Support Vector Machine --------
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    train_and_log(
        model_name="SupportVectorMachine",
        model=svm,
        experiment_name="SVM_Experiment",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=model_dir
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train three ML experiments with MLflow")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--val_path", type=str, default="data/processed/val.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    train_experiments(args.train_path, args.val_path, args.model_dir)