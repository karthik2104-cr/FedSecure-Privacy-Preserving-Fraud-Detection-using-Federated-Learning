
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fraud_config import FEATURES, TARGET, CLASS_DESCRIPTION


RANDOM_STATE = 42


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def add_noise(X, y, feature_noise=0.08, label_noise=0.05, missing_rate=0.03):
    X_noisy = X.copy().astype(float)

    std = np.std(X_noisy, axis=0)
    std[std == 0] = 1.0
    noise = np.random.normal(0, feature_noise * std, X_noisy.shape)
    X_noisy = X_noisy + noise

    mask = np.random.rand(*X_noisy.shape) < missing_rate
    X_noisy[mask] = np.nan

    X_noisy = pd.DataFrame(X_noisy, columns=FEATURES)
    X_noisy = X_noisy.fillna(X_noisy.median())

    y_noisy = y.copy()
    n_flip = int(label_noise * len(y_noisy))
    if n_flip > 0:
        flip_idx = np.random.choice(len(y_noisy), n_flip, replace=False)
        unique_labels = np.unique(y_noisy)

        for i in flip_idx:
            alternatives = unique_labels[unique_labels != y_noisy[i]]
            y_noisy[i] = np.random.choice(alternatives)

    return X_noisy, y_noisy


def split_clients(X: pd.DataFrame, y: np.ndarray, n_clients: int, non_iid: bool = True):
    client_data = []

    data = X.copy()
    data[TARGET] = y

    if non_iid:

        data = data.sort_values(TARGET).reset_index(drop=True)

    shards = np.array_split(data.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True), n_clients)

    for idx, shard in enumerate(shards):
        X_client = shard[FEATURES].copy()
        y_client = shard[TARGET].to_numpy()
        client_data.append((f"Bank_{idx + 1}", X_client, y_client))

    return client_data


def init_global_model(n_features: int, n_classes: int) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=64,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )


    X_dummy = np.random.rand(max(n_classes, 10), n_features)
    y_dummy = np.arange(max(n_classes, 10)) % n_classes
    model.fit(X_dummy, y_dummy)
    return model


def get_model_weights(model: MLPClassifier):
    return {
        "coefs_": [w.copy() for w in model.coefs_],
        "intercepts_": [b.copy() for b in model.intercepts_],
    }


def set_model_weights(model: MLPClassifier, weights: dict):
    model.coefs_ = [w.copy() for w in weights["coefs_"]]
    model.intercepts_ = [b.copy() for b in weights["intercepts_"]]


def scale_client_update(weights: dict, factor: float):
    return {
        "coefs_": [w * factor for w in weights["coefs_"]],
        "intercepts_": [b * factor for b in weights["intercepts_"]],
    }


def add_weights(w1: dict, w2: dict):
    return {
        "coefs_": [a + b for a, b in zip(w1["coefs_"], w2["coefs_"])],
        "intercepts_": [a + b for a, b in zip(w1["intercepts_"], w2["intercepts_"])],
    }


def zeros_like(weights: dict):
    return {
        "coefs_": [np.zeros_like(w) for w in weights["coefs_"]],
        "intercepts_": [np.zeros_like(b) for b in weights["intercepts_"]],
    }


def generate_mask_like(weights: dict):
    return {
        "coefs_": [np.random.normal(0, 0.01, size=w.shape) for w in weights["coefs_"]],
        "intercepts_": [np.random.normal(0, 0.01, size=b.shape) for b in weights["intercepts_"]],
    }


def subtract_weights(w1: dict, w2: dict):
    return {
        "coefs_": [a - b for a, b in zip(w1["coefs_"], w2["coefs_"])],
        "intercepts_": [a - b for a, b in zip(w1["intercepts_"], w2["intercepts_"])],
    }


def fedavg_aggregate(client_payloads: list[tuple[dict, int]]):
    total_samples = sum(n for _, n in client_payloads)
    agg = zeros_like(client_payloads[0][0])

    for weights, n_samples in client_payloads:
        scaled = scale_client_update(weights, n_samples / total_samples)
        agg = add_weights(agg, scaled)

    return agg


def fedsecure_aggregate(client_payloads: list[tuple[dict, int]]):

    total_samples = sum(n for _, n in client_payloads)

    masked_payloads = []
    masks = []

    for weights, n_samples in client_payloads:
        scaled = scale_client_update(weights, n_samples / total_samples)
        mask = generate_mask_like(scaled)
        masked = add_weights(scaled, mask)
        masked_payloads.append(masked)
        masks.append(mask)

    agg_masked = zeros_like(masked_payloads[0])
    for masked in masked_payloads:
        agg_masked = add_weights(agg_masked, masked)

    total_mask = zeros_like(masks[0])
    for mask in masks:
        total_mask = add_weights(total_mask, mask)

    recovered = subtract_weights(agg_masked, total_mask)
    return recovered


def train_local_model(global_weights: dict, X_client: pd.DataFrame, y_client: np.ndarray, n_classes: int):
    local_model = init_global_model(X_client.shape[1], n_classes)
    set_model_weights(local_model, global_weights)


    for _ in range(3):
        local_model.fit(X_client, y_client)

    return get_model_weights(local_model), len(X_client)


def evaluate_global_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, y_pred


def main(dataset_path: str, output_dir: str, n_clients: int, rounds: int, secure: bool):
    np.random.seed(RANDOM_STATE)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(dataset_path))

    X = df[FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    X, y = add_noise(X.values, y)
    X = pd.DataFrame(X, columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

    client_splits = split_clients(X_train_scaled, y_train, n_clients=n_clients, non_iid=True)

    global_model = init_global_model(X_train_scaled.shape[1], len(label_encoder.classes_))
    global_weights = get_model_weights(global_model)

    history = []

    for rnd in range(1, rounds + 1):
        client_payloads = []

        for client_name, X_client, y_client in client_splits:
            local_weights, n_samples = train_local_model(
                global_weights,
                X_client,
                y_client,
                n_classes=len(label_encoder.classes_),
            )
            client_payloads.append((local_weights, n_samples))

        if secure:
            global_weights = fedsecure_aggregate(client_payloads)
            aggregation_type = "FedSecure"
        else:
            global_weights = fedavg_aggregate(client_payloads)
            aggregation_type = "FedAvg"

        set_model_weights(global_model, global_weights)

        round_metrics, _ = evaluate_global_model(global_model, X_test_scaled, y_test, label_encoder)
        round_metrics["round"] = rnd
        round_metrics["aggregation"] = aggregation_type
        history.append(round_metrics)

        print(
            f"Round {rnd:02d} | {aggregation_type} | "
            f"Accuracy={round_metrics['accuracy']:.4f} | "
            f"F1={round_metrics['f1_macro']:.4f}"
        )

    final_metrics, y_pred = evaluate_global_model(global_model, X_test_scaled, y_test, label_encoder)

    artifact = {
        "model": global_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": FEATURES,
        "best_model_name": "Federated MLP",
        "class_labels": list(label_encoder.classes_),
        "aggregation_type": "FedSecure" if secure else "FedAvg",
        "federated_rounds": rounds,
        "num_clients": n_clients,
        "metrics": final_metrics,
        "history": history,
        "class_descriptions": CLASS_DESCRIPTION,
    }

    joblib.dump(artifact, out_dir / "federated_fraud_model_artifact.joblib")

    with open(out_dir / "federated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_metrics": final_metrics,
                "history": history,
                "aggregation_type": "FedSecure" if secure else "FedAvg",
                "num_clients": n_clients,
                "rounds": rounds,
            },
            f,
            indent=2,
        )

    holdout_df = X_test.copy()
    holdout_df["true_label"] = label_encoder.inverse_transform(y_test)
    holdout_df["predicted_label"] = label_encoder.inverse_transform(y_pred)
    holdout_df.to_csv(out_dir / "federated_holdout_predictions.csv", index=False)

    print("\nFederated training complete.")
    print(f"Aggregation: {'FedSecure' if secure else 'FedAvg'}")
    print(json.dumps(final_metrics, indent=2)[:1500])
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated fraud training simulation")
    parser.add_argument("--dataset", default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated banks")
    parser.add_argument("--rounds", type=int, default=10, help="Federated rounds")
    parser.add_argument("--secure", action="store_true", help="Use FedSecure masked aggregation")
    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        n_clients=args.clients,
        rounds=args.rounds,
        secure=args.secure,
    )
