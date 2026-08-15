from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from fraud_config import CLASS_LABELS, CLASS_DESCRIPTION, FEATURES, FRAUD_SIGNAL_MAP, TARGET


MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=280, max_depth=16, min_samples_split=4, min_samples_leaf=2,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=320, max_depth=18, min_samples_split=4, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'SVM (RBF)': SVC(probability=True, class_weight='balanced', C=2.5, gamma='scale', random_state=42),
}


def add_noise(X, y, feature_noise=0.1, label_noise=0.08, missing_rate=0.05):
    X_noisy = X.copy()


    noise = np.random.normal(0, feature_noise * np.std(X_noisy, axis=0), X_noisy.shape)
    X_noisy = X_noisy + noise


    mask = np.random.rand(*X_noisy.shape) < missing_rate
    X_noisy[mask] = np.nan

    X_noisy = pd.DataFrame(X_noisy, columns=FEATURES)
    X_noisy = X_noisy.fillna(X_noisy.median())


    y_noisy = y.copy()
    n_flip = int(label_noise * len(y))
    flip_idx = np.random.choice(len(y), n_flip, replace=False)

    unique_labels = np.unique(y)

    for i in flip_idx:
        if y[i] == 0:
            y_noisy[i] = np.random.choice(unique_labels[1:])
        else:
            y_noisy[i] = 0

    return X_noisy, y_noisy


def make_pipeline(model):
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, FEATURES)],
        remainder='drop',
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def evaluate_model(model_name, pipeline, x_train, y_train, x_test, y_test, label_encoder):
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'roc_auc_ovr': float(roc_auc_score(y_test, y_proba, multi_class='ovr')),
        'classification_report': classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        ),
    }
    return metrics, y_pred, y_proba


def plot_model_comparison(results: dict, output_dir: Path):
    comparison_df = pd.DataFrame(results).T[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']]
    ax = comparison_df.plot(kind='bar', figsize=(11, 6), rot=20)
    ax.set_title('Algorithm Comparison on Synthetic Fraud Dataset')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png')
    plt.close()


def plot_confusion(y_true, y_pred, label_encoder, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap='Blues')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()


def plot_multiclass_roc(y_test, y_proba, label_encoder, output_dir: Path):
    classes = np.arange(len(label_encoder.classes_))
    y_test_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(10, 7))
    for idx, class_name in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_proba[:, idx])
        auc_value = roc_auc_score(y_test_bin[:, idx], y_proba[:, idx])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_value:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png')
    plt.close()


def plot_feature_importance(best_pipeline, output_dir: Path):
    model = best_pipeline.named_steps['model']
    if not hasattr(model, 'feature_importances_'):
        return

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    importances.head(15).sort_values().plot(kind='barh', figsize=(9, 6))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()


def create_research_metadata(output_dir: Path):
    research_notes = {
        'problem_scope': 'Multiclass fraud detection across six major fraud mechanisms and one legitimate class.',
        'feature_count': len(FEATURES),
        'classes': CLASS_LABELS,
        'class_descriptions': CLASS_DESCRIPTION,
        'signals_per_class': {k: len(v) for k, v in FRAUD_SIGNAL_MAP.items()},
        'recommended_hypothesis': (
            'Behavioral, transactional, geospatial, device, and network signals '
            'can jointly distinguish heterogeneous financial fraud typologies '
            'in a multiclass setting under noisy conditions.'
        ),
    }
    with open(output_dir / 'research_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(research_notes, f, indent=2)


def main(dataset_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)

    x = df[FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    x, y = add_noise(x.values, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    results = {}
    fitted_models = {}
    best_score = -np.inf

    for model_name, model in MODELS.items():
        pipeline = make_pipeline(model)
        metrics, y_pred, y_proba = evaluate_model(
            model_name, pipeline, x_train, y_train, x_test, y_test, label_encoder
        )

        results[model_name] = metrics
        fitted_models[model_name] = pipeline

        if metrics['f1_macro'] > best_score:
            best_score = metrics['f1_macro']
            best_model = pipeline
            best_pred = y_pred
            best_proba = y_proba
            best_name = model_name

    plot_model_comparison(results, out_dir)
    plot_confusion(y_test, best_pred, label_encoder, out_dir)
    plot_multiclass_roc(y_test, best_proba, label_encoder, out_dir)
    plot_feature_importance(best_model, out_dir)
    create_research_metadata(out_dir)

    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': FEATURES,
        'best_model_name': best_name,
        'metrics': results,
        'class_labels': list(label_encoder.classes_)
    }, out_dir / 'fraud_model_artifact.joblib')

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest Model: {best_name}")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset.csv')
    parser.add_argument('--output-dir', default='artifacts')
    args = parser.parse_args()

    main(args.dataset, args.output_dir)
