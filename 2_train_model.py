"""Script 2 — Entrena y exporta el clasificador de señas (scikit-learn).

Uso:
    python 2_train_model.py                # usa data/dataset.npz (generado por 1_capture_data.py)
    python 2_train_model.py --synthetic    # datos sintéticos: solo para validar el pipeline sin cámara
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from core.landmarks import FRAME_DIM, sequences_to_features

RNG = np.random.default_rng(42)


def synthetic_dataset(n_per_class: int = 40):
    """Trayectorias sintéticas distintas por clase (movimiento de manos) + ruido. NO sirve para producción."""
    T = config.SEQ_LEN
    t = np.linspace(0, 1, T)[:, None]
    base = RNG.normal(0, 0.3, FRAME_DIM)
    X, y = [], []
    for label in range(len(config.SIGNS)):
        direction = RNG.normal(0, 1, FRAME_DIM)
        freq = 1 + label
        for _ in range(n_per_class):
            motion = np.sin(2 * np.pi * freq * t + RNG.uniform(0, 0.5)) * direction * 0.2
            X.append(base + motion + RNG.normal(0, 0.05, (T, FRAME_DIM)))
            y.append(label)
    return np.array(X, np.float32), np.array(y), list(config.SIGNS), "synthetic"


def load_dataset():
    if not config.DATASET_PATH.exists():
        raise SystemExit(f"No existe {config.DATASET_PATH}. Ejecuta 1_capture_data.py o usa --synthetic.")
    d = np.load(config.DATASET_PATH, allow_pickle=False)
    return d["X"], d["y"], [str(s) for s in d["labels"]], str(d["backend"])


def augment(X_seq: np.ndarray, y: np.ndarray, copies: int):
    """Jitter gaussiano + escala global + time-warp leve. Solo se aplica al conjunto de entrenamiento."""
    if copies <= 0:
        return X_seq, y
    T = X_seq.shape[1]
    out_X, out_y = [X_seq], [y]
    for _ in range(copies):
        scale = RNG.uniform(0.9, 1.1, (len(X_seq), 1, 1))
        noise = RNG.normal(0, 0.02, X_seq.shape)
        mask = (X_seq != 0)  # no "inventar" manos ausentes
        aug = (X_seq * scale + noise) * mask
        warp = np.clip(np.linspace(0, 1, T) ** RNG.uniform(0.85, 1.15), 0, 1) * (T - 1)
        aug = aug[:, np.round(warp).astype(int)]
        out_X.append(aug.astype(np.float32))
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--augment", type=int, default=3, help="copias aumentadas por muestra de train")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    X_seq, y, labels, backend = synthetic_dataset() if args.synthetic else load_dataset()
    print(f"Dataset: {X_seq.shape}  clases={labels}  backend={backend}")
    print("Muestras por clase:", {l: int((y == i).sum()) for i, l in enumerate(labels)})

    Xs_tr, Xs_te, y_tr, y_te = train_test_split(X_seq, y, test_size=args.test_size, stratify=y, random_state=42)
    X_tr, X_te = sequences_to_features(Xs_tr), sequences_to_features(Xs_te)

    candidates = {
        "logreg": LogisticRegression(C=0.5, max_iter=3000),
        "svc_rbf": SVC(C=5.0, gamma="scale", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    n_splits = max(2, min(5, np.bincount(y_tr).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {}
    for name, clf in candidates.items():  # CV sin aumento para no filtrar copias entre folds
        s = cross_val_score(make_pipeline(StandardScaler(), clf), X_tr, y_tr, cv=cv, scoring="f1_macro")
        scores[name] = s.mean()
        print(f"  {name:<14} F1-macro CV = {s.mean():.3f} ± {s.std():.3f}")
    best = max(scores, key=scores.get)
    print(f"Mejor modelo: {best}")

    Xa, ya = augment(Xs_tr, y_tr, args.augment)
    pipe = make_pipeline(StandardScaler(), candidates[best]).fit(sequences_to_features(Xa), ya)
    y_pred = pipe.predict(X_te)
    print("\nEvaluación en test (hold-out):")
    print(classification_report(y_te, y_pred, target_names=labels, digits=3))
    print("Matriz de confusión (filas = real):\n", confusion_matrix(y_te, y_pred))
    test_acc = float((y_pred == y_te).mean())

    # Modelo final: todos los datos (+ aumento)
    Xa, ya = augment(X_seq, y, args.augment)
    final = make_pipeline(StandardScaler(), candidates[best]).fit(sequences_to_features(Xa), ya)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": final, "labels": labels, "model_name": best,
        "seq_len": int(X_seq.shape[1]), "frame_dim": FRAME_DIM,
        "feature_version": config.FEATURE_VERSION, "backend": backend,
        "cv_f1": float(scores[best]), "test_acc": test_acc,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, config.MODEL_PATH)
    print(f"\nModelo exportado -> {config.MODEL_PATH}")
    if args.synthetic:
        print("AVISO: entrenado con datos SINTÉTICOS; no reconocerá señas reales. Captura datos con 1_capture_data.py.")


if __name__ == "__main__":
    main()
