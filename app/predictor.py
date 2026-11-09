from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
model = None


def load_model(path=MODEL_PATH):
    global model
    model = joblib.load(path)
    return model


def get_model():
    if model is None:
        if MODEL_PATH.exists():
            return load_model()
        raise RuntimeError(
            "Модель не загружена. Сначала обучите её или загрузите .pkl."
        )
    return model


def predict_records(records):
    clf = get_model()
    frame = pd.DataFrame(records)
    expected = (
        list(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else None
    )
    # Pipeline feature_names_in_ contains original input columns.
    if expected:
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"Не хватает признаков: {missing}")
        frame = frame[expected]
    pred = clf.predict(frame)
    probs = clf.predict_proba(frame)
    classes = list(map(str, clf.classes_))
    output = []
    for i, label in enumerate(pred):
        output.append(
            {
                "features": records[i],
                "predicted_position": str(label),
                "probabilities": {
                    classes[j]: round(float(probs[i][j]), 5)
                    for j in range(len(classes))
                },
            }
        )
    return output
