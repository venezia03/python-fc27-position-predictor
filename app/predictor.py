
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"

FEATURES = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physicality",
]

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
            "Модель не найдена. Сначала запустите train.py."
        )
    return model


def predict_records(records):
    clf = get_model()
    frame = pd.DataFrame(records)

    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Не хватает характеристик: {missing}")

    frame = frame[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if frame.isnull().any().any():
        raise ValueError("Все 6 характеристик должны быть числами.")

    if ((frame < 0) | (frame > 99)).any().any():
        raise ValueError("Значения характеристик должны быть от 0 до 99.")

    predictions = clf.predict(frame)
    probabilities = clf.predict_proba(frame)
    classes = list(map(str, clf.classes_))

    output = []

    for i, label in enumerate(predictions):
        output.append({
            "features": frame.iloc[i].to_dict(),
            "predicted_position": str(label),
            "probabilities": {
                classes[j]: round(float(probabilities[i][j]), 5)
                for j in range(len(classes))
            },
        })

    return output
