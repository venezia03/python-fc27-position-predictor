
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/players.csv")
MODEL_PATH = Path("models/best_model.pkl")
REPORT_PATH = Path("reports")

FEATURES = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physicality",
]

TARGET = "position"

REPORT_PATH.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET])

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df[TARGET].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )),
])

print("Обучение модели на 6 основных характеристиках...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

try:
    auc = roc_auc_score(
        y_test,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro",
    )
except ValueError:
    auc = 0.0

joblib.dump(model, MODEL_PATH)

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

(REPORT_PATH / "six_stats_report.txt").write_text(
    report,
    encoding="utf-8",
)

print("\nМОДЕЛЬ ОБУЧЕНА")
print(f"Количество признаков: {len(FEATURES)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Модель сохранена: {MODEL_PATH}")
