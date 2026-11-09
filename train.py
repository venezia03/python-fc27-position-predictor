import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Пути
DATA_PATH = Path("data/players.csv")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports")

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

# Загружаем данные
df = pd.read_csv(DATA_PATH)

TARGET = "position"

numeric_features = [
    "pace", "shooting", "passing",
    "dribbling", "defending", "physicality",
    "skill_moves", "weak_foot",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes"
]

categorical_features = ["preferred_foot"]

features = numeric_features + categorical_features

X = df[features].copy()
y = df[TARGET].astype(str)

# Преобразование числовых данных
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Предобработка
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Модели
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

print("\nНАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ\n")

for name, model in models.items():

    print(f"Обучение: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        labels=pipeline.classes_,
        multi_class="ovr",
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}\n")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    (REPORT_PATH / f"{name.replace(' ', '_')}_report.txt").write_text(
        report,
        encoding="utf-8"
    )

# Сравнение моделей
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_PATH / "model_comparison.csv",
    index=False
)

# Выбираем лучшую модель
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

# Сохраняем модель
joblib.dump(
    best_model,
    MODEL_PATH / "best_model.pkl"
)

print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_name}")
print("Модель сохранена в models/best_model.pkl")
