import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем данные
df = pd.read_csv("data/players.csv")

print("Исходный размер:", df.shape)

# Целевая переменная
TARGET = "position"

# Удаляем строки без целевой переменной
df = df.dropna(subset=[TARGET])

# Выбираем числовые характеристики
numeric_features = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physicality",
    "skill_moves",
    "weak_foot",
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
    "goalkeeping_reflexes",
]

# Категориальные признаки
categorical_features = ["preferred_foot"]

# Оставляем необходимые столбцы
features = numeric_features + categorical_features

X = df[features].copy()
y = df[TARGET].copy()

# Приводим числовые признаки к числовому типу
for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Удаляем полностью пустые признаки
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    print("Полностью пустые признаки:", empty_columns)
    X = X.drop(columns=empty_columns)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nПОДГОТОВКА ЗАВЕРШЕНА")
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
print("Количество классов:", y.nunique())

# Сохраняем подготовленные данные
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nФайлы успешно сохранены!")
