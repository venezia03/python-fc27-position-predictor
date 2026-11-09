from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Пути к файлам
DATA_PATH = Path("data/players.csv")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

# 1. Загрузка датасета
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 50)

print(f"Количество игроков: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# 2. Типы данных
print("\nТИПЫ ДАННЫХ:")
print(df.dtypes)

# 3. Пропущенные значения
print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

# 4. Проверка дубликатов
print("\nДУБЛИКАТЫ:")
print("Полные дубликаты:", df.duplicated().sum())
print("Повторяющиеся ID:", df["player_id"].duplicated().sum())

# 5. Распределение игроков по позициям
print("\nРАСПРЕДЕЛЕНИЕ ПО ПОЗИЦИЯМ:")
position_counts = df["position"].value_counts()

print(position_counts)

# Сохраняем распределение
position_counts.to_csv(REPORTS_PATH / "position_distribution.csv")

# 6. Статистика характеристик
print("\nСТАТИСТИКА ХАРАКТЕРИСТИК:")

stats = df[
    ["pace", "shooting", "passing", "dribbling", "defending", "physicality"]
].describe()

print(stats)

stats.to_csv(REPORTS_PATH / "player_statistics.csv")

# 7. Визуализация распределения позиций
plt.figure(figsize=(12, 6))

sns.countplot(data=df, x="position", order=position_counts.index)

plt.title("Распределение футболистов по игровым позициям")
plt.xlabel("Игровая позиция")
plt.ylabel("Количество игроков")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORTS_PATH / "position_distribution.png")
plt.show()

# 8. Распределение основных характеристик
attributes = ["pace", "shooting", "passing", "dribbling", "defending", "physicality"]

df[attributes].hist(figsize=(12, 8), bins=20)

plt.suptitle("Распределение основных характеристик игроков")
plt.tight_layout()

plt.savefig(REPORTS_PATH / "attributes_distribution.png")
plt.show()

print("\nАНАЛИЗ ДАННЫХ ЗАВЕРШЁН!")
