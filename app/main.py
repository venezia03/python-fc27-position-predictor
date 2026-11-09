
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import (
    FEATURES,
    MODEL_PATH,
    get_model,
    load_model,
    predict_records,
)

app = FastAPI(title="FC27 SCOUT", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "FC27 SCOUT",
        "features": FEATURES,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "features_count": len(FEATURES),
    }


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(
        (".pkl", ".joblib")
    ):
        raise HTTPException(400, "Ожидается файл .pkl или .joblib")

    content = await file.read()

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "Файл слишком большой")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(content)

    try:
        load_model()
    except Exception as exc:
        MODEL_PATH.unlink(missing_ok=True)
        raise HTTPException(400, f"Ошибка загрузки модели: {exc}")

    return {"status": "success", "filename": file.filename}


@app.post("/predict")
def predict(payload: dict):
    try:
        players = payload.get("players", [payload])

        if not isinstance(players, list) or not players:
            raise ValueError("Передайте данные игрока или список players")

        for player in players:
            if not isinstance(player, dict):
                raise ValueError("Данные игрока должны быть объектом")

            if set(player.keys()) != set(FEATURES):
                raise ValueError(
                    f"Нужно передать ровно 6 признаков: {FEATURES}"
                )

        results = predict_records(players)
        return {"predictions": results}

    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Ожидается CSV-файл")

    try:
        df = pd.read_csv(file.file)

        missing = [name for name in FEATURES if name not in df.columns]
        if missing:
            raise ValueError(f"В CSV отсутствуют признаки: {missing}")

        records = df[FEATURES].to_dict(orient="records")
        predictions = predict_records(records)

        result = df.copy()
        result["predicted_position"] = [
            item["predicted_position"] for item in predictions
        ]

        result = result.astype(object).where(pd.notna(result), None)

        return {
            "count": len(result),
            "rows": result.to_dict(orient="records"),
        }

    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))
