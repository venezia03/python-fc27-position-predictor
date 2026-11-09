from pathlib import Path
import tempfile
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.predictor import load_model, get_model, predict_records, MODEL_PATH

app = FastAPI(title="FC27 Position Predictor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"service": "FC27 Position Predictor", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".pkl", ".joblib")):
        raise HTTPException(400, "Ожидается файл .pkl или .joblib")
    # Only load model files you trust: pickle/joblib can execute code.
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "Файл слишком большой")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_bytes(content)
    try:
        load_model()
    except Exception as exc:
        MODEL_PATH.unlink(missing_ok=True)
        raise HTTPException(400, f"Не удалось загрузить модель: {exc}")
    return {"status": "success", "filename": file.filename}

@app.post("/predict")
def predict(payload: dict):
    try:
        players = payload.get("players")
        if players is None:
            players = [payload]
        if not isinstance(players, list) or not players:
            raise ValueError("Передайте объект игрока или список в поле players")
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
        true_y = df["position"].astype(str) if "position" in df.columns else None
        model = get_model()
        info_cols = list(model.feature_names_in_)
        missing = [c for c in info_cols if c not in df.columns]
        if missing:
            raise ValueError(f"В CSV отсутствуют признаки: {missing}")
        X = df[info_cols]
        pred = model.predict(X)
        result = df.copy()
        result["predicted_position"] = pred
        result = result.astype(object).where(pd.notna(result), None)
        response = {"rows": result.to_dict(orient="records"), "count": len(result)}
        
        if true_y is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            metrics = {
                "accuracy": float(accuracy_score(true_y, pred)),
                "f1_macro": float(f1_score(true_y, pred, average="macro", zero_division=0))
            }
            try:
                metrics["roc_auc_ovr_macro"] = float(roc_auc_score(
                    true_y, model.predict_proba(X), labels=model.classes_,
                    multi_class="ovr", average="macro"
                ))
            except ValueError:
                metrics["roc_auc_ovr_macro"] = None
                metrics["roc_auc_note"] = "ROC-AUC не рассчитан: в переданном CSV отсутствуют некоторые классы или недостаточно примеров."
            response["metrics"] = metrics
        return response
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))
