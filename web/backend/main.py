from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sys
import torch
import uuid
import threading

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.cnn_bilstm import CNN_BiLSTM
from web.backend.llm_advisor import USE_LLM_ADVISOR, generate_advice_with_llm

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(BASE_DIR, "checkpoints", "cnn_bilstm_best.pth"),
).strip()

MODEL_INPUT_DIM = int(os.getenv("MODEL_INPUT_DIM", "6"))
model = CNN_BiLSTM(input_dim=MODEL_INPUT_DIM, hidden_dim=64, num_classes=2).to(device)
class_names = ["Normal Traffic", "Malicious Traffic"]

if not os.path.exists(MODEL_CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_CHECKPOINT_PATH}. "
        "Please run training first: python -m src.training.train"
    )

state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()

advice_store = {}
advice_lock = threading.Lock()


def generate_and_store_advice(request_id, result, confidence, features):
    try:
        advice = generate_advice_with_llm(result=result, confidence=confidence, features=features)
        status = "done"
    except Exception as e:
        advice = f"LLM建议生成失败: {e}"
        status = "error"

    with advice_lock:
        if request_id in advice_store:
            advice_store[request_id]["status"] = status
            advice_store[request_id]["advice"] = advice

# Mount static files
static_dir = os.path.join(BASE_DIR, "web", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(BASE_DIR, "web", "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_traffic(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(temp_file)

        # Always use local classifier for final label.
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        result = class_names[pred_idx]

        request_id = str(uuid.uuid4())
        with advice_lock:
            advice_store[request_id] = {"status": "disabled", "advice": ""}

        if USE_LLM_ADVISOR:
            with advice_lock:
                advice_store[request_id]["status"] = "pending"
            background_tasks.add_task(
                generate_and_store_advice,
                request_id,
                result,
                confidence,
                features,
            )

        return {
            "filename": file.filename,
            "result": result,
            "confidence": confidence,
            "request_id": request_id,
            "advice_status": "pending" if USE_LLM_ADVISOR else "disabled",
            "advice": "",
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@app.get("/advice/{request_id}")
def get_advice(request_id: str):
    with advice_lock:
        item = advice_store.get(request_id)

    if not item:
        raise HTTPException(status_code=404, detail="request_id not found")

    return {
        "request_id": request_id,
        "status": item.get("status", "unknown"),
        "advice": item.get("advice", ""),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
