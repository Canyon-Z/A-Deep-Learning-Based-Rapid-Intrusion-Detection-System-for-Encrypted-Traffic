from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sys

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocessing.feature_extraction import FeatureExtractor
# TODO (周嘉辉): 导入模型类
# from src.models.cnn_bilstm import CNN_BiLSTM 

app = FastAPI()

# TODO (周嘉辉): 加载预训练好的模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN_BiLSTM(...)
# model.load_state_dict(torch.load("path/to/best_model.pth", map_location=device))
# model.eval()

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
async def analyze_traffic(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process
    try:
        extractor = FeatureExtractor()
        # TODO (周嘉辉): 
        # 1. 确保 feature_extraction 输出格式符合模型输入 (例如 numpy -> tensor)
        # 2. 调用模型推理: output = model(features)
        # 3. 解析结果: pred = torch.argmax(output, dim=1)
        
        features = extractor.extract_features(temp_file)
        
        # Mock prediction (临时占位，等待模型接入)
        result = "Normal Traffic" 
        confidence = 0.95
        
        return {"filename": file.filename, "result": result, "confidence": confidence}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
