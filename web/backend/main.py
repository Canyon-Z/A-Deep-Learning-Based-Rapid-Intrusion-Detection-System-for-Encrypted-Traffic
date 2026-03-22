from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
import shutil
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import base64
import io

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.cnn_bilstm import CNN_BiLSTM

app = FastAPI()

# Mount static files
static_dir = os.path.join(BASE_DIR, "web", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(BASE_DIR, "web", "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# Global variables for model
model = None
device = None
loading_error = None
import time

class_names = ['Benign', 'Malware'] # Default, should match training

def load_model():
    global model, device, class_names, loading_error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(BASE_DIR, "checkpoints", "final_model.pth")
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Check if checkpoint is the new dict format or old state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Loading model from new checkpoint format...")
                state_dict = checkpoint['model_state_dict']
                num_classes = checkpoint.get('num_classes', 2)
                class_names = checkpoint.get('class_names', ['Benign', 'Malware'])
            else:
                print("Loading model from legacy state_dict...")
                state_dict = checkpoint
                # Auto-detect num_classes from fc.weight to avoid mismatch
                if 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].size(0)
                    print(f"Detected {num_classes} classes from legacy checkpoint.")
                else:
                    num_classes = 2 # Fallback
                
            model = CNN_BiLSTM(num_classes=num_classes, hidden_dim=64)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            loading_error = None
            print(f"Model loaded successfully. Classes: {class_names}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            loading_error = str(e)
            model = None
    else:
        msg = f"Model file not found at {model_path}. Please run training first."
        print(msg)
        loading_error = msg
        model = None

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/model-status")
def get_model_status():
    return {
        "loaded": model is not None,
        "error": loading_error,
        "classes": class_names if model else []
    }

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "model_status": {
            "loaded": model is not None,
            "error": loading_error
        }
    })

@app.post("/analyze")
async def analyze_traffic(file: UploadFile = File(...)):
    if model is None:
        error_msg = loading_error if loading_error else "Model not loaded. Please train the model first."
        return JSONResponse(status_code=500, content={"error": error_msg})
        
    start_time = time.time()
    temp_file = f"temp_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Feature Extraction
        extractor = FeatureExtractor(truncate_len=784)
        # Modified to handle both old and new FeatureExtractor signature if needed
        # But we know we just updated it to return tuple
        pcap_result = extractor.pcap_to_sessions(temp_file)
        
        if isinstance(pcap_result, tuple):
             sessions, timestamps = pcap_result
        else:
             sessions = pcap_result
             timestamps = {}

        if not sessions:
            return JSONResponse(content={"status": "No valid sessions found in pcap", "confidence": 0.0})

        # Process sessions
        malware_count = 0
        total_sessions = 0
        max_malware_conf = 0.0
        malicious_session_image_b64 = None
        first_session_image_b64 = None
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Get session timestamp (first packet time) if available
        session_start_time = "Unknown"
        if timestamps and len(timestamps) > 0:
            first_ts = min(timestamps.values())
            # Convert to readable string in Local Time
            session_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_ts))

        for session_key, session_bytes in sessions.items():
            total_sessions += 1
            # 2. Preprocess (784 bytes -> 28x28 img)
            img_array = extractor.process_session(session_bytes)
            
            # Convert Session Image to Base64 for Visualization
            try:
                # Convert numpy (28,28) to PIL Image
                pil_img = Image.fromarray(img_array, mode='L')
                # Resize to (140, 140) for better visibility in web (5x scale)
                display_img = pil_img.resize((140, 140), Image.NEAREST)
                buffered = io.BytesIO()
                display_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                if total_sessions == 1:
                    first_session_image_b64 = img_str
            except Exception as e:
                print(f"Error converting image: {e}")
                img_str = None
            
            # Convert to Tensor (similar to DataLoader)
            img = Image.fromarray(img_array, mode='L')
            img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dim -> (1, 1, 28, 28)
            
            # 3. Model Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                prediction_idx = pred.item()
                confidence = conf.item()
                
                # Use class_names map if available, else assume 1 is Malware
                # Heuristic: 'Malware' or 'Darknet' usually index 1 if sorted alphabetically with 'Benign'
                is_malicious = False
                if len(class_names) > 1:
                     pred_label = class_names[prediction_idx]
                     if 'Malware' in pred_label or 'Malicious' in pred_label or prediction_idx == 1:
                         is_malicious = True
                else:
                     if prediction_idx == 1: is_malicious = True

                if is_malicious:
                    malware_count += 1
                    max_malware_conf = max(max_malware_conf, confidence)
                    # Capture the first malicious image found for display
                    if malicious_session_image_b64 is None and img_str:
                        malicious_session_image_b64 = img_str

        # Final Decision Logic
        if malware_count > 0:
            result_status = "Malicious Traffic Detected"
            final_conf = max_malware_conf
            display_image = malicious_session_image_b64
        else:
            result_status = "Benign Traffic"
            final_conf = 1.0 # Or average confidence of benign sessions
            display_image = first_session_image_b64 # Show the first session if nothing bad found
        
        end_time = time.time()
        elapsed_time = f"{(end_time - start_time):.4f}s"

        return JSONResponse(content={
            "status": result_status,
            "confidence": f"{final_conf:.4f}",
            "details": f"Analyzed {total_sessions} sessions, {malware_count} flagged as malicious.",
            "execution_time": elapsed_time,
            "capture_time": session_start_time,
            "image_data": display_image 
        })

        # Final Decision Logic
        if malware_count > 0:
            result_status = "Malicious Traffic Detected"
            final_conf = max_malware_conf
        else:
            result_status = "Benign Traffic"
            final_conf = 1.0 # Or average confidence of benign sessions
            
        return JSONResponse(content={
            "status": result_status,
            "confidence": f"{final_conf:.4f}",
            "details": f"Analyzed {total_sessions} sessions, {malware_count} flagged as malicious."
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
