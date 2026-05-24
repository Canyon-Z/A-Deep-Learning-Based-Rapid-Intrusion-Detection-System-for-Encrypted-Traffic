from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from torch import nn
import shutil
import os
import sys
import numpy as np
from typing import Any, Optional
from PIL import Image
try:
    from PIL.Image import Resampling
except ImportError:
    Resampling = None
from torchvision import transforms
import base64
import io

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.cnn_bilstm import CNN_BiLSTM
from src.models.classic_cnn import ClassicCNN
from src.models.mlp import MLP
from src.models.transformer import TrafficTransformer
from src.models.lightweight_cnn_bilstm import Lightweight_CNN_BiLSTM

# 定义在 run_training_all.py 里使用的同一个纯 BiLSTM Wrapper
class PureBiLSTM(nn.Module):
    def __init__(self, input_size=28, hidden_dim=128, num_layers=2, num_classes=2):
        super(PureBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)
        output, _ = self.bilstm(x)
        last_state = output[:, -1, :]
        out = self.fc(last_state)
        return out

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

# Global variables for models
models: dict[str, Any] = {
    'cnn_bilstm': None,
    'classic_cnn': None,
    'lightweight_cnn': None,
    'pure_bilstm': None,
    'mlp': None,
    'transformer': None
}
device = None
loading_error = None
import time

class_names = ['Benign', 'Malware'] # Default, should match training

# Per-model decision thresholds. Transformer is more conservative on this dataset,
# so it needs a lower malware cutoff than the CNN family.
MODEL_MALWARE_THRESHOLDS = {
    'cnn_bilstm': 0.65,
    'classic_cnn': 0.65,
    'lightweight_cnn': 0.60,
    'pure_bilstm': 0.60,
    'mlp': 0.60,
    'transformer': 0.35,
}

MODEL_DISPLAY_NAMES = {
    'cnn_bilstm': 'CNN-BiLSTM',
    'classic_cnn': 'Classic-CNN',
    'lightweight_cnn': 'Lightweight CNN-BiLSTM',
    'pure_bilstm': 'Pure BiLSTM',
    'mlp': 'MLP',
    'transformer': 'Transformer',
    'ensemble_vote': '6-model Majority Vote Ensemble',
}

ENSEMBLE_MODEL_ORDER = [
    'cnn_bilstm',
    'classic_cnn',
    'lightweight_cnn',
    'pure_bilstm',
    'mlp',
    'transformer',
]

ENSEMBLE_MODEL_TYPE = 'ensemble_vote'

# Package-level aggregation policy:
# If the fraction of sessions flagged as malicious >= PACKAGE_MALWARE_RATIO_THRESHOLD
# OR the absolute number of malicious sessions >= PACKAGE_MALICIOUS_COUNT_MIN,
# the whole capture is considered malicious. Adjust these to be more/less strict.
PACKAGE_MALWARE_RATIO_THRESHOLD = float(os.environ.get('PACKAGE_MALWARE_RATIO_THRESHOLD', 0.20))
PACKAGE_MALICIOUS_COUNT_MIN = int(os.environ.get('PACKAGE_MALICIOUS_COUNT_MIN', 3))
# Package-level confidence threshold: require average malware-session confidence >= this
# before elevating package-level decision to 'Malicious'. Default 0.5 (50%).
PACKAGE_CONFIDENCE_THRESHOLD = float(os.environ.get('PACKAGE_CONFIDENCE_THRESHOLD', 0.50))

def load_model():
    global models, device, class_names, loading_error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_model_paths = [
        os.path.join(BASE_DIR, "checkpoints", "cnn_bilstm.pth"),
        os.path.join(BASE_DIR, "checkpoints", "final_model.pth"),
    ]
    existing_candidates = [p for p in candidate_model_paths if os.path.exists(p)]

    if existing_candidates:
        try:
            best_choice = None

            for candidate_path in existing_candidates:
                checkpoint = torch.load(candidate_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    candidate_state_dict = checkpoint['model_state_dict']
                    candidate_num_classes = checkpoint.get('num_classes', 2)
                    candidate_class_names = checkpoint.get('class_names', ['Benign', 'Malware'])
                else:
                    candidate_state_dict = checkpoint
                    if 'fc.weight' in candidate_state_dict:
                        candidate_num_classes = candidate_state_dict['fc.weight'].size(0)
                    else:
                        candidate_num_classes = 2
                    candidate_class_names = ['Benign', 'Malware']

                probe = CNN_BiLSTM(num_classes=candidate_num_classes, hidden_dim=64)
                probe_result = probe.load_state_dict(candidate_state_dict, strict=False)
                missing_count = len(getattr(probe_result, 'missing_keys', []))
                unexpected_count = len(getattr(probe_result, 'unexpected_keys', []))
                mismatch_score = missing_count + unexpected_count

                print(
                    f"Checkpoint probe: {os.path.basename(candidate_path)} | "
                    f"missing={missing_count}, unexpected={unexpected_count}"
                )

                if best_choice is None or mismatch_score < best_choice['score']:
                    best_choice = {
                        'path': candidate_path,
                        'state_dict': candidate_state_dict,
                        'num_classes': candidate_num_classes,
                        'class_names': candidate_class_names,
                        'missing_count': missing_count,
                        'unexpected_count': unexpected_count,
                        'score': mismatch_score,
                    }

                if mismatch_score == 0:
                    break

            if best_choice is None:
                raise RuntimeError("No valid CNN checkpoint could be parsed.")

            model_path = best_choice['path']
            state_dict = best_choice['state_dict']
            num_classes = best_choice['num_classes']
            class_names = best_choice['class_names']

            print(f"Selected main model checkpoint: {os.path.basename(model_path)}")
            if best_choice['score'] > 0:
                print(
                    "Warning: selected checkpoint still has partial mismatch "
                    f"(missing={best_choice['missing_count']}, unexpected={best_choice['unexpected_count']})."
                )
                
            # Initialize CNN-BiLSTM (主力模型，对齐训练脚本的使用参数)
            m1 = CNN_BiLSTM(num_classes=num_classes, hidden_dim=64)
            if best_choice['score'] == 0:
                m1.load_state_dict(state_dict)
            else:
                load_result = m1.load_state_dict(state_dict, strict=False)
                missing = getattr(load_result, 'missing_keys', None)
                unexpected = getattr(load_result, 'unexpected_keys', None)
                if missing:
                    print('Warning: missing keys when loading CNN_BiLSTM:', missing)
                if unexpected:
                    print('Warning: unexpected keys when loading CNN_BiLSTM:', unexpected)
            m1.to(device)
            m1.eval()
            models['cnn_bilstm'] = m1
            
            # --- Initialize Classic-CNN as Standby (如果找不到对应权重文件，就用随机初始化代替演示) ---
            try:
                m2 = ClassicCNN(num_classes=num_classes)
                classic_pwd = os.path.join(BASE_DIR, "checkpoints", "classic_cnn.pth")
                if os.path.exists(classic_pwd):
                    m2.load_state_dict(torch.load(classic_pwd, map_location=device).get('model_state_dict', torch.load(classic_pwd, map_location=device)))
                m2.to(device)
                m2.eval()
                models['classic_cnn'] = m2
            except Exception as e:
                print("Could not fully load Classic CNN, fallback to raw.")
                m2 = ClassicCNN(num_classes=num_classes)
                m2.to(device)
                m2.eval()
            # --- Initialize Pure BiLSTM ---
            try:
                m4 = PureBiLSTM(num_classes=num_classes, hidden_dim=64, num_layers=2)
                pure_lstm_pwd = os.path.join(BASE_DIR, "checkpoints", "pure_bilstm.pth")
                if os.path.exists(pure_lstm_pwd):
                    m4.load_state_dict(torch.load(pure_lstm_pwd, map_location=device).get('model_state_dict', torch.load(pure_lstm_pwd, map_location=device)))
                m4.to(device)
                m4.eval()
                models['pure_bilstm'] = m4
            except Exception as e:
                print(f"Could not load Pure BiLSTM: {e}")

            # --- Initialize MLP ---
            try:
                mlp_pwd = os.path.join(BASE_DIR, "checkpoints", "mlp.pth")
                m6 = None
                if os.path.exists(mlp_pwd):
                    mlp_ckpt = torch.load(mlp_pwd, map_location=device)
                    mlp_state_dict = mlp_ckpt.get('model_state_dict', mlp_ckpt) if isinstance(mlp_ckpt, dict) else mlp_ckpt
                    mlp_num_classes = mlp_ckpt.get('num_classes', num_classes) if isinstance(mlp_ckpt, dict) else num_classes

                    if isinstance(mlp_ckpt, dict) and 'class_names' in mlp_ckpt:
                        class_names = mlp_ckpt.get('class_names', class_names)

                    m6 = MLP(num_classes=mlp_num_classes)
                    m6.load_state_dict(mlp_state_dict)
                    print(
                        "MLP loaded successfully with config: "
                        f"input_size=784, hidden_sizes=[512, 256, 128], num_classes={mlp_num_classes}"
                    )
                else:
                    m6 = MLP(num_classes=num_classes)

                m6.to(device)
                m6.eval()
                models['mlp'] = m6
            except Exception as e:
                print(f"Could not load MLP: {e}")
                
            # --- Initialize Transformer ---
            try:
                trans_pwd = os.path.join(BASE_DIR, "checkpoints", "transformer.pth")
                m5 = None
                if os.path.exists(trans_pwd):
                    trans_ckpt = torch.load(trans_pwd, map_location=device)
                    trans_state_dict = trans_ckpt.get('model_state_dict', trans_ckpt) if isinstance(trans_ckpt, dict) else trans_ckpt
                    trans_num_classes = trans_ckpt.get('num_classes', num_classes) if isinstance(trans_ckpt, dict) else num_classes

                    if isinstance(trans_ckpt, dict) and 'class_names' in trans_ckpt:
                        class_names = trans_ckpt.get('class_names', class_names)

                    candidate_cfgs = []
                    ckpt_model_cfg = trans_ckpt.get('model_config') if isinstance(trans_ckpt, dict) else None
                    if isinstance(ckpt_model_cfg, dict):
                        candidate_cfgs.append({
                            'input_dim': int(ckpt_model_cfg.get('input_dim', 28)),
                            'd_model': int(ckpt_model_cfg.get('d_model', 64)),
                            'nhead': int(ckpt_model_cfg.get('nhead', 4)),
                            'num_layers': int(ckpt_model_cfg.get('num_layers', 2)),
                            'dropout': float(ckpt_model_cfg.get('dropout', 0.3)),
                        })

                    # Web-serving default config (used by run_training_all.py)
                    candidate_cfgs.append({'input_dim': 28, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3})
                    # Legacy/single-script training config (train_transformer.py)
                    candidate_cfgs.append({'input_dim': 28, 'd_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.05})

                    dedup_cfgs = []
                    seen = set()
                    for cfg in candidate_cfgs:
                        key = (cfg['input_dim'], cfg['d_model'], cfg['nhead'], cfg['num_layers'], cfg['dropout'])
                        if key in seen:
                            continue
                        seen.add(key)
                        dedup_cfgs.append(cfg)

                    last_error = None
                    for cfg in dedup_cfgs:
                        try:
                            candidate_model = TrafficTransformer(
                                input_dim=cfg['input_dim'],
                                d_model=cfg['d_model'],
                                nhead=cfg['nhead'],
                                num_layers=cfg['num_layers'],
                                num_classes=trans_num_classes,
                                dropout=cfg['dropout']
                            )
                            candidate_model.load_state_dict(trans_state_dict)
                            m5 = candidate_model
                            print(
                                "Transformer loaded with config: "
                                f"input_dim={cfg['input_dim']}, d_model={cfg['d_model']}, "
                                f"nhead={cfg['nhead']}, num_layers={cfg['num_layers']}, "
                                f"dropout={cfg['dropout']}"
                            )
                            break
                        except Exception as e:
                            last_error = e

                    if m5 is None:
                        raise RuntimeError(f"No compatible Transformer architecture found for checkpoint. Last error: {last_error}")
                else:
                    # Fallback to a randomly initialized transformer so the UI can still run.
                    m5 = TrafficTransformer(input_dim=28, d_model=64, nhead=4, num_layers=2, num_classes=num_classes, dropout=0.3)

                m5.to(device)
                m5.eval()
                models['transformer'] = m5
            except Exception as e:
                print(f"Could not load Transformer: {e}")
                
            # --- Initialize Lightweight CNN-BiLSTM as Standby (用较低 hidden_dim 模拟) ---
            try:
                m3 = Lightweight_CNN_BiLSTM(num_classes=num_classes, hidden_dim=32)
                light_pwd = os.path.join(BASE_DIR, "checkpoints", "lightweight.pth")
                if os.path.exists(light_pwd):
                    m3.load_state_dict(torch.load(light_pwd, map_location=device).get('model_state_dict', torch.load(light_pwd, map_location=device)))
                m3.to(device)
                m3.eval()
                models['lightweight_cnn'] = m3
            except Exception as e:
                print("Could not fully load Lightweight CNN, fallback to raw.")
                m3 = Lightweight_CNN_BiLSTM(num_classes=num_classes, hidden_dim=32)
                m3.to(device)
                m3.eval()
                models['lightweight_cnn'] = m3

            loading_error = None
            print(f"Models loaded successfully. Classes: {class_names}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            loading_error = str(e)
            models['cnn_bilstm'] = None
    else:
        msg = "No CNN checkpoint found (checked: checkpoints/cnn_bilstm.pth and checkpoints/final_model.pth). Please run training first."
        print(msg)
        loading_error = msg
        models['cnn_bilstm'] = None

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/model-status")
def get_model_status():
    loaded_models = [name for name, model in _get_loaded_detection_models() if model is not None]
    return {
        "loaded": bool(loaded_models),
        "error": loading_error,
        "classes": class_names if loaded_models else [],
        "loaded_models": loaded_models,
        "ensemble_ready": bool(loaded_models),
    }


def _find_class_index(target_keywords, default_index=0):
    for idx, cname in enumerate(class_names):
        lowered = str(cname).lower()
        if any(keyword in lowered for keyword in target_keywords):
            return idx
    return default_index


def _get_loaded_detection_models():
    return [(model_name, models.get(model_name)) for model_name in ENSEMBLE_MODEL_ORDER if models.get(model_name) is not None]


def _predict_model_probabilities(model, batch_tensor, chunk_size=128):
    all_probs = []
    with torch.no_grad():
        for start_idx in range(0, batch_tensor.size(0), chunk_size):
            chunk = batch_tensor[start_idx:start_idx + chunk_size]
            outputs = model(chunk)
            all_probs.append(torch.nn.functional.softmax(outputs, dim=1))

    if not all_probs:
        return None

    return torch.cat(all_probs, dim=0).detach().cpu()


def _vote_with_ensemble(batch_tensor, chunk_size=128):
    loaded_models = _get_loaded_detection_models()
    if not loaded_models:
        raise RuntimeError('No detection models are available for ensemble voting.')

    malware_idx = _find_class_index(['malware', 'malicious'], default_index=1 if len(class_names) > 1 else 0)
    benign_idx = _find_class_index(['benign', 'normal'], default_index=0)

    model_probabilities = {}
    for model_name, model in loaded_models:
        probs = _predict_model_probabilities(model, batch_tensor, chunk_size=chunk_size)
        if probs is not None:
            model_probabilities[model_name] = probs

    if not model_probabilities:
        raise RuntimeError('Ensemble voting failed because no model produced outputs.')

    total_sessions = batch_tensor.size(0)
    session_votes = []

    for session_idx in range(total_sessions):
        model_votes = {}
        malware_conf_scores = []
        benign_conf_scores = []
        malicious_votes = 0
        benign_votes = 0

        for model_name in ENSEMBLE_MODEL_ORDER:
            probs = model_probabilities.get(model_name)
            if probs is None or session_idx >= probs.size(0):
                continue

            session_probs = probs[session_idx]
            malware_conf = float(session_probs[malware_idx].item()) if malware_idx < session_probs.shape[0] else float(session_probs.max().item())
            benign_conf = float(session_probs[benign_idx].item()) if benign_idx < session_probs.shape[0] else float(session_probs.max().item())
            threshold = MODEL_MALWARE_THRESHOLDS.get(model_name, 0.5)
            is_malicious = malware_conf >= threshold

            malicious_votes += int(is_malicious)
            benign_votes += int(not is_malicious)
            malware_conf_scores.append(malware_conf)
            benign_conf_scores.append(benign_conf)
            model_votes[model_name] = {
                'predicted_label': 'Malicious' if is_malicious else 'Benign',
                'malware_confidence': f'{malware_conf:.4f}',
                'benign_confidence': f'{benign_conf:.4f}',
                'threshold': threshold,
            }

        if malicious_votes > benign_votes:
            final_label = 'Malicious'
            winning_scores = malware_conf_scores
        elif benign_votes > malicious_votes:
            final_label = 'Benign'
            winning_scores = benign_conf_scores
        else:
            avg_malware_conf = float(np.mean(malware_conf_scores)) if malware_conf_scores else 0.0
            final_label = 'Malicious' if avg_malware_conf >= 0.5 else 'Benign'
            winning_scores = malware_conf_scores if final_label == 'Malicious' else benign_conf_scores

        final_confidence = float(np.mean(winning_scores)) if winning_scores else 0.0
        session_votes.append({
            'predicted_label': final_label,
            'predicted_confidence': final_confidence,
            'malware_votes': malicious_votes,
            'benign_votes': benign_votes,
            'vote_count': len(model_votes),
            'vote_ratio': (malicious_votes / len(model_votes)) if model_votes else 0.0,
            'malware_confidences': malware_conf_scores,
            'benign_confidences': benign_conf_scores,
            'model_votes': model_votes,
        })

    return {
        'malware_idx': malware_idx,
        'benign_idx': benign_idx,
        'available_models': [model_name for model_name, _ in loaded_models],
        'session_votes': session_votes,
    }

import json
import threading
import importlib
import tempfile
import uuid
try:
    scapy_all = importlib.import_module('scapy.all')
    sniff = scapy_all.sniff
    get_if_list = scapy_all.get_if_list
    IP = getattr(scapy_all, 'IP', None)
    TCP = getattr(scapy_all, 'TCP', None)
    UDP = getattr(scapy_all, 'UDP', None)
    scapy_windows = importlib.import_module('scapy.arch.windows')
    get_windows_if_list = scapy_windows.get_windows_if_list
    resolve_iface = scapy_windows.resolve_iface
    HAS_SCAPY = True
except Exception:
    HAS_SCAPY = False
    IP = None
    TCP = None
    UDP = None
    def get_if_list():
        return []
    def get_windows_if_list():
        return []
    def resolve_iface(value):
        return value

try:
    psutil = importlib.import_module('psutil')
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False
@app.get("/api/metrics")
def get_training_metrics():
    """读取训练阶段保存的真实评价参数"""
    metrics_path = os.path.join(BASE_DIR, "checkpoints", "model_metrics.json")
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return JSONResponse(content={"status": "success", "data": data})
        except Exception as e:
            return JSONResponse(content={"status": "error", "message": f"Failed to read metrics: {e}"})
    else:
         return JSONResponse(content={"status": "error", "message": "Metrics file not found. Please train models first."})

@app.get("/")
def read_root(request: Request):
    try:
        template_path = os.path.join(templates_dir, "index.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()

        status_block = ""
        if models['cnn_bilstm'] is None:
            error_text = loading_error or "Model not loaded. Please train the model first."
            status_block = f'''
            <div class="col-span-12">
                <div class="bg-red-900/50 border border-red-500 text-red-200 px-6 py-4 rounded-xl relative shadow-neon-red flex items-start gap-4" role="alert">
                    <i class="fa-solid fa-triangle-exclamation text-2xl text-red-500 mt-1"></i>
                    <div class="flex-1">
                        <strong class="font-bold text-lg">System Error: Model Not Loaded</strong>
                        <p class="mt-1">Back-end AI model failed to initialize. Prediction services are unavailable.</p>
                        <div class="mt-3 p-3 bg-black/30 rounded border border-red-500/30 font-mono text-xs text-red-300 overflow-x-auto">
                            {error_text}
                        </div>
                        <p class="mt-3 text-sm text-gray-400">Please check if 'checkpoints/final_model.pth' exists and is compatible.</p>
                    </div>
                </div>
            </div>
            '''

        placeholder = "<!-- MODEL_STATUS_PLACEHOLDER -->"
        if placeholder in html:
            html = html.replace(placeholder, status_block, 1)

        return HTMLResponse(content=html)
    except Exception as exc:
        return HTMLResponse(content=f"Homepage render failed: {exc}", status_code=500)

@app.post("/analyze")
async def analyze_traffic(
    file: UploadFile = File(...),
    model_type: str = Form("cnn_bilstm"),
    package_threshold: Optional[float] = Form(None, description="Optional package-level malware fraction threshold (0-1) to override server default"),
    package_conf_threshold: Optional[float] = Form(None, description="Optional package-level average malware-confidence threshold (0-1) to require before declaring package malicious")
):
    if models.get('cnn_bilstm') is None:
        error_msg = loading_error if loading_error else "Model not loaded. Please train the model first."
        return JSONResponse(status_code=500, content={"error": error_msg})
        
    start_time = time.time()
    # Create a unique temp file to avoid name collisions and ensure uploads don't block each other
    suffix = os.path.splitext(file.filename)[1] if file.filename else ''
    fd = None
    temp_file = None
    try:
        content = await file.read()
        fd, temp_file = tempfile.mkstemp(prefix='upload_', suffix=suffix, dir='.')
        os.close(fd)
        with open(temp_file, 'wb') as buffer:
            buffer.write(content)
            
        # 1. Feature Extraction
        extractor = FeatureExtractor(truncate_len=784)
        # Modified to handle both old and new FeatureExtractor signature if needed
        # But we know we just updated it to return tuple
        pcap_result = extractor.pcap_to_sessions(temp_file)
        
        if isinstance(pcap_result, tuple):
            sessions = pcap_result[0]
            timestamps = pcap_result[1] if len(pcap_result) > 1 else {}
            session_parse_report = pcap_result[2] if len(pcap_result) > 2 else {}
        else:
            sessions = pcap_result
            timestamps = {}
            session_parse_report = {}

        if not sessions:
            return JSONResponse(content={
                "status": "No valid sessions found in pcap",
                "confidence": 0.0,
                "session_parse_report": session_parse_report,
            })

        # Process sessions
        malware_count = 0
        total_sessions = 0
        malicious_session_image_b64 = None
        first_session_image_b64 = None
        malicious_session_hist = None
        first_session_hist = None
        malware_scores = []
        benign_scores = []
        predicted_scores = []
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Output list for multiple flow details (Front-End Features)
        flow_details = []
        
        # Get session timestamp (first packet time) if available
        session_start_time = "Unknown"
        if timestamps and len(timestamps) > 0:
            valid_ts = [t for t in timestamps.values() if t > 0]
            if valid_ts:
                first_ts = min(valid_ts)
                # Convert to readable string in Local Time
                session_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_ts))

        # Batch Processing for Speed Optimization
        batch_tensors = []
        session_keys = []
        img_strs = []
        hist_lists = []
        
        for session_key, session_bytes in sessions.items():
            total_sessions += 1
            # 2. Preprocess (784 bytes -> 28x28 img)
            img_array = extractor.process_session(session_bytes)
            
            # Convert Session Image to Base64 for Visualization
            try:
                hist, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 256))
                hist_lists.append(hist.tolist())

                pil_img = Image.fromarray(img_array, mode='L')
                resample_nearest = getattr(Resampling, 'NEAREST', 0)
                display_img = pil_img.resize((140, 140), resample_nearest)
                buffered = io.BytesIO()
                display_img.save(buffered, format="PNG")
                img_strs.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            except Exception as e:
                print(f"Error converting image: {e}")
                img_strs.append(None)
                hist_lists.append(None)
            
            # Add to batch
            img = Image.fromarray(img_array, mode='L')
            batch_tensors.append(transform(img))
            session_keys.append((session_key, len(session_bytes)))
            
        if total_sessions > 0:
            first_session_image_b64 = img_strs[0]
            first_session_hist = hist_lists[0]

            # 3. Model Inference (Batched for speed)
            if not batch_tensors:
                return JSONResponse(status_code=400, content={
                    "error": "No valid session images produced from uploaded pcap",
                    "session_parse_report": session_parse_report
                })

            try:
                batch_tensor = torch.stack(batch_tensors).to(device)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print("Failed to stack batch tensors:\n", tb)
                return JSONResponse(status_code=500, content={
                    "error": "Failed to prepare input tensor batch",
                    "detail": str(e),
                    "trace": tb,
                    "session_parse_report": session_parse_report
                })

            requested_model_type = str(model_type or '').strip().lower()
            run_ensemble = requested_model_type in ('', ENSEMBLE_MODEL_TYPE)
            effective_model_type = ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type
            ensemble_result = None

            if run_ensemble:
                try:
                    ensemble_result = _vote_with_ensemble(batch_tensor)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print("Model inference aggregation failed:\n", tb)
                    return JSONResponse(status_code=500, content={
                        "error": "Model inference aggregation failed",
                        "detail": str(e),
                        "trace": tb,
                        "model_type": model_type,
                        "ensemble": True,
                    })

                session_votes = ensemble_result['session_votes']

                for i in range(total_sessions):
                    vote_result = session_votes[i]
                    confidence = float(vote_result['predicted_confidence'])
                    malware_conf = float(np.mean(vote_result['malware_confidences'])) if vote_result['malware_confidences'] else confidence
                    benign_conf = float(np.mean(vote_result['benign_confidences'])) if vote_result['benign_confidences'] else confidence
                    predicted_label = vote_result['predicted_label']

                    # 六模型投票：多数派决定最终标签，平票时用平均恶意置信度兜底。
                    is_malicious = predicted_label == 'Malicious'

                    predicted_scores.append(confidence)
                    malware_scores.append(malware_conf)
                    benign_scores.append(benign_conf)

                    key, byte_len = session_keys[i]
                    src_ip, sport, dst_ip, dport, proto = key
                    proto_str = "TCP" if proto == 6 else ("UDP" if proto == 17 else str(proto))
                    
                    flow_details.append({
                        "src_ip": src_ip,
                        "src_port": sport,
                        "dst_ip": dst_ip,
                        "dst_port": dport,
                        "protocol": proto_str,
                        "bytes": byte_len,
                        "is_malicious": is_malicious,
                        "predicted_label": predicted_label,
                        "predicted_confidence": f"{confidence:.4f}",
                        "malware_confidence": f"{malware_conf:.4f}",
                        "benign_confidence": f"{benign_conf:.4f}",
                        "vote_summary": {
                            "malware_votes": vote_result['malware_votes'],
                            "benign_votes": vote_result['benign_votes'],
                            "vote_ratio": f"{vote_result['vote_ratio']:.4f}",
                            "model_votes": vote_result['model_votes'],
                        }
                    })

                    if is_malicious:
                        malware_count += 1
                        # Capture the first malicious image found for display
                        if malicious_session_image_b64 is None and img_strs[i]:
                            malicious_session_image_b64 = img_strs[i]
                            malicious_session_hist = hist_lists[i]
            else:
                selected_model = models.get(effective_model_type)
                if selected_model is None:
                    return JSONResponse(status_code=400, content={
                        "error": f"Unknown or unavailable model_type: {effective_model_type}",
                        "available_model_types": [k for k, v in models.items() if v is not None] + [ENSEMBLE_MODEL_TYPE]
                    })

                malware_threshold = MODEL_MALWARE_THRESHOLDS.get(effective_model_type, 0.65)
                malware_idx = _find_class_index(['malware', 'malicious'], default_index=1 if len(class_names) > 1 else 0)
                benign_idx = _find_class_index(['benign', 'normal'], default_index=0)

                try:
                    probs = _predict_model_probabilities(selected_model, batch_tensor)
                    if probs is None:
                        raise RuntimeError('Single-model inference returned no probabilities.')
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print("Single model inference failed:\n", tb)
                    return JSONResponse(status_code=500, content={
                        "error": "Single model inference failed",
                        "detail": str(e),
                        "trace": tb,
                        "model_type": effective_model_type,
                        "ensemble": False,
                    })

                conf, pred = torch.max(probs, 1)

                for i in range(total_sessions):
                    prediction_idx = int(pred[i].item())
                    confidence = float(conf[i].item())
                    session_probs = probs[i]
                    malware_conf = float(session_probs[malware_idx].item()) if malware_idx < session_probs.shape[0] else confidence
                    benign_conf = float(session_probs[benign_idx].item()) if benign_idx < session_probs.shape[0] else confidence
                    predicted_label = class_names[prediction_idx] if prediction_idx < len(class_names) else str(prediction_idx)

                    is_malicious = False
                    if len(class_names) > 1:
                        if ('Malware' in predicted_label or 'Malicious' in predicted_label or prediction_idx == malware_idx) and malware_conf >= malware_threshold:
                            is_malicious = True
                    else:
                        if prediction_idx == malware_idx and malware_conf >= malware_threshold:
                            is_malicious = True

                    predicted_scores.append(confidence)
                    malware_scores.append(malware_conf)
                    benign_scores.append(benign_conf)

                    key, byte_len = session_keys[i]
                    src_ip, sport, dst_ip, dport, proto = key
                    proto_str = "TCP" if proto == 6 else ("UDP" if proto == 17 else str(proto))

                    flow_details.append({
                        "src_ip": src_ip,
                        "src_port": sport,
                        "dst_ip": dst_ip,
                        "dst_port": dport,
                        "protocol": proto_str,
                        "bytes": byte_len,
                        "is_malicious": is_malicious,
                        "predicted_label": predicted_label,
                        "predicted_confidence": f"{confidence:.4f}",
                        "malware_confidence": f"{malware_conf:.4f}",
                        "benign_confidence": f"{benign_conf:.4f}",
                    })

                    if is_malicious:
                        malware_count += 1
                        if malicious_session_image_b64 is None and img_strs[i]:
                            malicious_session_image_b64 = img_strs[i]
                            malicious_session_hist = hist_lists[i]

        # Final Decision Logic (package-level aggregation)
        # Consider the whole capture malicious only if either:
        #  - malicious session fraction >= PACKAGE_MALWARE_RATIO_THRESHOLD, OR
        #  - malicious session absolute count >= PACKAGE_MALICIOUS_COUNT_MIN
        malicious_fraction = (malware_count / total_sessions) if total_sessions > 0 else 0.0

        # Allow per-request override of the package-level fraction threshold (value between 0 and 1)
        if package_threshold is not None:
            try:
                pt = float(package_threshold)
                if pt < 0.0:
                    pt = 0.0
                if pt > 1.0:
                    pt = 1.0
            except Exception:
                pt = PACKAGE_MALWARE_RATIO_THRESHOLD
        else:
            pt = PACKAGE_MALWARE_RATIO_THRESHOLD

        # Compute final average malware confidence (used to gate package-level decision)
        avg_malware_conf = float(np.mean(malware_scores)) if malware_scores else 0.0

        # Per-request override for package confidence threshold
        if package_conf_threshold is not None:
            try:
                pct_conf = float(package_conf_threshold)
                if pct_conf < 0.0:
                    pct_conf = 0.0
                if pct_conf > 1.0:
                    pct_conf = 1.0
            except Exception:
                pct_conf = PACKAGE_CONFIDENCE_THRESHOLD
        else:
            pct_conf = PACKAGE_CONFIDENCE_THRESHOLD

        # Only declare the package malicious if (count or ratio condition) AND
        # the average malware-session confidence meets the package confidence threshold.
        if ((malware_count >= PACKAGE_MALICIOUS_COUNT_MIN) or (malicious_fraction >= pt)) and (avg_malware_conf >= pct_conf):
            result_status = "Malicious Traffic Detected"
            chosen_scores = malware_scores if malware_scores else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = malicious_session_image_b64
            display_hist = malicious_session_hist
        else:
            result_status = "Benign Traffic"
            chosen_scores = benign_scores if benign_scores else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = first_session_image_b64  # Show the first session if nothing bad found
            display_hist = first_session_hist
        
        end_time = time.time()
        elapsed_time = f"{(end_time - start_time):.4f}s"

        return JSONResponse(content={
            "status": result_status,
            "confidence": f"{final_conf:.4f}",
            "package_threshold_used": float(pt),
            "confidence_label": result_status,
            "model_type": effective_model_type,
            "model_name": MODEL_DISPLAY_NAMES.get(effective_model_type, effective_model_type),
            "requested_model_type": model_type,
            "malware_sessions": malware_count,
            "total_sessions": total_sessions,
            "details": f"Analyzed {total_sessions} sessions, {malware_count} flagged as malicious.",
            "execution_time": elapsed_time,
            "capture_time": session_start_time,
            "image_data": display_image,
            "payload_dist": display_hist,
            "session_parse_report": session_parse_report,
            "flows": flow_details,
            "ensemble": bool(run_ensemble),
            "ensemble_models": ensemble_result['available_models'] if run_ensemble and ensemble_result else [effective_model_type],
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})
    finally:
        try:
            await file.close()
        except Exception:
            pass
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ====== Real-time capture globals ======
capture_thread: Optional[threading.Thread] = None
capture_stop_event: Optional[threading.Event] = None
capture_lock = threading.Lock()
capture_stats = {
    'running': False,
    'malware_count': 0,
    'total_packets': 0,
    'last_update': None,
    'last_error': None
}
recent_flows = []  # circular buffer of recent detections
MAX_RECENT = 50
live_debug_logs = []
MAX_DEBUG_LOGS = 300


@app.get('/api/interfaces')
def list_interfaces():
    """Return a list of available network interfaces (best-effort)."""
    try:
        result = []
        seen = set()

        def is_noise_interface(name: str, description: str) -> bool:
            text = f"{name} {description}".lower()
            deny_terms = [
                'npcap packet driver',
                'wfp native mac layer lightweight filter',
                'wfp 802.3 mac layer lightweight filter',
                'qos packet scheduler',
                'huorong ndis filter driver',
                'virtual wifi filter driver',
                'native wifi filter driver',
                'microsoft wi-fi direct virtual adapter',
                'virtual adapter',
                'loopback pseudo-interface',
                'loopback interface',
                'wan miniport',
                'teredo',
                'ip-https',
                '6to4 adapter',
                'kernel debug',
                'remote ndis based internet sharing device',
            ]
            return any(term in text for term in deny_terms)

        # Prefer Windows adapter friendly names when available
        if HAS_SCAPY:
            try:
                for iface in get_windows_if_list():
                    display_name = iface.get('description') or iface.get('name') or iface.get('guid')
                    capture_name = iface.get('name') or iface.get('guid') or display_name
                    if is_noise_interface(str(capture_name or ''), str(display_name or '')):
                        continue
                    if display_name and capture_name and capture_name not in seen:
                        seen.add(capture_name)
                        result.append({
                            'name': display_name,
                            'value': capture_name,
                            'description': iface.get('description', ''),
                            'guid': iface.get('guid', '')
                        })
            except Exception:
                pass

        # Prefer psutil friendly names as fallback on non-Windows or if scapy list is empty
        if (not result) and HAS_PSUTIL and psutil is not None:
            try:
                for name in psutil.net_if_addrs().keys():
                    if name not in seen:
                        seen.add(name)
                        result.append({'name': name, 'value': name, 'description': '', 'guid': ''})
            except Exception:
                pass

        # If nothing found, return empty list
        return JSONResponse(content={'status': 'success', 'interfaces': result})
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


def _append_recent(flow):
    with capture_lock:
        recent_flows.insert(0, flow)
        if len(recent_flows) > MAX_RECENT:
            recent_flows.pop()


@app.post('/api/live/start')
def start_live_capture(iface: str = Form(...), model_type: str = Form('cnn_bilstm')):
    global capture_thread, capture_stop_event, capture_stats
    if capture_stats.get('running'):
        return JSONResponse(content={'status': 'already_running'})

    if not HAS_SCAPY:
        capture_stats['running'] = False
        return JSONResponse(status_code=500, content={'status': 'error', 'message': 'scapy not available on server'})

    capture_stop_event = threading.Event()
    capture_stats['last_error'] = None
    # Clear previous session data so a fresh live session starts quickly
    try:
        with capture_lock:
            recent_flows.clear()
            live_debug_logs.clear()
            capture_stats['malware_count'] = 0
            capture_stats['total_packets'] = 0
            capture_stats['last_update'] = None
    except Exception:
        # best-effort; do not fail start if clearing fails
        pass

    def packet_handler(pkt):
        try:
            capture_stats['total_packets'] += 1
            try:
                msg = f"[live] pkt #{capture_stats['total_packets']} received, len={len(bytes(pkt))}"
                live_debug_logs.append(msg)
                if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                print(msg)
            except Exception:
                pass
            # best-effort 5-tuple
            src = pkt[IP].src if IP is not None and IP in pkt else '0.0.0.0'
            dst = pkt[IP].dst if IP is not None and IP in pkt else '0.0.0.0'
            proto = None
            sport = None
            dport = None
            if TCP is not None and TCP in pkt:
                proto = 6
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP is not None and UDP in pkt:
                proto = 17
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            else:
                proto = int(pkt.proto) if hasattr(pkt, 'proto') else 0

            raw = bytes(pkt)

            # Use FeatureExtractor to make an image-like array
            extractor = FeatureExtractor(truncate_len=784)
            try:
                img_arr = extractor.process_session(raw)
            except Exception:
                # Log processing error for debugging
                try:
                    msg = f"[live] FeatureExtractor.process_session failed, pkt_len={len(raw)}"
                    live_debug_logs.append(msg)
                    if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                    print(msg)
                except Exception:
                    pass
                return

            # Prepare tensor and run model
            # Also prepare a small visualization (thumbnail PNG) and payload histogram
            image_b64 = None
            payload_hist = None
            try:
                hist, _ = np.histogram(img_arr.flatten(), bins=16, range=(0, 256))
                payload_hist = hist.tolist()
                pil_img = Image.fromarray(img_arr, mode='L')
                resample_nearest = getattr(Resampling, 'NEAREST', 0)
                display_img = pil_img.resize((140, 140), resample_nearest)
                buffered = io.BytesIO()
                display_img.save(buffered, format="PNG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception:
                # best-effort: if visualization fails, continue without it
                try:
                    live_debug_logs.append('[live] failed to build image/ histogram for pkt')
                    if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                except Exception:
                    pass
            transform = transforms.Compose([transforms.ToTensor()])
            img = Image.fromarray(img_arr, mode='L')
            tensor_img = transform(img)
            if not isinstance(tensor_img, torch.Tensor):
                return
            tensor = torch.unsqueeze(tensor_img, 0).to(device)
            requested_model_type = str(model_type or '').strip().lower()
            run_ensemble = requested_model_type in ('', ENSEMBLE_MODEL_TYPE)
            vote_result = None

            if run_ensemble:
                with torch.no_grad():
                    ensemble_result = _vote_with_ensemble(tensor)
                    vote_result = ensemble_result['session_votes'][0]
                    malware_conf = float(np.mean(vote_result['malware_confidences'])) if vote_result['malware_confidences'] else 0.0
                    is_malicious = vote_result['predicted_label'] == 'Malicious'
            else:
                selected_model = models.get(requested_model_type)
                if selected_model is None:
                    return
                malware_idx = _find_class_index(['malware', 'malicious'], default_index=1 if len(class_names) > 1 else 0)
                threshold = MODEL_MALWARE_THRESHOLDS.get(requested_model_type, 0.65)
                with torch.no_grad():
                    probs = _predict_model_probabilities(selected_model, tensor)
                    if probs is None:
                        return
                    session_probs = probs[0]
                    malware_conf = float(session_probs[malware_idx].item()) if malware_idx < session_probs.shape[0] else 0.0
                    is_malicious = malware_conf >= threshold

            flow = {
                'src': src,
                'dst': dst,
                'sport': sport,
                'dport': dport,
                'proto': proto,
                'malware_conf': round(malware_conf, 4),
                'is_malicious': bool(is_malicious),
                'model_type': ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type,
                'model_name': MODEL_DISPLAY_NAMES.get(ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type, ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type),
                'captured_at': time.time(),
                # Attach visualization and histogram similar to upload flow output
                'image_data': image_b64,
                'payload_dist': payload_hist
            }

            if vote_result is not None:
                flow['vote_summary'] = {
                    'malware_votes': vote_result['malware_votes'],
                    'benign_votes': vote_result['benign_votes'],
                    'vote_ratio': round(float(vote_result['vote_ratio']), 4),
                    'model_votes': vote_result['model_votes'],
                }

            if is_malicious:
                capture_stats['malware_count'] += 1
            _append_recent(flow)
            try:
                msg = f"[live] appended flow src={src} dst={dst} proto={proto} malicious={is_malicious} conf={flow.get('malware_conf')}"
                live_debug_logs.append(msg)
                if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                print(msg)
            except Exception:
                pass
            capture_stats['last_update'] = time.time()
        except Exception:
            try:
                import traceback
                msg = '[live] packet_handler exception: ' + traceback.format_exc()
                live_debug_logs.append(msg)
                if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                print(msg)
            except Exception:
                pass
            return

    def capture_loop():
        try:
            global capture_thread, capture_stop_event
            capture_stats['running'] = True
            capture_iface = resolve_iface(iface) if HAS_SCAPY else iface
            stop_event = capture_stop_event
            # Run sniff in short-timeout loops so we can respond to stop requests
            while stop_event is not None and not stop_event.is_set():
                try:
                    # Use a short timeout so the loop can check the stop_event frequently
                    sniff(iface=capture_iface, prn=packet_handler, store=False, timeout=1)
                except Exception as e:
                    capture_stats['last_error'] = str(e)
                    print('Live capture error during sniff:', e)
                    break
        except Exception as e:
            capture_stats['last_error'] = str(e)
            print('Live capture error:', e)
        finally:
            capture_stats['running'] = False
            # Clean up globals so subsequent start/stop calls see the correct state
            try:
                with capture_lock:
                    capture_thread = None
                    capture_stop_event = None
            except Exception:
                pass

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    return JSONResponse(content={'status': 'started'})


@app.post('/api/live/stop')
def stop_live_capture():
    global capture_stop_event, capture_thread
    if capture_stop_event is None:
        return JSONResponse(content={'status': 'not_running'})
    try:
        capture_stop_event.set()
        # Give the capture thread a short time to exit so running flag is cleared
        if capture_thread is not None and isinstance(capture_thread, threading.Thread):
            capture_thread.join(timeout=1.0)
    except Exception as e:
        capture_stats['last_error'] = str(e)
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})
    return JSONResponse(content={'status': 'stopped'})


@app.get('/api/live/stats')
def live_stats():
    return JSONResponse(content={'status': 'success', 'stats': capture_stats, 'recent': recent_flows})


@app.get('/api/live/debug')
def live_debug():
    try:
        return JSONResponse(content={'status': 'success', 'stats': capture_stats, 'recent': recent_flows, 'debug': live_debug_logs[-200:]})
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})
