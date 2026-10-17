from fastapi import FastAPI, File, UploadFile, Request, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
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
import json
import hashlib
import hmac
import html
import secrets
import threading
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.cnn_bilstm import CNN_BiLSTM
from src.models.classic_cnn import ClassicCNN
from src.models.mlp import MLP
from src.models.transformer import TrafficTransformer
from src.models.lightweight_cnn_bilstm import Lightweight_CNN_BiLSTM

# 代码审查第 31 条：PureBiLSTM 原先在本文件里另有一份拷贝（默认 hidden_dim=128、无 dropout），
# 与训练脚本里的那份不一致；现统一从 src/models/pure_bilstm.py 导入唯一定义。
from src.models.pure_bilstm import PureBiLSTM


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期：启动时同步加载 6 份权重。

    代码审查第 37 条：原先用已弃用的 @app.on_event("startup")，改为 lifespan。
    函数体在应用启动时执行，此时模块已完全导入，因此可以引用下方的 load_model。
    """
    load_model()
    yield


app = FastAPI(lifespan=lifespan)

# 挂载静态文件目录
static_dir = os.path.join(BASE_DIR, "web", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 模板目录
templates_dir = os.path.join(BASE_DIR, "web", "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# 模型全局变量
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
# 每个已加载模型对应的权重文件（相对仓库根目录），供 /model-status 展示来源
model_sources: dict[str, str] = {}
import time

class_names = ['Benign', 'Malware']  # 默认类别名称，加载模型后会被覆盖


def _env_float(name, default, min_value=None, max_value=None):
    """安全解析浮点环境变量：非法值回退默认并告警，越界则夹取。"""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"Warning: environment variable {name}={raw!r} is not a number, falling back to {default}.")
        return default
    if min_value is not None and value < min_value:
        print(f"Warning: environment variable {name}={value} below {min_value}, clamping.")
        value = min_value
    if max_value is not None and value > max_value:
        print(f"Warning: environment variable {name}={value} above {max_value}, clamping.")
        value = max_value
    return value


def _env_int(name, default, min_value=None, max_value=None):
    """安全解析整型环境变量：非法值回退默认并告警，越界则夹取。"""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        print(f"Warning: environment variable {name}={raw!r} is not an integer, falling back to {default}.")
        return default
    if min_value is not None and value < min_value:
        print(f"Warning: environment variable {name}={value} below {min_value}, clamping.")
        value = min_value
    if max_value is not None and value > max_value:
        print(f"Warning: environment variable {name}={value} above {max_value}, clamping.")
        value = max_value
    return value


# 代码审查第 42 条：下面这些阈值直接决定最终判决，集中在此处并注明来源。
# 来源：在验证集上按「恶意类召回优先、误报可控」逐模型调参得到（USTC-TFC2016 + 自建混合集）。
# 每个模型有各自的恶意判决阈值。Transformer 在这个数据集上比较保守，
# 所以对恶意类别的概率门槛比 CNN 系列低一些。
MODEL_MALWARE_THRESHOLDS = {
    'cnn_bilstm': 0.65,
    'classic_cnn': 0.65,
    'lightweight_cnn': 0.60,
    'pure_bilstm': 0.60,
    'mlp': 0.60,
    'transformer': 0.35,
}

# 未知/未配置模型名时回退用的默认恶意阈值（与 CNN 系列一致）
DEFAULT_MALWARE_THRESHOLD = 0.65

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

# PCAP 包级聚合策略：
# 如果被标记为恶意的会话占比 >= PACKAGE_MALWARE_RATIO_THRESHOLD，
# 或者恶意会话的绝对数量 >= PACKAGE_MALICIOUS_COUNT_MIN，
# 则将整个 PCAP 判定为恶意。调整这些值可以控制判定的严格程度。
PACKAGE_MALWARE_RATIO_THRESHOLD = _env_float('PACKAGE_MALWARE_RATIO_THRESHOLD', 0.20, 0.0, 1.0)
PACKAGE_MALICIOUS_COUNT_MIN = _env_int('PACKAGE_MALICIOUS_COUNT_MIN', 3, 1)
# 包级置信度阈值：只有当恶意会话的平均置信度 >= 该值时，
# 才会将包级判定提升为'恶意'，默认 0.5 (50%)。
PACKAGE_CONFIDENCE_THRESHOLD = _env_float('PACKAGE_CONFIDENCE_THRESHOLD', 0.50, 0.0, 1.0)
# 会话样本量门槛：会话数达到该值时才按"恶意占比"判定，
# 低于该值时回退到"恶意会话绝对数量"判定，避免大 pcap 上几个误报就整体判恶意。
PACKAGE_MIN_SESSIONS_FOR_RATIO = _env_int('PACKAGE_MIN_SESSIONS_FOR_RATIO', 5, 1)

# 上传限制：只接受 pcap/pcapng，并限制单文件体积（MB，默认 512）。
MAX_UPLOAD_MB = _env_int('DLIDS_MAX_UPLOAD_MB', 512, 1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {'.pcap', '.pcapng'}

# /analyze 响应里最多回传多少条流详情（代码审查第 55 条：原先全量回传，
# 万级会话的 PCAP 会生成几 MB 的 JSON）。截断时响应同时给出 flows_total /
# flows_truncated，前端显示「已显示前 N 条 / 共 M 条」，不静默丢数据。
MAX_RESPONSE_FLOWS = _env_int('DLIDS_MAX_RESPONSE_FLOWS', 200, 1)

# 每个模型对应的候选权重文件（cnn_bilstm 有两个候选，由 load_model 显式记录实际选中项）。
MODEL_CHECKPOINT_FILES = {
    'cnn_bilstm': ('cnn_bilstm.pth', 'final_model.pth'),
    'classic_cnn': ('classic_cnn.pth',),
    'lightweight_cnn': ('lightweight.pth',),
    'pure_bilstm': ('pure_bilstm.pth',),
    'mlp': ('mlp.pth',),
    'transformer': ('transformer.pth',),
}


def _record_model_sources():
    """记录每个「已加载」模型对应的权重文件，供 /model-status 展示（代码审查第 33 条）。"""
    for name, model in models.items():
        if model is None:
            model_sources.pop(name, None)
            continue
        if model_sources.get(name):
            continue
        for filename in MODEL_CHECKPOINT_FILES.get(name, ()):
            candidate = os.path.join(BASE_DIR, "checkpoints", filename)
            if os.path.exists(candidate):
                model_sources[name] = os.path.relpath(candidate, BASE_DIR).replace(os.sep, '/')
                break


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
                
            # 主力模型：CNN-BiLSTM，与训练脚本参数保持一致
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
            model_sources['cnn_bilstm'] = os.path.relpath(model_path, BASE_DIR).replace(os.sep, '/')
            
            # --- Initialize Classic-CNN ---
            # 只有真正加载到训练权重才注册该模型：随机初始化的网络不参与投票，
            # 否则未训练权重会污染集成结果，而前端仍显示"6 模型就绪"。
            classic_pwd = os.path.join(BASE_DIR, "checkpoints", "classic_cnn.pth")
            if os.path.exists(classic_pwd):
                try:
                    m2 = ClassicCNN(num_classes=num_classes)
                    classic_ckpt = torch.load(classic_pwd, map_location=device)
                    classic_state = classic_ckpt.get('model_state_dict', classic_ckpt) if isinstance(classic_ckpt, dict) else classic_ckpt
                    m2.load_state_dict(classic_state)
                    m2.to(device)
                    m2.eval()
                    models['classic_cnn'] = m2
                except Exception as e:
                    print(f"Could not load Classic CNN weights, model skipped: {e}")
                    models['classic_cnn'] = None
            else:
                print("Classic CNN checkpoint not found, model skipped.")
                models['classic_cnn'] = None
            # --- Initialize Pure BiLSTM ---
            # --- Initialize Pure BiLSTM ---
            pure_lstm_pwd = os.path.join(BASE_DIR, "checkpoints", "pure_bilstm.pth")
            if os.path.exists(pure_lstm_pwd):
                try:
                    m4 = PureBiLSTM(num_classes=num_classes, hidden_dim=64, num_layers=2)
                    pure_ckpt = torch.load(pure_lstm_pwd, map_location=device)
                    pure_state = pure_ckpt.get('model_state_dict', pure_ckpt) if isinstance(pure_ckpt, dict) else pure_ckpt
                    m4.load_state_dict(pure_state)
                    m4.to(device)
                    m4.eval()
                    models['pure_bilstm'] = m4
                except Exception as e:
                    print(f"Could not load Pure BiLSTM weights, model skipped: {e}")
                    models['pure_bilstm'] = None
            else:
                print("Pure BiLSTM checkpoint not found, model skipped.")
                models['pure_bilstm'] = None

            # --- Initialize MLP ---
            mlp_pwd = os.path.join(BASE_DIR, "checkpoints", "mlp.pth")
            if os.path.exists(mlp_pwd):
                try:
                    mlp_ckpt = torch.load(mlp_pwd, map_location=device)
                    mlp_state_dict = mlp_ckpt.get('model_state_dict', mlp_ckpt) if isinstance(mlp_ckpt, dict) else mlp_ckpt
                    mlp_num_classes = mlp_ckpt.get('num_classes', num_classes) if isinstance(mlp_ckpt, dict) else num_classes

                    if isinstance(mlp_ckpt, dict) and 'class_names' in mlp_ckpt:
                        class_names = mlp_ckpt.get('class_names', class_names)

                    m6 = MLP(num_classes=mlp_num_classes)
                    m6.load_state_dict(mlp_state_dict)
                    m6.to(device)
                    m6.eval()
                    models['mlp'] = m6
                    print(
                        "MLP loaded successfully with config: "
                        f"input_size=784, hidden_sizes=[512, 256, 128], num_classes={mlp_num_classes}"
                    )
                except Exception as e:
                    print(f"Could not load MLP weights, model skipped: {e}")
                    models['mlp'] = None
            else:
                print("MLP checkpoint not found, model skipped.")
                models['mlp'] = None
                
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

                    # Web 端默认配置（与 run_training_all.py 一致）
                    candidate_cfgs.append({'input_dim': 28, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3})
                    # 旧版单脚本训练配置（train_transformer.py）
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
                    # 没有权重就不注册该模型，绝不让随机初始化的网络参与投票
                    print("Transformer checkpoint not found, model skipped.")
                    m5 = None

                if m5 is not None:
                    m5.to(device)
                    m5.eval()
                    models['transformer'] = m5
            except Exception as e:
                print(f"Could not load Transformer weights, model skipped: {e}")
                models['transformer'] = None
                
            # --- Initialize Lightweight CNN-BiLSTM ---
            light_pwd = os.path.join(BASE_DIR, "checkpoints", "lightweight.pth")
            if os.path.exists(light_pwd):
                try:
                    m3 = Lightweight_CNN_BiLSTM(num_classes=num_classes, hidden_dim=32)
                    light_ckpt = torch.load(light_pwd, map_location=device)
                    light_state = light_ckpt.get('model_state_dict', light_ckpt) if isinstance(light_ckpt, dict) else light_ckpt
                    m3.load_state_dict(light_state)
                    m3.to(device)
                    m3.eval()
                    models['lightweight_cnn'] = m3
                except Exception as e:
                    print(f"Could not load Lightweight CNN weights, model skipped: {e}")
                    models['lightweight_cnn'] = None
            else:
                print("Lightweight CNN checkpoint not found, model skipped.")
                models['lightweight_cnn'] = None

            loading_error = None
            _record_model_sources()
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

# --- Auth Logic ---
USER_DB_FILE = os.path.join(BASE_DIR, "web", "backend", "users.json")
SESSION_SECRET_FILE = os.path.join(BASE_DIR, "web", "backend", ".session_secret")
SESSION_COOKIE_NAME = "session_user"
SESSION_MAX_AGE_SECONDS = _env_int('DLIDS_SESSION_MAX_AGE', 86400, 60)

PBKDF2_PREFIX = 'pbkdf2_sha256'
PBKDF2_ITERATIONS = 200000


def _load_session_secret():
    """会话签名密钥：环境变量优先，其次本地密钥文件，最后随机生成并落盘。"""
    env_secret = os.environ.get('DLIDS_SESSION_SECRET')
    if env_secret and env_secret.strip():
        return env_secret.strip().encode('utf-8')
    try:
        if os.path.exists(SESSION_SECRET_FILE):
            with open(SESSION_SECRET_FILE, 'rb') as f:
                data = f.read().strip()
            if data:
                return data
        new_secret = secrets.token_bytes(48)
        with open(SESSION_SECRET_FILE, 'wb') as f:
            f.write(new_secret)
        print(f"Generated a new session secret at {SESSION_SECRET_FILE} (keep it out of version control).")
        return new_secret
    except Exception as e:
        print(f"Warning: could not persist session secret ({e}); using an in-memory secret. Sessions reset on restart.")
        return secrets.token_bytes(48)


SESSION_SECRET = _load_session_secret()


def hash_password(password, salt=None):
    """PBKDF2-HMAC-SHA256 哈希，格式 pbkdf2_sha256$iterations$salt_hex$hash_hex。"""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', str(password).encode('utf-8'), salt, PBKDF2_ITERATIONS)
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """校验口令：支持 PBKDF2 哈希；历史上以明文存储的值仍可校验（仅用于一次性兼容）。"""
    stored = str(stored or '')
    if stored.startswith(PBKDF2_PREFIX + '$'):
        try:
            _, iterations, salt_hex, digest_hex = stored.split('$', 3)
            digest = hashlib.pbkdf2_hmac(
                'sha256', str(password).encode('utf-8'), bytes.fromhex(salt_hex), int(iterations)
            )
            return hmac.compare_digest(digest.hex(), digest_hex)
        except Exception:
            return False
    return hmac.compare_digest(stored, str(password))


def _read_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    try:
        with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Failed to read user data: {e}")
        return {}


def _write_users(users):
    try:
        tmp_path = USER_DB_FILE + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, USER_DB_FILE)
        return True
    except Exception as e:
        print(f"Failed to write user data: {e}")
        return False


def verify_user(username, password):
    """校验用户名口令。旧版明文口令在首次成功登录后自动升级为 PBKDF2 哈希。"""
    if not username or password is None:
        return False

    # 环境变量优先，便于在不改文件的前提下注入凭据
    env_user = os.environ.get('DLIDS_ADMIN_USER')
    env_password = os.environ.get('DLIDS_ADMIN_PASSWORD')
    if env_user and env_password and hmac.compare_digest(str(username), env_user):
        return hmac.compare_digest(str(password), env_password)

    users = _read_users()
    stored = users.get(username)
    if stored is None:
        # 消耗近似的时间，避免用响应耗时枚举用户名
        hashlib.pbkdf2_hmac('sha256', str(password).encode('utf-8'), b'dlids-absent-user', 1000)
        return False

    if not verify_password(password, stored):
        return False

    if not str(stored).startswith(PBKDF2_PREFIX + '$'):
        users[username] = hash_password(password)
        if _write_users(users):
            print(f"Upgraded plaintext password for user {username!r} to a PBKDF2 hash.")
    return True


def _sign_session(username):
    """签名会话令牌：username:issued_at:hmac_sha256。"""
    issued_at = int(time.time())
    payload = f"{username}:{issued_at}"
    signature = hmac.new(SESSION_SECRET, payload.encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def _verify_session(token):
    if not token:
        return None
    try:
        username, issued_at_raw, signature = token.rsplit(':', 2)
        issued_at = int(issued_at_raw)
    except (ValueError, AttributeError):
        return None
    payload = f"{username}:{issued_at}"
    expected = hmac.new(SESSION_SECRET, payload.encode('utf-8'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    if time.time() - issued_at > SESSION_MAX_AGE_SECONDS:
        return None
    return username


def require_session(request: Request):
    """FastAPI 依赖：校验签名 cookie，未登录/被篡改/过期一律 401。"""
    username = _verify_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或会话已过期")
    return username


# 登录失败限速（单进程内存实现）
LOGIN_MAX_FAILURES = _env_int('DLIDS_LOGIN_MAX_FAILURES', 5, 1)
LOGIN_LOCKOUT_SECONDS = _env_int('DLIDS_LOGIN_LOCKOUT_SECONDS', 60, 1)
_login_failures = {}
_login_lock = threading.Lock()

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    context = {"request": request, "error": ""}
    try:
        # 兼容最新版 FastAPI (Starlette 0.28.0+)
        return templates.TemplateResponse(request=request, name="login.html", context=context)
    except Exception:
        # 兼容旧版本 FastAPI
        return templates.TemplateResponse("login.html", context)

def _login_error_response(request, context):
    try:
        return templates.TemplateResponse(request=request, name="login.html", context=context)
    except Exception:
        return templates.TemplateResponse("login.html", context)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    client_host = request.client.host if request.client else 'unknown'
    now = time.time()

    # 失败次数过多则短暂锁定，避免口令被在线暴力枚举
    with _login_lock:
        record = _login_failures.get(client_host)
        if record and now < float(record.get('locked_until', 0)):
            wait_seconds = int(record['locked_until'] - now) + 1
            return _login_error_response(
                request, {"request": request, "error": f"登录尝试过于频繁，请 {wait_seconds} 秒后重试"}
            )

    if verify_user(username, password):
        with _login_lock:
            _login_failures.pop(client_host, None)
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=_sign_session(username),
            max_age=SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite='lax',
        )
        return response

    with _login_lock:
        record = _login_failures.get(client_host) or {'count': 0, 'locked_until': 0}
        record['count'] = int(record.get('count', 0)) + 1
        if record['count'] >= LOGIN_MAX_FAILURES:
            record['locked_until'] = now + LOGIN_LOCKOUT_SECONDS
            record['count'] = 0
        _login_failures[client_host] = record

    return _login_error_response(request, {"request": request, "error": "用户名或密码不正确"})

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response

@app.get("/model-status")
def get_model_status(session_user: str = Depends(require_session)):
    loaded_models = [name for name, model in _get_loaded_detection_models() if model is not None]
    # 代码审查第 33 条：ensemble_ready 原先写成 bool(loaded_models)，只要有一个模型就报
    # 「六模型投票可用」。现在要求 6 个模型全部就位，并额外给出数量、缺失项与各自权重来源。
    missing_models = [name for name in ENSEMBLE_MODEL_ORDER if name not in loaded_models]
    return {
        "loaded": bool(loaded_models),
        "error": loading_error,
        "classes": class_names if loaded_models else [],
        "loaded_models": loaded_models,
        "loaded_count": len(loaded_models),
        "expected_count": len(ENSEMBLE_MODEL_ORDER),
        "missing_models": missing_models,
        "model_sources": {name: model_sources.get(name) for name in loaded_models},
        "ensemble_ready": len(loaded_models) == len(ENSEMBLE_MODEL_ORDER),
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
            threshold = MODEL_MALWARE_THRESHOLDS.get(model_name, DEFAULT_MALWARE_THRESHOLD)
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
# 指标文件缓存（代码审查第 38 条）：原先每个请求都重新读盘 + json.load；
# 现在按 mtime 判断文件是否变化，未变化就复用内存里的那份。
_metrics_cache = {"mtime": None, "path": None, "data": None}


@app.get("/api/metrics")
def get_training_metrics(session_user: str = Depends(require_session)):
    """训练阶段保存的真实评价参数。

    代码审查第 38/疑点 3：这是全站指标的**单一数据源**
    （BASE_DIR/checkpoints/model_metrics.json），前端不再另取静态副本。
    """
    metrics_path = os.path.join(BASE_DIR, "checkpoints", "model_metrics.json")

    if not os.path.exists(metrics_path):
        return JSONResponse(content={"status": "error", "message": "Metrics file not found. Please train models first."})
    try:
        mtime = os.path.getmtime(metrics_path)
        if (_metrics_cache["data"] is None
                or _metrics_cache["path"] != metrics_path
                or _metrics_cache["mtime"] != mtime):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                _metrics_cache["data"] = json.load(f)
            _metrics_cache["path"] = metrics_path
            _metrics_cache["mtime"] = mtime
        return JSONResponse(content={
            "status": "success",
            "source": "checkpoints/model_metrics.json",
            "data": _metrics_cache["data"],
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"Failed to read metrics: {e}"})

@app.get("/")
def read_root(request: Request):
    return RedirectResponse(url="/login")


# ── 页面路由 ──────────────────────────────────────────────────────────────
# 5 个页面共用 base.html 外壳（侧边导航 + 顶栏常驻抓包控制），只有 content 块不同。
# 页面之间是整页跳转，前端不再做任何 DOM 搬移，因此也不会再有「切页丢状态」的问题。
_PAGES = {
    "overview": ("index.html", "总览", "系统健康、流量趋势与最近检测记录"),
    "upload": ("upload.html", "上传检测", "拖入 PCAP 文件，批量分析加密流量"),
    "live": ("live.html", "实时抓包", "选择网卡启动抓包，主机通信关系实时可视化"),
    "records": ("records.html", "检测记录", "上传与实时抓包的全部检测结果"),
    "models": ("models.html", "模型评估", "6 个模型的准确率、延迟与对比指标"),
}


def _build_status_block() -> str:
    """模型未就绪时的红色告警块，5 个页面共用同一份文案。"""
    if models['cnn_bilstm'] is not None:
        return ""
    # 该文本可能来自异常信息，必须转义后再拼进 HTML
    error_text = html.escape(str(loading_error or "Model not loaded. Please train the model first."))
    return f'''
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


def _render_page(request: Request, page_key: str):
    # 会话校验：签名不合法/被篡改/已过期都当作未登录
    session_user = _verify_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not session_user:
        return RedirectResponse(url="/login")

    template_name, page_title, page_subtitle = _PAGES[page_key]
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "status_block": _build_status_block(),
            "active_page": page_key,
            "page_title": page_title,
            "page_subtitle": page_subtitle,
        }
    )


@app.get("/dashboard")
def dashboard(request: Request):
    return _render_page(request, "overview")


@app.get("/upload")
def upload_page(request: Request):
    return _render_page(request, "upload")


@app.get("/live")
def live_page(request: Request):
    return _render_page(request, "live")


@app.get("/records")
def records_page(request: Request):
    return _render_page(request, "records")


@app.get("/models")
def models_page(request: Request):
    return _render_page(request, "models")

@app.post("/analyze")
def analyze_traffic(
    file: UploadFile = File(...),
    model_type: str = Form(ENSEMBLE_MODEL_TYPE),
    package_threshold: Optional[float] = Form(None, description="Optional package-level malware fraction threshold (0-1) to override server default"),
    package_conf_threshold: Optional[float] = Form(None, description="Optional package-level average malware-confidence threshold (0-1) to require before declaring package malicious"),
    session_user: str = Depends(require_session)
):
    # 只要被请求的模型可用就允许分析，不再要求 cnn_bilstm 必须存在
    requested_model_type = str(model_type or '').strip().lower()
    run_ensemble = requested_model_type in ('', ENSEMBLE_MODEL_TYPE)
    if run_ensemble:
        if not _get_loaded_detection_models():
            error_msg = loading_error if loading_error else "No detection model is loaded. Please train the models first."
            return JSONResponse(status_code=503, content={"error": error_msg})
    elif models.get(requested_model_type) is None:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown or unavailable model_type: {requested_model_type}",
            "available_model_types": [k for k, v in models.items() if v is not None] + [ENSEMBLE_MODEL_TYPE],
        })

    start_time = time.time()
    # 上传校验：只接受 pcap/pcapng，边读边落盘，超过大小上限立即中止
    suffix = os.path.splitext(file.filename or '')[1].lower()
    if suffix not in ALLOWED_UPLOAD_EXTS:
        allowed = ', '.join(sorted(ALLOWED_UPLOAD_EXTS))
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported file type '{suffix or '(none)'}': only {allowed} are accepted.",
        })

    fd = None
    temp_file = None
    try:
        # 临时文件放在系统临时目录，避免上传残留堆积在项目根目录
        fd, temp_file = tempfile.mkstemp(prefix='dlids_upload_', suffix=suffix, dir=tempfile.gettempdir())
        os.close(fd)
        written_bytes = 0
        with open(temp_file, 'wb') as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written_bytes += len(chunk)
                if written_bytes > MAX_UPLOAD_BYTES:
                    buffer.close()
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                    temp_file = None
                    return JSONResponse(status_code=413, content={
                        "error": f"Uploaded file exceeds the {MAX_UPLOAD_MB} MB limit.",
                    })
                buffer.write(chunk)
            
        # 1. Feature Extraction
        extractor = FeatureExtractor()
        # 兼容新旧版本 FeatureExtractor 的返回签名（新版本返回元组）
        # 新版本 FeatureExtractor 已改为返回元组
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

        # 遍历所有会话
        malware_count = 0
        total_sessions = 0
        malicious_session_image_b64 = None
        first_session_image_b64 = None
        malicious_session_hist = None
        first_session_hist = None
        malware_scores = []
        benign_scores = []
        predicted_scores = []
        # 只收集"被判定为恶意"的会话的恶意概率，用于包级门控（见下方严重 26 的修正）
        flagged_malware_confs = []
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # 批量输出多个流详情（供前端展示）
        flow_details = []
        
        # 获取会话时间戳（首个数据包的时间）
        session_start_time = "Unknown"
        if timestamps and len(timestamps) > 0:
            valid_ts = [t for t in timestamps.values() if t > 0]
            if valid_ts:
                first_ts = min(valid_ts)
                # 转换为本地时间字符串
                session_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_ts))

        # 批量推理以加速
        batch_tensors = []
        session_keys = []
        img_strs = []
        hist_lists = []
        
        for session_key, session_bytes in sessions.items():
            total_sessions += 1
            # 2. Preprocess (784 bytes -> 28x28 img)
            img_array = extractor.process_session(session_bytes)
            # 代码审查第 36 条：Pillow≥10 已弃用 Image.fromarray(..., mode='L')，
            # 这里先把数组规整成 uint8（process_session 本就返回 uint8，仅作兜底）。
            img_array = np.ascontiguousarray(img_array, dtype=np.uint8)
            
            # 将会话图片转成 Base64，便于前端可视化
            try:
                hist, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 256))
                hist_lists.append(hist.tolist())

                pil_img = Image.fromarray(img_array)
                resample_nearest = getattr(Resampling, 'NEAREST', 0)
                display_img = pil_img.resize((140, 140), resample_nearest)
                buffered = io.BytesIO()
                display_img.save(buffered, format="PNG")
                img_strs.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            except Exception as e:
                print(f"Error converting image: {e}")
                img_strs.append(None)
                hist_lists.append(None)
            
            # 收集到批次中
            img = Image.fromarray(img_array)
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
                print("Failed to stack batch tensors:", e)
                return JSONResponse(status_code=500, content={
                    "error": "Failed to prepare input tensor batch",
                    "session_parse_report": session_parse_report
                })

            effective_model_type = ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type
            ensemble_result = None

            if run_ensemble:
                try:
                    ensemble_result = _vote_with_ensemble(batch_tensor)
                except Exception as e:
                    print("Model inference aggregation failed:", e)
                    return JSONResponse(status_code=500, content={
                        "error": "Model inference aggregation failed",
                        "detail": str(e),
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
                        flagged_malware_confs.append(malware_conf)
                        # 记录第一个发现的恶意图片，用于前端预览
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

                malware_threshold = MODEL_MALWARE_THRESHOLDS.get(effective_model_type, DEFAULT_MALWARE_THRESHOLD)
                malware_idx = _find_class_index(['malware', 'malicious'], default_index=1 if len(class_names) > 1 else 0)
                benign_idx = _find_class_index(['benign', 'normal'], default_index=0)

                try:
                    probs = _predict_model_probabilities(selected_model, batch_tensor)
                    if probs is None:
                        raise RuntimeError('Single-model inference returned no probabilities.')
                except Exception as e:
                    print("Single model inference failed:", e)
                    return JSONResponse(status_code=500, content={
                        "error": "Single model inference failed",
                        "detail": str(e),
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
                        flagged_malware_confs.append(malware_conf)
                        if malicious_session_image_b64 is None and img_strs[i]:
                            malicious_session_image_b64 = img_strs[i]
                            malicious_session_hist = hist_lists[i]

        # 包级聚合判决逻辑（严重 10 修正版）
        # 旧实现是「恶意会话绝对数 >= 3」或「恶意会话比例 >= 阈值」二者满足其一即整包判恶意：
        # 在万级会话的 PCAP 上，只要出现 3 条被误判的恶意流就会把整个文件判成恶意；
        # 而在只有两三个会话的小 PCAP 上，比例条件又太容易抖动。现在按会话规模分档：
        #  - 会话数 < PACKAGE_MIN_SESSIONS_FOR_RATIO：以恶意比例为准（小样本下比例本身就对少量恶意敏感）
        #  - 会话数 >= 该阈值：除比例达标外，还要求恶意会话绝对数 >= PACKAGE_MALICIOUS_COUNT_MIN
        malicious_fraction = (malware_count / total_sessions) if total_sessions > 0 else 0.0

        # 允许通过请求参数覆盖包级比例阈值（取值 0~1）
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

        # 门控置信度：只对被标记为恶意的会话取平均。旧实现把全部会话的恶意概率一起平均，
        # 大量良性会话会把均值压低，导致门控几乎永远通不过。
        avg_malware_conf = float(np.mean(flagged_malware_confs)) if flagged_malware_confs else 0.0

        # 允许通过请求参数覆盖包级置信度阈值
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

        ratio_ok = bool(total_sessions > 0 and malware_count > 0 and malicious_fraction >= pt)
        if not ratio_ok:
            flagged_by_volume = False
        elif total_sessions < PACKAGE_MIN_SESSIONS_FOR_RATIO:
            flagged_by_volume = True
        else:
            flagged_by_volume = malware_count >= PACKAGE_MALICIOUS_COUNT_MIN

        # 体积条件与门控置信度同时满足，才判定整个 PCAP 为恶意
        if flagged_by_volume and (avg_malware_conf >= pct_conf):
            result_status = "Malicious Traffic Detected"
            chosen_scores = flagged_malware_confs if flagged_malware_confs else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = malicious_session_image_b64
            display_hist = malicious_session_hist
        else:
            result_status = "Benign Traffic"
            chosen_scores = benign_scores if benign_scores else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = first_session_image_b64  # 没发现恶意流量时展示第一个会话的图片
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
            # 代码审查第 55 条：流详情按上限截断后再回传，避免大 PCAP 生成几 MB 的响应；
            # 同时给出总数与是否截断，前端据此显示「已显示前 N 条 / 共 M 条」。
            "flows": flow_details[:MAX_RESPONSE_FLOWS],
            "flows_total": len(flow_details),
            "flows_truncated": len(flow_details) > MAX_RESPONSE_FLOWS,
            "ensemble": bool(run_ensemble),
            "ensemble_models": ensemble_result['available_models'] if run_ensemble and ensemble_result else [effective_model_type],
        })

    except Exception as e:
        # 堆栈只写服务端日志，不随响应回传给浏览器
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Analyze failed: {type(e).__name__}"})
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

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
recent_flows = []  # 最近检测结果的环形缓冲区
# 最近流量环形缓冲条数：前端 /api/live/stats 每 1.5 秒轮询一次
# （见 web/static/js/main.js 的 LIVE_POLL_INTERVAL_MS = 1500），
# 50 条足以覆盖一两个轮询周期；调整这个值时要同步考虑前端轮询间隔。
MAX_RECENT = 50
live_debug_logs = []
MAX_DEBUG_LOGS = 300

# 实时抓包的流级聚合参数。
# 训练和离线链路都是取会话前 28 个包、每包前 28 字节（MAC/IP 已抹）拼 28x28 图，
# 实时这边得照同样的规则攒包，不然喂进模型的跟训练时不是一回事。
# 攒够 28 个包（正好一张图）就推理；攒不满的等空闲超时或停止抓包时补零提交。
LIVE_SESSION_PACKETS = _env_int('DLIDS_LIVE_SESSION_PACKETS', 28, 1)
LIVE_SESSION_MIN_PACKETS = _env_int('DLIDS_LIVE_SESSION_MIN_PACKETS', 8, 1)
LIVE_MAX_TRACKED_SESSIONS = _env_int('DLIDS_LIVE_MAX_SESSIONS', 256, 1)
LIVE_SESSION_IDLE_SECONDS = _env_float('DLIDS_LIVE_SESSION_IDLE', 5.0, 0.5, 600.0)

# 单调递增的流 id：前端用它做增量拉取（since_id），避免每轮重传全部缩略图
latest_flow_id = 0
live_session_buffers: dict = {}   # 五元组 -> {'pkts': [bytes(28)], 'packets': int, 'bytes': int, 'updated': float, 'meta': dict}

# 特征提取器与张量变换只创建一次（原来是每个包都新建一个 FeatureExtractor）
live_extractor = None
live_transform = transforms.Compose([transforms.ToTensor()])


def _get_live_extractor():
    global live_extractor
    if live_extractor is None:
        live_extractor = FeatureExtractor()
    return live_extractor


@app.get('/api/interfaces')
def list_interfaces(session_user: str = Depends(require_session)):
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

        # Windows 下优先使用适配器的友好名称
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

        # 非 Windows 或 scapy 列表为空时，用 psutil 的友好名称兜底
        if (not result) and HAS_PSUTIL and psutil is not None:
            try:
                for name in psutil.net_if_addrs().keys():
                    if name not in seen:
                        seen.add(name)
                        result.append({'name': name, 'value': name, 'description': '', 'guid': ''})
            except Exception:
                pass

        # 什么都没有找到就返回空列表
        return JSONResponse(content={'status': 'success', 'interfaces': result})
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


def _append_recent(flow):
    """给流分配单调递增 id 后放入环形缓冲，返回该 id。"""
    global latest_flow_id
    with capture_lock:
        latest_flow_id += 1
        flow['id'] = latest_flow_id
        recent_flows.insert(0, flow)
        if len(recent_flows) > MAX_RECENT:
            recent_flows.pop()
        return latest_flow_id


def _snapshot_recent_flows():
    """在锁内做浅拷贝，避免序列化过程中抓包线程继续修改列表。"""
    with capture_lock:
        return [dict(item) for item in recent_flows]


@app.post('/api/live/start')
def start_live_capture(iface: str = Form(...), model_type: str = Form(ENSEMBLE_MODEL_TYPE), session_user: str = Depends(require_session)):
    global capture_thread, capture_stop_event, capture_stats
    if capture_stats.get('running'):
        return JSONResponse(content={'status': 'already_running'})

    if not HAS_SCAPY:
        capture_stats['running'] = False
        return JSONResponse(status_code=500, content={'status': 'error', 'message': 'scapy not available on server'})

    capture_stop_event = threading.Event()
    capture_stats['last_error'] = None
    # 清除上一次会话的残留数据，确保新的实时会话快速启动
    try:
        with capture_lock:
            recent_flows.clear()
            live_session_buffers.clear()
            live_debug_logs.clear()
            capture_stats['malware_count'] = 0
            capture_stats['total_packets'] = 0
            capture_stats['last_update'] = None
    except Exception:
        # 尽力而为，清理失败不影响启动
        pass

    def _parse_frame(raw):
        """从以太网帧里取五元组和 IP 头偏移，规则跟 feature_extraction 那段一样。

        返回 (src, sport, dst, dport, proto, offset)；非 IPv4 或太短的帧返回 None。
        """
        if len(raw) < 34:
            return None
        eth_type = (raw[12] << 8) | raw[13]
        offset = 14
        if eth_type == 0x8100:                       # VLAN 标签
            if len(raw) < 38:
                return None
            eth_type = (raw[16] << 8) | raw[17]
            offset = 18
        if eth_type != 0x0800:                       # 只处理 IPv4
            return None
        if offset + 20 > len(raw):
            return None
        ip_hl = (raw[offset] & 0x0F) * 4
        proto = raw[offset + 9]
        src = '%d.%d.%d.%d' % tuple(raw[offset + 12:offset + 16])
        dst = '%d.%d.%d.%d' % tuple(raw[offset + 16:offset + 20])
        sport = 0
        dport = 0
        if proto in (6, 17):
            t = offset + ip_hl
            if len(raw) >= t + 4:
                sport = (raw[t] << 8) | raw[t + 1]
                dport = (raw[t + 2] << 8) | raw[t + 3]
        return src, sport, dst, dport, proto, offset

    def _infer_and_append(pkts, meta, raw_bytes, requested_model_type):
        """把攒够的会话送进模型，结果写进最近缓冲。

        pkts 是每包 28 字节的列表，转成 tuple 后 process_session 才会按"每包一行"填图。
        """
        extractor = _get_live_extractor()
        try:
            img_arr = extractor.process_session(tuple(pkts))
            img_arr = np.ascontiguousarray(img_arr, dtype=np.uint8)
        except Exception:
            try:
                live_debug_logs.append('[live] process_session failed, pkts=%d' % len(pkts))
                if len(live_debug_logs) > MAX_DEBUG_LOGS:
                    live_debug_logs.pop(0)
            except Exception:
                pass
            return

        image_b64 = None
        payload_hist = None
        try:
            hist, _ = np.histogram(img_arr.flatten(), bins=16, range=(0, 256))
            payload_hist = hist.tolist()
            pil_img = Image.fromarray(img_arr)
            resample_nearest = getattr(Resampling, 'NEAREST', 0)
            display_img = pil_img.resize((140, 140), resample_nearest)
            buffered = io.BytesIO()
            display_img.save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception:
            try:
                live_debug_logs.append('[live] build image/histogram failed')
                if len(live_debug_logs) > MAX_DEBUG_LOGS:
                    live_debug_logs.pop(0)
            except Exception:
                pass

        img = Image.fromarray(img_arr)
        tensor_img = live_transform(img)
        if not isinstance(tensor_img, torch.Tensor):
            return
        tensor = torch.unsqueeze(tensor_img, 0).to(device)

        requested = str(requested_model_type or '').strip().lower()
        run_ensemble = requested in ('', ENSEMBLE_MODEL_TYPE)
        vote_result = None

        if run_ensemble:
            with torch.no_grad():
                ensemble_result = _vote_with_ensemble(tensor)
                vote_result = ensemble_result['session_votes'][0]
                confs = vote_result['malware_confidences']
                malware_conf = float(np.mean(confs)) if confs else 0.0
                is_malicious = vote_result['predicted_label'] == 'Malicious'
        else:
            selected_model = models.get(requested)
            if selected_model is None:
                return
            threshold = MODEL_MALWARE_THRESHOLDS.get(requested, DEFAULT_MALWARE_THRESHOLD)
            malware_idx = _find_class_index(['malware', 'malicious'],
                                            default_index=1 if len(class_names) > 1 else 0)
            with torch.no_grad():
                probs = _predict_model_probabilities(selected_model, tensor)
                if probs is None:
                    return
                session_probs = probs[0]
                malware_conf = (float(session_probs[malware_idx].item())
                                if malware_idx < session_probs.shape[0] else 0.0)
                is_malicious = malware_conf >= threshold

        flow = {
            'src': meta.get('src'),
            'dst': meta.get('dst'),
            'sport': meta.get('sport'),
            'dport': meta.get('dport'),
            'proto': meta.get('proto'),
            'packets': len(pkts),
            'bytes': int(raw_bytes or 0),
            'malware_conf': round(malware_conf, 4),
            'is_malicious': bool(is_malicious),
            'model_type': ENSEMBLE_MODEL_TYPE if run_ensemble else requested,
            'model_name': MODEL_DISPLAY_NAMES.get(
                ENSEMBLE_MODEL_TYPE if run_ensemble else requested,
                ENSEMBLE_MODEL_TYPE if run_ensemble else requested),
            'captured_at': time.time(),
            # 附上可视化和直方图，与文件上传分析的输出格式一致
            'image_data': image_b64,
            'payload_dist': payload_hist,
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
        capture_stats['last_update'] = time.time()
        try:
            live_debug_logs.append(
                '[live] flow %s:%s -> %s:%s pkts=%d malicious=%s conf=%s'
                % (flow['src'], flow['sport'], flow['dst'], flow['dport'],
                   flow['packets'], flow['is_malicious'], flow['malware_conf']))
            if len(live_debug_logs) > MAX_DEBUG_LOGS:
                live_debug_logs.pop(0)
        except Exception:
            pass

    def _sweep_sessions(force=False):
        """空闲或停止抓包时，把攒不满 28 包的会话按现有包数补零提交。

        force=True 用于停止时清空残留；包数不够下限的直接丢掉，免得噪声会话进列表。
        """
        now_ts = time.time()
        ready = []
        with capture_lock:
            for key in list(live_session_buffers.keys()):
                s = live_session_buffers[key]
                if s['packets'] < LIVE_SESSION_MIN_PACKETS:
                    if force:
                        live_session_buffers.pop(key, None)
                    continue
                if force or (now_ts - s['updated']) >= LIVE_SESSION_IDLE_SECONDS:
                    ready.append((list(s['pkts']), dict(s['meta']), s['bytes']))
                    live_session_buffers.pop(key, None)
        for pkts, meta, raw_bytes in ready:
            _infer_and_append(pkts, meta, raw_bytes, model_type)

    def packet_handler(pkt):
        try:
            capture_stats['total_packets'] += 1
            raw = bytes(pkt)
            parsed = _parse_frame(raw)
            if parsed is None:
                return
            src, sport, dst, dport, proto, offset = parsed

            # 上下行并成一条会话，键里 IP 小的放前面，跟离线那边一致
            if src <= dst:
                session_key = (src, sport, dst, dport, proto)
            else:
                session_key = (dst, dport, src, sport, proto)

            extractor = _get_live_extractor()
            now_ts = time.time()
            snapshot = None
            with capture_lock:
                session = live_session_buffers.get(session_key)
                if session is None:
                    # 会话表过大时丢弃最旧的条目，避免长时间抓包导致内存持续增长
                    if len(live_session_buffers) >= LIVE_MAX_TRACKED_SESSIONS:
                        oldest_key = min(live_session_buffers,
                                         key=lambda k: live_session_buffers[k]['updated'])
                        live_session_buffers.pop(oldest_key, None)
                    session = {'pkts': [], 'packets': 0, 'bytes': 0,
                               'updated': now_ts,
                               'meta': {'src': src, 'sport': sport, 'dst': dst,
                                        'dport': dport, 'proto': proto}}
                    live_session_buffers[session_key] = session

                # 每包存掩码后的前 28 字节。掩码直接调离线那个方法，
                # 两边字节才会完全一样，不然又会跑偏。
                masked = extractor._mask_packet_headers(bytearray(raw), offset)
                session['pkts'].append(bytes(masked[:extractor.bytes_per_packet]))
                session['packets'] += 1
                session['bytes'] += len(raw)
                session['updated'] = now_ts

                if session['packets'] >= LIVE_SESSION_PACKETS:
                    snapshot = (list(session['pkts']), dict(session['meta']),
                                session['bytes'])
                    live_session_buffers.pop(session_key, None)

            if snapshot is not None:
                _infer_and_append(snapshot[0], snapshot[1], snapshot[2], model_type)
        except Exception:
            try:
                import traceback
                msg = '[live] packet_handler exception: ' + traceback.format_exc()
                live_debug_logs.append(msg)
                if len(live_debug_logs) > MAX_DEBUG_LOGS:
                    live_debug_logs.pop(0)
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
            # 短超时循环抓包，方便及时响应停止请求
            while stop_event is not None and not stop_event.is_set():
                try:
                    # 用短超时让循环能频繁检查 stop_event
                    sniff(iface=capture_iface, prn=packet_handler, store=False, timeout=1)
                    # 每轮结束后把空闲会话补算掉，避免短会话一直等不到第 28 个包
                    _sweep_sessions()
                except Exception as e:
                    capture_stats['last_error'] = str(e)
                    print('Live capture error during sniff:', e)
                    break
        except Exception as e:
            capture_stats['last_error'] = str(e)
            print('Live capture error:', e)
        finally:
            capture_stats['running'] = False
            # 停止前把没攒满的会话也判掉，别丢最后几条
            try:
                _sweep_sessions(force=True)
            except Exception:
                pass
            # 清理全局状态，让后续的 start/stop 调用看到正确状态
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
def stop_live_capture(session_user: str = Depends(require_session)):
    global capture_stop_event, capture_thread
    if capture_stop_event is None:
        return JSONResponse(content={'status': 'not_running'})
    try:
        capture_stop_event.set()
        # 给抓包线程一点时间退出，确保 running 标志被清除
        if capture_thread is not None and isinstance(capture_thread, threading.Thread):
            capture_thread.join(timeout=1.0)
    except Exception as e:
        capture_stats['last_error'] = str(e)
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})
    return JSONResponse(content={'status': 'stopped'})


@app.get('/api/live/stats')
def live_stats(since_id: Optional[int] = None, session_user: str = Depends(require_session)):
    """实时检测结果。

    不传 since_id：返回缓冲区里的全部流（兼容旧前端）。
    传 since_id：只返回 id 大于它的新流，前端因此不必每 1.5 秒重传全部缩略图。
    """
    flows = _snapshot_recent_flows()
    if since_id is not None and since_id > 0:
        payload_flows = sorted(
            (f for f in flows if int(f.get('id', 0)) > since_id),
            key=lambda f: int(f.get('id', 0)),
        )
        # 增量模式下不再重复下发整份缓冲，避免带宽浪费
        payload_recent = []
    else:
        payload_flows = sorted(flows, key=lambda f: int(f.get('id', 0)))
        payload_recent = payload_flows

    latest_id = max((int(f.get('id', 0)) for f in flows), default=0)
    return JSONResponse(content={
        'status': 'success',
        'stats': dict(capture_stats),
        'recent': payload_recent,
        'flows': payload_flows,
        'latest_id': latest_id,
    })


@app.get('/api/live/debug')
def live_debug(session_user: str = Depends(require_session)):
    try:
        return JSONResponse(content={'status': 'success', 'stats': capture_stats, 'recent': recent_flows, 'debug': live_debug_logs[-200:]})
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    # 默认只监听本机，避免把控制台直接暴露到局域网；
    # 需要对外提供服务时显式设置 DLIDS_HOST / DLIDS_PORT。
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get('DLIDS_HOST', '127.0.0.1'),
        port=_env_int('DLIDS_PORT', 8000, 1, 65535),
    )
