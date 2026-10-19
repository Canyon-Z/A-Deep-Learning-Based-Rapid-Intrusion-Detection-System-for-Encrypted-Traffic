from fastapi import FastAPI, File, UploadFile, Request, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
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
import math
import hmac
import html
import secrets
import threading
import asyncio
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
# 每个模型从自己 ckpt 里声明的 class_names（缺陷 D-09）。
# 全局 class_names 只由主力模型（CNN-BiLSTM）决定，其余模型的一致性在这里核对。
model_class_names: dict[str, list] = {}
class_names_conflicts: list[str] = []


def _register_class_names(model_name, names):
    """登记某个模型自带的 class_names，并与主力模型比对（不一致只告警不覆盖）。"""
    if not names:
        return
    resolved = [str(x) for x in names]
    model_class_names[model_name] = resolved
    if resolved != list(class_names):
        msg = (f"{model_name} 的 class_names={resolved} 与主力模型 {list(class_names)} 不一致；"
               f"该模型的恶意/良性下标可能错位，投票与阈值判定结果不可信。")
        if msg not in class_names_conflicts:
            class_names_conflicts.append(msg)
            print('Warning:', msg)



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


def _env_flag(name, default=False):
    """安全解析布尔环境变量：'1'/'true'/'yes'/'on' 为真，其余为假。"""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


# 推理并发闸门（缺陷 M6）。
# 现状：/analyze 与 /analyze-flow 都是 CPU 密集（解析 pcap + torch 前向）。原先没有
# 任何排队机制，N 个请求同时进来就同时开跑，互相抢 GIL 与 torch 线程池 ——
# 实测 4 路并发把单次 5.8s 拖到 30.7s（≈5 倍劣化），界面上还看不到任何
# 「正在排队 / 请稍后」的反馈。/analyze-flow 是 async def，同步 CPU 段落直接卡住
# 事件循环，连轮询接口都会被一起拖死。
# 现在：同一时刻只放行 MAX_CONCURRENT_INFERENCE 个推理请求，其余排队；排队超过
# INFERENCE_QUEUE_TIMEOUT 秒回 429 + Retry-After，前端能明确提示而不是干等。
MAX_CONCURRENT_INFERENCE = _env_int('DLIDS_MAX_CONCURRENT_INFERENCE', 1, 1)
INFERENCE_QUEUE_TIMEOUT = _env_float('DLIDS_INFERENCE_QUEUE_TIMEOUT', 120.0, 1.0, 3600.0)
INFERENCE_GUARDED_PATHS = {'/analyze', '/analyze-flow'}
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
_inference_queue_stats = {'active': 0, 'waiting': 0, 'rejected': 0, 'peak_waiting': 0}


@app.middleware("http")
async def inference_concurrency_guard(request: Request, call_next):
    """给推理入口排队；排队超时回 429 而不是无限等待（缺陷 M6）。"""
    if request.url.path not in INFERENCE_GUARDED_PATHS:
        return await call_next(request)

    _inference_queue_stats['waiting'] += 1
    _inference_queue_stats['peak_waiting'] = max(
        _inference_queue_stats['peak_waiting'], _inference_queue_stats['waiting']
    )
    acquired = False
    try:
        try:
            await asyncio.wait_for(_inference_semaphore.acquire(), timeout=INFERENCE_QUEUE_TIMEOUT)
            acquired = True
        except asyncio.TimeoutError:
            _inference_queue_stats['rejected'] += 1
            return JSONResponse(
                status_code=429,
                content={
                    'error': 'The inference queue is busy. Retry shortly.',
                    'queue_timeout_seconds': INFERENCE_QUEUE_TIMEOUT,
                    'max_concurrent_inference': MAX_CONCURRENT_INFERENCE,
                    'queue': dict(_inference_queue_stats),
                },
                headers={'Retry-After': str(int(INFERENCE_QUEUE_TIMEOUT))},
            )
        _inference_queue_stats['waiting'] -= 1
        _inference_queue_stats['active'] += 1
        try:
            return await call_next(request)
        finally:
            _inference_queue_stats['active'] -= 1
            _inference_semaphore.release()
    finally:
        if not acquired:
            _inference_queue_stats['waiting'] = max(0, _inference_queue_stats['waiting'] - 1)


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

# 级联检测（cascade）：先用主力单模型跑一遍，批次平均置信度不足时再升级为 6 模型集成。
# 单模型在"简单样本"上已经够用，只有模型自己拿不准的批次才值得付集成成本。
#
# 阈值从 0.90 上调到 0.995（实测依据，见 tools/probe_cascade_escalation.py）：
# 拿 10 份**分布外**的工控协议 pcap（BACnet/DNP3/Modbus/Zigbee）跑级联，
# 批次平均置信度最低 0.9585、中位 0.9995 —— 阈值 0.90 甚至 0.95 都是 0/8 触发，
# 级联等于永远不升级。调到 0.995 后是 2/8 触发，才符合「少数可疑批次送复核」的设计意图。
# 注意这也说明模型对没见过的协议同样过度自信，单靠 softmax 判风险不够，
# 还需要熵/距离类判据（见 CASCADE_ENTROPY_THRESHOLD）。
CASCADE_MODEL_TYPE = 'cascade'
CASCADE_CONFIDENCE_THRESHOLD = 0.995
# 归一化预测熵阈值：H(p)/log(C)。二分类下 max(softmax)=0.995 时熵约 0.04，
# 取 0.25 相当于「两类概率差距不足一个数量级」，比只看 max 更能抓住「都不像」的批次。
CASCADE_ENTROPY_THRESHOLD = 0.25

# 六分类流特征链路的档位阈值：**唯一配置源**（缺陷 D-07）。
# 之前 /analyze-flow 的表单默认是 0.90，而 pcap 链路与实时链路调用
# _classify_flow_matrix 时不传阈值、吃的是函数默认 0.80 —— 同一个模型在两个
# 入口给出两套 auto/review/manual 分布，产物 logs/flow_multiclass_metrics.json
# 里的「自动处置（置信 >= 0.90）」也就无法映射回线上。现在两条链路都从这里取。
FLOW_CONFIDENCE_HIGH = _env_float('DLIDS_FLOW_CONF_HIGH', 0.90, 0.0, 1.0)
FLOW_CONFIDENCE_LOW = _env_float('DLIDS_FLOW_CONF_LOW', 0.50, 0.0, 1.0)
if FLOW_CONFIDENCE_LOW > FLOW_CONFIDENCE_HIGH:
    print(f"Warning: DLIDS_FLOW_CONF_LOW={FLOW_CONFIDENCE_LOW} > "
          f"DLIDS_FLOW_CONF_HIGH={FLOW_CONFIDENCE_HIGH}, swapping them.")
    FLOW_CONFIDENCE_HIGH, FLOW_CONFIDENCE_LOW = FLOW_CONFIDENCE_LOW, FLOW_CONFIDENCE_HIGH

# 二分类判决的标签域：唯一真源（缺陷 D-06 / D-09）。
# 集成分支过去把 'Malicious'/'Benign' 写死，单模型分支却用 class_names 里的原始串，
# 于是前端 `predicted_label === 'Malware'` 这类比较会在单模型链路静默失效。
DECISION_POSITIVE_LABEL = 'Malicious'
DECISION_NEGATIVE_LABEL = 'Benign'
# 判定「哪个下标是恶意类」只认这一组关键词；找不到时按二分类约定回退。
MALWARE_CLASS_KEYWORDS = ('malware', 'malicious', 'attack', 'abnormal', 'anomaly')
BENIGN_CLASS_KEYWORDS = ('benign', 'normal', 'clean')
# 六分类里显式认定为「攻击类」的名单（缺陷 D-13：旧写法「非 benign 即攻击」，
# 一旦加入新的良性子类（如 Benign-Web）就会被误算成攻击）。
FLOW_ATTACK_CLASSES = ('Bot', 'DDoS-HOIC', 'DoS-GoldenEye', 'DoS-Slowloris', 'DDoS-LOIC-UDP')
FLOW_BENIGN_CLASSES = ('Benign', 'Normal')

# 分布外判据：对角马氏距离。softmax 无法表达「没见过这种东西」——实测拿 8 份
# 工控协议 pcap（BACnet/DNP3/Modbus/Zigbee）去跑，批次平均置信度 0.9585~0.9998，
# 比同源流量还自信。距离判据才能识别分布外样本：
#   dist = mean_i(((x_i - mu_i) / sigma_i)^2)
# mu/sigma 与 p99 由 tools/compute_dist_stats.py 在训练集上估出来（checkpoints/dist_stats.npz）。
DIST_STATS_PATH = os.path.join(BASE_DIR, 'checkpoints', 'dist_stats.npz')
_DIST_MEAN = None
_DIST_STD = None
_DIST_P99 = None
# 缺陷 D-02：dist_stats.npz 曾退化成 mean 全 0 / std 全 1，此时
# dist = mean(x_i^2) <= 1 恒小于 p99，距离支路永不升级，级联实际只剩两个条件。
# 这里做自洽校验并把结论暴露到 /model-status，不再让退化产物静默通过。
_DIST_STATS_STATUS = {'loaded': False, 'degenerate': False, 'reason': None}
try:
    if os.path.exists(DIST_STATS_PATH):
        with np.load(DIST_STATS_PATH) as _dist_data:
            _DIST_MEAN = _dist_data['mean'].astype(np.float32)
            _DIST_STD = _dist_data['std'].astype(np.float32)
            _DIST_P99 = float(_dist_data['p99'])
        print(f"Distance stats loaded: dim={_DIST_MEAN.shape[0]}, p99={_DIST_P99:.4f}")
        _DIST_STATS_STATUS['loaded'] = True
        _dist_reason = None
        if not np.any(_DIST_MEAN):
            _dist_reason = 'mean 全为 0'
        elif not np.any(_DIST_STD != 1.0):
            _dist_reason = 'std 全为 1'
        elif _DIST_P99 <= float(np.mean(_DIST_MEAN ** 2)) + 1e-6:
            _dist_reason = (f'p99={_DIST_P99:.4f} 不大于训练分布自身均值 '
                            f'{float(np.mean(_DIST_MEAN ** 2)):.4f}，距离判据恒不触发')
        if _dist_reason:
            _DIST_STATS_STATUS['degenerate'] = True
            _DIST_STATS_STATUS['reason'] = _dist_reason
            print(f"Warning: dist_stats.npz is degenerate ({_dist_reason}). "
                  "The cascade distance gate will never escalate until it is rebuilt: "
                  "python tools/compute_dist_stats.py")
    else:
        print("Distance stats not found, cascade will rely on confidence only:", DIST_STATS_PATH)
        _DIST_STATS_STATUS['reason'] = 'dist_stats.npz 不存在'
except Exception as _dist_err:
    print("Failed to load distance stats:", _dist_err)
    _DIST_STATS_STATUS['reason'] = f'加载失败: {_dist_err}'



# ---- 多分类（流特征分支）：CSV 上传入口 ----
# 这条分支与 pcap 链路完全不同：pcap 链路吃 28x28 字节图（784 维），这一条吃
# CICFlowMeter 风格的 78 维流统计特征。它由 tools/train_multiclass.py 训练，
# 支持 Benign/Bot/DDoS-HOIC/DoS-GoldenEye/DoS-Slowloris/DDoS-LOIC-UDP 六类，
# 并附带温度校准参数（让 softmax 输出接近真实正确率）。
# 为什么单独开接口而不并进 /analyze：输入格式（CSV vs pcap）、模型、特征口径全不同，
# 强行合并会让两条链路互相污染。
FLOW_MULTICLASS_PATH = os.path.join(BASE_DIR, 'checkpoints', 'flow_multiclass.pth')
# pcap 链路跑六分类时最多处理多少条流：一条流要算 78 维统计，几十万条流会把一次
# pcap 上传拖到分钟级；取包数最多的前 N 条已经能代表这份抓包的攻击构成。
FLOW_THREAT_MAX_FLOWS = _env_int('DLIDS_FLOW_THREAT_MAX_FLOWS', 4000, 50)
flow_multiclass = None
flow_multiclass_meta = None
# 缺陷 D-01：环境校准（mean/std 被 flow_env_calib.npz 覆盖）与主权重自带的
# normalization 一旦不一致，重启后同一份 CSV 会给出不同预测，而产物里的指标
# 对应的是哪套参数完全不可追溯。这里把差异显式记录下来暴露到 /model-status。
_CALIBRATION_STATUS = {
    'checkpoint_normalization': None,
    'active_source': None,
    'mismatch': False,
    'mean_l1_mean_diff': None,
    'std_ratio_median': None,
}
try:
    if os.path.exists(FLOW_MULTICLASS_PATH):
        from src.models.flow_mlp import FlowMLP as _FlowMLP
        _fm_raw = torch.load(FLOW_MULTICLASS_PATH, map_location='cpu')
        _fm_model = _FlowMLP(input_dim=_fm_raw['input_dim'],
                             hidden_sizes=tuple(_fm_raw.get('hidden_sizes', (256, 128))),
                             num_classes=_fm_raw['num_classes'])
        _fm_model.load_state_dict(_fm_raw['model_state_dict'])
        _fm_model.eval()
        flow_multiclass = _fm_model
        _ckpt_mean = np.array(_fm_raw['normalization']['mean'], dtype=np.float32)
        _ckpt_std = np.array(_fm_raw['normalization']['std'], dtype=np.float32)
        flow_multiclass_meta = {
            'feature_names': list(_fm_raw['feature_names']),
            'class_names': list(_fm_raw['class_names']),
            'temperature': float(_fm_raw.get('temperature', 1.0)),
            'mean': _ckpt_mean.copy(),
            'std': _ckpt_std.copy(),
            # 权重是怎么切分出来的（缺陷 D-08）：block=类内时间块、无泄漏；
            # random=按行随机、同一 pcap 的流会横跨 train/test。产物里没写就留 None，
            # 让上层（/model-status、回归脚本）能如实区分「不知道」与「block」。
            'split_strategy': _fm_raw.get('split_strategy'),
            'leakage_controlled': (_fm_raw.get('split_strategy') == 'block'),
            'normalization_scope': _fm_raw.get('normalization_scope'),
            'days': list(_fm_raw.get('days', []) or []),
        }
        _CALIBRATION_STATUS['checkpoint_normalization'] = 'checkpoints/flow_multiclass.pth'
        _CALIBRATION_STATUS['active_source'] = 'checkpoints/flow_multiclass.pth'
        print(f"Flow multiclass model loaded: {len(flow_multiclass_meta['class_names'])} classes "
              f"{flow_multiclass_meta['class_names']}, T={flow_multiclass_meta['temperature']:.4f}")
        # 环境校准留下的标准化参数：/api/flow/calibrate 会把 mean/std 换成目标环境的
        # 统计量，这里读回来 —— 否则重启一次，适配就白做了，实时流量又成片进「未知攻击」。
        #
        # 缺陷 D-01（P0）：这一步曾无条件覆盖主权重自带的 normalization，而
        #   ① checkpoints/flow_class_centers.npz 的类中心是在 **主权重** 标准化空间里算的；
        #   ② logs/flow_multiclass_metrics.json 的指标也是在那个空间里算的。
        # 覆盖之后，输入被映射到 calib 空间、类中心还留在 ckpt 空间，两个坐标系不一致，
        # 「到最近类中心的距离」与它的 31.53 阈值全部失效（实测 44.3% 的样本被判成未知攻击），
        # 指标也不再可复现。现在改为**空间一致性守门**：只有 calib 与类中心同空间才启用。
        _env_calib_path = os.path.join(BASE_DIR, 'checkpoints', 'flow_env_calib.npz')
        if os.path.isfile(_env_calib_path):
            try:
                _ec = np.load(_env_calib_path)
                _env_mean = np.asarray(_ec['mean'], dtype=np.float32)
                _env_std = np.asarray(_ec['std'], dtype=np.float32)
                if _env_mean.shape == _ckpt_mean.shape and _env_std.shape == _ckpt_std.shape:
                    _mean_diff = float(np.mean(np.abs(_env_mean - _ckpt_mean)))
                    _safe_ckpt_std = np.where(np.abs(_ckpt_std) < 1e-12, 1.0, _ckpt_std)
                    _std_ratio = float(np.median(np.abs(_env_std / _safe_ckpt_std)))
                    _CALIBRATION_STATUS['mean_l1_mean_diff'] = round(_mean_diff, 4)
                    _CALIBRATION_STATUS['std_ratio_median'] = round(_std_ratio, 4)
                    # 判定阈：均值差超过 1% 的量级或 std 比值偏离 20% 即视为不一致
                    _mismatch = bool(
                        _mean_diff > max(1e-3, 0.01 * float(np.mean(np.abs(_ckpt_mean))))
                        or not (0.8 <= _std_ratio <= 1.2)
                    )
                    _CALIBRATION_STATUS['mismatch'] = _mismatch
                    # 类中心所属空间（npz 里随中心一起落盘，缺失则视为未知）
                    _centers_space = None
                    try:
                        _csp = os.path.join(BASE_DIR, 'checkpoints', 'flow_class_centers.npz')
                        if os.path.isfile(_csp):
                            with np.load(_csp, allow_pickle=False) as _csp_npz:
                                if 'mean' in _csp_npz.files and 'std' in _csp_npz.files:
                                    _centers_space = (np.asarray(_csp_npz['mean'], dtype=np.float32),
                                                      np.asarray(_csp_npz['std'], dtype=np.float32))
                    except Exception as _csp_err:  # noqa: BLE001
                        print('Failed to read class-center normalization space:', _csp_err)
                    _space_ok = None
                    if _centers_space is not None:
                        _cs_mean, _cs_std = _centers_space
                        _space_ok = bool(
                            _cs_mean.shape == _env_mean.shape
                            and _cs_std.shape == _env_std.shape
                            and np.allclose(_cs_mean, _env_mean, rtol=1e-4, atol=1e-6)
                            and np.allclose(_cs_std, _env_std, rtol=1e-4, atol=1e-6)
                        )
                    _CALIBRATION_STATUS['centers_space_matches_env_calib'] = _space_ok
                    _force_calib = _env_flag('DLIDS_FLOW_USE_ENV_CALIB', False)
                    if _space_ok:
                        _use_calib = True
                    elif _force_calib:
                        # 强制启用：坐标不一致时距离支路必须关掉，不能拿错尺子量
                        _use_calib = True
                        print('Warning: DLIDS_FLOW_USE_ENV_CALIB=1 forces flow_env_calib.npz, '
                              'but the class centers were built in a different normalization '
                              'space. The 未知攻击 distance gate will be DISABLED until '
                              '`python tools/build_flow_centers.py` is re-run.')
                    else:
                        _use_calib = False
                        print('Flow env calib IGNORED: it is not the space the class centers '
                              'were built in (centers-space match=%s). Predictions and '
                              'logs/flow_multiclass_metrics.json therefore stay reproducible '
                              'against checkpoints/flow_multiclass.pth. Set '
                              'DLIDS_FLOW_USE_ENV_CALIB=1 to force it, or re-run '
                              'tools/build_flow_centers.py after calibrating.'
                              % ('unknown' if _space_ok is None else 'no'))
                    if _use_calib:
                        flow_multiclass_meta['mean'] = _env_mean
                        flow_multiclass_meta['std'] = _env_std
                        _CALIBRATION_STATUS['active_source'] = 'checkpoints/flow_env_calib.npz'
                        _CALIBRATION_STATUS['env_calib_applied'] = True
                        print('Flow env calib loaded:', os.path.relpath(_env_calib_path, BASE_DIR),
                              f'(mean|Δ|={_mean_diff:.4g}, std ratio median={_std_ratio:.4f})')
                    else:
                        _CALIBRATION_STATUS['env_calib_applied'] = False
                else:
                    print('Warning: flow_env_calib.npz shape', _env_mean.shape,
                          'does not match checkpoint normalization', _ckpt_mean.shape, '- ignored.')
            except Exception as _ec_err:  # noqa: BLE001
                print('Flow env calib load failed:', _ec_err)

    else:
        print("Flow multiclass checkpoint not found:", FLOW_MULTICLASS_PATH)
except Exception as _fm_err:
    print("Failed to load flow multiclass model:", _fm_err)


# 「未知攻击」判据的参照数据：每类在标准化空间里的中心，以及训练分布内距离的分位数。
# 没有这份参照，模型对没见过的攻击（Brute Force / XSS / SQLi / Infiltration）只能
# 硬塞进最像的那一类，看上去置信度还挺高——这是六分类最危险的失败方式。
FLOW_CENTERS_PATH = os.path.join(BASE_DIR, 'checkpoints', 'flow_class_centers.npz')
flow_centers = None
# 类中心所属的标准化空间（build_flow_centers.py 会把它随中心一起落盘）。
# 缺陷 D-01：没有它就无法判断「现在推理用的标准化」和「中心所在的坐标」是否同一个，
# 而两者不一致时距离与阈值全部失效且**不会报错**。
flow_centers_space = None
try:
    if os.path.exists(FLOW_CENTERS_PATH):
        _fc = np.load(FLOW_CENTERS_PATH, allow_pickle=False)
        flow_centers = {
            'centers': _fc['centers'].astype(np.float32),
            'class_names': [str(x) for x in _fc['class_names']],
            'thresholds': _fc['thresholds'].astype(np.float32),
            'percentiles': _fc['percentiles'].astype(np.float32),
        }
        if 'mean' in _fc.files and 'std' in _fc.files:
            flow_centers_space = (
                np.asarray(_fc['mean'], dtype=np.float32),
                np.asarray(_fc['std'], dtype=np.float32),
            )
        _p99 = float(flow_centers['thresholds'][2]) if len(flow_centers['thresholds']) > 2 else None
        _p999 = float(flow_centers['thresholds'][3]) if len(flow_centers['thresholds']) > 3 else None
        print(f"Flow class centers loaded: {len(flow_centers['class_names'])} classes"
              + (f", p99 distance={_p99:.4f}" if _p99 is not None else "")
              + (f", p99.9 distance={_p999:.4f}" if _p999 is not None else ""))
        # 与当前 active 标准化参数对比：不同空间就直接把 OOD 支路判为不可用，
        # 而不是拿 31.53 这种在另一个坐标系里标定出来的阈值去量。
        if flow_centers_space is not None and flow_multiclass_meta is not None:
            _cs_mean, _cs_std = flow_centers_space
            _act_mean = np.asarray(flow_multiclass_meta['mean'], dtype=np.float32)
            _act_std = np.asarray(flow_multiclass_meta['std'], dtype=np.float32)
            _same_space = bool(
                _cs_mean.shape == _act_mean.shape
                and _cs_std.shape == _act_std.shape
                and np.allclose(_cs_mean, _act_mean, rtol=1e-4, atol=1e-6)
                and np.allclose(_cs_std, _act_std, rtol=1e-4, atol=1e-6)
            )
            flow_centers['space_matches_active'] = _same_space
            if not _same_space:
                print('Warning: flow_class_centers.npz was built in a different normalization '
                      'space than the active one. The 未知攻击 distance gate is DISABLED '
                      '(thresholds would be meaningless). Re-run '
                      '`python tools/build_flow_centers.py` after changing the calibration.')
        else:
            flow_centers['space_matches_active'] = None
    else:
        print("Flow class centers not found (未知攻击检测关闭):", FLOW_CENTERS_PATH)
except Exception as _fc_err:
    print("Failed to load flow class centers:", _fc_err)


# 系统默认推理模型。选型依据：跨数据集（ICS）评估里 Pure BiLSTM 是唯一同时做到
# 100% 恶意检出与低误报的模型（99.37% / 误报 0.63%），而 CNN-BiLSTM 虽然在同源
# USTC 测试集上 99.96%，跨域却只有 18.55%（81.46% 良性被误判为恶意）。
# 前端 KPI 带下拉的默认选中项必须与之保持一致（web/templates/partials/stats_panel.html）。
DEFAULT_MODEL_TYPE = 'pure_bilstm'

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

# 分布外协议门控（2026-10-18 加入）。
# 背景：训练集是 USTC-TFC2016（加密/互联网流量），完全没有工业控制协议。
# 实测 48 个工控良性 pcap 有 37 个被判成「恶意」，且 6 模型一致、置信度 0.83~0.9997，
# 包长分布、28x28 窗口信息量、对角马氏距离在工控与训练分布之间都不可分
# （实测记录见 logs/probe_input_size*.log；探针脚本已随一次性工具清理），
# 说明这是「模型没见过」而不是「模型不确定」——softmax 无法表达这件事。
# 端口是目前唯一强可分的信号：工控流量的 46%~100% 落在下列端口，
# 而 USTC 全部 pcap 的命中率都 <= 0.1%（随机高端口巧合）。
# 因此对命中率超过阈值的 pcap，不再输出「检测到恶意流量」，改为
# 「不在模型适用范围内」，让误报不再伪装成结论。
# 工业控制协议端口。**只有这张表参与「工业域」判定** —— 见下面的拆分说明。
INDUSTRIAL_PROTOCOL_PORTS = {
    502: 'Modbus/TCP',
    102: 'S7comm/MMS',
    2222: 'EtherNet/IP',
    44818: 'EtherNet/IP',
    20000: 'DNP3',
    19999: 'DNP3',
    47808: 'BACnet',
    47809: 'BACnet',
    2404: 'IEC-104',
    2405: 'IEC-104',
    4840: 'OPC-UA',
    4843: 'OPC-UA/TLS',
    1962: 'PCworx',
    9600: 'OMRON FINS',
    2455: 'CODESYS',
    789: 'Red Lion',
    # 下面这批是补充：实测发现的漏网工业/工业仿真协议
    8002: 'CIGI',
    8004: 'CIGI',
    8005: 'CIGI',
    34962: 'PROFINET',
    34963: 'PROFINET',
    34964: 'PROFINET',
    5094: 'HART-IP',
    5007: 'MELSEC',
    18245: 'GE SRTP',
    18246: 'GE SRTP',
    48898: 'Beckhoff ADS',
    1911: 'Niagara Fox',
    4911: 'Niagara Fox',
    1089: 'FF-HSE',
}
# 训练集同样没覆盖、但**不属于工控**的基础设施协议。
# 拆开的原因（实测踩到）：原来一张表混装，DHCP 的 67/68 命中会让 domain_score 直接
# 到 0.6 —— dhcp.pcap 被判成「工业域」并套上 ICS 权重，而它跟工控毫无关系。
# 「模型没见过这个协议」与「这是工业域」是两件事，混在一起加权就是错的。
FOREIGN_PROTOCOL_PORTS = {
    67: 'DHCP',
    68: 'DHCP',
    161: 'SNMP',
    162: 'SNMP',
    123: 'NTP',
    554: 'RTSP',
    5060: 'SIP',
    5061: 'SIP',
    5671: 'AMQP/TLS',
    5672: 'AMQP',
    1883: 'MQTT',
    8883: 'MQTT/TLS',
    5683: 'CoAP',
}
# 兼容视图：判「整包是否落在模型没训练过的协议族」（分布外拒判仍用它）。
OUT_OF_TRAINING_PORTS = dict(INDUSTRIAL_PROTOCOL_PORTS)
OUT_OF_TRAINING_PORTS.update(FOREIGN_PROTOCOL_PORTS)# 会话命中上述端口的比例达到该值，即认为整包属于模型未训练过的协议族。
# 取 0.30 而不是 0.6：混合流量（工控会话 + 少量其它会话）在 0.3 以下仍会漏过，
# 但阈值再低会把只有零星工控会话的普通抓到包也拒判。0.3 是实测的折中点：
# test_ics 里 19/37 的误报文件命中率 >=0.667，其余多为 0%~0.5 的真混合流量。
OOD_INDUSTRIAL_RATIO_THRESHOLD = _env_float('DLIDS_OOD_INDUSTRIAL_RATIO', 0.30, 0.0, 1.0)
# 像素级分布外判据：不依赖协议名，直接看会话图的稀疏程度是否远离训练分布。
# 基线取自 USTC-TFC2016 训练集实测：像素均值 0.1226、非零占比 34.4%。
# 工控/工业协议流量实测只有 0.0241 / 7.55%（test_ics 的 55 个会话），即基线的 1/5 与 1/4.5。
# 这类输入会让 softmax 过度自信（实测 0.999+）却判错，正好补上 industrial_ratio 在
# 「混合流量里工业会话只占 6%」那批文件上的漏网。两个比值都是「实测 / 基线」。
# 像素统计基线：取自 USTC-TFC2016 训练集 1,957,416 个会话的实测值。
# 这组数字只用于前端展示「本文件的像素材质离训练集有多远」，**不参与分布外判定**。
# 原因（实测）：USTC 域内单个文件的像素均值跨度就有 122 倍
# （BitTorrent 0.0040 → WorldOfWarcraft 0.4899，非零占比 1.38%~95.78%），
# 比 USTC 与 ICS 之间的差距还大。任何全局阈值都会把 USTC 自己的良性文件
# （Facetime 0.0040 / Gmail 0.0116 / Skype 0.0078）一起误判成分布外。
PIXEL_BASELINE_MEAN = 0.14296
PIXEL_BASELINE_NONZERO = 0.3488
# 会话数低于该值时，单纯的「恶意比例」证据不足，仅作为低置信提示（不改判定）
SMALL_SAMPLE_SESSIONS = _env_int('DLIDS_SMALL_SAMPLE_SESSIONS', 5, 1)

# 上传限制：只接受 pcap/pcapng，并限制单文件体积（MB，默认 512）。
MAX_UPLOAD_MB = _env_int('DLIDS_MAX_UPLOAD_MB', 512, 1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {'.pcap', '.pcapng'}

# 空闲超时（秒），用于统一两条链路的"流"定义。
# 六分类链路（src/preprocessing/flow_features.py 的 FLOW_TIMEOUT_US）固定按 120 秒空闲
# 切分；二分类链路（FeatureExtractor.pcap_to_sessions）历史上不按时间切分，同一个
# 五元组无论停顿多久都算一条会话。于是同一份 pcap 上「二分类卡片说的会话数」与
# 「威胁类型分布说的流数」天然不相等，看起来像数据不一致。
# 默认 None = 保持历史口径（已训练权重与既有指标就是这个口径下得到的）；
# 显式设 DLIDS_FLOW_IDLE_TIMEOUT=120 即让二分类按同一规则切流。
_flow_idle_env = os.environ.get('DLIDS_FLOW_IDLE_TIMEOUT', '').strip()
try:
    FLOW_IDLE_TIMEOUT_SECONDS = float(_flow_idle_env) if _flow_idle_env else None
except ValueError:
    FLOW_IDLE_TIMEOUT_SECONDS = None

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
    """记录每个「已加载」模型对应的权重文件，供 /model-status 展示（代码审查第 33 条）。

    缺陷 D-31：加载函数会显式回填它**实际**用到的文件（cnn_bilstm 有两个候选，
    best_choice 可能不是候选列表里的第一个）。这里对已回填的值只做存在性校验，
    不再用"候选里第一个存在的"去覆盖，避免展示的来源与实际权重不一致。
    """
    for name, model in models.items():
        if model is None:
            model_sources.pop(name, None)
            continue
        recorded = model_sources.get(name)
        if recorded:
            if os.path.exists(os.path.join(BASE_DIR, recorded)):
                continue
            print(f"Warning: recorded checkpoint for {name} does not exist: {recorded}")
            model_sources.pop(name, None)
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
            class_names = list(best_choice['class_names'])
            # 主力模型的 class_names 是全局真源，先登记自己（此时比对必然一致）
            model_class_names['cnn_bilstm'] = list(class_names)

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
                        # 缺陷 D-09：旧实现直接覆盖全局 class_names，加载顺序是
                        # MLP → Transformer，最终值取决于这两个 ckpt；顺序一颠倒，
                        # 恶意/良性下标就错位且不报错。现在只记录、不覆盖。
                        _register_class_names('mlp', mlp_ckpt.get('class_names'))

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
                        _register_class_names('transformer', trans_ckpt.get('class_names'))

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

def _login_error_response(request, context, status_code=200, headers=None):
    """渲染登录页的错误态。

    ``status_code`` / ``headers`` 是给「被限流」这类语义化失败用的（缺陷 M7）：
    浏览器照旧显示同一张登录页，但调用方能从状态码与 Retry-After 知道该等多久。
    """
    try:
        return templates.TemplateResponse(request=request, name="login.html",
                                          context=context, status_code=status_code,
                                          headers=headers)
    except TypeError:
        # 老版本 starlette 的模板响应不支持 status_code/headers 关键字
        return templates.TemplateResponse("login.html", context)
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
            # 缺陷 M7：锁定过去返回 200 + 一个错误文案。脚本/监控无法把「被限流」
            # 与「口令错」区分开，也读不到还要等多久。这里按 HTTP 语义回 429 并带
            # Retry-After；页面（浏览器表单）仍然走同一份 HTML，只是状态码正确。
            return _login_error_response(
                request,
                {"request": request,
                 "error": f"登录尝试过于频繁，请 {wait_seconds} 秒后重试",
                 "locked": True,
                 "retry_after": wait_seconds},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={'Retry-After': str(wait_seconds)},
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
        # 缺陷 D-09：每个模型自带声明的类别名与不一致告警，前端可据此提示「下标可能错位」。
        "model_class_names": {name: model_class_names.get(name) for name in loaded_models},
        "class_names_conflicts": list(class_names_conflicts),
        # 缺陷 D-01：校准参数（env_calib vs 主权重）是否一致，重启后预测是否可复现。
        "calibration": dict(_CALIBRATION_STATUS),
        "calibration_mismatch": bool(_CALIBRATION_STATUS.get('mismatch')),
        # 缺陷 D-02：dist_stats 退化会让级联的距离支路永不再触发。
        "dist_stats": dict(_DIST_STATS_STATUS),
        "dist_stats_degenerate": bool(_DIST_STATS_STATUS.get('degenerate')),
        # 缺陷 D-07：两条链路实际生效的档位阈值，产物标签里的阈值必须与之一致。
        "flow_thresholds": {
            "confidence_high": FLOW_CONFIDENCE_HIGH,
            "confidence_low": FLOW_CONFIDENCE_LOW,
            "ood_threshold": (float(flow_centers['thresholds'][3])
                              if flow_centers is not None else None),
        },
    }



def _find_class_index(target_keywords, default_index=0):
    for idx, cname in enumerate(class_names):
        lowered = str(cname).lower()
        if any(keyword in lowered for keyword in target_keywords):
            return idx
    return default_index


def _resolve_class_indices(names=None):
    """解析「恶意类 / 良性类」下标。传入 names 时用于校验某个模型自己的类别表。"""
    table = list(names) if names else list(class_names)
    malware_idx = benign_idx = None
    for idx, cname in enumerate(table):
        lowered = str(cname).lower()
        if malware_idx is None and any(k in lowered for k in MALWARE_CLASS_KEYWORDS):
            malware_idx = idx
        elif benign_idx is None and any(k in lowered for k in BENIGN_CLASS_KEYWORDS):
            benign_idx = idx
    if malware_idx is None:
        malware_idx = 1 if len(table) > 1 else 0
    if benign_idx is None:
        benign_idx = 0 if malware_idx != 0 else (1 if len(table) > 1 else 0)
    return malware_idx, benign_idx


def _decision_label(is_malicious):
    """判定结论 -> 展示标签。全系统唯一一处把布尔判定翻译成标签（缺陷 D-06）。"""
    return DECISION_POSITIVE_LABEL if is_malicious else DECISION_NEGATIVE_LABEL


def _decide_single_session(argmax_idx, probs, malware_idx, threshold):
    """单模型会话判决：**先按阈值算判定，再由此决定标签**（缺陷 D-06）。

    旧实现在单模型链路里把 ``predicted_label = class_names[argmax]`` 直接当结论回传，
    于是 ``argmax=Malware`` 且 ``0.5 < malware_conf < 阈值`` 时，标签写着 Malware、
    ``is_malicious`` 却是 False，前端弹窗与统计表各说一套。
    现在返回 ``(is_malicious, argmax_label, decision_label)``，前端只渲染 decision_label。
    """
    malware_conf = float(probs[malware_idx]) if malware_idx < len(probs) else float(max(probs))
    is_malicious = bool(argmax_idx == malware_idx and malware_conf >= threshold)
    argmax_label = class_names[argmax_idx] if argmax_idx < len(class_names) else str(argmax_idx)
    return is_malicious, argmax_label, _decision_label(is_malicious)



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


# ── 按「数据域」加权的模型投票 ────────────────────────────────────────
# 用户提出的观察：不同模型对不同类型数据的判断能力差别很大，可靠性高的模型该给更高
# 的票权。这不是印象，仓库里已有的评估产物就能量出来 —— 同一批模型，换成工业控制
# 流量，F1 从 51 掉到 0.10：
#   checkpoints/model_metrics.json               同源测试集：6 个模型全 99.9x%，等权与加权几乎一样
#   logs/external_eval_test_ics.json             工业控制（test_ics）：Pure_BiLSTM 准确率 99.37% /
#                                                MLP 0.10% / CNN_BiLSTM 18.55%
#   logs/external_eval_img_cse_cic_ids2018.json  CSE 加密流量：Classic_CNN 60.38% 最好，
#                                                CNN_BiLSTM 21.50% 最差
# 结论：等权投票只在同源数据上站得住。到了域外流量，等权等于让四个近乎瞎猜的模型把
# 最强那个投掉。所以权重按「模型 × 域」的实测 F1 来定，而不是拍脑袋给系数。
#
# 默认 off：加权会改变既有检测结论，开启前必须先在真实数据上对比检出与误报。
# 打开方式：环境变量 DLIDS_ENSEMBLE_WEIGHTING=domain，或在「模型评估」页直接切换。
ENSEMBLE_WEIGHTING_MODE = (os.environ.get('DLIDS_ENSEMBLE_WEIGHTING') or 'off').strip().lower()
# 界面只暴露 off / domain 两个状态；auto 留给后续「按批次置信一致性自适应」的策略。
ENSEMBLE_WEIGHTING_MODES = ('off', 'domain', 'auto')
# 权重下限：F1 再差也保留 1.0 的底票（一个模型在某个域失手，不代表它对每个样本都错），
# 但差一个数量级时它的票几乎不改变结果。
ENSEMBLE_WEIGHT_FLOOR = 1.0
_DOMAIN_PAYLOAD_CACHE = {'sig': None, 'data': None}

# 评估产物里用的是大写模型名，代码里是小写 key，且 Lightweight 那个名字对不上，只能写表。
_METRIC_MODEL_KEYS = {
    'CNN_BiLSTM': 'cnn_bilstm',
    'Classic_CNN': 'classic_cnn',
    'Lightweight_CNN_BiLSTM': 'lightweight_cnn',
    'Pure_BiLSTM': 'pure_bilstm',
    'MLP': 'mlp',
    'Transformer': 'transformer',
}

_DOMAIN_WEIGHT_SOURCES = {
    'default': os.path.join('checkpoints', 'model_metrics.json'),
    'ics': os.path.join('logs', 'external_eval_test_ics.json'),
    'cse': os.path.join('logs', 'external_eval_img_cse_cic_ids2018.json'),
}

_DOMAIN_LABELS = {
    'default': '同源（USTC-TFC2016 测试集）',
    'ics': '工业控制（test_ics）',
    'cse': 'CSE 加密流量（CIC-IDS2018 图像）',
}


def _domain_metric_float(value):
    """评估产物里数值有两种写法：``"99.91%"`` 字符串与 float，统一成 float。"""
    if isinstance(value, str):
        value = value.strip().rstrip('%')
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _domain_payloads():
    """``{域: {available, error, models, total, class_counts}}``，按文件 mtime 缓存。

    读不到某个域的文件就把 available 置 False —— 调用方据此退回等权，绝不用「同源权重」
    冒充「域外同样可靠」，那正是等权投票犯的错。
    """
    sig = []
    for path in _DOMAIN_WEIGHT_SOURCES.values():
        try:
            sig.append(os.path.getmtime(os.path.join(BASE_DIR, path)))
        except OSError:
            sig.append(None)
    sig = tuple(sig)
    if _DOMAIN_PAYLOAD_CACHE['sig'] == sig and _DOMAIN_PAYLOAD_CACHE['data'] is not None:
        return _DOMAIN_PAYLOAD_CACHE['data']

    data = {}
    for domain, path in _DOMAIN_WEIGHT_SOURCES.items():
        full = os.path.join(BASE_DIR, path)
        try:
            with open(full, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            print('domain weights unavailable ({}): {}'.format(domain, e))
            data[domain] = {'available': False, 'error': str(e), 'models': {}, 'total': None,
                            'class_counts': None}
            continue
        models = payload if domain == 'default' else payload.get('models')
        if not isinstance(models, dict):
            data[domain] = {'available': False, 'error': 'models section missing',
                            'models': {}, 'total': None, 'class_counts': None}
            continue
        normalized = {}
        for metric_name, row in models.items():
            key = _METRIC_MODEL_KEYS.get(str(metric_name))
            if not key or not isinstance(row, dict):
                continue
            normalized[key] = {
                'f1': _domain_metric_float(row.get('f1_score')),
                'accuracy': _domain_metric_float(row.get('accuracy', row.get('test_accuracy'))),
                'precision': _domain_metric_float(row.get('precision')),
                'recall': _domain_metric_float(row.get('recall')),
                'fpr': _domain_metric_float(row.get('fpr')),
            }
        data[domain] = {
            'available': bool(normalized),
            'error': None if normalized else 'no usable f1_score',
            'models': normalized,
            'total': payload.get('total_sessions') or payload.get('sample_count'),
            'class_counts': payload.get('class_counts'),
        }

    _DOMAIN_PAYLOAD_CACHE['sig'] = sig
    _DOMAIN_PAYLOAD_CACHE['data'] = data
    return data


def _normalize_weights(raw):
    """把 {模型: F1} 变成「均值 1」的权重：保留相对可靠性，不改变整体尺度。"""
    weights = {}
    for name in ENSEMBLE_MODEL_ORDER:
        weights[name] = max(float(raw.get(name) or 0.0), ENSEMBLE_WEIGHT_FLOOR)
    mean = (sum(weights.values()) / len(weights)) if weights else 1.0
    if mean <= 0:
        mean = 1.0
    return {k: round(v / mean, 4) for k, v in weights.items()}


def _domain_model_weights():
    """``{域: {模型: 权重}}``，域为 default / ics / cse；没有依据的域为 None。"""
    table = {}
    for domain, payload in _domain_payloads().items():
        if not payload['available']:
            table[domain] = None
            continue
        raw = {k: (v.get('f1') or 0.0) for k, v in payload['models'].items()}
        table[domain] = _normalize_weights(raw) if raw else None
    return table


def _weights_for_domain(domain):
    """当前开关下该用哪套权重；off 或该域没有评估依据时返回 None（= 等权，行为不变）。"""
    if ENSEMBLE_WEIGHTING_MODE not in ('domain', 'auto'):
        return None
    return (_domain_model_weights().get(domain) or None)


# ── 域外程度：从「二选一」改成「连续插值」────────────────────────────────
# 原来只有一条硬判断：工业端口占比 ≥ 阈值就用 ics 权重，否则用 default 权重。
# 这条路对端口伪装（工控跑 8080、或把 Modbus 隧道进 443）完全失效 —— 命中率 0，
# 判定直接落到 default，于是「模型压根没见过这种流量」这件事一个字都没被表达出来。
# 这里补一条与端口无关的证据：二分类主力模型算出的对角马氏距离（越远＝越陌生）。
# 两个信号合成 0~1 的域外分数，再在 default 与 ics 两套权重之间线性插值。
# 分数 0 时插值结果就是 default 权重（等权），所以端口误判最坏也只是退回等权。
ENSEMBLE_DOMAIN_SIGNAL_WEIGHTS = {'port': 0.6, 'dist': 0.4}
# 距离比值 = 批次平均马氏距离 / 训练集 p99。0.5 以下视为正常不给分，2.0 以上饱和。
ENSEMBLE_DIST_RATIO_LOW = 0.5
ENSEMBLE_DIST_RATIO_HIGH = 2.0

# ── 分歧度调权 ────────────────────────────────────────────────────────
# 6 个模型分成 3:3 时，等权与域权重都在赌运气。这种局面更该听「换一个域也还站得住」
# 的模型（ICS 与 CSE 两域 F1 取较小值，见 _robust_weights）。
ENSEMBLE_CONTEST_START = 0.34       # 4:2（0.33）之前不介入
ENSEMBLE_CONTEST_FULL = 0.99        # 3:3 完全介入
ENSEMBLE_CONTEST_STRENGTH = 0.6     # 最多把 60% 的分量换成跨域稳健权重
_ROBUST_WEIGHTS_CACHE = {'sig': None, 'weights': None}


def _vote_contest(malware_votes, benign_votes):
    """票数有多僵持：0 = 一边倒，1 = 完全对半。"""
    total = int(malware_votes) + int(benign_votes)
    if total <= 0:
        return 0.0
    return round(1.0 - abs(int(malware_votes) - int(benign_votes)) / float(total), 4)


def _domain_out_score(port_ratio=None, dist_ratio=None, unknown_ratio=None):
    """0~1 的「这批输入有多像训练分布之外」，与文件名、路径无关。

    端口只是其中一个信号：缺失或为 0 时，距离信号照样会说话。
    任一信号传 None 就只在剩下的信号里重新归一化权重 —— 六分类没加载、
    或请求直接走集成（没有探针距离）时依然算得出来。
    返回 ``(分数, {信号: 归一化后的值})``。
    """
    parts = {}
    if port_ratio is not None:
        parts['port'] = max(0.0, min(1.0, float(port_ratio) / max(OOD_INDUSTRIAL_RATIO_THRESHOLD, 1e-6)))
    if dist_ratio is not None:
        span = max(ENSEMBLE_DIST_RATIO_HIGH - ENSEMBLE_DIST_RATIO_LOW, 1e-6)
        parts['dist'] = max(0.0, min(1.0, (float(dist_ratio) - ENSEMBLE_DIST_RATIO_LOW) / span))
    if unknown_ratio is not None:
        parts['unknown'] = max(0.0, min(1.0, float(unknown_ratio) / 0.5))
    if not parts:
        return 0.0, {}
    weight_sum = sum(ENSEMBLE_DOMAIN_SIGNAL_WEIGHTS.get(k, 0.0) for k in parts)
    if weight_sum <= 0:
        return 0.0, {}
    score = sum(ENSEMBLE_DOMAIN_SIGNAL_WEIGHTS.get(k, 0.0) * v for k, v in parts.items()) / weight_sum
    return round(score, 4), {k: round(v, 4) for k, v in parts.items()}


def _blend_weights(base_weights, target_weights, t):
    """按 t∈[0,1] 在两套权重之间线性插值（两套都是均值 1，插值后仍是均值 1）。"""
    if not base_weights or not target_weights:
        return base_weights
    t = max(0.0, min(1.0, float(t)))
    out = {}
    for name in set(base_weights) | set(target_weights):
        base = float(base_weights.get(name, 1.0))
        goal = float(target_weights.get(name, base))
        out[name] = round((1.0 - t) * base + t * goal, 4)
    return out


def _robust_weights():
    """「换一个域也别崩」的兜底权重：两域最坏误报率越低，权重越高。

    为什么不用 F1（这是踩过的坑）：
      · ICS 的恶意样本只有 4 条（45,497 良性 : 4 恶意），F1 完全被单条样本的运气主导；
      · CSE 的 AUC 只有 0.33~0.61，排序能力已经接近随机，F1 同样是噪声。
    两域里唯一统计稳定的量是**误报率**（45,497 + 11,354 条良性样本支撑），
    所以依据取 ``1 - max(FPR_ics, FPR_cse)``：语义是「陌生流量上最少误报的那个更可信」。
    取两域的最大误报率而不是平均，是因为一个模型在某个域彻底失守就不该被听，
    平均会把这件失守摊平掉（MLP 在 ICS 上 FPR 99.91%）。
    """
    payloads = _domain_payloads()
    sig = _DOMAIN_PAYLOAD_CACHE.get('sig')
    if _ROBUST_WEIGHTS_CACHE['sig'] == sig and _ROBUST_WEIGHTS_CACHE['weights'] is not None:
        return _ROBUST_WEIGHTS_CACHE['weights']
    worst_fpr = {}
    worst_f1 = {}
    for domain in ('ics', 'cse'):
        payload = payloads.get(domain) or {}
        if not payload.get('available'):
            continue
        for name, metric in (payload.get('models') or {}).items():
            fpr = metric.get('fpr')
            if fpr is not None:
                worst_fpr[name] = fpr if name not in worst_fpr else max(worst_fpr[name], fpr)
            f1 = metric.get('f1')
            if f1 is not None:
                worst_f1[name] = f1 if name not in worst_f1 else min(worst_f1[name], f1)
    basis = {}
    for name in ENSEMBLE_MODEL_ORDER:
        if name in worst_fpr:
            # 1 - 最坏 FPR，换算成百分制，量级与 _normalize_weights 的下限 1.0 匹配
            basis[name] = max(0.0, 100.0 - float(worst_fpr[name]))
        elif name in worst_f1:
            # 没有 FPR 产物时才退回 F1（已知不可靠，仅作最后兜底）
            basis[name] = float(worst_f1[name])
    weights = _normalize_weights(basis) if basis else None
    _ROBUST_WEIGHTS_CACHE['sig'] = sig
    _ROBUST_WEIGHTS_CACHE['weights'] = weights
    return weights


def _contest_adjusted_weights(base_weights, contest, strength=None):
    """分歧度越高，越把权重往「跨域稳健」那一侧掺。

    返回 ``(生效权重, 说明)``；说明里带 blend 系数，事后核对这一票被改了多少时有据可查。
    """
    if not base_weights:
        return base_weights, None
    robust = _robust_weights()
    if not robust:
        return base_weights, {'contest': round(float(contest), 4), 'blend': 0.0,
                              'reason': 'no_robust_weights'}
    span = max(ENSEMBLE_CONTEST_FULL - ENSEMBLE_CONTEST_START, 1e-6)
    level = max(0.0, min(1.0, (float(contest) - ENSEMBLE_CONTEST_START) / span))
    cap = ENSEMBLE_CONTEST_STRENGTH if strength is None else max(0.0, min(1.0, float(strength)))
    blend = round(level * cap, 4)
    info = {'contest': round(float(contest), 4), 'blend': blend}
    if blend <= 0.0:
        return base_weights, info
    info['robust_weights'] = {k: round(float(v), 4) for k, v in robust.items()}
    return _blend_weights(base_weights, robust, blend), info


def _batch_mahalanobis(batch_tensor):
    """批次平均对角马氏距离；距离统计缺失或维度不符时返回 None。

    这是全链路唯一「与协议名无关」的陌生度证据：softmax 对没见过的流量照样给 99%，
    距离不会。cascade 探针与需求 A 的域外分数共用它，所以抽出来。
    """
    if _DIST_MEAN is None or _DIST_STD is None:
        return None
    try:
        flat = batch_tensor.reshape(batch_tensor.size(0), -1).cpu().numpy()
        if flat.shape[1] != _DIST_MEAN.shape[0]:
            return None
        z = (flat - _DIST_MEAN) / _DIST_STD
        return float(np.mean((z ** 2).mean(axis=1)))
    except Exception as e:
        print('batch distance failed:', e)
        return None


def _vote_with_ensemble(batch_tensor, chunk_size=128, weights=None, contest_strength=None):
    loaded_models = _get_loaded_detection_models()
    if not loaded_models:
        raise RuntimeError('No detection models are available for ensemble voting.')

    malware_idx, benign_idx = _resolve_class_indices()

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
        # 缺陷 D-11：只对「投了这一票」的模型取置信度均值，输出才与票数同语义。
        malicious_conf_of_voters = []
        benign_conf_of_voters = []
        malicious_votes = 0
        benign_votes = 0
        # 加权票：权重来自「模型 × 域」的实测 F1（见 _domain_model_weights）。
        # 注意两套计数必须并存：malware_votes 是「几个模型投了恶意」，
        # malicious_weight 是「投恶意的模型权重之和」，混用会把 3 票说成 2.1 票。
        malicious_weight = 0.0
        benign_weight = 0.0
        # 这一样本里各模型怎么投的、各自的基础权重是多少。
        # 分歧度调权必须在「票数已经数完」之后才算得出来，所以先收集、后加权。
        voters = []

        for model_name in ENSEMBLE_MODEL_ORDER:
            probs = model_probabilities.get(model_name)
            if probs is None or session_idx >= probs.size(0):
                continue

            session_probs = probs[session_idx]
            malware_conf = float(session_probs[malware_idx].item()) if malware_idx < session_probs.shape[0] else float(session_probs.max().item())
            benign_conf = float(session_probs[benign_idx].item()) if benign_idx < session_probs.shape[0] else float(session_probs.max().item())
            threshold = MODEL_MALWARE_THRESHOLDS.get(model_name, DEFAULT_MALWARE_THRESHOLD)
            is_malicious = malware_conf >= threshold

            model_weight = float(weights.get(model_name, 1.0)) if weights else 1.0
            malicious_votes += int(is_malicious)
            benign_votes += int(not is_malicious)
            voters.append((model_name, bool(is_malicious), model_weight))
            if is_malicious:
                malicious_weight += model_weight
            else:
                benign_weight += model_weight
            malware_conf_scores.append(malware_conf)
            benign_conf_scores.append(benign_conf)
            (malicious_conf_of_voters if is_malicious else benign_conf_of_voters).append(
                malware_conf if is_malicious else benign_conf)
            model_votes[model_name] = {
                # 标签仍用统一的判决标签域（缺陷 D-09：不再出现第三套串）
                'predicted_label': _decision_label(is_malicious),
                'malware_confidence': f'{malware_conf:.4f}',
                'benign_confidence': f'{benign_conf:.4f}',
                'threshold': threshold,
                # 该模型在这一票上占多少权重（等权模式恒为 1.0）
                'vote_weight': round(model_weight, 4),
            }

        # 分歧度调权：只在有域权重时生效，off 模式的行为一个字都不变。
        # 票数越接近（3:3 最僵），越把权重掺向「换域也不崩」的那几个模型。
        contest = _vote_contest(malicious_votes, benign_votes)
        contest_info = None
        if weights and voters:
            adjusted, contest_info = _contest_adjusted_weights(weights, contest, contest_strength)
            if contest_info and contest_info.get('blend', 0.0) > 0.0:
                malicious_weight = 0.0
                benign_weight = 0.0
                for voter_name, voter_malicious, voter_base in voters:
                    voter_weight = float(adjusted.get(voter_name, voter_base))
                    if voter_malicious:
                        malicious_weight += voter_weight
                    else:
                        benign_weight += voter_weight
                    if voter_name in model_votes:
                        model_votes[voter_name]['vote_weight'] = round(voter_weight, 4)

        tie_break = None
        # 有域权重时比权重，否则比票数 —— 两者都是在回答「哪一方的意见更重」。
        malicious_side = malicious_weight if weights else float(malicious_votes)
        benign_side = benign_weight if weights else float(benign_votes)
        if malicious_side > benign_side:
            is_malicious = True
            winning_scores = malicious_conf_of_voters or malware_conf_scores
        elif benign_side > malicious_side:
            is_malicious = False
            winning_scores = benign_conf_of_voters or benign_conf_scores
        else:
            # 缺陷 D-10：平票不再用与单模型阈值无关的 0.5 兜底，改用「参与投票模型的
            # 恶意阈值均值」——它与各模型的调参依据同源，语义上是「按同一标准再看一次」。
            voter_thresholds = [MODEL_MALWARE_THRESHOLDS.get(n, DEFAULT_MALWARE_THRESHOLD)
                                for n in model_votes]
            tie_threshold = float(np.mean(voter_thresholds)) if voter_thresholds else 0.5
            avg_malware_conf = float(np.mean(malware_conf_scores)) if malware_conf_scores else 0.0
            is_malicious = avg_malware_conf >= tie_threshold
            tie_break = {
                'policy': 'tie_by_mean_threshold',
                'avg_malware_confidence': round(avg_malware_conf, 4),
                'tie_threshold': round(tie_threshold, 4),
                'resolved_to': _decision_label(is_malicious),
            }
            winning_scores = (malicious_conf_of_voters if is_malicious else benign_conf_of_voters) \
                or malware_conf_scores

        final_confidence = float(np.mean(winning_scores)) if winning_scores else 0.0
        vote_count = len(model_votes)
        session_votes.append({
            # 判定结论标签（唯一真源），与 is_malicious 永不矛盾
            'predicted_label': _decision_label(is_malicious),
            'is_malicious': bool(is_malicious),
            # 语义见 confidence_semantics：这是「投该票的模型在该类上的平均概率」，
            # **不是**投票比例（投票比例看 vote_ratio），两者不可互相替代（缺陷 D-11）。
            'predicted_confidence': final_confidence,
            'malware_votes': malicious_votes,
            'benign_votes': benign_votes,
            'malware_weight': round(malicious_weight, 4),
            'benign_weight': round(benign_weight, 4),
            'weighting': 'domain' if weights else 'uniform',
            'vote_count': vote_count,
            # 这一票有多僵 + 分歧度把权重改了多少（blend=0 表示没改）
            'vote_contest': contest,
            'contest_adjust': contest_info,
            'vote_ratio': ((malicious_weight / (malicious_weight + benign_weight))
                           if (weights and (malicious_weight + benign_weight) > 0)
                           else ((malicious_votes / vote_count) if vote_count else 0.0)),
            'malware_confidences': malware_conf_scores,
            'benign_confidences': benign_conf_scores,
            'model_votes': model_votes,
            'tie_break': tie_break,
        })

    return {
        'malware_idx': malware_idx,
        'benign_idx': benign_idx,
        'available_models': [model_name for model_name, _ in loaded_models],
        'confidence_semantics': 'mean probability of the winning class over voting models',
        # 这次投票到底用的是等权还是域权重 —— 事后核对结论时必须能查到
        'weighting': 'domain' if weights else 'uniform',
        'weights_used': ({k: round(float(v), 4) for k, v in weights.items()} if weights else None),
        # 分歧度调权的强度上限（0 = 关闭），便于回放时解释「这一票为什么这么算」
        'contest_strength': (ENSEMBLE_CONTEST_STRENGTH if contest_strength is None
                             else float(contest_strength)),
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


def _ensemble_weights_payload():
    """三域权重 + 每模型原始指标，供「模型评估」页画表，也是接口的唯一返回体。"""
    weights = _domain_model_weights()
    payloads = _domain_payloads()
    domains = []
    for name in ('default', 'ics', 'cse'):
        p = payloads.get(name) or {}
        domains.append({
            'domain': name,
            'label': _DOMAIN_LABELS.get(name, name),
            'source': _DOMAIN_WEIGHT_SOURCES.get(name),
            'available': bool(p.get('available')),
            'error': p.get('error'),
            'total': p.get('total'),
            'class_counts': p.get('class_counts'),
        })
    rows = []
    for name in ENSEMBLE_MODEL_ORDER:
        rows.append({
            'model': name,
            'display': MODEL_DISPLAY_NAMES.get(name, name),
            'weights': {d['domain']: (weights.get(d['domain']) or {}).get(name) for d in domains},
            'metrics': {
                d['domain']: (((payloads.get(d['domain']) or {}).get('models') or {}).get(name))
                for d in domains
            },
        })
    return {
        'status': 'success',
        'mode': ENSEMBLE_WEIGHTING_MODE,
        'enabled': ENSEMBLE_WEIGHTING_MODE in ('domain', 'auto'),
        'floor': ENSEMBLE_WEIGHT_FLOOR,
        'models': list(ENSEMBLE_MODEL_ORDER),
        'domains': domains,
        'rows': rows,
        'note': ('权重 = max(该域实测 F1, 下限) 后按均值归一化（均值=1，只改相对分量，'
                 '不改整体尺度）。why F1：ICS 是 45497:4 的极端不平衡集，'
                 '全判良性的准确率也有 99.91%，只有 F1 反映真检出能力。'),
        # ── 以下是「这套权重在推理时怎么被用」的运行参数 ──
        # 域权重（上表）回答「某个域里各模型多可信」；下面的 robust/contest 回答
        # 「不知道是哪个域、或者模型们吵起来的时候，该听谁的」。
        'robust': {
            # 稳健权重刻意不用 F1：ICS 的 F1 只有 4 条恶意样本支撑，CSE 的 AUC 接近随机。
            'basis': '两域最大误报率：1 − max(FPR_ics, FPR_cse)',
            'weights': _robust_weights(),
        },
        'contest': {
            'start': ENSEMBLE_CONTEST_START,
            'full': ENSEMBLE_CONTEST_FULL,
            'strength': ENSEMBLE_CONTEST_STRENGTH,
        },
        'domain_score': {
            'signals': dict(ENSEMBLE_DOMAIN_SIGNAL_WEIGHTS),
            'dist_ratio_low': ENSEMBLE_DIST_RATIO_LOW,
            'dist_ratio_high': ENSEMBLE_DIST_RATIO_HIGH,
            'port_threshold': OOD_INDUSTRIAL_RATIO_THRESHOLD,
        },
        'plan_doc': 'docs/cross_domain_improvement_plan.md',
    }


@app.get('/api/ensemble/weights')
def ensemble_weights_api(session_user: str = Depends(require_session)):
    """六模型投票的加权依据：每个域里各模型的实测指标与归一化权重。

    数据来自仓库里已有的评估产物（按文件 mtime 缓存），不在请求时重算 ——
    这是「权重为什么长这样」的可核对来源。
    """
    return JSONResponse(content=_ensemble_weights_payload())


@app.post('/api/ensemble/weights')
def set_ensemble_weights_api(mode: str = Form('off'),
                             session_user: str = Depends(require_session)):
    """切换加权投票开关。改了就是全局生效（下一次推理就用新权重），不需要重启。"""
    global ENSEMBLE_WEIGHTING_MODE
    target = str(mode or 'off').strip().lower()
    if target not in ENSEMBLE_WEIGHTING_MODES:
        return JSONResponse(status_code=400, content={
            'status': 'error',
            'message': 'mode must be one of: ' + ' / '.join(ENSEMBLE_WEIGHTING_MODES),
        })
    if target in ('domain', 'auto') and not any(_domain_model_weights().values()):
        # 一份域外评估都没有时，加权无从谈起：直接拒绝，别让界面显示「已开启」而实际等权。
        return JSONResponse(status_code=409, content={
            'status': 'error',
            'message': '没有可用的域外评估产物，无法按域加权',
            'domains': _ensemble_weights_payload()['domains'],
        })
    ENSEMBLE_WEIGHTING_MODE = target
    print('[ensemble] weighting mode -> {}'.format(ENSEMBLE_WEIGHTING_MODE))
    return JSONResponse(content=_ensemble_weights_payload())


# 跨域评估的数据源：全部来自实际跑过的实验日志，不在这里现算。
# 这些 JSON 由 tools/evaluate_external.py（图像分支）、tools/evaluate_external_images.py、
# tools/train_flow_model.py（流特征）、tools/finetune_local.py、tools/train_autoencoder.py 产出。
_CROSS_DOMAIN_FILES = {
    "image_ics": os.path.join("logs", "external_eval_test_ics.json"),
    "image_cse": os.path.join("logs", "external_eval_img_cse_cic_ids2018.json"),
    "flow_multiclass": os.path.join("logs", "flow_multiclass_metrics.json"),
    "flow_finetune": os.path.join("logs", "finetune_local.json"),
    "flow_adanb": os.path.join("logs", "adanb_eval.json"),
    "flow_autoencoder": os.path.join("logs", "flow_ae_metrics.json"),
}

# 流特征跨天对照：数字摘自 logs/pipeline_b2_summary.txt / b3 / b4 / b5，
# 不再回到 CSV 重算（每次重跑要几十分钟，而结论是稳定的）。
#
# 单位约定（缺陷 D-03）：四项**全部是百分比**，fpr 也是百分比。
# 三类日志的原始口径本来不一致 —— b2 打印的是比例（0.00342857），b3/b4/b5 打印的是
# 带 % 的百分比（0.15%）。之前的表把两者混着填，于是 A 组「fpr 0.0034」看起来比
# P3 组「fpr 15.35」小四个数量级，读者会误以为跨天误报率极低。这里统一换算成百分比，
# 并在 fpr 旁注明原始值，避免再被比例/百分比混用骗一次。
_FLOW_CROSS_DAY = [
    {"group": "A", "desc": "15-02+21-02 训练 → 02-03（跨天，不加权）",
     # pipeline_b2: accuracy 72.5983581053589 / recall 0.0 / fpr 0.00342856841444535(比例) / auc 0.5000772095478796
     "accuracy": 72.60, "detection_rate": 0.00, "fpr": 0.3429, "auc": 0.5001},
    {"group": "B", "desc": "15-02 训练 → 15-02（同天，加权）",
     # pipeline_b2: accuracy 99.99038967928438 / recall 100.0 / fpr 0.010120945296290674(比例)
     "accuracy": 99.99, "detection_rate": 100.00, "fpr": 1.0121, "auc": 0.99999},
    {"group": "C", "desc": "15-02 训练 → 15-02（同天，不加权）",
     # pipeline_b2: accuracy 99.99115850494162 / recall 99.97142748295174 / fpr 0.0077931278781438185(比例)
     "accuracy": 99.99, "detection_rate": 99.97, "fpr": 0.7793, "auc": 0.99999},
    {"group": "F", "desc": "02-03+15-02 训练 → 21-02（跨天镜像）",
     # pipeline_b3: 34.41% / 0.00% / 0.00% / AUC 0.2420
     "accuracy": 34.41, "detection_rate": 0.00, "fpr": 0.00, "auc": 0.2420},
    {"group": "G", "desc": "02-03+21-02 训练 → 15-02（跨天镜像）",
     # pipeline_b3: 94.94% / 0.00% / 0.01% / AUC 0.2969（原先写成 0.0 丢了精度）
     "accuracy": 94.94, "detection_rate": 0.00, "fpr": 0.01, "auc": 0.2969},
    {"group": "P1", "desc": "剔除 Dst Port + 全局标准化（跨天）",
     # pipeline_b5: P1_dropport_global 72.49% / 0.00% / 0.15% / AUC 0.6098
     # （原先 acc 写成 72.59 —— 那是 b4 的 global 组，不是 P1）
     "accuracy": 72.49, "detection_rate": 0.00, "fpr": 0.15, "auc": 0.6098},
    {"group": "P3", "desc": "剔除端口 + log 时间特征 + 批内标准化（跨天）",
     # pipeline_b5: P3_all_batch 70.09% / 31.52% / 15.35% / AUC 0.4969
     # （原先 acc 写成 60.76，与日志不符）
     "accuracy": 70.09, "detection_rate": 31.52, "fpr": 15.35, "auc": 0.4969},
]


def _cross_domain_caveats(name, data):
    """给每份跨域评估结果贴上「这个数字能不能用」的说明（缺陷 M4）。

    起因：`/api/cross-domain` 把 image_ics 上 MLP 的 accuracy 0.10% / FPR 99.91%
    原样摆出来，页面上没有任何风险提示。那批数据里 Malware 只有 4 条、Benign 45497 条，
    于是「全部判成 Benign」就能拿到 99.99% 的准确率 —— 这个 accuracy 不是能力指标，
    而是类别不平衡的读数。同理，0.10% 说明该模型几乎把所有流都判成恶意。

    这里不改任何原始指标（日志是实验产物，改动等于篡改证据），只把解读所需的
    上下文补进响应：少数类基数、不平衡比、以及按指标本身能得出的定性结论。
    """
    out = {'dataset': name, 'severity': 'info', 'notes': [],
           'metrics_reliable': True, 'minority_count': None, 'imbalance_ratio': None,
           'rate_unit_note': None}
    if not isinstance(data, dict):
        return out

    counts = data.get('class_counts') or {}
    if isinstance(counts, dict) and counts:
        try:
            vals = {k: int(v) for k, v in counts.items() if v is not None}
        except Exception:
            vals = {}
        if vals:
            lo_k = min(vals, key=vals.get)
            hi_k = max(vals, key=vals.get)
            out['minority_count'] = vals[lo_k]
            if vals[lo_k] > 0:
                out['imbalance_ratio'] = round(vals[hi_k] / vals[lo_k], 1)
            if vals[lo_k] < 100:
                out['metrics_reliable'] = False
                out['severity'] = 'high'
                out['notes'].append(
                    f"少数类只有 {vals[lo_k]} 条（{lo_k}），{hi_k} 有 {vals[hi_k]} 条。"
                    f"此时 accuracy / precision 主要由多数类决定，不能当作模型能力；"
                    f"请改看 recall 与 FPR，且 recall 在只有 {vals[lo_k]} 条样本时波动极大。")

    models = data.get('models')
    if isinstance(models, dict) and models:
        bad = []
        for mname, m in models.items():
            if not isinstance(m, dict):
                continue
            acc = m.get('accuracy')
            fpr = m.get('fpr')
            if isinstance(acc, (int, float)) and acc < 60:
                bad.append(f"{mname} accuracy {acc:.2f}%"
                           + (f" / FPR {fpr:.2f}%" if isinstance(fpr, (int, float)) else ''))
        if bad:
            out['severity'] = 'high'
            out['metrics_reliable'] = False
            out['notes'].append(
                "以下模型在该数据集上已接近或低于随机水平，属于跨域失效而非能力指标："
                + '；'.join(bad) + "。")

    # 六分类：随机切分导致同一条 pcap 的流同时出现在训练集与测试集（缺陷 D-08），
    # 99.94% 是泄漏后的数字，必须标注，否则会被当成真实泛化能力。
    if name == 'flow_multiclass':
        if data.get('split_strategy') in ('grouped', 'block') or data.get('leakage_controlled'):
            pass
        else:
            out['severity'] = 'high'
            out['metrics_reliable'] = False
            out['notes'].append(
                "该指标的切分方式为按行随机（train_test_split + stratify），同一条 pcap 的"
                "流会同时落入训练集与测试集，test_accuracy 存在数据泄漏，不能代表跨环境泛化能力。"
                "重跑 tools/train_multiclass.py --split block 可得到无泄漏口径。")

    # 比率字段的单位与越界问题（缺陷 D-18）。
    # 这里先纠正一个误读：`logs/flow_ae_metrics.json` 里 `default_threshold.fpr=1.91`、
    # `precision=2.24` 并不是「0–1 比例口径下大于 1」，而是 `tools/train_autoencoder.py`
    # 里统一 `*100` 出来的**百分比**（1.91% / 2.24%），与同一份产物的 workpoints 一致。
    # 真正会出错的只有两种情况，分开判：
    #   1. > 100% —— 一定是算错（分子分母混用），必须报；
    #   2. 同一份产物里同一个字段出现两种单位 —— 比较时必然错位，也必须报。
    _scan_targets = [('root', data)]
    for _k in ('default_threshold', 'baseline'):
        if isinstance(data.get(_k), dict):
            _scan_targets.append((_k, data[_k]))
    if isinstance(data.get('workpoints'), list):
        for _i, _wp in enumerate(data['workpoints'][:16]):
            if isinstance(_wp, dict):
                _scan_targets.append((f'workpoints[{_i}]', _wp))
    if isinstance(data.get('models'), dict):
        for _mn, _mv in list(data['models'].items())[:16]:
            if isinstance(_mv, dict):
                _scan_targets.append((f'models.{_mn}', _mv))

    _over = []
    _has_rate = False
    for _where, _obj in _scan_targets:
        for key in ('fpr', 'false_positive_rate', 'precision', 'detection_rate'):
            v = _obj.get(key)
            if not isinstance(v, (int, float)):
                continue
            _has_rate = True
            if v > 100.0:
                _over.append(f"{key}={v}（{_where}）超过 100%")
    if _over:
        out['severity'] = 'high'
        out['metrics_reliable'] = False
        out['notes'].append(
            '；'.join(_over) + '。该字段口径有误（分子分母混用了样本数与比例），'
            '不要直接与其他表格的百分比比较。')
    elif _has_rate and not (data.get('rate_unit') or data.get('unit')):
        # 单位未声明本身不是错误，但会让读者把 1.91 读成「191%」或「0–1 比例下越界」（缺陷 D-18）。
        # 放进独立字段而不是 notes，避免把「一切正常」的结论挤掉。
        out['rate_unit_note'] = (
            '该组比率字段未在产物里声明单位；按 tools/ 下脚本的口径均为百分比（0–100，'
            '如 fpr=1.91 表示 1.91%），比较时请统一按百分比读。')

    # 全零结果（缺陷 D-19）：adaBN 评估里出现满屏 0，是评估脚本没接上标签而非模型归零。
    baseline = data.get('baseline') if isinstance(data.get('baseline'), dict) else {}
    adanb = data.get('adanb')
    if isinstance(adanb, list) and adanb:
        first = adanb[0] if isinstance(adanb[0], dict) else {}
        zeros = all(first.get(k) in (0, 0.0, None)
                    for k in ('accuracy', 'detection_rate', 'precision'))
        if zeros and baseline and baseline.get('accuracy'):
            out['severity'] = 'high'
            out['metrics_reliable'] = False
            out['notes'].append(
                "adaBN 各轮指标全为 0 而 baseline 非零，说明评估环节没有正确接上标签/预测，"
                "这一组数不能用于判断自适应是否有效。")

    # 零检出（比「全零」更常见也更重要）：有恶意样本却一个都没抓到。
    # 这不是口径错误，是「该配置在这个目标域上不可用」的直接证据，
    # 必须显式说出来，否则 72.59% 的 accuracy 会被读成「基本可用」。
    _zero_recall = []
    for _where, _node in (('adanb', adanb[0] if isinstance(adanb, list) and adanb else None),
                          ('baseline', baseline or None)):
        if not isinstance(_node, dict):
            continue
        _fn = _node.get('fn')
        _tp = _node.get('tp')
        if isinstance(_fn, int) and isinstance(_tp, int) and _fn > 0 and _tp == 0:
            _dr = _node.get('detection_rate')
            _zero_recall.append(
                f"{_where} 检出率为 {0 if _dr is None else _dr}%（TP=0 / FN={_fn}）")
    if _zero_recall:
        out['severity'] = 'high'
        out['metrics_reliable'] = False
        out['notes'].append(
            '；'.join(_zero_recall) + '。该配置在目标域上一条恶意流量都没检出，'
            'accuracy 完全由良性样本贡献，不要当作可用性证据。')

    if not out['notes']:
        out['notes'].append('该组指标的口径与样本量正常，可直接引用。')
    return out


@app.get("/api/cross-domain")
def get_cross_domain(session_user: str = Depends(require_session)):
    """跨域评估汇总：同源基线 / 三个跨域实验 / 已尝试的适配手段。

    代码审查报告的结论是「模型在同源数据上 99%+，换环境即失效」，
    这张面板就是把这个结论的原始数字摆出来，避免只看同源指标误判能力。
    """
    def _load(key):
        path = os.path.join(BASE_DIR, _CROSS_DOMAIN_FILES[key])
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    image_ics = _load("image_ics")
    image_cse = _load("image_cse")
    flow_mc = _load("flow_multiclass")
    flow_ft = _load("flow_finetune")
    flow_adanb = _load("flow_adanb")
    flow_ae = _load("flow_autoencoder")

    availability = {
        "image_ics": image_ics is not None,
        "image_cse": image_cse is not None,
        "flow_multiclass": flow_mc is not None,
        "flow_finetune": flow_ft is not None,
        "flow_adanb": flow_adanb is not None,
        "flow_autoencoder": flow_ae is not None,
    }

    return JSONResponse(content={
        "status": "success",
        "availability": availability,
        # 同源基线用 model_metrics.json（/api/metrics 已经提供），这里只给跨域部分
        "image_ics": image_ics,
        "image_cse": image_cse,
        "flow_cross_day": _FLOW_CROSS_DAY,
        "flow_multiclass": flow_mc,
        "flow_finetune": flow_ft,
        "flow_adanb": flow_adanb,
        "flow_autoencoder": flow_ae,
        # 缺陷 M4：每份结果的可用性说明（样本量、不平衡、泄漏、口径错误）。
        # 前端必须把它显示在对应表格上方，否则 0.10% / 99.91% 这种数字
        # 与 99.37% 并排出现，读者无从判断哪个是失效模型、哪个是真实能力。
        "caveats": {
            "image_ics": _cross_domain_caveats('image_ics', image_ics),
            "image_cse": _cross_domain_caveats('image_cse', image_cse),
            "flow_multiclass": _cross_domain_caveats('flow_multiclass', flow_mc),
            "flow_finetune": _cross_domain_caveats('flow_finetune', flow_ft),
            "flow_adanb": _cross_domain_caveats('flow_adanb', flow_adanb),
            "flow_autoencoder": _cross_domain_caveats('flow_autoencoder', flow_ae),
        },
    })



@app.get("/")
def read_root(request: Request):
    return RedirectResponse(url="/login")


# ── 页面路由 ──────────────────────────────────────────────────────────────
# 5 个页面共用 base.html 外壳（侧边导航 + 顶栏常驻抓包控制），只有 content 块不同。
# 页面之间是整页跳转，前端不再做任何 DOM 搬移，因此也不会再有「切页丢状态」的问题。
_PAGES = {
    "overview": ("index.html", "总览", "系统健康、流量趋势与最近检测记录"),
    # 上传检测与实时抓包是同一套检测能力的两种输入方式，2026-11-01 起合并成一个「检测工作台」：
    # 侧栏只保留一项，页面内用顶部 tab 切换两个子视图（detect.html 的 data-detect-pane）。
    # /upload 与 /live 作为旧链接保留，重定向到 /detect?tab=...，避免书签失效。
    "detect": ("detect.html", "检测工作台", "拖入 PCAP 批量检测，或选择网卡实时抓包"),
    "records": ("records.html", "检测记录", "上传与实时抓包的全部检测结果"),
    # 复核中心：上传与实时两条链路里所有「待确认 / 人工复核」的流量都在这里处置，
    # 可以让本地 LLM 先给建议，也可以人工逐条定案。过去这块散在实时页的侧栏里，
    # 上传检测产出的可疑流量根本无处处置。
    "review": ("review.html", "复核中心", "待确认与人工复核流量的统一处置台"),
    "models": ("models.html", "模型评估", "6 个模型的准确率、延迟与对比指标"),
    # 流量分类走的是 CSV 流特征分支（78 维 CICFlowMeter 特征 + 六分类模型），
    # 与 pcap 字节图像链路完全独立，所以单独一页而不是并进检测工作台。
    "flow": ("flow.html", "流量分类", "上传流特征 CSV，按攻击类型归类并分流复核"),
}


def _build_status_block() -> str:
    """模型未就绪时的红色告警块，5 个页面共用同一份文案。"""
    if models.get(DEFAULT_MODEL_TYPE) is not None or models.get('cnn_bilstm') is not None:
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
            # 缺陷 M5：上传页文案写着 500MB，而后端实际按 MAX_UPLOAD_MB（默认 512）拒收。
            # 从前端拿到的是「文案 500 / 行为 512」两套数，用户传到 505MB 会看到一句
            # 「超过 512MB」而页面上压根没提过 512。现在两处共用同一个值。
            "max_upload_mb": MAX_UPLOAD_MB,
        }
    )


@app.get("/dashboard")
def dashboard(request: Request):
    return _render_page(request, "overview")


@app.get("/detect")
def detect_page(request: Request, tab: str = ""):
    # tab 只用于「旧链接跳转过来时先展开哪个子视图」，页面本身两个子视图都在，
    # 前端 detect_impl 会读 ?tab= 决定初始高亮；非法值一律回落到默认（upload）。
    return _render_page(request, "detect")


@app.get("/upload")
def upload_page(request: Request):
    # 旧地址：合并进检测工作台后保留跳转，书签与外部链接不至于 404
    return RedirectResponse(url="/detect?tab=upload", status_code=302)


@app.get("/live")
def live_page(request: Request):
    return RedirectResponse(url="/detect?tab=live", status_code=302)


@app.get("/records")
def records_page(request: Request):
    return _render_page(request, "records")


@app.get("/review")
def review_page(request: Request):
    return _render_page(request, "review")


# ---- 检测记录的服务端持久化（缺口 L2） ----
# 原先「检测记录」页完全没有后端：数据只存在浏览器的 localStorage 里。
# 后果是换机器/换浏览器/清理站点数据就全丢，也没法把一次演练的记录交给别人看。
# 这里补一个最小落盘：每个登录用户一个 JSON 文件，原子写入，容量与条数都有上限。
#
# 目录放在 cache/records：它是运行时缓存，不是随代码分发的数据，
# 与 data/（训练/评估产物）分开，便于整体清理与加入 .gitignore。
# data/records 是旧位置，读取时自动迁移过来，避免升级后丢历史记录。
RECORDS_DIR = os.path.join(BASE_DIR, 'cache', 'records')
RECORDS_LEGACY_DIR = os.path.join(BASE_DIR, 'data', 'records')
RECORDS_MAX_BYTES = _env_int('DLIDS_RECORDS_MAX_BYTES', 16 * 1024 * 1024, 4096)


def _records_path(session_user: str) -> str:
    """把用户名映射成安全的文件名，避免路径穿越。"""
    safe = hashlib.sha256(str(session_user or 'anonymous').encode('utf-8')).hexdigest()[:32]
    return os.path.join(RECORDS_DIR, f'records_{safe}.json')


@app.get('/api/records')
def get_records(session_user: str = Depends(require_session)):
    """读取当前用户的检测记录；没有落盘文件时返回空结构而不是 404。"""
    path = _records_path(session_user)
    empty = {'live': [], 'upload': [], 'updated_at': None, 'storage': 'server'}
    if not os.path.exists(path):
        # 旧位置兼容：记录从 data/records 迁到 cache/records，第一次读取时搬过来，
        # 否则升级后用户会看到「记录全没了」（文件还在，只是换了目录）。
        legacy = os.path.join(RECORDS_LEGACY_DIR, os.path.basename(path))
        if not os.path.exists(legacy):
            return empty
        try:
            os.makedirs(RECORDS_DIR, exist_ok=True)
            os.replace(legacy, path)
            print(f'migrated records {legacy} -> {path}')
        except Exception as e:      # noqa: BLE001 - 迁移失败不该让页面 500
            print(f'migrate records failed for {session_user}: {e}')
            try:
                with open(legacy, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data['storage'] = 'server'
                    return data
            except Exception as e2:  # noqa: BLE001
                print(f'read legacy records failed for {session_user}: {e2}')
            return empty
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:      # noqa: BLE001 - 文件损坏不该让页面 500
        print(f'read records failed for {session_user}: {e}')
        return {**empty, 'error': 'stored records are unreadable'}
    if not isinstance(data, dict):
        return empty
    data['storage'] = 'server'
    return data


@app.post('/api/records')
async def save_records(request: Request, session_user: str = Depends(require_session)):
    """覆盖写入当前用户的检测记录（前端持久化后调用）。"""
    raw = await request.body()
    if len(raw) > RECORDS_MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={'error': 'records payload too large',
                     'max_bytes': RECORDS_MAX_BYTES, 'received_bytes': len(raw)},
        )
    try:
        payload = json.loads(raw.decode('utf-8'))
    except Exception:
        return JSONResponse(status_code=400, content={'error': 'body must be JSON'})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={'error': 'body must be a JSON object'})
    body = {
        'live': payload.get('live') if isinstance(payload.get('live'), list) else [],
        'upload': payload.get('upload') if isinstance(payload.get('upload'), list) else [],
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'storage': 'server',
    }
    path = _records_path(session_user)
    try:
        os.makedirs(RECORDS_DIR, exist_ok=True)
        tmp = f'{path}.tmp.{int(time.time() * 1000)}'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(body, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:      # noqa: BLE001
        print(f'save records failed for {session_user}: {e}')
        return JSONResponse(status_code=500, content={'error': 'failed to persist records'})
    return {'ok': True, 'storage': 'server', 'updated_at': body['updated_at'],
            'live': len(body['live']), 'upload': len(body['upload']),
            'bytes': os.path.getsize(path)}


@app.delete('/api/records')
def delete_records(session_user: str = Depends(require_session)):
    """清空当前用户的服务端检测记录。"""
    path = _records_path(session_user)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:      # noqa: BLE001
        print(f'delete records failed for {session_user}: {e}')
        return JSONResponse(status_code=500, content={'error': 'failed to delete records'})
    return {'ok': True, 'storage': 'server', 'deleted': True}


@app.get("/models")
def models_page(request: Request):
    return _render_page(request, "models")


@app.get("/flow")
def flow_page(request: Request):
    return _render_page(request, "flow")

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
    is_cascade = requested_model_type == CASCADE_MODEL_TYPE
    if run_ensemble or is_cascade:
        if not _get_loaded_detection_models():
            error_msg = loading_error if loading_error else "No detection model is loaded. Please train the models first."
            return JSONResponse(status_code=503, content={"error": error_msg})
    elif models.get(requested_model_type) is None:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown or unavailable model_type: {requested_model_type}",
            "available_model_types": [k for k, v in models.items() if v is not None] + [ENSEMBLE_MODEL_TYPE, CASCADE_MODEL_TYPE],
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
        # 空闲超时：默认不按时间切分（保持已训练模型的历史口径）。设 DLIDS_FLOW_IDLE_TIMEOUT
        # 后二分类与会按同一超时切流，两张卡片的流基数才严格对齐。
        extractor = FeatureExtractor(idle_timeout=FLOW_IDLE_TIMEOUT_SECONDS)
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
        # 分布外协议门控用的统计（见 INDUSTRIAL_PROTO_PORTS 处的说明）
        industrial_sessions = 0
        industrial_hits = {}
        # 「训练集外基础设施协议」单独计数：它们说明模型没见过，但**不**代表这是工业域。
        foreign_sessions = 0
        foreign_hits = {}
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

            # 分布外协议门控：按端口统计多少会话属于模型未训练过的工业/基础设施协议
            try:
                _sport, _dport = int(session_key[1]), int(session_key[3])
            except Exception:
                _sport, _dport = 0, 0
            _hit = INDUSTRIAL_PROTOCOL_PORTS.get(_sport) or INDUSTRIAL_PROTOCOL_PORTS.get(_dport)
            if _hit:
                industrial_sessions += 1
                industrial_hits[_hit] = industrial_hits.get(_hit, 0) + 1
            else:
                _fhit = FOREIGN_PROTOCOL_PORTS.get(_sport) or FOREIGN_PROTOCOL_PORTS.get(_dport)
                if _fhit:
                    foreign_sessions += 1
                    foreign_hits[_fhit] = foreign_hits.get(_fhit, 0) + 1
            
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

            # 级联：先用主力单模型探一次批次平均置信度，再决定要不要升级到 6 模型集成。
            # 目的不是提高精度，而是在精度基本不变的前提下省掉大部分集成开销——
            # 高置信批次（绝大多数）走单模型，只有模型自己拿不准的批次才付集成成本。
            cascade_escalated = False
            cascade_avg_confidence = None
            cascade_avg_entropy = None
            cascade_avg_dist = None
            probe_type = None
            probe_probs = None
            if is_cascade:
                probe_type = DEFAULT_MODEL_TYPE if models.get(DEFAULT_MODEL_TYPE) is not None else None
                if probe_type is None:
                    loaded_types = _get_loaded_detection_models()
                    probe_type = loaded_types[0] if loaded_types else None
                probe_model = models.get(probe_type) if probe_type else None
                if probe_model is None:
                    # 一个可用的单模型都没有，只能直接走集成
                    run_ensemble = True
                else:
                    try:
                        probe_probs = _predict_model_probabilities(probe_model, batch_tensor)
                        probe_conf, _ = torch.max(probe_probs, 1)
                        cascade_avg_confidence = float(probe_conf.mean().item())
                        # 归一化预测熵 H(p)/log(C)：max(softmax) 只看"最像的那个有多像"，
                        # 熵还看"其余的类是不是也差不多像"。分布外样本常常所有类都给高分，
                        # max 很高但熵也高——这种情况下 max 判据会漏，熵能抓住。
                        num_cls = max(int(probe_probs.size(1)), 2)
                        logp = torch.log(probe_probs.clamp_min(1e-12))
                        entropy = -(probe_probs * logp).sum(dim=1) / math.log(num_cls)
                        cascade_avg_entropy = float(entropy.mean().item())
                        # 对角马氏距离：唯一能识别「没见过这种流量」的判据。
                        # softmax 对分布外样本依然给 99% 置信度，距离不会。
                        cascade_avg_dist = _batch_mahalanobis(batch_tensor)
                    except Exception as e:
                        print("cascade probe failed:", e)
                        cascade_avg_confidence = None
                        cascade_avg_entropy = None
                    run_ensemble = (
                        cascade_avg_confidence is None
                        or cascade_avg_confidence < CASCADE_CONFIDENCE_THRESHOLD
                        or (cascade_avg_entropy is not None
                            and cascade_avg_entropy > CASCADE_ENTROPY_THRESHOLD)
                        or (cascade_avg_dist is not None and _DIST_P99 is not None
                            and cascade_avg_dist > _DIST_P99)
                    )
                cascade_escalated = bool(run_ensemble)
                if not run_ensemble and probe_type:
                    # 不升级：这次判定就由探针模型给出
                    requested_model_type = probe_type

            effective_model_type = ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type
            ensemble_result = None

            if run_ensemble:
                try:
                    # 域权重（需求 A）：不再二选一，改成连续插值。
                    # 原来只看「工业端口占比是否过阈值」，端口伪装（工控跑 8080、
                    # 或隧道进 443）会让判定直接落到 default，模型没见过这种流量
                    # 这件事完全表达不出来。现在把与端口无关的马氏距离一起折算成
                    # 0~1 的域外分数，在 default 与 ics 两套权重之间线性插值：
                    # 分数 0 → 就是 default 权重（等权），所以最坏也不比现在差。
                    _port_ratio = (industrial_sessions / total_sessions) if total_sessions > 0 else 0.0
                    # 距离信号：cascade 探针已经算过就直接用；直接走集成（ensemble_vote）
                    # 时探针没跑，这里补算一次 —— 只需一次 reshape，不付前向成本。
                    _batch_dist = cascade_avg_dist
                    if _batch_dist is None:
                        _batch_dist = _batch_mahalanobis(batch_tensor)
                    _dist_ratio = None
                    if _batch_dist is not None and _DIST_P99:
                        _dist_ratio = float(_batch_dist) / float(_DIST_P99)
                    _out_score, _out_signals = _domain_out_score(
                        port_ratio=_port_ratio, dist_ratio=_dist_ratio)
                    _default_weights = _weights_for_domain('default')
                    _ics_weights = _weights_for_domain('ics')
                    _ensemble_weights = (
                        _blend_weights(_default_weights, _ics_weights, _out_score)
                        if (_default_weights and _ics_weights) else _default_weights)
                    ensemble_result = _vote_with_ensemble(
                        batch_tensor, weights=_ensemble_weights)
                    if isinstance(ensemble_result, dict):
                        # 事后核对必须能查到：这次是按哪个域外分数、用哪套权重投的
                        ensemble_result['domain_score'] = _out_score
                        ensemble_result['domain_signals'] = _out_signals
                        ensemble_result['weights_used'] = (
                            {k: round(float(v), 4) for k, v in _ensemble_weights.items()}
                            if _ensemble_weights else None)
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
                    # 级联在"不升级"分支已经用同一个模型算过一次，直接复用探针结果，
                    # 否则会白跑第二遍推理（实测 0.012s 变成 0.36s，慢 30 倍）。
                    if probe_probs is not None and probe_type == effective_model_type:
                        probs = probe_probs
                    else:
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

        # 分布外判定：两条独立证据，任一成立就降级可靠性。
        #  ① 协议族：模型从未在训练中见过这些协议（工业/基础设施），任何 softmax 结论都不可信
        #  ② 像素分布：会话图的稀疏程度远离训练基线 —— 补齐「工业会话只占 6%、协议占比够不到
        #     0.30」那批文件的漏网（实测这类输入 softmax 仍给 0.999+ 却判错）
        industrial_ratio = (industrial_sessions / total_sessions) if total_sessions > 0 else 0.0
        industrial_top = sorted(industrial_hits.items(), key=lambda kv: -kv[1])
        foreign_ratio = (foreign_sessions / total_sessions) if total_sessions > 0 else 0.0
        foreign_top = sorted(foreign_hits.items(), key=lambda kv: -kv[1])
        # 像素统计只作为信息展示，不参与判定（原因见 PIXEL_BASELINE_MEAN 处注释）。
        _pixel_ood_unused, pixel_info = _pixel_ood_stats(batch_tensor)
        out_of_domain = bool(industrial_ratio >= OOD_INDUSTRIAL_RATIO_THRESHOLD)

        def _ood_reason(benign_verdict=False):
            """把命中的分布外证据拼成人话；两条都命中时都列出来，便于判断该往哪查。"""
            _parts = []
            if industrial_ratio >= OOD_INDUSTRIAL_RATIO_THRESHOLD:
                _parts.append(
                    f"{industrial_ratio * 100:.1f}% 的会话属于模型未训练过的工业控制协议"
                    f"（{'、'.join('%s×%d' % (n, c) for n, c in industrial_top[:3])}）")
            _tail = ("「良性」结论同样只是模型对该协议族无分辨力，不代表流量安全。"
                     if benign_verdict else "模型的判定不可采信。")
            return "；".join(_parts) + "，训练集 USTC-TFC2016 不含此类流量，" + _tail
        # 体积条件与门控置信度同时满足，才考虑判定整个 PCAP 为恶意
        would_flag_malicious = bool(flagged_by_volume and (avg_malware_conf >= pct_conf))

        if would_flag_malicious and out_of_domain:
            # 关键修正：把「模型没见过」与「检测到恶意」区分开。
            # 之前这类流量会以 0.9+ 的置信度被报成 Malicious Traffic Detected，
            # 实测 37/48 的工控良性 pcap 都栽在这里。
            result_status = "Out-of-Distribution Traffic"
            chosen_scores = flagged_malware_confs if flagged_malware_confs else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = malicious_session_image_b64 or first_session_image_b64
            display_hist = malicious_session_hist or first_session_hist
            reliability = "out_of_domain"
            reliability_reason = _ood_reason()
        elif would_flag_malicious:
            result_status = "Malicious Traffic Detected"
            chosen_scores = flagged_malware_confs if flagged_malware_confs else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = malicious_session_image_b64
            display_hist = malicious_session_hist
            if total_sessions < SMALL_SAMPLE_SESSIONS:
                reliability = "low"
                reliability_reason = (
                    f"仅解析出 {total_sessions} 个会话，样本量低于 {SMALL_SAMPLE_SESSIONS}，"
                    f"建议抓取更长流量复核。"
                )
            else:
                reliability = "normal"
                reliability_reason = ""
        else:
            result_status = "Benign Traffic"
            chosen_scores = benign_scores if benign_scores else predicted_scores
            final_conf = float(np.mean(chosen_scores)) if chosen_scores else 0.0
            display_image = first_session_image_b64  # 没发现恶意流量时展示第一个会话的图片
            display_hist = first_session_hist
            if out_of_domain:
                reliability = "out_of_domain"
                reliability_reason = _ood_reason(benign_verdict=True)
            else:
                reliability = "normal"
                reliability_reason = ""
        
        # ---- 六分类威胁类型分布（pcap 链路上唯一的类别来源）----
        # 28x28 字节图那条链路只有「正常/恶意」两档，说不出攻击类型；要让
        # 「威胁类型分布」卡片名副其实，必须把这份 pcap 再解析成 78 维流统计，
        # 喂进 CSV 链路那个六分类模型。同一个文件读两遍是刻意取舍：会话图那条
        # 链路只保留前 28 包 × 前 28 字节，信息量不足以算流特征；而把那套截断逻辑
        # 改成保留全量包，会动到训练与评估脚本共用的 FeatureExtractor。
        # 六分类模型没加载时 threat_summary 保持 None，前端据此隐藏区块而不是报错。
        threat_summary = None
        threat_meta = None
        threat_flows = []
        # 缺陷 D-12：pcap 链路此前不回 tier_summary，前端在这一页看不到
        # 「自动处置 / 待确认 / 人工复核」三档分布，只有 CSV 页有。
        threat_tier_summary = None
        try:
            if flow_multiclass is not None and flow_multiclass_meta is not None:
                from src.preprocessing.flow_features import (
                    pcap_to_flow_features, features_to_matrix,
                )
                _ff = pcap_to_flow_features(temp_file, max_flows=FLOW_THREAT_MAX_FLOWS)
                if _ff:
                    _flow_list, _Xf = features_to_matrix(_ff)
                    threat_summary, _t_detail = _classify_flow_matrix(_Xf)
                    if threat_summary is not None:
                        threat_tier_summary = _tier_summary(_t_detail['tier'], _t_detail['conf'])
                        # 逐流明细：卡片上点开某个分类看的就是它。
                        # 必须从 _flow_list 取五元组——_classify_flow_matrix 只拿到矩阵，
                        # 它并不知道每一行原本是哪条连接。
                        # two_class_map 用同一次请求的二分类明细 join（flow_details
                        # 的 protocol 是 'TCP' 字符串，_conn_sort_key 内部已归一到数字），
                        # 前端据此压制「二分类说正常、六分类说攻击」的矛盾行。
                        threat_flows = _threat_flow_rows(
                            _flow_list, _t_detail, limit=0,
                            two_class_map=_two_class_map_from(flow_details))
                        # 分歧统计基于全量，展示只取前 N 条（缺陷 B2/B3、L2）：
                        # 过去 flow_rows 截断到 200 却说成全部 4000 条的结论，卡片数字
                        # 与「点开能看到的条数」对不上。
                        _consistency = _threat_consistency(threat_flows, threat_summary)
                        threat_meta = {
                            'flows': int(len(_Xf)),
                            'flows_total': int(len(_flow_list)),
                            'unknown_count': int(_t_detail['is_unknown'].sum()),
                            'ood_threshold': (round(float(_t_detail['ood_threshold']), 2)
                                              if _t_detail['ood_threshold'] is not None else None),
                            'ood_enabled': bool(_t_detail.get('ood_enabled', True)),
                            'ood_disabled_reason': _t_detail.get('ood_disabled_reason'),
                            'class_names': list(flow_multiclass_meta['class_names']),
                            # 明细条数 vs 总流数：前端据此在弹窗里说明「只列出前 N 条」
                            'flow_rows': min(len(threat_flows), THREAT_FLOW_ROWS_LIMIT),
                            'flows_shown': min(len(threat_flows), THREAT_FLOW_ROWS_LIMIT),
                            'flows_classified': len(threat_flows),
                            # 两套模型的分歧量：前端必须据此写清「不同模型、不同口径」
                            'consistency': _consistency,
                            # 两条链路的"流"定义不同，基数不该被当成同一个数（缺陷 D-23）
                            'stream_definition': _stream_definition(),
                            # 与 /analyze-flow 的 tier_summary 同一实现（缺陷 D-12）
                            'tier_summary': threat_tier_summary,
                        }
                        # 上传检测（pcap）也要进复核队列：过去只有实时抓包链路入队，
                        # 「上传检测出需要复核的流量」在复核中心里根本看不到。
                        try:
                            _sync_review_queue(
                                _flow_list, _t_detail, source='upload',
                                meta={'file': str(file.filename or ''), 'channel': 'pcap'},
                                two_class_map=_two_class_map_from(flow_details))
                        except Exception as _review_err:  # noqa: BLE001
                            print('pcap review enqueue failed:', _review_err)
                        threat_flows = threat_flows[:THREAT_FLOW_ROWS_LIMIT]

        except Exception as _threat_err:
            print('pcap threat classification failed:', _threat_err)

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
            # 判定可靠性：normal / low（样本不足）/ out_of_domain（协议不在训练分布内）
            "reliability": reliability,
            "reliability_reason": reliability_reason,
            "distribution": {
                "in_domain": not out_of_domain,
                "industrial_sessions": industrial_sessions,
                "industrial_ratio": round(industrial_ratio * 100, 2),
                "industrial_protocols": [{"protocol": n, "sessions": c} for n, c in industrial_top],
                # 「模型没见过但不属于工控」的协议单列，免得被当成工业域证据
                "foreign_sessions": foreign_sessions,
                "foreign_ratio": round(foreign_ratio * 100, 2),
                "foreign_protocols": [{"protocol": n, "sessions": c} for n, c in foreign_top],
                "threshold": OOD_INDUSTRIAL_RATIO_THRESHOLD,
                # 像素统计只作信息展示，不参与判定 —— 基线 0.14296 / 34.88% 是训练集
                # 全局值，而域内单文件跨度达 122 倍，任何全局阈值都会误伤
                # （详见 PIXEL_BASELINE_MEAN 处注释）。前端可用它解释「这批流量长什么样」。
                "pixel_stats": pixel_info,
            },
            "details": f"Analyzed {total_sessions} sessions, {malware_count} flagged as malicious."
                       + (f" {reliability_reason}" if reliability_reason else ""),
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
            # 需求 A/C 的取证出口：这次判定按哪个域外分数、插值出哪套权重投的，
            # 以及每一票有多僵、分歧度把权重改了多少。只在集成路径下有值。
            "ensemble_weighting": ({
                "weighting": ensemble_result.get('weighting'),
                "domain_score": ensemble_result.get('domain_score'),
                "domain_signals": ensemble_result.get('domain_signals'),
                # 实际算出来的批次马氏距离与它的标定阈值，便于核对「为什么给/不给分」
                "batch_distance": (round(_batch_dist, 4) if _batch_dist is not None else None),
                "distance_threshold": (round(_DIST_P99, 4) if _DIST_P99 is not None else None),
                "weights_used": ensemble_result.get('weights_used'),
                "contest_strength": ensemble_result.get('contest_strength'),
                "sessions": [
                    {"contest": vote.get('vote_contest'), "adjust": vote.get('contest_adjust')}
                    for vote in (ensemble_result.get('session_votes') or [])[:20]
                ],
            } if (run_ensemble and ensemble_result) else None),
            # 六分类威胁类型分布：结构与 /analyze-flow 的 class_summary 完全一致
            # （{类名: {count, avg_confidence, unknown_count}}，外加一项「未知攻击」），
            # 前端用同一条渲染路径，两个页面才不会各显示一套口径。
            "threat_summary": threat_summary,
            "threat_meta": threat_meta,
            # 三档处置分布（缺陷 D-12）：与 CSV 链路的 tier_summary 同一实现
            "threat_tier_summary": threat_tier_summary,
            # 逐流明细：卡片上每个分类点开看的就是它。只放轻量字段
            # （五元组 + 分类 + 置信度 + OOD 距离 + 包数），不带 78 维特征。
            "threat_flows": threat_flows,
            # 级联轨迹：前端据此提示「单模型直接判定」还是「已升级 6 模型复核」
            "cascade": {
                "requested": True,
                "escalated": cascade_escalated,
                "probe_model": probe_type,
                # 缺陷 M2：请求 cascade 而未触发升级时，顶层 model_type 会是探针模型名
                # （例如 pure_bilstm），前端把两个字段并排显示就成了「请求 cascade、
                # 返回 pure_bilstm」的假矛盾。这里把「谁在真正判定」写清楚，
                # 让界面能显示「cascade 未升级 · 由 pure_bilstm 直接判定」。
                "executed_by": ("ensemble_vote" if cascade_escalated else probe_type),
                "execution_note": (
                    "已升级：6 模型投票复核" if cascade_escalated
                    else f"未升级：批次置信度满足阈值，由 {probe_type} 直接判定"),
                "avg_confidence": (round(cascade_avg_confidence, 4)
                                   if cascade_avg_confidence is not None else None),
                "avg_entropy": (round(cascade_avg_entropy, 4)
                                if cascade_avg_entropy is not None else None),
                "avg_distance": (round(cascade_avg_dist, 4)
                                 if cascade_avg_dist is not None else None),
                "confidence_threshold": CASCADE_CONFIDENCE_THRESHOLD,
                "entropy_threshold": CASCADE_ENTROPY_THRESHOLD,
                "distance_threshold": (round(_DIST_P99, 4) if _DIST_P99 is not None else None),
                # 缺陷 P0-2：dist_stats 退化时距离支路恒不触发，三条件实际只剩两条。
                # 不显式说明的话，读者会以为「没升级 = 三项都通过」。
                "distance_gate_active": bool(
                    _DIST_P99 is not None and not _DIST_STATS_STATUS.get('degenerate')),
                "distance_gate_reason": (
                    None if (_DIST_P99 is not None and not _DIST_STATS_STATUS.get('degenerate'))
                    else _DIST_STATS_STATUS.get('reason') or 'dist_stats unavailable'),
            } if is_cascade else None,

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
    # 本次抓包见过的不同五元组数。前端 live 那一栏读的是 stats.session_count，
    # 缺了它 Number.isFinite(undefined) 为假，页面上那几格会一直是 0——
    # 抓包明明在跑却全是 0，看起来就像功能坏了。
    'session_count': 0,
    # 当前跟踪表里还有多少条会话（受 LIVE_MAX_TRACKED_SESSIONS 限制）。
    # 它和 session_count 不是一回事：session_count 是本次抓包**累计见过**的五元组数，
    # sessions_tracked 是此刻还留在内存里的条数（缺陷 D-29）。
    'sessions_tracked': 0,
    'session_count_approximate': False,
    'last_update': None,
    'last_error': None
}
recent_flows = []  # 最近检测结果的环形缓冲区
# 最近流量环形缓冲条数：前端 /api/live/stats 每 1.5 秒轮询一次
# （见 web/static/js/main.js 的 LIVE_POLL_INTERVAL_MS = 1500），
# 50 条足以覆盖一两个轮询周期；调整这个值时要同步考虑前端轮询间隔。
MAX_RECENT = 500
# 上面这个值同时决定了两个数字能不能对上：前端「实时抓包记录」表只拿得到 recent_flows
# 里那批流，而右栏「威胁类型分布」卡片统计的是 LiveFlowTracker 里全部流（按双向五元组
# 每条连接一条）。原来只有 50 条时，一次抓包四十多个连接、最近 50 条事件却只覆盖到二十来个，
# 于是卡片说 41 条、表格却只有 20 行。放到 500 后两者在正常抓包规模下同口径。
live_debug_logs = []
MAX_DEBUG_LOGS = 300

# 流级聚合参数：过去每抓到一个包就单独跑一次模型推理，一次浏览网页就能产生上万次推理；
# 现在按五元组把包攒成"流"，攒够包数或字节数才推理一次。
LIVE_SESSION_MAX_BYTES = _env_int('DLIDS_LIVE_SESSION_MAX_BYTES', 4096, 64)
LIVE_SESSION_MIN_PACKETS = _env_int('DLIDS_LIVE_SESSION_MIN_PACKETS', 8, 1)
LIVE_MAX_TRACKED_SESSIONS = _env_int('DLIDS_LIVE_MAX_SESSIONS', 256, 1)
# 缺陷 D-29：本次抓包「见过的五元组」集合。原先 session_count 是在会话被淘汰后
# 重新建表时又 +1，长抓包一路把数字推到远超真实会话数；改用集合去重后，
# 淘汰只会让 sessions_tracked 变小，累计值只增不重复。
# 集合本身也要有上限，否则长抓包会无限涨内存；超过上界后退化为「只增计数」，
# 并置 session_count_approximate=True 让界面说明这个数是估算的。
LIVE_SEEN_SESSION_CAP = _env_int('DLIDS_LIVE_SEEN_SESSION_CAP', 20000, 100)
live_seen_sessions: set = set()
live_seen_overflow = 0
LIVE_SESSION_IDLE_SECONDS = _env_float('DLIDS_LIVE_SESSION_IDLE', 10.0, 0.5, 600.0)
# 单条会话缓冲最多保留的包数。process_session 只取前 28 行（max_packets），
# 多留的包既不影响图像（流特征由 live_flow_tracker 独立逐包累加），也只会白占内存。
LIVE_SESSION_PACKET_CAP = _env_int('DLIDS_LIVE_SESSION_PACKET_CAP', 56, 1)

# 单调递增的流 id：前端用它做增量拉取（since_id），避免每轮重传全部缩略图
latest_flow_id = 0
live_session_buffers: dict = {}   # 五元组 -> {'pkts': list[bytes], 'bytes': int, 'packets': int, 'updated': float}

# 实时抓包的 78 维流特征表。「威胁类型分布」卡片要出六分类，就必须让实时链路也算出
# 与训练 CSV 同口径的 78 维，否则这一页只会有二分类的 is_malicious，卡片永远是空的。
# 抓包线程每收一个包喂一次，接口按 version 判断要不要重算，避免 1.5 秒一次的无谓推理。
live_flow_tracker = None          # 延迟到开始抓包时创建，import 阶段不碰 numpy/模型
# 'flows' 存逐流明细（用户点开某个分类时看的就是它）。它跟着 version 一起缓存，
# 否则前端每次轮询都会重新构造一遍几千条字典。
live_threat_cache = {'version': -1, 'model_sig': None, 'recent_sig': None,
                     'summary': None, 'meta': None, 'flows': []}
# 逐流明细的下发上限：卡片上点「DDoS-HOIC」只是想看是哪几条，
# 不是要把 4000 条流一次性拉到浏览器里。超出部分只报数量。
THREAT_FLOW_ROWS_LIMIT = _env_int('DLIDS_THREAT_FLOW_ROWS', 200, 10)
# 复核队列：六分类判成 review/manual 档、或被标为「未知攻击」的流自动挂进来。
# 用 dict 按流五元组去重 —— 同一条流每来一个包都会重算一次，不去重几分钟就能堆出几千条。
live_review_queue = {}
LIVE_REVIEW_QUEUE_LIMIT = 200

# 特征提取器与张量变换只创建一次（原来是每个包都新建一个 FeatureExtractor）
live_extractor = None
live_transform = transforms.Compose([transforms.ToTensor()])


def _get_live_extractor():
    global live_extractor
    if live_extractor is None:
        live_extractor = FeatureExtractor()
    return live_extractor


def _mask_live_packet(buf):
    """实时链路单包的头部掩码 + 包边界保留。

    离线链路在 ``FeatureExtractor.pcap_to_sessions`` 里对每个包调
    ``_mask_packet_headers``（掩掉前 12 字节 MAC 与 IPv4 头内 +12 起的 8 字节地址），
    再把每个包裁成前 28 字节存进会话。实时链路原先缺这两步：把整条会话的字节直接
    ``bytearray.extend`` 拼起来，交给 ``process_session`` 后按 28 字节一刀切——
    既没有掩码头，切片边界也和真实包边界无关。结果是同一条流在在线/离线两条入口上
    得到的 28x28 字节图不同（离线是按包对齐 + 掩码，在线是连续字节流硬切），
    模型看到的输入分布不一致，在线指标与离线指标因此不可直接比较。

    这里补上掩码；``packet_handler`` 改为按真实包累积成列表，与离线复用同一个
    ``process_session`` 接口。
    """
    data = bytes(buf or b"")
    if not data:
        return data
    try:
        offset = 14
        if len(data) >= 18 and ((data[12] << 8) | data[13]) == 0x8100:
            offset = 18
        return bytes(_get_live_extractor()._mask_packet_headers(bytearray(data), offset))
    except Exception:
        # 掩码失败不能丢包：宁可带着原始头部进去，也不要让整条流消失
        return data


def _classify_flow_matrix(X_raw, confidence_high=None, confidence_low=None):
    """把 (N, 78) 原始流特征跑成六分类汇总：标准化 -> 推理 -> 温度校准 -> OOD -> 汇总。

    返回 ``(class_summary, detail)``；模型没加载或没有数据行时返回 ``(None, None)``。
    detail 里带逐行的 pred/conf/is_unknown/tier，调用方要明细时自己取。

    **两条链路共用这一段**：CSV 上传（/analyze-flow）与 pcap 上传（/analyze）都必须
    走同一个函数，否则同一个模型会在两个入口给出两套口径——六分类的阈值、温度、
    OOD 判据任何一处不同，前端那张「威胁类型分布」卡片就会在两页之间对不上。

    缺陷 D-07：档位阈值不再是本函数的私有默认值（旧值 0.80/0.50 与 /analyze-flow
    表单的 0.90 并存），缺省即取全站唯一配置源 FLOW_CONFIDENCE_HIGH/LOW。
    """
    meta = flow_multiclass_meta
    if flow_multiclass is None or meta is None:
        return None, None
    if X_raw is None or len(X_raw) == 0:
        return None, None

    if confidence_high is None:
        confidence_high = FLOW_CONFIDENCE_HIGH
    if confidence_low is None:
        confidence_low = FLOW_CONFIDENCE_LOW
    if confidence_low > confidence_high:
        confidence_high, confidence_low = confidence_low, confidence_high

    Xn = ((X_raw - meta['mean'].reshape(1, -1)) / meta['std'].reshape(1, -1)).astype(np.float32)
    class_names = meta['class_names']
    temperature = meta['temperature'] or 1.0

    # 模型在模块级加载时留在 CPU（启动阶段不碰 CUDA），首次推理时才搬到 device。
    # 不能只在加载时搬一次：device 常量在本文件里是在模型加载之后才解析出来的。
    flow_multiclass.to(device)
    with torch.no_grad():
        logits = flow_multiclass(torch.from_numpy(Xn).to(device)).cpu().numpy()

    # 温度校准：argmax 不变，但让输出的概率数字接近真实正确率
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    prob = exp / exp.sum(axis=1, keepdims=True)

    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    tiers = np.where(conf >= confidence_high, 'auto',
                     np.where(conf >= confidence_low, 'review', 'manual'))


    # 未知攻击判据：到最近类中心的欧氏距离。
    # 阈值直接取自产物 checkpoints/flow_class_centers.npz 的 thresholds[3]（p99.9），
    # 不在这里手写数字 —— 缺陷 D-17：旧注释写的「p99≈22、p99.9≈67」与产物实测
    # （8.89 / 31.53）差两倍以上，注释里的数字会随重训漂移，只能引用产物字段。
    # 超过阈值说明这行更像「不属于任何已学过的类」，而不是某个类的正常成员。
    ood_distance = None
    is_unknown = np.zeros(len(Xn), dtype=bool)
    ood_threshold = None
    # 缺陷 D-16：维度不匹配时 OOD 实际未启用，但 meta 仍会回传阈值，前端据此误判
    # 「未知攻击 = 0 条」。用一个显式开关把「没启用」与「算出来是 0」区分开。
    ood_enabled = False
    ood_disabled_reason = None
    _space_ok = (flow_centers or {}).get('space_matches_active')
    if flow_centers is None:
        ood_disabled_reason = 'class_centers_missing'
    elif flow_centers['centers'].shape[1] != Xn.shape[1]:
        ood_disabled_reason = 'dimension_mismatch'
    elif _space_ok is False:
        # 缺陷 D-01：中心与当前标准化不在同一坐标系，距离与阈值都不可用。
        ood_disabled_reason = 'normalization_space_mismatch'
    if ood_disabled_reason is None:
        ood_enabled = True
        _centers = flow_centers['centers']
        _d = np.linalg.norm(Xn[:, None, :] - _centers[None, :, :], axis=2)
        ood_distance = _d.min(axis=1)
        # 取 p99.9 而不是 p99：Benign 类内部极度异质（几十种正常业务叠在一起），
        # p99 阈值会把接近两成的正常流量判成「未知」——那个误报量没人愿意接手。
        ood_threshold = float(flow_centers['thresholds'][3])
        is_unknown = ood_distance > ood_threshold

    # 判为未知攻击的一律压到人工复核：模型对它的类归属本就不可信，
    # 再用置信度决定处置等级没有意义（高置信的错误分类最危险）。
    tiers = np.where(is_unknown, 'manual', tiers)

    class_summary = {}
    for i, name in enumerate(class_names):
        mask = pred == i
        class_summary[name] = {
            'count': int(mask.sum()),
            'avg_confidence': round(float(conf[mask].mean()), 4) if mask.any() else None,
            'unknown_count': int((mask & is_unknown).sum()),
        }
    if is_unknown.any():
        # 单列一项「未知攻击」：这些行模型也给了六类之一的预测，但离该类中心太远，
        # 混在普通预测里会被忽略，必须单独占一行。
        class_summary['未知攻击'] = {
            'count': int(is_unknown.sum()),
            'avg_confidence': round(float(conf[is_unknown].mean()), 4),
            'unknown_count': int(is_unknown.sum()),
        }

    detail = {
        'pred': pred,
        'conf': conf,
        'tier': tiers,
        'is_unknown': is_unknown,
        'ood_distance': ood_distance,
        'ood_threshold': ood_threshold,
        'ood_enabled': ood_enabled,
        'ood_disabled_reason': ood_disabled_reason,
        'confidence_high': float(confidence_high),
        'confidence_low': float(confidence_low),
        'class_names': class_names,
        'prob': prob,
    }
    return class_summary, detail


def _pixel_ood_stats(batch_tensor):
    """统计输入会话图的像素材质，**仅作信息展示，不参与分布外判定**。

    返回 ``(False, info)``；``is_ood`` 恒为 False，保留该字段只为兼容前端既有读法。

    为什么不再做判定（实测数据说话）：USTC-TFC2016 训练集 1,957,416 个会话的全局
    像素均值 0.14296、非零占比 34.88%，但**单个文件**的均值跨度就有 122 倍
    （Benign/BitTorrent 0.0040 → Benign/WorldOfWarcraft 0.4899，非零 1.38%~95.78%）。
    域内方差远大于域间差异：任何全局阈值都会把 USTC 自己的良性文件
    （Facetime 0.0040 / Gmail 0.0116 / Skype 0.0078）判成分布外；而 ICS 良性实测
    0.0241 / 7.55% 恰好落在 USTC 良性范围正中，同样分不开。
    分布外判定因此只依赖协议族比例（OOD_INDUSTRIAL_RATIO_THRESHOLD）。
    """
    try:
        arr = batch_tensor.detach().float().cpu().numpy()
        if arr.size == 0:
            return False, {}
        mean = float(arr.mean())
        nonzero = float((arr > 0).mean())
    except Exception as e:      # noqa: BLE001 - 统计失败不该让上传接口 500
        print('pixel stats failed:', e)
        return False, {}

    mean_ratio = (mean / PIXEL_BASELINE_MEAN) if PIXEL_BASELINE_MEAN > 0 else 1.0
    nonzero_ratio = (nonzero / PIXEL_BASELINE_NONZERO) if PIXEL_BASELINE_NONZERO > 0 else 1.0
    return False, {
        'pixel_mean': round(mean, 4),
        'pixel_nonzero_ratio': round(nonzero * 100, 2),
        'baseline_mean': PIXEL_BASELINE_MEAN,
        'baseline_nonzero_ratio': round(PIXEL_BASELINE_NONZERO * 100, 2),
        'mean_vs_baseline': round(mean_ratio, 3),
        'nonzero_vs_baseline': round(nonzero_ratio, 3),
        'is_ood': False,
    }


def _two_class_map_from(rows):
    """从包级二分类明细里建 {双向连接键: is_malicious}。

    为什么要这个 map：卡片（流级六分类）和下方记录表（包级二分类）是两套独立模型，
    同一条连接经常一个说 Benign、一个说 DoS-GoldenEye。用户在界面上看到「记录表全绿、
    卡片里有攻击类」会直接认为系统自相矛盾。

    取不到就返回空 dict —— 调用方据此把 two_class_malicious 留 None，
    前端只在明确 False 时才压制（拿不到结论不等于结论是「正常」）。

    字段名两套都认：``flow_details``（上传链路）用 ``src_ip/src_port/protocol``，
    ``recent_flows``（实时链路）用 ``src/sport/proto``。
    """
    m = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        try:
            src = r.get('src') if r.get('src') is not None else r.get('src_ip')
            dst = r.get('dst') if r.get('dst') is not None else r.get('dst_ip')
            sport = r.get('sport') if r.get('sport') is not None else r.get('src_port')
            dport = r.get('dport') if r.get('dport') is not None else r.get('dst_port')
            proto = r.get('proto') if r.get('proto') is not None else r.get('protocol')
            k = _conn_sort_key(src, int(sport or 0), dst, int(dport or 0), proto)
        except Exception:
            continue
        # 同一条连接在明细里可能出现多次：只要有一次被判成恶意，就按恶意算，
        # 否则会被「最后一次恰好是正常」覆盖掉。
        m[k] = bool(m.get(k)) or bool(r.get('is_malicious'))
    return m


def _threat_flow_rows(flow_list, detail, limit=None, two_class_map=None):
    """把 (flow_key, 78 维) 与推理结果拼成前端可点击查看的逐流明细。

    卡片上的每一行（Benign / Bot / DoS-GoldenEye …）都要能点开看「到底是哪几条」。
    汇总数字（class_summary）只知道每类多少条，说不出是哪几条，所以这里把
    ``_classify_flow_matrix`` 的逐行结果还原成一条条带五元组的记录。

    排序按「越可疑越靠前」：先未知攻击、再按离群距离降序、最后看置信度。
    和 ``_review_queue_payload`` 同一套排序理由——实时抓包在当前环境下大量流都会被
    判「未知攻击」，按时间排等于随机列，人点开也看不出该先核哪条。

    ``two_class_map`` 来自 ``_two_class_map_from``：给每行附上包级二分类的结论，
    前端据此把「二分类明确说正常、六分类却给攻击类」的行改计入域外流量，
    避免同一屏里两个数字打架。
    """
    if detail is None or not flow_list:
        return []
    # limit <= 0 表示「全量返回」，供一致性统计用：分歧比例必须基于全部流算，
    # 只统计前 200 条明细会让 B2/B3 那种「卡片数字与记录表打架」的判断失真。
    cap = int(limit if limit is not None else THREAT_FLOW_ROWS_LIMIT)
    truncate = cap > 0
    names = detail['class_names']
    preds = detail['pred']
    confs = detail['conf']
    tiers = detail['tier']
    is_unknown = detail['is_unknown']
    oods = detail['ood_distance']

    rows = []
    for i, item in enumerate(flow_list):
        try:
            key, feats = item[0], item[1]
            src, sport, dst, dport, proto = key
        except (TypeError, ValueError, IndexError):
            continue
        # snapshot(with_ts=True) 会给第三项「该流最后收包时刻（秒）」。
        # 前端点「清除日志」后要按这个时间戳把清除之前的流滤掉，否则下一轮轮询
        # 又把同一批流灌回来，用户会觉得清除按钮点了没用。
        captured_at = None
        if len(item) > 2:
            try:
                captured_at = round(float(item[2]), 3)
            except (TypeError, ValueError):
                captured_at = None
        if captured_at is None:
            # 上传链路（pcap → 78 维特征）分析的是离线文件，没有逐包时间可比，
            # 用「本次判定时刻」代替。留 None 的后果是前端清除一次之后，再上传的
            # 全部明细都被时间窗口滤掉（Number(null) === 0），卡片就再也不动了。
            captured_at = round(time.time(), 3)
        ood = None if oods is None else round(float(oods[i]), 2)
        two_mal = None
        if two_class_map:
            two_mal = two_class_map.get(_conn_sort_key(src, sport, dst, dport, proto))
        rows.append({
            'src_ip': str(src), 'src_port': int(sport),
            'dst_ip': str(dst), 'dst_port': int(dport),
            'proto': int(proto),
            'predicted': names[int(preds[i])],
            'confidence': round(float(confs[i]), 4),
            'tier': str(tiers[i]),
            'is_unknown': bool(is_unknown[i]),
            'ood_distance': ood,
            'captured_at': captured_at,
            # 包级二分类对同一条连接的结论。None = 没采到，前端据此不做压制。
            'two_class_malicious': (None if two_mal is None else bool(two_mal)),
            'packets': int((feats.get('Tot Fwd Pkts', 0) or 0)
                           + (feats.get('Tot Bwd Pkts', 0) or 0)),
        })

    rows.sort(key=lambda r: (1 if r['is_unknown'] else 0,
                             float(r['ood_distance'] or 0.0),
                             float(r['confidence'])),
              reverse=True)
    return rows[:cap] if truncate else rows


def _stream_definition():
    """两条链路的「流/会话」定义，供接口与界面显式说明（缺陷 D-23）。

    同一份 pcap 上，「包级二分类卡片里的会话数」和「威胁类型分布里的流数」本来
    就不该相等：
      · 二分类：五元组双向聚合，每会话最多取前 28 包；默认不按时间切分，
        同一五元组无论停顿多久都算一条（``DLIDS_FLOW_IDLE_TIMEOUT`` 可改成按时间切）。
      · 六分类：五元组 + 120 秒空闲超时，超时即另起一条流（CICFlowMeter 口径）。
    把这件事写进响应，界面就不必假装两个数字理应相同。
    """
    from src.preprocessing.flow_features import FLOW_TIMEOUT_US

    return {
        'two_class': '五元组双向聚合、每会话前 28 包；空闲超时=%s' % (
            '关闭（同一五元组不分段）' if FLOW_IDLE_TIMEOUT_SECONDS is None
            else f'{FLOW_IDLE_TIMEOUT_SECONDS:g} 秒'),
        'flow_six_class': f'五元组 + {FLOW_TIMEOUT_US / 1e6:g} 秒空闲超时切流，取前 200000 包',
        'counts_are_comparable': FLOW_IDLE_TIMEOUT_SECONDS == FLOW_TIMEOUT_US / 1e6,
    }


def _threat_consistency(rows, summary):
    """统计两套模型的结论分歧，供界面显式提示（缺陷 B2 / B3）。

    背景：一次上传里其实跑了两套完全独立的模型 ——
      · 包级二分类（28×28 字节图），回答「这批流量是否恶意」；
      · 流级六分类（78 维流统计），回答「属于哪类攻击」。
    两份输入都不一样，结论不一致是常态，不是 bug。真正的 bug 是**界面上零提示**：
    记录表全绿（99.95% 正常）的同时威胁卡里 DoS-GoldenEye 占大头，用户只能理解为
    系统自相矛盾（B2）；良性 Facetime 抓包被六分类判成 3180 条 DoS-GoldenEye
    置信度 1.0，页面也照实陈列（B3）。

    这里不压制、不删除任何一侧的结论（压掉就等于丢掉可能的漏报），只把分歧量算出来，
    让前端写清楚「这两个数字来自不同模型，它们不一致」。逐行的 two_class_malicious
    仍然照旧下发，前端若要压制单行还有个依据。

    三个桶：
      · agree     —— 二分类说恶意，六分类也给攻击类；
      · conflict  —— 二分类说正常，六分类却给攻击类（B3 那种情形）；
      · missed    —— 二分类说恶意，六分类说正常（反向不一致，同样是分歧）；
      · unjudged  —— 二分类没覆盖到这条流（五元组 join 不上），不参与统计。
    """
    agree = conflict = missed = unjudged = 0
    per_class_conflict = {}
    for r in rows or []:
        two = r.get('two_class_malicious')
        if two is None:
            unjudged += 1
            continue
        name = r.get('predicted')
        is_attack_pred = bool(name) and name != 'Benign' and not r.get('is_unknown')
        if two and is_attack_pred:
            agree += 1
        elif two and not is_attack_pred:
            missed += 1
        elif (not two) and is_attack_pred:
            conflict += 1
            per_class_conflict[name] = per_class_conflict.get(name, 0) + 1
        else:
            agree += 1
    judged = agree + conflict + missed
    ratio = (conflict / judged) if judged else 0.0
    return {
        'judged': judged,
        'agree': agree,
        'conflict': conflict,
        'missed': missed,
        'unjudged': unjudged,
        'conflict_ratio': round(ratio * 100, 2),
        'per_class_conflict': per_class_conflict,
        # 有分歧时前端必须写明「两个数字来自不同模型」；没分歧就不用多话
        'disagree': bool(conflict or missed),
        'note': ('包级二分类与流级六分类是两套独立模型（输入分别为 28×28 字节图与 78 维流统计），'
                 '本次两者结论不一致：请以「记录」页的明细为准核对，不要把这组数字当作同一口径。'
                 if (conflict or missed) else None),
    }


def _trim_review_queue():
    """超出上限就丢最早的，避免长时间抓包把内存和前端渲染拖死。"""
    if len(live_review_queue) <= LIVE_REVIEW_QUEUE_LIMIT:
        return
    ordered = sorted(
        live_review_queue.items(),
        key=lambda kv: float(kv[1].get('last_seen') or 0.0),
        reverse=True,
    )
    live_review_queue.clear()
    live_review_queue.update(ordered[:LIVE_REVIEW_QUEUE_LIMIT])


def _put_review_entry(key: str, entry: dict):
    live_review_queue[key] = entry
    _trim_review_queue()


def _sync_review_queue(flows, detail, source: str = 'live', meta=None, two_class_map=None):
    """把 review/manual 档与「未知攻击」的流挂进复核队列。

    判据选择依据：字节图上的置信度几乎不触发（实测 8 份工控 pcap 全在 0.95 以上），
    而流特征的六分类档位有实测依据（`tools/sweep_threshold.py` 的误报-检出曲线），
    所以用档位而不是二元模型置信度来决定「值不值得人看一眼」。

    未知攻击一律进队列：模型对它的类归属本就不可信，高置信的错误分类最危险。

    ``source`` 记录这条流是从哪条链路进来的（live 实时抓包 / upload pcap 上传 /
    csv 流特征上传）。复核中心按来源分组显示，同一条五元组在不同批次里也各留一份
    结论 —— 把它们的 key 混在一起会互相覆盖，等于悄悄吞掉一次检测结果。
    """
    if detail is None:
        return 0
    extra = dict(meta or {})
    added = 0
    new_keys = []
    try:
        tiers = detail['tier']
        preds = detail['pred']
        confs = detail['conf']
        isunk = detail['is_unknown']
        oods = detail['ood_distance']
        names = detail['class_names']
        now = time.time()
        for i, item in enumerate(flows):
            try:
                # snapshot(with_ts=True) 给的是三元组 (key, feats, ts)，没有时间戳时是二元组；
                # 这里只取前两项，不要直接 `for (key, feats) in flows` —— 三元组会直接抛
                # "too many values to unpack (expected 2)"，整条队列就永远进不去。
                key, feats = item[0], item[1]
                src, sport, dst, dport, proto = key
            except (TypeError, ValueError, IndexError):
                continue
            tier = str(tiers[i])
            unknown = bool(isunk[i])
            if tier == 'auto' and not unknown:
                continue
            k = '{}|{}:{}|{}:{}|{}'.format(source, src, sport, dst, dport, proto)
            entry = {
                'key': k,
                'source': source,
                'src_ip': src, 'src_port': int(sport),
                'dst_ip': dst, 'dst_port': int(dport), 'proto': int(proto),
                'predicted': names[int(preds[i])],
                'confidence': round(float(confs[i]), 4),
                'tier': tier,
                'is_unknown': unknown,
                'ood_distance': (None if oods is None else round(float(oods[i]), 2)),
                'flows': int((feats.get('Tot Fwd Pkts', 0) or 0)
                             + (feats.get('Tot Bwd Pkts', 0) or 0)),
                'last_seen': now,
            }
            # 二分类结论是「默认 6 模型投票」自动定案的唯一依据；拿不到就留 None，
            # 让自动策略退回「按六分类映射」并把这件事写进 note（None 不等于正常）。
            if two_class_map:
                entry['two_class_malicious'] = two_class_map.get(
                    _conn_sort_key(src, sport, dst, dport, proto))
            entry.update(extra)
            _put_review_entry(k, entry)
            added += 1
            new_keys.append(k)
    except Exception as e:  # noqa: BLE001 - 队列维护失败不该影响主链路
        print('sync review queue failed:', e)
    _apply_review_policy(new_keys)
    return added


def _enqueue_csv_review_rows(file_name: str, details, source: str = 'csv', limit: int = 40) -> int:
    """把流特征 CSV 批次里「待确认 / 人工复核 / 未知攻击」的行挂进复核队列。

    CSV 是匿名流特征（没有五元组），键用「来源 + 文件名 + 行号」。
    一个 CSV 动辄几百上千行，全量入队会把队列和前端一起拖死，所以只取最可疑的
    前 ``limit`` 行 —— ``details`` 由 /analyze-flow 按置信度升序产出，最可疑的在最前。
    """
    if not details:
        return 0
    now = time.time()
    added = 0
    new_keys = []
    for row in details:
        if added >= limit:
            break
        try:
            tier = str(row.get('tier') or '')
            unknown = bool(row.get('is_unknown'))
            # 与实时链路同一判据：自动档且不是未知攻击，才不需要人看
            if tier == 'auto' and not unknown:
                continue
            idx = row.get('row')
            k = '{}|{}#{}'.format(source, file_name, idx)
            probs = row.get('probabilities') or {}
            top = sorted(probs.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:4]
            ood = row.get('ood_distance')
            entry = {
                'key': k,
                'source': source,
                'file': file_name,
                'row': idx,
                'src_ip': '', 'src_port': 0, 'dst_ip': '', 'dst_port': 0, 'proto': 0,
                'predicted': row.get('predicted'),
                'confidence': round(float(row.get('confidence') or 0.0), 4),
                'tier': tier,
                'is_unknown': unknown,
                'ood_distance': (None if ood is None else round(float(ood), 2)),
                'probabilities': [
                    {'label': str(k2), 'value': round(float(v2 or 0.0), 4)} for k2, v2 in top
                ],
                'flows': 0,
                'last_seen': now,
            }
            _put_review_entry(k, entry)
            added += 1
            new_keys.append(k)
        except Exception as e:  # noqa: BLE001
            print('enqueue csv review row failed:', e)
    _apply_review_policy(new_keys)
    return added


def _review_queue_payload(limit=50):
    """复核队列的下发形态：越异常越靠前，最多 limit 条。

    排序不按时间而按可疑度。实时抓包在当前环境下绝大多数流都会被判「未知攻击」
    （实测 ood 距离是阈值的 2~17 倍），纯按时间排等于随机列 50 条，人工看不出该先核哪个。
    先排未知攻击，再按离群距离降序，同距离段再看时间新的在前。
    """
    try:
        items = sorted(
            live_review_queue.values(),
            key=lambda v: (1 if v.get('is_unknown') else 0,
                           float(v.get('ood_distance') or 0.0),
                           float(v.get('last_seen') or 0.0)),
            reverse=True,
        )
        return items[:max(1, int(limit))]
    except Exception as e:  # noqa: BLE001
        print('review queue payload failed:', e)
        return []


# ── 人工复核结果 ──────────────────────────────────────────────────
# 队列只解决「哪些该看」，这里解决「看完怎么记」——没有这一层，队列里的流
# 永远无法标记为已处理，只能等下一轮抓包整体清掉，「人工复核」是空标签。
# 三种结论的含义（统计口径不同，不能混）：
#   malicious  确认是攻击 → 可作为真实检出数
#   benign     确认是误报 → 可作为真实误报数（这才是可信的误报率来源）
#   ignore     看不出/无分析价值 → 不参与统计，只留痕
# 结果与 tracker 同生命周期，清空抓包不清它；要清得单独调用。
live_review_results = {}
LIVE_REVIEW_RESULT_LIMIT = 500


def _review_stats_payload():
    """复核队列与已复核结果的聚合计数，供页头显示。"""
    counts = {'malicious': 0, 'benign': 0, 'ignore': 0}
    for v in live_review_results.values():
        verdict = v.get('verdict')
        if verdict in counts:
            counts[verdict] += 1
    return {
        'pending': len(live_review_queue),
        'reviewed': len(live_review_results),
        'by_verdict': counts,
    }


REVIEW_VERDICTS = ('malicious', 'benign', 'ignore')

# ── 复核队列的默认处置策略 ────────────────────────────────────────
# 队列解决「哪些该看」，但并非每条都值得一个人坐下来点三个按钮：实时抓包下绝大多数
# 流都会被判「未知攻击」，队列几分钟就能堆到上限。这里给一个可切换的默认策略：
#   manual —— 全部人工复核（默认，与过去行为完全一致）
#   vote   —— 入队即按二分类投票结论自动定案（没有二分类结论时退化为按六分类预测映射）
#   llm    —— 入队即交给本地 LLM 判断；LLM 不可用时降级为 vote 并把原因记在条目上
# 自动结论与人工结论共用同一个落库入口（_apply_review_verdict），并且都标 reviewed_by，
# 否则统计出来的「真实误报率」无法区分是人判的还是机器判的。
REVIEW_POLICY_MODES = ('manual', 'vote', 'llm')
review_policy = {'mode': 'manual', 'updated_at': 0.0}
REVIEW_POLICY_META = {
    'manual': {
        'label': '人工复核（默认）',
        'desc': '所有待复核流量都留在队列里，等人逐条点结论。',
    },
    'vote': {
        'label': '默认 6 模型投票',
        'desc': '入队时立即用检测模型的投票结论自动定案，不再占队列（结论记为 auto:vote）。',
    },
    'llm': {
        'label': '默认 LLM 判断',
        'desc': '入队时交给本地 LLM 判断，失败或不可用时自动退回 6 模型投票（结论记为 auto:llm / auto:vote）。',
    },
}


def _review_policy_payload() -> dict:
    """当前策略 + 可选值，供复核中心渲染下拉框。"""
    mode = str(review_policy.get('mode') or 'manual')
    return {
        'mode': mode if mode in REVIEW_POLICY_MODES else 'manual',
        'modes': [
            {'value': m, 'label': REVIEW_POLICY_META[m]['label'], 'desc': REVIEW_POLICY_META[m]['desc']}
            for m in REVIEW_POLICY_MODES
        ],
        'updated_at': review_policy.get('updated_at') or 0.0,
    }


def _review_source_counts() -> dict:
    """按来源（live / upload / csv）统计待复核条数，复核中心按来源分组用。"""
    counts = {}
    for v in live_review_queue.values():
        src = str(v.get('source') or 'live')
        counts[src] = counts.get(src, 0) + 1
    return counts


def _apply_review_verdict(key: str, verdict: str, reviewed_by: str = 'human', note: str = None) -> dict:
    """把一条复核结论落库，并把该流移出待复核队列。

    人工点按钮、LLM 自主判断、默认策略自动定案共用这一个入口 —— 三条路径的落库口径
    必须一致，否则统计出来的「真实误报率」会把其中一种结论漏掉。

    ``reviewed_by`` 是这张表唯一能区分「人判的」和「机器判的」的字段：
    ``human`` / ``auto:vote`` / ``auto:llm``。默认策略自动定案时必须带上它，
    否则自动结论会混进人工统计里，误报率就再也说不清是谁的锅。
    """
    entry = dict(live_review_queue.get(key) or {'key': key})
    entry['key'] = key
    entry['verdict'] = verdict
    entry['reviewed_at'] = time.time()
    entry['reviewed_by'] = reviewed_by
    if note:
        entry['note'] = note
    live_review_results[key] = entry
    live_review_queue.pop(key, None)

    if len(live_review_results) > LIVE_REVIEW_RESULT_LIMIT:
        ordered = sorted(
            live_review_results.items(),
            key=lambda kv: float(kv[1].get('reviewed_at') or 0.0),
            reverse=True,
        )
        live_review_results.clear()
        live_review_results.update(ordered[:LIVE_REVIEW_RESULT_LIMIT])
    return entry


# 一次后台自动判断最多处理多少条：LLM 是逐条串行调用的，条数太多会一直占着一个
# 本地模型，抓包期间其它请求就只能排队等它。
REVIEW_AUTO_LLM_BATCH = 6


def _auto_verdict_for_entry(entry):
    """「默认 6 模型投票」口径的自动结论。

    优先级：条目自带的二分类投票结论（实时 / 上传 pcap 链路才有）> 按六分类预测映射。
    流特征 CSV 只有 78 维统计量、没有字节图输入，二分类模型用不上，只能按类别映射，
    这一点必须在 note 里写明，否则事后分不清「投票判的」和「映射猜的」。

    返回 ``(verdict, reason, confidence)``；verdict 为 None 表示没法自动判。
    """
    two = entry.get('two_class_malicious')
    conf = entry.get('confidence')
    if two is not None:
        verdict = 'malicious' if two else 'benign'
        return verdict, '二分类投票：{}'.format('恶意' if two else '良性'), conf
    pred = str(entry.get('predicted') or '')
    if not pred:
        return None, '缺少可用结论', None
    verdict = 'benign' if pred == 'Benign' else 'malicious'
    return verdict, '按六分类预测映射：{}（该链路无二分类输入）'.format(pred), conf


def _auto_llm_worker(keys):
    """后台线程：让本地 LLM 给一批条目自动定案，LLM 不可用时退回投票。

    必须放线程里跑 —— ``_llm_chat`` 单条最长等 LLM_TIMEOUT（默认 60 秒），在入队路径
    上同步等待会把抓包轮询和上传接口一起堵死。
    """
    try:
        ok, model, _names = _llm_probe()
    except Exception as e:  # noqa: BLE001
        ok, model = False, None
        print('review auto llm probe failed:', e)
    for key in keys[:REVIEW_AUTO_LLM_BATCH]:
        entry = live_review_queue.get(key)
        if not entry:
            continue
        verdict, conf, reason = None, None, None
        if ok:
            try:
                text = _llm_chat(_build_review_messages(entry), max_tokens=220)
                verdict, conf, reason = _parse_llm_verdict(text)
            except Exception as e:  # noqa: BLE001
                print('review auto llm failed:', e)
        if verdict in REVIEW_VERDICTS:
            _apply_review_verdict(
                key, verdict, reviewed_by='auto:llm',
                note='LLM {}：{}'.format(model or 'local', (reason or '未给出理由')))
        else:
            v2, why, _c2 = _auto_verdict_for_entry(entry)
            if v2:
                _apply_review_verdict(
                    key, v2, reviewed_by='auto:vote',
                    note='LLM 不可用或未给出结论，退回投票：' + why)


def _apply_review_policy(keys):
    """按当前默认策略处置刚入队的条目。

    ``manual`` 时直接返回 0 —— 默认策略是关的，行为与加这个功能之前完全一致。
    """
    mode = str(review_policy.get('mode') or 'manual')
    if mode not in REVIEW_POLICY_MODES or mode == 'manual' or not keys:
        return 0
    applied = 0
    if mode == 'vote':
        for key in keys:
            entry = live_review_queue.get(key)
            if not entry:
                continue
            verdict, why, _conf = _auto_verdict_for_entry(entry)
            if verdict:
                _apply_review_verdict(key, verdict, reviewed_by='auto:vote', note=why)
                applied += 1
    elif mode == 'llm':
        try:
            threading.Thread(target=_auto_llm_worker, args=(list(keys),), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            print('review auto llm thread failed:', e)
    return applied


def _live_threat_summary():
    """把实时抓包跟踪表里的活跃流算成六分类汇总。

    返回 ``(summary, meta)``；模型没加载、还没抓到可解析的流、或 tracker 尚未创建时
    返回 ``(None, None)``。

    缓存策略：``live_flow_tracker.version`` 每收一个包就 +1，只有它变了才重新推理。
    前端 1.5 秒轮询一次，不缓存的话一秒钟要做几十次 78 维 -> 六分类，纯粹浪费。

    缺陷 D-30：缓存键过去只有 ``version``，于是「抓包停了以后重新训练/重校准了
    六分类模型」这种版本号不变、结论却该变的情况会一直返回旧缓存，卡片显示的是
    上一个模型的判断。现在把「模型指纹」和「recent_flows 快照签名」一并纳入键：
    前者在模型对象/温度/OOD 开关变化时失效，后者在抓包停止后仍新增采样记录时失效。
    """
    tracker = live_flow_tracker
    if tracker is None or flow_multiclass is None or flow_multiclass_meta is None:
        return None, None

    try:
        with capture_lock:
            _recent_snapshot = list(recent_flows)
    except Exception as e:      # noqa: BLE001
        print('live flow sampling snapshot failed:', e)
        _recent_snapshot = []

    _model_sig = (
        id(flow_multiclass),
        float(flow_multiclass_meta.get('temperature') or 1.0),
        bool((flow_centers or {}).get('space_matches_active', True)),
    )
    _recent_sig = (
        len(_recent_snapshot),
        (_recent_snapshot[-1].get('id') if _recent_snapshot else None),
    )

    cached = live_threat_cache
    if (cached['version'] == tracker.version
            and cached.get('model_sig') == _model_sig
            and cached.get('recent_sig') == _recent_sig):
        return cached['summary'], cached['meta']

    try:
        flows = tracker.snapshot(active_only=bool(capture_stats.get('running')), with_ts=True)
    except Exception as e:      # noqa: BLE001 - 统计失败不该让轮询接口 500
        print('live flow snapshot failed:', e)
        return None, None

    # 只统计「已经被推理采样过的连接」。
    # tracker 里每条流在收包时就建好了，但要攒够流量才会触发一次推理、写进 recent_flows
    # —— 下方「实时抓包记录」表渲染的就是 recent_flows。实测一次抓包 4400 个包时
    # tracker 有 130 条流、真正被采样记录下来的只有 52 条。卡片如果按 130 算，
    # 就会永远比表格多出一大截，用户看到的就是「卡片说 130 条，表里只有 52 行」。
    # 按同一批连接过滤之后，两个数字才是同一件事。
    try:
        sampled = {_recent_conn_key(item) for item in _recent_snapshot}
        if sampled:
            flows = [kv for kv in flows if _conn_sort_key(*kv[0]) in sampled]
    except Exception as e:      # noqa: BLE001
        print('live flow sampling filter failed:', e)
        _recent_snapshot = []

    summary = meta = None
    rows = []
    if flows:
        try:
            from src.preprocessing.flow_features import features_to_matrix

            _flows, matrix = features_to_matrix(flows)
            summary, _detail = _classify_flow_matrix(matrix)
            # 队列维护独立于 summary：某次推理失败也不该把已入队的条目抹掉
            # two_class_map 让每条待复核条目带上二分类投票结论 —— 「默认 6 模型投票」
            # 这个自动策略就是靠它定案的，拿不到才退回按六分类映射。
            _sync_review_queue(_flows, _detail,
                               two_class_map=_two_class_map_from(_recent_snapshot))
            # 逐流明细：卡片上每一行点开看的就是它。和 summary 同一批 _flows，
            # 所以点进去的条数一定等于那一行显示的 count。
            # two_class_map 让每条明细带上包级二分类的结论，前端据此把
            # 「二分类说正常、六分类却给攻击类」的流改计入域外流量。
            rows = _threat_flow_rows(_flows, _detail, limit=0,
                                     two_class_map=_two_class_map_from(_recent_snapshot))
            if summary is not None:
                unknown = summary.get('未知攻击', {}).get('count', 0)
                meta = {
                    'flows': len(flows),
                    'unknown_count': int(unknown),
                    'ood_threshold': (None if flow_centers is None
                                      else float(flow_centers['thresholds'][3])),
                    'class_names': list(flow_multiclass_meta['class_names']),
                    'source': 'live',
                    'ood_enabled': bool((flow_centers or {}).get('space_matches_active', True)),
                    'ood_disabled_reason': (flow_centers or {}).get('space_mismatch_note'),
                    # 明细条数：弹窗据此说明「只列出前 N 条」。
                    # 正常情况下它等于 flows（每条流都有一条明细），
                    # 只有超过 THREAT_FLOW_ROWS_LIMIT 时才会更小。
                    'flow_rows': min(len(rows), THREAT_FLOW_ROWS_LIMIT),
                    'flows_shown': min(len(rows), THREAT_FLOW_ROWS_LIMIT),
                    'flows_classified': len(rows),
                    # 包级二分类与流级六分类的分歧量（缺陷 B2/B3）：界面上必须写明
                    # 「这两个数字来自不同模型」，否则用户只会认为系统自相矛盾。
                    'consistency': _threat_consistency(rows, summary),
                    # 实时链路的流定义就是六分类那一套（tracker 内部按 120s 空闲切分），
                    # 与包级二分类的会话数不可直接比较（缺陷 D-23）
                    'stream_definition': _stream_definition(),
                }
                rows = rows[:THREAT_FLOW_ROWS_LIMIT]

        except Exception as e:  # noqa: BLE001
            print('live threat classify failed:', e)
            summary = meta = None
            rows = []

    cached['version'] = tracker.version
    cached['model_sig'] = _model_sig
    cached['recent_sig'] = _recent_sig
    cached['summary'] = summary
    cached['meta'] = meta
    cached['flows'] = rows
    return summary, meta


@app.post("/analyze-flow")
async def analyze_flow(
    file: UploadFile | None = File(None),
    confidence_high: float = Form(FLOW_CONFIDENCE_HIGH, description="自动处置门槛（置信度 >= 此值直接定案）"),
    confidence_low: float = Form(FLOW_CONFIDENCE_LOW, description="人工复核门槛（置信度 < 此值进复核队列）"),
    max_rows: int = Form(0, description="最多分析多少行，0 表示全部"),
    session_user: str = Depends(require_session),
):
    """多分类流特征检测：上传 CICFlowMeter 风格的 CSV，返回攻击类型 + 校准置信度 + 风险档位。

    与 /analyze 的分工：那条走 pcap → 28x28 字节图 → 6 个二分类模型；这条走 CSV →
    78 维流统计 → 单个六分类模型。两者输入格式与特征口径完全不同，所以分开两个接口。
    """
    # 缺陷 D-07：这两个门槛过去在这里默认 0.90/0.50，与 pcap 链路、实时链路的
    # 0.90/0.50 各自独立写死，改一处漏两处，同一个模型在三个入口给出不同档位。
    # 现在缺省值直接取全站唯一配置源；显式传参仍然允许，但要过范围与顺序校验。
    if confidence_high is None:
        confidence_high = FLOW_CONFIDENCE_HIGH
    if confidence_low is None:
        confidence_low = FLOW_CONFIDENCE_LOW
    if not (0.0 <= confidence_high <= 1.0) or not (0.0 <= confidence_low <= 1.0):
        return JSONResponse(status_code=400, content={
            "error": "confidence_high / confidence_low must be within [0, 1].",
            "confidence_high": confidence_high,
            "confidence_low": confidence_low,
        })
    if confidence_low > confidence_high:
        # 过去照收：低门槛高于高门槛时档位判定整段反过来，所有样本都落进 review，
        # 页面看起来"全部需人工复核"却不报错，排查成本极高。
        return JSONResponse(status_code=400, content={
            "error": "confidence_low must be <= confidence_high.",
            "confidence_high": confidence_high,
            "confidence_low": confidence_low,
        })
    # file 从 `File(...)` 改成 `File(None)`：必填校验若留在依赖解析阶段，
    # FastAPI 会在进入函数体之前就吐 422，上面两条门槛校验永远跑不到 ——
    # 「参数非法」会被报成「缺文件」，用户按提示补文件后才发现真正的问题。
    # 现在先校验参数、再在这里补必填判断。
    if file is None:
        return JSONResponse(status_code=422, content={
            "error": "file is required",
            "detail": "提供一个 CICFlowMeter 风格的 CSV 文件。",
        })
    if flow_multiclass is None:
        return JSONResponse(status_code=503, content={
            "error": "Multiclass flow model is not loaded. Train it first: "
                     "python tools/train_multiclass.py --days 15-02,21-02,02-03"
        })

    suffix = os.path.splitext(file.filename or '')[1].lower()
    if suffix not in ('.csv',):
        return JSONResponse(status_code=400, content={
            "error": f"Only .csv is accepted here (pcap goes to /analyze), got '{suffix or '(none)'}'.",
        })

    start_time = time.time()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={
            "error": f"CSV exceeds the {MAX_UPLOAD_MB} MB limit.",
        })

    try:
        import io as _io
        import pandas as _pd
        df = _pd.read_csv(_io.BytesIO(raw), low_memory=False, nrows=(max_rows or None))
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Failed to parse CSV", "detail": str(e)})

    meta = flow_multiclass_meta
    feature_names = meta['feature_names']
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return JSONResponse(status_code=400, content={
            "error": "CSV is missing required feature columns.",
            "missing": missing[:20],
            "missing_count": len(missing),
            "expected_count": len(feature_names),
        })

    # 多天数据简单拼接出的 CSV 常把重复表头当成数据行混进来，于是某一列里出现字符串
    # （典型报错：could not convert string to float: 'ACK Flag Cnt'）。强制转数值，
    # 坏行变 NaN 后由下面的 isfinite 统一剔除，不要因为一行脏数据把整次分析打成 500。
    numeric_block = df[feature_names].apply(_pd.to_numeric, errors='coerce')
    X = numeric_block.to_numpy(dtype=np.float32)
    finite = np.isfinite(X).all(axis=1)
    dropped = int((~finite).sum())
    X = X[finite]
    if len(X) == 0:
        return JSONResponse(status_code=400, content={
            "error": "No valid rows left after dropping NaN/inf.",
        })

    class_names = meta['class_names']
    temperature = meta['temperature'] or 1.0

    try:
        class_summary, _detail = _classify_flow_matrix(X, confidence_high, confidence_low)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Inference failed", "detail": str(e)})
    if _detail is None:
        return JSONResponse(status_code=503, content={
            "error": "Multiclass flow model is not loaded."
        })

    # 逐行结果统一从共用函数里取：六分类的 pred/conf/OOD/tier 只在一处计算，
    # 免得 pcap 链路与 CSV 链路各写一份、阈值或口径悄悄分叉。
    conf = _detail['conf']
    pred = _detail['pred']
    tiers = _detail['tier']
    is_unknown = _detail['is_unknown']
    ood_distance = _detail['ood_distance']
    ood_threshold = _detail['ood_threshold']
    prob = _detail['prob']

    tier_summary = _tier_summary(tiers, conf)

    # 明细按置信度升序输出，最可疑的排在最前，方便直接进人工复核队列
    order = np.argsort(conf)
    limit = min(len(order), 200)
    details = []
    for idx in order[:limit]:
        details.append({
            'row': int(idx),
            'predicted': class_names[int(pred[idx])],
            'confidence': round(float(conf[idx]), 4),
            'tier': str(tiers[idx]),
            'is_unknown': bool(is_unknown[idx]),
            'ood_distance': round(float(ood_distance[idx]), 2) if ood_distance is not None else None,
            'probabilities': {name: round(float(prob[idx][j]), 4)
                              for j, name in enumerate(class_names)},
        })

    malware_idx = [i for i, n in enumerate(class_names) if n.strip().lower() != 'benign']
    malware_ratio = float(np.isin(pred, malware_idx).mean()) * 100 if malware_idx else 0.0

    # 留一份供「AI 分析」用的上下文：只存明细里出现过的行 + 批次中位数。
    # 全量 X 可能是百万行（几百 MB），而明细最多 200 条，没必要为解释功能留全量；
    # 批次中位数则是判断「这条偏高还是正常」的唯一参照。
    global _LAST_FLOW_BATCH
    try:
        _LAST_FLOW_BATCH = {
            'file': file.filename,
            'rows': {int(i): X[i] for i in order[:limit]},
            'batch_median': np.median(X, axis=0),
            'feature_names': feature_names,
            'class_names': class_names,
            'time': time.time(),
        }
    except Exception as _snap_err:
        print('Failed to cache flow batch for explain:', _snap_err)
        _LAST_FLOW_BATCH = None

    # 流特征 CSV 批次同样要进复核队列：这张明细表就是「待确认 / 人工复核」的来源，
    # 只在页面上筛一遍、不落到统一队列里，换页后就再也找不到了。
    try:
        _enqueue_csv_review_rows(str(file.filename or ''), details,
                                 source='csv', limit=40)
    except Exception as _review_err:  # noqa: BLE001
        print('csv review enqueue failed:', _review_err)

    return {
        "file": file.filename,
        "rows": int(len(X)),
        "dropped_rows": dropped,
        "classes": class_names,
        "temperature": temperature,
        "malware_ratio": round(malware_ratio, 2),
        "class_summary": class_summary,
        "tier_summary": tier_summary,
        "thresholds": {"auto": confidence_high, "review": confidence_low},
        "unknown": {
            "enabled": ood_distance is not None,
            "count": int(is_unknown.sum()) if ood_distance is not None else 0,
            "ratio": round(float(is_unknown.mean()) * 100, 2) if ood_distance is not None else 0.0,
            "threshold": ood_threshold,
        },
        "details": details,
        "details_truncated": bool(len(order) > limit),
        "execution_time": f"{time.time() - start_time:.4f}s",
    }


# ---------------------------------------------------------------- 环境校准
# 跨数据集实测结论：源域训练出来的模型换到没见过的环境，恶意检出率会掉到 0%
# （AdaBN、批内标准化、特征剔除、AutoEncoder 全都试过，救不回来）。
# 唯一确定有效的办法是让模型看目标环境自己的数据：500~1000 条人工标注 + 8 轮微调，
# 检出率就能从 0% 回到 97%（见 tools/finetune_local.py、logs/finetune_local.json）。
# 这个接口把那条路做成闭环：上传带 Label 的 CSV → 微调 → 指标不降就热切换。
FLOW_LABEL_ALIASES = {
    'benign': 'Benign',
    'bot': 'Bot',
    'ddos attack-hoic': 'DDoS-HOIC',
    'dos attacks-goldeneye': 'DoS-GoldenEye',
    'dos attacks-slowloris': 'DoS-Slowloris',
    'ddos attack-loic-udp': 'DDoS-LOIC-UDP',
}


def _flow_label_index(raw: Any, class_names: list) -> int:
    """把 CSV 的 Label 映射成模型类别下标，认不出返回 -1。"""
    text = str(raw).strip().lower()
    for i, name in enumerate(class_names):
        if text == str(name).lower():
            return i
    alias = FLOW_LABEL_ALIASES.get(text)
    if alias is not None:
        for i, name in enumerate(class_names):
            if alias.lower() == str(name).lower():
                return i
    return -1


def _flow_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """准确率 + macro F1 + 每类召回。类别不平衡时只看准确率会高估真实水平。

    macro F1 只对「评估集里真的出现过」的类求平均。上传的标注批次常常只覆盖少数
    几类，把没样本的类当 0 分计入平均，会把结果压成 33%（6 类里 4 类缺席时），
    那个数字对判断校准有没有用完全没有参考价值。
    """
    acc = float((y_true == y_pred).mean()) * 100 if len(y_true) else 0.0
    f1s, per_class = [], {}
    for i, name in enumerate(class_names):
        support = int((y_true == i).sum())
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if support > 0:
            f1s.append(f1)
        per_class[name] = {
            'support': support,
            'recall': round(recall * 100, 2),
            'precision': round(precision * 100, 2),
            'f1': round(f1 * 100, 2),
        }
    return {
        'samples': int(len(y_true)),
        'accuracy': round(acc, 2),
        'macro_f1': round(float(np.mean(f1s)) * 100, 2) if f1s else 0.0,
        'classes_present': len(f1s),
        'per_class': per_class,
    }


def _tier_summary(tiers, conf):
    """三档处置的汇总（自动 / 待确认 / 人工复核）。

    缺陷 D-12：这段原本只写在 `/analyze-flow`（CSV）里，pcap 上传与实时链路
    拿不到 tier_summary，同一个模型在两个入口给出两种可见程度。抽成一处后
    两条链路共用，阈值语义与档位名称不会再分叉。
    """
    tiers = np.asarray(tiers)
    conf = np.asarray(conf, dtype=np.float64)
    out = {}
    for key, label in (('auto', '自动处置'), ('review', '告警待确认'), ('manual', '人工复核')):
        mask = tiers == key
        out[key] = {
            'label': label,
            'count': int(mask.sum()),
            'ratio': round(float(mask.mean()) * 100, 2),
            'avg_confidence': round(float(conf[mask].mean()), 4) if mask.any() else None,
        }
    return out


def _persist_flow_multiclass_checkpoint(model, mean, std, temperature, extra=None):
    """把**当前生效**的六分类权重与标准化参数原子写回主权重文件。

    为什么必须有这一步（缺陷 D-01 / P0）：
    过去 /api/flow/calibrate 热切换只改内存，新权重仅以
    ``checkpoints/flow_local_<ts>.pth`` 的形式留档，而**没有任何代码加载那份文件**。
    于是「校准成功」只活到下一次重启：重启后加载的还是 flow_multiclass.pth 里的旧模型，
    而 flow_env_calib.npz 又把标准化参数换掉了，三者各说一套，指标再也无法复现。

    这里做三件事：
      1. 把原主权重备份成 flow_multiclass.pth.bak.<时间戳>（可回滚，不销毁原始产物）；
      2. 保留原 checkpoint 的全部元数据（feature_names / hidden_sizes / class_names ...），
         只替换 model_state_dict、normalization、temperature 与本次微调记录；
      3. 先写临时文件再 os.replace，避免写一半断电留下半个模型。
    """
    path = FLOW_MULTICLASS_PATH
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup = None
    try:
        base = {}
        if os.path.exists(path):
            base = torch.load(path, map_location='cpu', weights_only=False)
            backup = f'{path}.bak.{ts}'
            shutil.copy2(path, backup)
        payload = dict(base)
        payload['model_state_dict'] = {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}
        payload['normalization'] = {
            'mean': np.asarray(mean, dtype=np.float64).tolist(),
            'std': np.asarray(std, dtype=np.float64).tolist(),
        }
        payload['temperature'] = float(temperature or 1.0)
        payload['input_dim'] = int(payload.get('input_dim', len(np.asarray(mean).ravel())))
        payload['num_classes'] = int(payload.get('num_classes', len(class_names)))
        payload['class_names'] = list(payload.get('class_names') or list(flow_multiclass_meta['class_names']))
        payload['feature_names'] = list(payload.get('feature_names') or list(flow_multiclass_meta['feature_names']))
        payload['hidden_sizes'] = tuple(payload.get('hidden_sizes', (256, 128)))
        for k, v in (extra or {}).items():
            payload[k] = v
        payload['persisted_at'] = ts

        tmp = f'{path}.tmp.{ts}'
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return {
            'ok': True,
            'path': os.path.relpath(path, BASE_DIR),
            'backup': os.path.relpath(backup, BASE_DIR) if backup else None,
            'bytes': os.path.getsize(path),
            'persisted_at': ts,
        }
    except Exception as e:  # noqa: BLE001
        return {'ok': False, 'error': str(e), 'path': os.path.relpath(path, BASE_DIR)}


@app.post("/api/flow/calibrate")
async def calibrate_flow_model(
    file: UploadFile = File(...),
    epochs: int = Form(8, description="微调轮数"),
    lr: float = Form(1e-3, description="学习率"),
    holdout: float = Form(0.2, description="留出多少比例做评估"),
    activate: bool = Form(False, description="指标不降时是否热切换到新权重"),
    adapt_normalization: bool = Form(
        False,
        description="是否用本机流量统计量替换标准化参数（会改变类中心坐标系，"
                    "需要重跑 tools/build_flow_centers.py 才能恢复未知攻击判据）"),

    session_user: str = Depends(require_session),
):
    """用目标环境上传的标注 CSV 微调流特征模型，产出本环境专用权重。

    与 /analyze-flow 的分工：那条是「用现成模型判」，这条是「让模型适应这个环境」。
    换环境后先用几百条标注跑一次校准，检出能力才会回到可用水平。
    """
    if flow_multiclass is None or flow_multiclass_meta is None:
        return JSONResponse(status_code=503, content={
            "error": "Multiclass flow model is not loaded. Train it first: "
                     "python tools/train_multiclass.py --days 15-02,21-02,02-03"
        })

    suffix = os.path.splitext(file.filename or '')[1].lower()
    if suffix != '.csv':
        return JSONResponse(status_code=400, content={
            "error": f"Only .csv is accepted here, got '{suffix or '(none)'}'.",
        })

    import copy
    import io as _io
    import pandas as _pd

    start_time = time.time()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={
            "error": f"CSV exceeds the {MAX_UPLOAD_MB} MB limit.",
        })

    try:
        df = _pd.read_csv(_io.BytesIO(raw), low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Failed to parse CSV", "detail": str(e)})

    if 'Label' not in df.columns:
        return JSONResponse(status_code=400, content={
            "error": "校准数据必须带 Label 列（CICFlowMeter 的 CSV 格式）。",
            "first_columns": [str(c) for c in df.columns[:10]],
        })

    meta = flow_multiclass_meta
    feature_names = meta['feature_names']
    class_names = meta['class_names']
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return JSONResponse(status_code=400, content={
            "error": "CSV is missing required feature columns.",
            "missing": missing[:20],
            "missing_count": len(missing),
            "expected_count": len(feature_names),
        })

    block = df[feature_names].apply(_pd.to_numeric, errors='coerce')
    block = block.replace([np.inf, -np.inf], np.nan)
    keep = block.notna().all(axis=1)
    # 区分「本机实时流量」与「锚点样本」：前者用来搬 OOD 判据，后者保证每类都有梯度
    local_flag = None
    if '_live_local' in df.columns:
        local_flag = (_pd.to_numeric(df['_live_local'], errors='coerce')
                      .fillna(0).to_numpy().astype(np.int64) == 1)
    X_raw = block[keep].to_numpy(dtype=np.float32)
    raw_labels = df.loc[keep, 'Label']
    y_all = np.array([_flow_label_index(v, class_names) for v in raw_labels], dtype=np.int64)
    valid = y_all >= 0
    dropped = int((~keep).sum() + (~valid).sum())
    X_raw, y_all = X_raw[valid], y_all[valid]
    if local_flag is not None:
        local_flag = local_flag[keep.to_numpy()][valid]
    if len(X_raw) < 50:
        return JSONResponse(status_code=400, content={
            "error": "可用于校准的带标签行太少（至少 50 行，建议 500~1000 行）。",
            "usable_rows": int(len(X_raw)),
            "dropped_rows": dropped,
        })

    Xn = ((X_raw - meta['mean'].reshape(1, -1)) / meta['std'].reshape(1, -1)).astype(np.float32)

    # 分层留出：每类都切一点进评估集，否则少数类在评估里缺位
    rng = np.random.default_rng(42)
    eval_parts, fit_parts = [], []
    for cls in np.unique(y_all):
        idx = np.where(y_all == cls)[0]
        rng.shuffle(idx)
        n_eval = max(1, int(len(idx) * holdout)) if len(idx) > 4 else 0
        eval_parts.append(idx[:n_eval])
        fit_parts.append(idx[n_eval:])
    eval_idx = np.concatenate(eval_parts) if eval_parts else np.array([], dtype=int)
    fit_idx = np.concatenate(fit_parts) if fit_parts else np.array([], dtype=int)
    if len(eval_idx) == 0 or len(fit_idx) < 20:
        return JSONResponse(status_code=400, content={"error": "样本量不足以拆分训练/评估集。"})

    X_fit, y_fit = Xn[fit_idx], y_all[fit_idx]
    X_eval, y_eval = Xn[eval_idx], y_all[eval_idx]

    def _predict(model, X):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                out.append(model(torch.from_numpy(X[s:s + 1024])).argmax(dim=1).numpy())
        return np.concatenate(out) if out else np.array([], dtype=np.int64)

    baseline_model = copy.deepcopy(flow_multiclass).to('cpu')
    before = _flow_metrics(y_eval, _predict(baseline_model, X_eval), class_names)

    tuned = copy.deepcopy(flow_multiclass).to('cpu')
    tuned.train()
    opt = torch.optim.Adam(tuned.parameters(), lr=lr)
    counts = np.bincount(y_fit, minlength=len(class_names)).astype(np.float64)
    weights = counts.sum() / (len(class_names) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    Xt, yt = torch.from_numpy(X_fit), torch.from_numpy(y_fit)
    history = []
    for _epoch in range(max(1, min(int(epochs), 50))):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for s in range(0, len(perm), 64):
            idx = perm[s:s + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = criterion(tuned(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(round(total / len(Xt), 4))

    after = _flow_metrics(y_eval, _predict(tuned, X_eval), class_names)

    ts = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(BASE_DIR, 'checkpoints', f'flow_local_{ts}.pth')
    torch.save({
        'model_state_dict': tuned.state_dict(),
        'input_dim': len(feature_names),
        'hidden_sizes': (256, 128),
        'num_classes': len(class_names),
        'class_names': class_names,
        'feature_names': feature_names,
        'normalization': {'mean': meta['mean'].tolist(), 'std': meta['std'].tolist()},
        'temperature': meta['temperature'],
        'local_finetune': {
            'created_at': ts,
            'source_file': file.filename,
            'usable_rows': int(len(X_raw)),
            'fit_samples': int(len(X_fit)),
            'eval_samples': int(len(X_eval)),
            'epochs': int(epochs),
            'lr': lr,
            'metrics_before': before,
            'metrics_after': after,
        },
    }, save_path)

    # 只有指标不降才切换：上传数据质量差时自动切换会把线上模型一起带坏。
    # 缺陷 D-15：这里原本是 `>=`，于是「完全没变」也被算成 improved 并触发切换 +
    # 落盘，日志上看起来像微调起了作用。改成严格大于，并把增量一起回传，
    # 让调用方能区分「真的变好」与「原地踏步」。
    _f1_delta = float(after['macro_f1']) - float(before['macro_f1'])
    improved = _f1_delta > 1e-9
    activated = False
    ood_adapt = None
    persisted = None
    if activate and improved:
        tuned.eval()
        flow_multiclass.load_state_dict(tuned.state_dict())
        flow_multiclass.to(device)
        flow_multiclass.eval()
        activated = True

        # 域适配：只微调分类权重、不动 OOD 判据的话，实时流量照样成片进「未知攻击」
        # （实测校准前后 unknown 比例 28.6% → 29.8%，等于没动）。这里用本机流量的
        # 一阶/二阶统计量替换标准化参数，把「离多远算远」这把尺子换成这个环境的。
        #
        # 缺陷 D-01（P0）：这一段过去是**无条件**执行的，而它改变的是整个六分类链路的
        # 坐标系 —— 类中心、未知攻击阈值、logs/flow_multiclass_metrics.json 的指标
        # 全部是在旧坐标系里标定的。结果：热切换当下看起来正常，重启后
        # 「新标准化 + 旧类中心」错位，未知攻击比例暴涨且指标不可复现。
        # 现在改为显式开关 adapt_normalization，默认只换权重、不动坐标系。
        if adapt_normalization and local_flag is not None and int(local_flag.sum()) >= 20:
            loc = np.asarray(X_raw[local_flag], dtype=np.float32)
            old_mean = np.asarray(meta['mean'], dtype=np.float32).reshape(1, -1)
            old_std = np.asarray(meta['std'], dtype=np.float32).reshape(1, -1)
            new_mean = loc.mean(axis=0).astype(np.float32)
            new_std = loc.std(axis=0).astype(np.float32)
            new_std = np.where(new_std < 1e-6, 1.0, new_std).astype(np.float32)
            meta['mean'] = new_mean
            meta['std'] = new_std
            # 类中心原先是在旧标准化坐标下算的：先反变换回原始尺度，再用新参数
            # 投回去，否则距离口径和新数据对不上，阈值就失去意义。
            center_note = 'unchanged'
            if flow_centers is not None and 'centers' in flow_centers:
                try:
                    centers_raw = np.asarray(flow_centers['centers'], dtype=np.float32) * old_std + old_mean
                    flow_centers['centers'] = ((centers_raw - new_mean) / new_std).astype(np.float32)
                    center_note = 'rescaled'
                except Exception as e:  # noqa: BLE001
                    print('center rescale failed:', e)
            try:
                np.savez_compressed(
                    os.path.join(os.path.dirname(save_path), 'flow_env_calib.npz'),
                    mean=new_mean, std=new_std,
                )
            except Exception as e:  # noqa: BLE001
                print('env calib save failed:', e)
            # 中心被搬到新坐标系后，原来的距离分位数（p99/p99.9）不再对应新尺度 ——
            # 距离在 σ 不同的空间里是加权欧氏距离，不能用单标量缩放修正。
            # 因此这里把 OOD 支路显式标记为 stale 并关掉，直到重跑 build_flow_centers。
            if flow_centers is not None:
                flow_centers['space_matches_active'] = False
            ood_adapt = {
                'local_rows': int(local_flag.sum()),
                'mean_shift_l1': float(np.abs(new_mean - old_mean.ravel()).mean()),
                'std_ratio_median': float(np.median(new_std / np.maximum(old_std.ravel(), 1e-6))),
                'centers': center_note,
                'distance_gate': 'disabled_until_rebuild',
                'rebuild_command': 'python tools/build_flow_centers.py',
            }

        # ---- 缺陷 D-01 的落盘部分：热切换必须同时改磁盘，否则重启即漂移 ----
        # 过去只把新权重存成 checkpoints/flow_local_<ts>.pth（没有任何代码加载它），
        # 主权重 checkpoints/flow_multiclass.pth 一直停在旧模型，重启后等于白校准。
        try:
            persisted = _persist_flow_multiclass_checkpoint(
                flow_multiclass,
                mean=meta['mean'],
                std=meta['std'],
                temperature=meta['temperature'],
                extra={
                    'local_finetune': {
                        'created_at': ts,
                        'source_file': file.filename,
                        'usable_rows': int(len(X_raw)),
                        'fit_samples': int(len(X_fit)),
                        'eval_samples': int(len(X_eval)),
                        'epochs': int(epochs),
                        'lr': lr,
                        'metrics_before': before,
                        'metrics_after': after,
                    },
                },
            )
        except Exception as e:  # noqa: BLE001
            print('persist flow checkpoint failed:', e)
            persisted = {'ok': False, 'error': str(e)}

    return {
        "file": file.filename,
        "usable_rows": int(len(X_raw)),
        "dropped_rows": dropped,
        "fit_samples": int(len(X_fit)),
        "eval_samples": int(len(X_eval)),
        "epochs": int(epochs),
        "loss_history": history,
        "before": before,
        "after": after,
        "improved": bool(improved),
        # 缺陷 D-15：把增量单列出来，「没变」与「变好」才区分得开
        "macro_f1_delta": round(_f1_delta, 4),
        "improved_rule": "macro_f1_after > macro_f1_before（严格大于，相等视为未改进）",
        "activated": activated,
        "ood_adapt": ood_adapt,
        "checkpoint": os.path.relpath(save_path, BASE_DIR),
        # 归档权重（仅供审计/回滚），**不是**当前生效的模型
        "checkpoint_role": "archive",
        "active_checkpoint": (persisted or {}).get('path', 'checkpoints/flow_multiclass.pth'),
        "persisted": persisted,
        "execution_time": f"{time.time() - start_time:.2f}s",
    }


# ---------------------------------------------------------------- 复核辅助（本地 LLM）
# 定位：这是「辅助解释」，不是「二次判决」。
# 实验已经证明判别力全在 78 维数字里（跨天时检出率 0.00%，AUC≈0.5），把同样的
# 数字转成文字喂给 LLM，信息量一点没变——它不可能比训练过的模型判得更准。所以
# 这里只让它做人擅长、模型不擅长的事：把「这条为什么被标可疑」讲成人话。
# 判定链路（置信度、风险档位、是否未知）完全不受这段影响，纯只读。
LLM_BASE_URL = os.environ.get('DLIDS_LLM_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
LLM_MODEL = os.environ.get('DLIDS_LLM_MODEL', '')
LLM_TIMEOUT = float(os.environ.get('DLIDS_LLM_TIMEOUT', '60'))

# LM Studio 的 Local Server 端口是可以改的（实测这台机器上就不是默认的 1234，
# 而是 11451）。只认一个写死的端口，用户换个端口就得改环境变量重启后端。
# 所以没有显式配置时，在几个常见端口间自动找一个能响应的，结果缓存一会儿，
# 避免每次点「AI 分析」都去轮询四个端口。
LLM_CANDIDATE_BASE_URLS = [
    'http://127.0.0.1:1234/v1',
    'http://127.0.0.1:11451/v1',
    'http://127.0.0.1:8080/v1',
    'http://127.0.0.1:5000/v1',
]
_LLM_RESOLVED = {'url': '', 'at': 0.0}
_LLM_RESOLVE_TTL = 20.0


def _llm_base_url() -> str:
    """返回当前可用的 OpenAI 兼容接口地址（显式配置优先，否则自动探测）。"""
    import time as _time

    if os.environ.get('DLIDS_LLM_URL'):
        return LLM_BASE_URL
    now = _time.time()
    if _LLM_RESOLVED['url'] and (now - _LLM_RESOLVED['at']) < _LLM_RESOLVE_TTL:
        return _LLM_RESOLVED['url']
    import urllib.request

    for base in LLM_CANDIDATE_BASE_URLS:
        try:
            with urllib.request.urlopen(f'{base}/models', timeout=1.5) as resp:
                if resp.status == 200:
                    _LLM_RESOLVED['url'] = base
                    _LLM_RESOLVED['at'] = now
                    return base
        except Exception:
            continue
    _LLM_RESOLVED['url'] = LLM_CANDIDATE_BASE_URLS[0]
    _LLM_RESOLVED['at'] = now
    return _LLM_RESOLVED['url']

# 最近一次 CSV 分析的上下文：只留明细里出现过的行 + 批次中位数。
# 有这几个数，LLM 才能说出「这条和同批其它流比，哪里不对」——光给一条流的
# 绝对数值，任何模型都判断不了它是偏高还是正常。
_LAST_FLOW_BATCH = None


def _llm_probe() -> tuple:
    """探测本地 LLM 服务（LM Studio / Ollama 的 OpenAI 兼容接口）。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f'{_llm_base_url()}/models', timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        names = [m.get('id') for m in data.get('data', []) if m.get('id')]
        model = LLM_MODEL or (names[0] if names else '')
        return bool(model), model, names
    except Exception as e:
        return False, '', [f'{type(e).__name__}: {e}']


def _llm_chat(messages: list, max_tokens: int = 512) -> str:
    import urllib.request

    ok, model, _ = _llm_probe()
    if not ok:
        raise RuntimeError('本地 LLM 未就绪')
    body = json.dumps({
        'model': model,
        'messages': messages,
        'temperature': 0.1,
        'max_tokens': max_tokens,
        'stream': False,
    }).encode('utf-8')
    req = urllib.request.Request(
        f'{_llm_base_url()}/chat/completions',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    return ((payload.get('choices') or [{}])[0].get('message') or {}).get('content', '').strip()


def _build_explain_messages(row_index: int, detail: dict, batch: dict = None) -> list:
    import math

    # batch 参数让实时抓包也能复用这段提示词构造：那条链路的批次快照不在
    # _LAST_FLOW_BATCH 里（它是 CSV 上传时缓存的），必须由调用方直接传进来。
    batch = batch or _LAST_FLOW_BATCH
    feature_names = batch['feature_names']
    vec = batch['rows'].get(int(row_index))
    median = batch['batch_median']

    facts = [f"文件：{batch['file']}"]
    facts.append(f"模型判定：{detail.get('predicted')}，置信度 {float(detail.get('confidence') or 0):.2%}")
    facts.append(f"风险档位：{detail.get('tier')}")
    ood = detail.get('ood_distance')
    if ood is not None:
        # 阈值必须来自当前这次推理，不能写死：CSV 链路和实时链路共用同一套类中心，
        # 但阈值随重算会变，写死一个数字会让 LLM 说出与实际不符的参照。
        _thr = detail.get('ood_threshold')
        if _thr is not None:
            facts.append(f"离群距离：{float(ood):.2f}（训练分布 p99.9 = {float(_thr):.2f}，越大越不像已学过的类）")
        else:
            facts.append(f"离群距离：{float(ood):.2f}（越大越不像已学过的类）")
    if detail.get('is_unknown'):
        facts.append("该行被标为「未知攻击」：离最近的类中心过远。")
    probs = detail.get('probabilities') or {}
    if probs:
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
        facts.append("概率分布：" + " / ".join(f"{k} {float(v):.2%}" for k, v in top))

    if vec is not None:
        scored = []
        for i, name in enumerate(feature_names):
            m = float(median[i])
            v = float(vec[i])
            if not math.isfinite(v):
                continue
            # 中位数很小时「相对偏差」会失控：中位数 0.5 → 本行 88280 会被算成
            # 十七万倍。这种数字对 LLM 毫无意义，只会把长尾当成爆炸式异常。
            # 用「中位数与数值本身量级的较大者」做尺度，并给倍数设上限。
            scale = max(abs(m), abs(v) * 0.01, 1e-6)
            dev = min(abs(v - m) / scale, 999.0)
            scored.append((dev, name, m, v))
        scored.sort(key=lambda t: -t[0])
        # 只列真正偏离的维度：偏差不到 0.5 倍的排进来只会当噪声，
        # 让 LLM 把注意力浪费在和批次中位数几乎一样的列上。
        picked = [t for t in scored if t[0] > 0.5][:6]
        if picked:
            facts.append("偏离批次中位数最大的特征：")
            for dev, name, m, v in picked:
                if abs(m) <= 1e-3:
                    facts.append(f"  {name}: 同批多数流为 0，本行 {v:.4g}")
                elif dev >= 50:
                    facts.append(f"  {name}: 批次中位数 {m:.4g} → 本行 {v:.4g}（50 倍以上）")
                else:
                    facts.append(f"  {name}: 批次中位数 {m:.4g} → 本行 {v:.4g}（{dev:.1f} 倍）")
        else:
            facts.append("各特征与批次中位数基本一致，没有明显偏离的维度。")

    system = (
        "你是一名网络流量安全分析师，正在协助人工复核。必须遵守以下约束：\n"
        "1. 你只能看到用户消息里列出的数据。不要提及任何未列出的特征名或数值，"
        "不要推测协议细节、端口用途、攻击工具或数据包内容。\n"
        "2. 输出纯文本：不要用 markdown、不要加粗、不要编号列表、不要小标题。\n"
        "3. 总共不超过 3 句，不要重复罗列已经给出的数字。\n"
        "4. 不要输出「误报」「真实攻击」这类结论词。是否存在攻击由模型判定和人工复核决定，"
        "你只解释这条流的特征偏离程度是支持还是削弱了模型给出的判定。"
    )
    user = (
        "【本次分类结果与特征】\n" + "\n".join(facts) +
        "\n\n请按顺序用三句话回答：\n"
        "第一句：最值得注意的偏离，必须引用上面给出的某个特征名。\n"
        "第二句：这些偏离是否足以支持上面的模型判定；若不足以支持，说明理由"
        "（例如偏离只是分布中位数附近的抖动，或该取值在模型判定的类别里本来就很常见）。\n"
        "第三句：人工复核时优先核对什么。"
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


@app.get('/api/flow/llm-status')
def flow_llm_status(session_user: str = Depends(require_session)):
    """本地 LLM 是否就绪。前端据此决定要不要显示「AI 分析」按钮。"""
    ok, model, names = _llm_probe()
    return {
        'available': ok,
        'base_url': _llm_base_url(),
        'model': model,
        'models': names if ok else [],
        'hint': '' if ok else f'在 LM Studio 里打开 Server（默认 {LLM_BASE_URL}），或设置 DLIDS_LLM_URL',
        'has_batch': _LAST_FLOW_BATCH is not None,
    }


@app.post('/api/flow/explain')
async def explain_flow_row(request: Request, session_user: str = Depends(require_session)):
    """让本地 LLM 解释某一条分类明细。只读，不改判定结果。"""
    if _LAST_FLOW_BATCH is None:
        return JSONResponse(status_code=409, content={
            'error': '还没有可解释的分类结果，请先上传 CSV 完成一次分类。',
        })

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={'error': '请求体必须是 JSON'})

    try:
        row_index = int(payload.get('row_index', -1))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={'error': 'row_index 必须是整数'})

    if row_index not in _LAST_FLOW_BATCH['rows']:
        return JSONResponse(status_code=400, content={
            'error': f'行号 {row_index} 不在最近一次分析的明细里（可能重新上传过数据）。',
        })

    ok, model, names = _llm_probe()
    if not ok:
        return JSONResponse(status_code=503, content={
            'error': '本地 LLM 未就绪',
            'hint': f'在 LM Studio 里打开 Server（默认 {LLM_BASE_URL}），或设置环境变量 DLIDS_LLM_URL',
            'detail': names[:3],
        })

    detail = payload.get('detail') or {}
    start = time.time()
    try:
        text = _llm_chat(_build_explain_messages(row_index, detail))
    except Exception as e:
        return JSONResponse(status_code=502, content={'error': 'LLM 调用失败', 'detail': str(e)})

    return {
        'row_index': row_index,
        'model': model,
        'content': text or '（模型没有返回内容）',
        'elapsed': f'{time.time() - start:.2f}s',
        'disclaimer': 'LLM 辅助分析，仅供参考，不参与判定',
    }


@app.post('/api/live/explain')
async def explain_live_flow(request: Request, session_user: str = Depends(require_session)):
    """对实时抓包里的一条流做 LLM 辅助解释。

    与 `/api/flow/explain` 的差别：那条读的是 CSV 上传时缓存的批次快照，
    这条现算——实时链路的流在 `live_flow_tracker` 里活着，没有持久化批次。

    只解释、不判定：六分类模型给类型与置信度，LLM 说明「为什么这条值得看」。
    """
    tracker = live_flow_tracker
    if tracker is None:
        return JSONResponse(status_code=409, content={
            'error': '实时抓包尚未启动，没有可解释的流',
        })
    if flow_multiclass is None or flow_multiclass_meta is None:
        return JSONResponse(status_code=503, content={
            'error': '六分类模型未加载，无法解释',
        })

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={'error': '请求体必须是 JSON'})

    src_ip = str(payload.get('src_ip') or '').strip()
    dst_ip = str(payload.get('dst_ip') or '').strip()
    try:
        src_port = int(payload.get('src_port') or 0)
        dst_port = int(payload.get('dst_port') or 0)
        proto = int(payload.get('proto') or 6)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={'error': '端口与协议号必须是整数'})
    if not src_ip or not dst_ip:
        return JSONResponse(status_code=400, content={'error': '缺少 src_ip / dst_ip'})

    # tracker 的 key 以「IP 字典序小的一侧」为前向，所以两个方向都要试：
    # 用户看到的记录里方向可能是反的（服务端 → 客户端）。
    want = {
        (src_ip, src_port, dst_ip, dst_port, proto),
        (dst_ip, dst_port, src_ip, src_port, proto),
    }
    try:
        flows = tracker.snapshot(active_only=False)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={'error': '读取实时流失败', 'detail': str(e)})

    matched = [item for item in flows if item[0] in want]
    if not matched:
        return JSONResponse(status_code=404, content={
            'error': '这条流已从跟踪表里过期或从未被解析',
            'hint': '实时流只在收到包后存活若干秒；抓包停止后过久就无法再解释',
            'active_flows': len(flows),
        })

    try:
        from src.preprocessing.flow_features import features_to_matrix

        _all_flows, all_matrix = features_to_matrix(flows)
        _one, one_matrix = features_to_matrix(matched)
        summary, detail = _classify_flow_matrix(one_matrix, confidence_high=0.90, confidence_low=0.50)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={'error': '六分类推理失败', 'detail': str(e)})
    if detail is None:
        return JSONResponse(status_code=500, content={'error': '六分类推理未返回结果'})

    class_names = list(flow_multiclass_meta['class_names'])
    row = {
        'row': 0,
        'predicted': class_names[int(detail['pred'][0])],
        'confidence': float(detail['conf'][0]),
        'tier': str(detail['tier'][0]),
        'is_unknown': bool(detail['is_unknown'][0]),
        'ood_distance': (None if detail['ood_distance'] is None
                         else float(detail['ood_distance'][0])),
        'ood_threshold': (None if detail.get('ood_threshold') is None
                          else float(detail['ood_threshold'])),
        'probabilities': {name: float(detail['prob'][0][i]) for i, name in enumerate(class_names)},
    }

    # 批次中位数用「当前所有活跃流」算，而不是这一条流自己：
    # 单条流的绝对值没有参照系，LLM 无从判断哪个数字算高。
    batch = {
        'file': '实时抓包（{}）'.format(capture_stats.get('interface') or '未命名网卡'),
        'rows': {0: one_matrix[0]},
        'batch_median': (np.median(all_matrix, axis=0) if len(all_matrix)
                         else np.median(one_matrix, axis=0)),
        'feature_names': list(flow_multiclass_meta['feature_names']),
        'class_names': class_names,
        'time': time.time(),
    }

    ok, model, names = _llm_probe()
    if not ok:
        return JSONResponse(status_code=503, content={
            'error': '本地 LLM 未就绪',
            'hint': f'在 LM Studio 里打开 Server（默认 {LLM_BASE_URL}），或设置环境变量 DLIDS_LLM_URL',
            'detail': names[:3],
        })

    start = time.time()
    try:
        text = _llm_chat(_build_explain_messages(0, row, batch))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=502, content={'error': 'LLM 调用失败', 'detail': str(e)})

    return {
        'flow': {
            'src_ip': src_ip, 'src_port': src_port,
            'dst_ip': dst_ip, 'dst_port': dst_port, 'proto': proto,
        },
        'classification': row,
        'model': model,
        'content': text or '（模型没有返回内容）',
        'elapsed': f'{time.time() - start:.2f}s',
        'disclaimer': 'LLM 辅助分析，仅供参考，不参与判定',
    }


def _collect_interface_rows():
    """收集可用网卡（name=友好名、value=抓包名），供 /api/interfaces 与启动校验共用。

    抽出来是为了让 ``/api/live/start`` 能校验 iface 是否是真实存在的网卡：
    过去启动环节完全不看这个参数，传一个不存在的名字也回 ``{"status":"started"}``，
    只有等 scapy 线程内部抛错、去 ``/api/live/debug`` 的 last_error 才能看到，
    前端拿到 200 只能显示「已开始」，用户以为在抓包，其实一个包都没收到。
    """
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

    return result


@app.get('/api/interfaces')
def list_interfaces(session_user: str = Depends(require_session)):
    """Return a list of available network interfaces (best-effort)."""
    try:
        return JSONResponse(content={'status': 'success', 'interfaces': _collect_interface_rows()})
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})



# 协议号归一表：flow_details 存的是 'TCP' 字符串，tracker 与 recent_flows 存的是数字。
_PROTO_NUM = {
    'icmp': 1, 'igmp': 2, 'tcp': 6, 'udp': 17, 'gre': 47, 'esp': 50, 'ah': 51,
    'icmpv6': 58, 'sctp': 132,
}


def _proto_num(v):
    """把 'TCP' / 'tcp' / 6 / '6' 统一成数字 6。

    两套明细的协议字段格式不同——上传链路的 ``flow_details`` 存 ``'TCP'``，
    实时链路的 ``recent_flows`` 与 tracker 键存数字。不归一的话按五元组 join 永远对不上，
    「二分类明确判正常 vs 六分类给出攻击类」的压制逻辑会静默失效（界面上照旧矛盾）。
    """
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    try:
        if isinstance(v, float):
            return int(v)
    except Exception:
        pass
    s = str(v).strip().lower()
    if s.isdigit():
        return int(s)
    return _PROTO_NUM.get(s, 0)


def _conn_sort_key(src, sport, dst, dport, proto):
    """双向五元组归一键：IP 字符串字典序小的一侧固定为前向。

    与 src/preprocessing/flow_features.py 的 _flow_key 同一条规则，另外把 proto
    归一成数字（见 ``_proto_num``），好让 tracker 的键、recent_flows 的条目、
    以及 flow_details 里的 'TCP' 字符串能直接比。
    """
    a, b = str(src), str(dst)
    ap, bp = int(sport or 0), int(dport or 0)
    p = _proto_num(proto)
    return (a, ap, b, bp, p) if a <= b else (b, bp, a, ap, p)


def _recent_conn_key(flow):
    """recent_flows 里一条记录的双向五元组键。"""
    return _conn_sort_key(flow.get('src'), flow.get('sport'),
                          flow.get('dst'), flow.get('dport'),
                          flow.get('proto') or '')


def _append_recent(flow):
    """给流分配单调递增 id 后放入环形缓冲，返回该 id。

    同一个连接（双向五元组）只保留最新一条，而不是每检测一次就追加一行。
    右栏「威胁类型分布」卡片统计的是 LiveFlowTracker 里每条连接的最新状态，原来这里是
    「每次检测结果一条」，一条长连接会反复出现、很快把缓冲占满：实测抓到 4675 个包时
    缓冲里 500 条记录仅覆盖到 21 个连接，而卡片说 69 条 —— 两个数字必然对不上。
    改成覆盖式之后，缓冲里天然就是「每条连接一条」，与卡片同口径。
    """
    global latest_flow_id
    with capture_lock:
        latest_flow_id += 1
        flow['id'] = latest_flow_id
        key = _recent_conn_key(flow)
        for idx, item in enumerate(recent_flows):
            if _recent_conn_key(item) == key:
                recent_flows.pop(idx)
                break
        recent_flows.insert(0, flow)
        if len(recent_flows) > MAX_RECENT:
            recent_flows.pop()
        # KPI 的「恶意流量 / 恶意占比」与记录表、辐射图、威胁卡片必须同一口径：
        # 缓冲里每条连接只保留最新一条，所以这里按当前缓冲重算，而不是在推理处按次数累加。
        capture_stats['malware_count'] = sum(1 for item in recent_flows if item.get('is_malicious'))
        return latest_flow_id


def _snapshot_recent_flows():
    """在锁内做浅拷贝，避免序列化过程中抓包线程继续修改列表。"""
    with capture_lock:
        return [dict(item) for item in recent_flows]


def _reset_live_threat_tracker():
    """开始新一轮抓包时重置六分类流表与缓存。"""
    global live_flow_tracker
    from src.preprocessing.flow_features import LiveFlowTracker

    if live_flow_tracker is None:
        live_flow_tracker = LiveFlowTracker()
    else:
        live_flow_tracker.reset()
    live_threat_cache['version'] = -1
    live_threat_cache['model_sig'] = None
    live_threat_cache['recent_sig'] = None
    live_threat_cache['summary'] = None
    live_threat_cache['meta'] = None
    live_threat_cache['flows'] = []
    # 新一轮抓包重新开始，上一轮的复核队列不该跟着延续
    live_review_queue.clear()
    return live_flow_tracker


@app.post('/api/live/start')
def start_live_capture(iface: str = Form(...), model_type: str = Form(ENSEMBLE_MODEL_TYPE), session_user: str = Depends(require_session)):
    global capture_thread, capture_stop_event, capture_stats, live_seen_overflow
    if capture_stats.get('running'):
        return JSONResponse(content={'status': 'already_running'})

    if not HAS_SCAPY:
        capture_stats['running'] = False
        return JSONResponse(status_code=500, content={'status': 'error', 'message': 'scapy not available on server'})

    # 缺陷（验收报告）：不校验 iface 就返回 started，错误被吞到 last_error 里。
    # 这里先确认这台机器上确实有这个名字（或友好名）的网卡，否则直接 400。
    # 取不到网卡列表时（异常/权限）放行，避免把「列不出来」误判成「不存在」。
    try:
        _ifaces = _collect_interface_rows()
    except Exception:
        _ifaces = []
    if _ifaces:
        _values = {str(r.get('value')) for r in _ifaces}
        _names = {str(r.get('name')) for r in _ifaces}
        if str(iface) not in _values and str(iface) not in _names:
            return JSONResponse(status_code=400, content={
                'status': 'error',
                'message': f"Interface '{iface}' is not available on this host.",
                'available': [{'name': r.get('name'), 'value': r.get('value')} for r in _ifaces],
            })

    # 模型名也要校验：未知名字不会报错，只会在推理分支里静默 return，
    # 表现为「抓包在跑、威胁卡有数，但实时明细与辐射图永久空白」——
    # 前端默认模型是 cascade，这里必须认得它，认不出就直接 400 说清楚。
    _requested_model = str(model_type or '').strip().lower()
    _known_model = _requested_model in ('', ENSEMBLE_MODEL_TYPE, CASCADE_MODEL_TYPE) or models.get(_requested_model) is not None
    if not _known_model:
        return JSONResponse(status_code=400, content={
            'status': 'error',
            'message': f"Unknown or unavailable model_type: {_requested_model}",
            'available': [k for k, v in models.items() if v is not None] + [ENSEMBLE_MODEL_TYPE, CASCADE_MODEL_TYPE],
        })

    capture_stop_event = threading.Event()
    capture_stats['last_error'] = None
    # 清除上一次会话的残留数据，确保新的实时会话快速启动
    try:
        with capture_lock:
            recent_flows.clear()
            live_session_buffers.clear()
            # 缺陷 D-29：会话计数改用「见过的五元组集合」，淘汰后重新出现不再重复计数
            live_seen_sessions.clear()
            live_seen_overflow = 0
            live_debug_logs.clear()
            capture_stats['malware_count'] = 0
            capture_stats['total_packets'] = 0
            capture_stats['session_count'] = 0
            capture_stats['sessions_tracked'] = 0
            capture_stats['session_count_approximate'] = False
            capture_stats['last_update'] = None
    except Exception:
        # 尽力而为，清理失败不影响启动
        pass
    # 六分类流特征表也一并重置：上一轮抓包的流不该混进这一轮的威胁分布
    try:
        _reset_live_threat_tracker()
    except Exception:
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
            # 尽力提取五元组
            has_ip = IP is not None and IP in pkt
            src = pkt[IP].src if has_ip else '0.0.0.0'
            dst = pkt[IP].dst if has_ip else '0.0.0.0'
            # 没有 IP 层的包（ARP、部分 mDNS/LLMNR 组播）会得到 0.0.0.0 → 0.0.0.0，
            # 这种"流"既算不出 78 维特征也没有分析价值，还会在实时记录里刷屏。
            if not has_ip:
                return
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
            # 与离线链路同口径：先掩掉 MAC/IP 头部再入会话缓冲（见 _mask_live_packet）。
            # live_flow_tracker 仍吃未掩码的 raw —— 它算的是包长/时长/标志位统计，
            # 需要真实头部（尤其是 IP 总长与传输层端口）。
            raw_masked = _mask_live_packet(raw)

            # 每个包都喂一次 78 维累加器：六分类靠的是包的时长/长度/标志位统计，
            # 不是 28x28 字节图，两者口径不同，所以这里独立于下面那段二分类聚合。
            try:
                tracker = live_flow_tracker
                if tracker is not None:
                    tracker.add_packet(raw, float(pkt.time))
            except Exception:
                pass

            # ---- 流级聚合（严重 6 修正）：按五元组把包攒成一条流，攒够才推理 ----
            session_key = (src, sport, dst, dport, proto)
            now_ts = time.time()
            session = live_session_buffers.get(session_key)
            if session is None:
                # 会话表过大时丢弃最旧的条目，避免长时间抓包导致内存持续增长
                if len(live_session_buffers) >= LIVE_MAX_TRACKED_SESSIONS:
                    oldest_key = min(live_session_buffers, key=lambda k: live_session_buffers[k]['updated'])
                    live_session_buffers.pop(oldest_key, None)
                # 按真实包边界累积（而不是把字节拼成一条连续流再硬切），
                # 这样在线与离线的 28x28 图逐行对齐同一批包。
                session = {'pkts': [], 'bytes': 0, 'packets': 0, 'updated': now_ts}
                live_session_buffers[session_key] = session
                # 缺陷 D-29：累计会话数按「见过的五元组」去重，不按建表次数。
                # 旧实现只在新建 session 时 +1，被 LIVE_MAX_TRACKED_SESSIONS 淘汰掉
                # 的连接再次出现会被当成新会话再加一次，长抓包的数字因此虚高。
                global live_seen_overflow
                if session_key not in live_seen_sessions:
                    if len(live_seen_sessions) < LIVE_SEEN_SESSION_CAP:
                        live_seen_sessions.add(session_key)
                    else:
                        live_seen_overflow += 1
                        capture_stats['session_count_approximate'] = True
                capture_stats['session_count'] = len(live_seen_sessions) + live_seen_overflow
                capture_stats['sessions_tracked'] = len(live_session_buffers)

            session['pkts'].append(raw_masked)
            session['bytes'] += len(raw_masked)
            session['packets'] += 1
            session['updated'] = now_ts

            enough_packets = session['packets'] >= LIVE_SESSION_MIN_PACKETS
            enough_bytes = session['bytes'] >= LIVE_SESSION_MAX_BYTES
            if not (enough_packets or enough_bytes):
                # 这条流还没攒够，等后面的包；此处直接返回，避免每包一次模型推理
                return
            if len(session['pkts']) > LIVE_SESSION_PACKET_CAP * 2:
                session['pkts'] = session['pkts'][-LIVE_SESSION_PACKET_CAP:]

            flow_packets = session['pkts']
            # 本轮送进模型的这一批包的累计字节数，必须在重置计数器之前取出来。
            # 下面 flow['bytes'] 原来是 `len(flow_bytes)`，而 flow_bytes 早已在重构里
            # 被删掉 —— 每轮推理都会在构造 flow 字典时抛 NameError，被外层 except 吃掉，
            # 结果 recent_flows 永远是空的：辐射图、实时明细、实时记录表全部停在空态。
            flow_bytes_len = session['bytes']
            session['pkts'] = []
            session['bytes'] = 0
            session['packets'] = 0

            # 用 FeatureExtractor 把包列表转成图片（与离线 pcap_to_sessions 同一个入口）
            extractor = _get_live_extractor()
            try:
                img_arr = extractor.process_session(flow_packets)
                # 代码审查第 36 条：同上，先规整成 uint8 再交给 Pillow。
                img_arr = np.ascontiguousarray(img_arr, dtype=np.uint8)
            except Exception:
                # 记录处理错误便于调试
                try:
                    msg = f"[live] FeatureExtractor.process_session failed, pkt_len={len(raw)}"
                    live_debug_logs.append(msg)
                    if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                    print(msg)
                except Exception:
                    pass
                return

            # 准备张量输入模型，同时生成缩略图和载荷直方图
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
                # 尽力而为，可视化失败不影响主流程
                try:
                    live_debug_logs.append('[live] failed to build image/ histogram for pkt')
                    if len(live_debug_logs) > MAX_DEBUG_LOGS: live_debug_logs.pop(0)
                except Exception:
                    pass
            img = Image.fromarray(img_arr)
            tensor_img = live_transform(img)
            if not isinstance(tensor_img, torch.Tensor):
                return
            tensor = torch.unsqueeze(tensor_img, 0).to(device)
            requested_model_type = str(model_type or '').strip().lower()
            # 「级联检测」在上传链路上是 CASCADE_MODEL_TYPE 这个别名（见 /analyze 的
            # is_cascade），但实时链路原先只认 ensemble_vote。前端默认模型就是 cascade，
            # 于是 models.get('cascade') 恒为 None → 下面 `if selected_model is None: return`
            # 静默返回：包数在涨、六分类也有数，可 recent_flows 一条都不产出，
            # 辐射图与实时明细就永久空白。这里把 cascade 归到同一个集成分支。
            run_ensemble = requested_model_type in ('', ENSEMBLE_MODEL_TYPE, CASCADE_MODEL_TYPE)
            vote_result = None

            if run_ensemble:
                with torch.no_grad():
                    # 实时链路没有 pcap 文件可做协议族统计，用同源权重；
                    # 开关关闭时 _weights_for_domain 返回 None（等权，行为与过去一致）。
                    ensemble_result = _vote_with_ensemble(
                        tensor, weights=_weights_for_domain('default'))
                    vote_result = ensemble_result['session_votes'][0]
                    malware_conf = float(np.mean(vote_result['malware_confidences'])) if vote_result['malware_confidences'] else 0.0
                    is_malicious = vote_result['predicted_label'] == 'Malicious'
            else:
                selected_model = models.get(requested_model_type)
                if selected_model is None:
                    return
                malware_idx = _find_class_index(['malware', 'malicious'], default_index=1 if len(class_names) > 1 else 0)
                threshold = MODEL_MALWARE_THRESHOLDS.get(requested_model_type, DEFAULT_MALWARE_THRESHOLD)
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
                # 该会话累计字节数，供前端实时明细列表「字节」列展示
                'bytes': flow_bytes_len,
                'malware_conf': round(malware_conf, 4),
                'is_malicious': bool(is_malicious),
                'model_type': ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type,
                'model_name': MODEL_DISPLAY_NAMES.get(ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type, ENSEMBLE_MODEL_TYPE if run_ensemble else requested_model_type),
                'captured_at': time.time(),
                # 附上可视化和直方图，与文件上传分析的输出格式一致
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

            # 恶意计数不再在这里按「推理次数」累加：长连接每攒满一批包就推理一次，
            # 按次数累加会把同一个连接重复计入，实测出现「会话 72 / 恶意 72 / 占比 100%」
            # 而明细里全是 Normal 的自相矛盾。改由 _append_recent 按缓冲重算（与记录表、
            # 辐射图、威胁卡同一口径：每条连接只算一次）。
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
            # 短超时循环抓包，方便及时响应停止请求
            while stop_event is not None and not stop_event.is_set():
                try:
                    # 用短超时让循环能频繁检查 stop_event
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


@app.post('/api/live/stats/reset')
def live_stats_reset(session_user: str = Depends(require_session)):
    """把实时统计整体清零。

    「清除日志」过去只清前端表格与威胁桶，但顶部 KPI 是后端 capture_stats 在算，
    下一轮 1.5 秒轮询又把同一个数字画回来，看起来像没清掉。
    抓包进行中调用也安全：只是丢弃已经统计过的会话，后续抓包照常计数。
    """
    global live_seen_overflow
    try:
        with capture_lock:
            recent_flows.clear()
            live_session_buffers.clear()
            live_seen_sessions.clear()
            live_seen_overflow = 0
            live_debug_logs.clear()
            capture_stats['malware_count'] = 0
            capture_stats['total_packets'] = 0
            capture_stats['session_count'] = 0
            capture_stats['sessions_tracked'] = 0
            capture_stats['session_count_approximate'] = False
            capture_stats['last_update'] = None
    except Exception as e:      # noqa: BLE001
        print('live stats reset failed:', e)
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})
    # 六分类流特征表也一起重置，否则威胁卡还挂着刚才那批连接
    try:
        _reset_live_threat_tracker()
    except Exception:      # noqa: BLE001
        pass
    return JSONResponse(content={'status': 'success', 'stats': dict(capture_stats)})


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
    # 六分类汇总：实时抓包的活跃流 -> 78 维 -> 六分类模型。算不出来就是 None，
    # 前端据此显示空态，而不是拿二分类的 is_malicious 假充六类。
    try:
        threat_summary, threat_meta = _live_threat_summary()
    except Exception as e:      # noqa: BLE001 - 轮询接口不能因为统计失败整页挂掉
        print('live threat summary failed:', e)
        threat_summary, threat_meta = None, None
    return JSONResponse(content={
        'status': 'success',
        'stats': dict(capture_stats),
        'recent': payload_recent,
        'flows': payload_flows,
        'latest_id': latest_id,
        'threat_summary': threat_summary,
        'threat_meta': threat_meta,
        # 逐流明细：卡片上每个分类点开看的就是它。
        # 和 threat_summary 同一批流、同一次推理，所以某一类的明细条数
        # 一定等于那一类的 count —— 两个数字对不上就没法解释给用户听。
        'threat_flows': list(live_threat_cache.get('flows') or []),
        'review_queue': _review_queue_payload(),
        'review_stats': _review_stats_payload(),
    })


@app.post('/api/live/review')
async def live_review(
    key: str = Form(...),
    verdict: str = Form(...),
    session_user: str = Depends(require_session),
):
    """记录一条人工复核结论，并把该流移出待复核队列。

    允许 key 不在队列里（可能已被上限淘汰、或中间重启过抓包）：复核结论本身仍值得留存，
    前端不该因为一次 404 就卡在原地。
    """
    if verdict not in ('malicious', 'benign', 'ignore'):
        raise HTTPException(status_code=400, detail="verdict must be malicious / benign / ignore")
    entry = _apply_review_verdict(key, verdict)
    return JSONResponse(content={
        'status': 'success',
        'reviewed': entry,
        'stats': _review_stats_payload(),
        'review_queue': _review_queue_payload(),
    })


@app.delete('/api/live/review')
def live_review_reset(session_user: str = Depends(require_session)):
    """清空已复核记录（待复核队列不动）。"""
    live_review_results.clear()
    return JSONResponse(content={'status': 'success', 'stats': _review_stats_payload()})


# ── 复核中心（/review）────────────────────────────────────────────────
# 上传检测与实时抓包产生的「待确认 / 人工复核」都落在同一份队列里，这个页面是唯一
# 的处置入口：既可以让本地 LLM 逐条给建议，也可以人工直接定案。两条路径最终都走
# _apply_review_verdict，结论口径一致。

_REVIEW_SYSTEM_PROMPT = (
    "你是网络入侵检测系统的人工复核分析师。系统用流特征六分类模型判了一条流量，"
    "但置信度不足以自动处置。请判断它更可能是真实攻击、正常流量，还是信息不足无法判断。"
    "只输出一行 JSON，不要任何解释文字，格式："
    '{"verdict":"malicious|benign|ignore","confidence":0.0,"reason":"不超过40字的中文理由"}'
)


def _build_review_messages(entry: dict) -> list:
    """把一条待复核流量整理成给 LLM 的事实清单。

    只喂它判不出来的那一层（预测类别、置信度、档位、离群距离、各类概率）。
    原始 78 维特征不塞进去 —— 模型的判断依据已经压在这一层，本地小模型看原始
    特征只会跑偏，还会把上下文撑爆。
    """
    source_label = {
        'live': '实时抓包',
        'upload': '上传检测（pcap）',
        'csv': '上传检测（流特征 CSV）',
    }
    facts = [f"来源：{source_label.get(str(entry.get('source') or 'live'), entry.get('source'))}"]
    if entry.get('file'):
        facts.append(f"文件：{entry.get('file')}")
    if entry.get('row') is not None:
        facts.append(f"CSV 行号：{entry.get('row')}")
    if entry.get('src_ip') or entry.get('dst_ip'):
        facts.append(
            "五元组：{}:{} -> {}:{} proto={}".format(
                entry.get('src_ip') or '?', entry.get('src_port') or '?',
                entry.get('dst_ip') or '?', entry.get('dst_port') or '?',
                entry.get('proto') or '?'))
        if entry.get('flows'):
            facts.append(f"包数：{entry.get('flows')}")
    facts.append(f"模型判定：{entry.get('predicted')}")
    facts.append(f"置信度：{float(entry.get('confidence') or 0):.2%}")
    tier_label = {'auto': '自动处置', 'review': '告警待确认', 'manual': '人工复核'}
    facts.append("分档：{}（越靠后越需要人确认）".format(
        tier_label.get(str(entry.get('tier') or ''), entry.get('tier'))))
    if entry.get('ood_distance') is not None:
        facts.append(f"离群距离：{entry.get('ood_distance')}（越大越远离所有已知类别）")
    if entry.get('is_unknown'):
        facts.append("系统标记：未知攻击（不属于训练集的任何已知攻击族）")
    probs = entry.get('probabilities')
    if isinstance(probs, dict) and probs:
        top = sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:4]
        facts.append("各类概率：" + '、'.join(f'{k} {float(v or 0):.2%}' for k, v in top))
    elif isinstance(probs, list) and probs:
        facts.append("各类概率：" + '、'.join(
            f"{p.get('label')} {float(p.get('value') or 0):.2%}" for p in probs[:4]))
    return [
        {'role': 'system', 'content': _REVIEW_SYSTEM_PROMPT},
        {'role': 'user', 'content': '请复核下面这条流量：\n' + '\n'.join(facts)},
    ]


def _parse_llm_verdict(text: str) -> tuple:
    """从 LLM 回复里抠出 (verdict, confidence, reason)。

    本地小模型经常不肯只吐 JSON（会先来一句「根据以上分析」），所以先按 JSON 解析，
    失败再退化成关键词匹配；都失败返回 None —— 绝不让它瞎猜一个结论去落库。
    """
    import re

    raw = str(text or '').strip()
    if not raw:
        return None, None, ''
    match = re.search(r'\{[^{}]*\}', raw, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            verdict = str(obj.get('verdict') or '').strip().lower()
            if verdict in REVIEW_VERDICTS:
                conf = obj.get('confidence')
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = None
                return verdict, conf, str(obj.get('reason') or '').strip()[:160]
    low = raw.lower()
    for verdict in REVIEW_VERDICTS:
        if verdict in low:
            return verdict, None, raw[:160]
    return None, None, raw[:160]


@app.get('/api/review/policy')
def review_policy_get(session_user: str = Depends(require_session)):
    """复核队列的默认处置策略：manual（人工）/ vote（6 模型投票）/ llm（LLM 判断）。"""
    return JSONResponse(content={
        'status': 'success',
        'policy': _review_policy_payload(),
        'stats': _review_stats_payload(),
    })


@app.post('/api/review/policy')
def review_policy_set(
    mode: str = Form(...),
    apply_existing: bool = Form(True),
    session_user: str = Depends(require_session),
):
    """切换默认处置策略；``apply_existing`` 为真时把当前队列里能自动判的一次性处理掉。

    切到 vote 会把队列里能自动定案的条目全部消化；切到 llm 只消化前
    ``REVIEW_AUTO_LLM_BATCH`` 条（LLM 逐条串行，一次全判会把本地模型占满）。
    """
    m = str(mode or '').strip().lower()
    if m not in REVIEW_POLICY_MODES:
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: " + " / ".join(REVIEW_POLICY_MODES))
    review_policy['mode'] = m
    review_policy['updated_at'] = time.time()
    handled = 0
    if apply_existing and m != 'manual':
        handled = _apply_review_policy(list(live_review_queue.keys()))
    return JSONResponse(content={
        'status': 'success',
        'policy': _review_policy_payload(),
        'handled': handled,
        'stats': _review_stats_payload(),
        'items': _review_queue_payload(),
    })


@app.get('/api/review/queue')
def review_queue_api(limit: int = 200, session_user: str = Depends(require_session)):
    """复核中心首屏：待复核队列（按可疑度降序）+ 统计 + 来源分布。"""
    return JSONResponse(content={
        'status': 'success',
        'items': _review_queue_payload(limit),
        'stats': _review_stats_payload(),
        'sources': _review_source_counts(),
        'queue_limit': LIVE_REVIEW_QUEUE_LIMIT,
    })


@app.get('/api/review/history')
def review_history_api(limit: int = 100, session_user: str = Depends(require_session)):
    """已复核结论，最新在前。"""
    items = sorted(
        live_review_results.values(),
        key=lambda v: float(v.get('reviewed_at') or 0.0),
        reverse=True,
    )
    return JSONResponse(content={
        'status': 'success',
        'items': items[:max(1, int(limit))],
        'stats': _review_stats_payload(),
    })


@app.post('/api/review/verdict')
def review_verdict_api(
    key: str = Form(...),
    verdict: str = Form(...),
    session_user: str = Depends(require_session),
):
    """人工定案。与 /api/live/review 等价，这里是复核中心的正式入口。"""
    if verdict not in REVIEW_VERDICTS:
        raise HTTPException(status_code=400, detail="verdict must be malicious / benign / ignore")
    entry = _apply_review_verdict(key, verdict)
    return JSONResponse(content={
        'status': 'success',
        'reviewed': entry,
        'stats': _review_stats_payload(),
        'items': _review_queue_payload(),
    })


@app.post('/api/review/verdict-batch')
def review_verdict_batch_api(
    keys: str = Form(...),
    verdict: str = Form(...),
    session_user: str = Depends(require_session),
):
    """批量定案。复核中心的多选操作走这里，一次请求处理整批，逐条记 reviewed_by='human'。

    keys 是逗号分隔的队列 key（key 里不含逗号：实时是 'src|ip:port|ip:port|proto'，
    CSV 是 'csv|文件#行号'，都安全）。
    """
    if verdict not in REVIEW_VERDICTS:
        raise HTTPException(status_code=400, detail="verdict must be malicious / benign / ignore")
    wanted = [item.strip() for item in str(keys or '').split(',') if item.strip()]
    if not wanted:
        raise HTTPException(status_code=400, detail='keys is empty')
    applied = 0
    missing = []
    for key in wanted:
        entry = _apply_review_verdict(key, verdict)
        if entry:
            applied += 1
        else:
            missing.append(key)
    return JSONResponse(content={
        'status': 'success',
        'reviewed': applied,
        'requested': len(wanted),
        'missing': missing[:20],
        'stats': _review_stats_payload(),
        'items': _review_queue_payload(),
    })


@app.delete('/api/review/queue')
def review_queue_clear(session_user: str = Depends(require_session)):
    """清空待复核队列（已复核结论保留，统计口径不变）。"""
    live_review_queue.clear()
    return JSONResponse(content={
        'status': 'success',
        'stats': _review_stats_payload(),
        'items': [],
    })


@app.post('/api/review/auto')
def review_auto_api(
    keys: str = Form(''),
    limit: int = Form(5),
    apply: bool = Form(False),
    session_user: str = Depends(require_session),
):
    """让本地 LLM 给出判断建议。

    ``keys`` 为空时按可疑度取前 ``limit`` 条；``apply=true`` 表示「委托给模型自主判断」，
    LLM 的结论直接落库并从队列移除；``apply=false`` 只给建议，等人工点确认。
    """
    ok, model, names = _llm_probe()
    if not ok:
        return JSONResponse(status_code=503, content={
            'error': '本地 LLM 不可用',
            'detail': '请先启动 LM Studio / Ollama 的 OpenAI 兼容接口，或用 DLIDS_LLM_URL 指向它。',
            'probe': list(names)[:3],
        })
    wanted = [k.strip() for k in str(keys or '').split(',') if k.strip()]
    if wanted:
        targets = [live_review_queue[k] for k in wanted if k in live_review_queue]
    else:
        targets = _review_queue_payload(max(1, min(int(limit), 20)))
    results = []
    for entry in targets:
        started = time.time()
        try:
            content = _llm_chat(_build_review_messages(entry), max_tokens=220)
        except Exception as e:  # noqa: BLE001 - 单条失败不该中断整批
            results.append({'key': entry.get('key'), 'error': str(e)})
            continue
        verdict, conf, reason = _parse_llm_verdict(content)
        results.append({
            'key': entry.get('key'),
            'verdict': verdict,
            'confidence': conf,
            'reason': reason,
            'model': model,
            'raw': content[:600],
            'elapsed': round(time.time() - started, 2),
        })
        if apply and verdict in REVIEW_VERDICTS:
            _apply_review_verdict(entry['key'], verdict)
    return JSONResponse(content={
        'status': 'success',
        'model': model,
        'applied': bool(apply),
        'results': results,
        'stats': _review_stats_payload(),
        'items': _review_queue_payload(),
    })


@app.get('/api/live/export-calib')
def export_live_calib(session_user: str = Depends(require_session)):
    """把当前实时抓到的流导出成一份可直接喂给 /api/flow/calibrate 的 CSV。

    实时流量没有标签，这里把本机抓到的流一律标 Benign —— 校准的语义就是「让模型认识
    这个环境的正常形状」，不是给它看攻击。但单类数据会让微调崩（类别权重把 Benign 压到
    1/6 以下，模型扭头全判攻击），所以再拼一份六类均衡样本当锚点，保证每类都有梯度。
    """
    if flow_multiclass_meta is None:
        return JSONResponse(status_code=503, content={"error": "Multiclass model not loaded."})
    tracker = live_flow_tracker
    if tracker is None:
        return JSONResponse(status_code=400, content={"error": "还没抓过包，先启动抓包再导出。"})

    import io as _io
    import pandas as _pd
    from src.preprocessing.flow_features import features_to_matrix

    try:
        flows = tracker.snapshot(active_only=False)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "snapshot failed", "detail": str(e)})
    if not flows:
        return JSONResponse(status_code=400, content={"error": "当前没有可用的流，多抓一会儿再导出。"})

    try:
        _used, matrix = features_to_matrix(flows)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "feature build failed", "detail": str(e)})
    if len(matrix) == 0:
        return JSONResponse(status_code=400, content={"error": "流特征为空，无法导出。"})

    feature_names = flow_multiclass_meta['feature_names']
    local = _pd.DataFrame(matrix, columns=feature_names)
    local['Label'] = 'Benign'
    # 标记来源：校准端点靠这一列认出「本机实时流量」，用它的统计量把 OOD 判据
    # （标准化参数 + 类中心）搬到这个环境；锚点行是 0，只负责保证每类都有梯度。
    local['_live_local'] = 1

    anchor_path = os.path.join(BASE_DIR, 'data', 'cse_csv', 'demo_mixed_6class.csv')
    frames = [local]
    anchor_rows = 0
    if os.path.isfile(anchor_path):
        try:
            anchor = _pd.read_csv(anchor_path, low_memory=False)
            anchor.columns = [str(c).strip() for c in anchor.columns]
            if 'Label' in anchor.columns and all(c in anchor.columns for c in feature_names):
                frames.append(anchor[feature_names + ['Label']])
                anchor_rows = int(len(anchor))
        except Exception as e:  # noqa: BLE001
            print('calib anchor load failed:', e)

    out = _pd.concat(frames, ignore_index=True)
    buf = _io.StringIO()
    out.to_csv(buf, index=False)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return Response(
        content=buf.getvalue().encode('utf-8'),
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename="live_calib_%s.csv"' % stamp,
            'X-Live-Rows': str(int(len(local))),
            'X-Anchor-Rows': str(int(anchor_rows)),
        },
    )


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
