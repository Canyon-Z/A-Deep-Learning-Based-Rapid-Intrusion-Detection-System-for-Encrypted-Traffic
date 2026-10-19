"""训练收尾自动化：等 6 个模型全部训练完成，同步 Web 权重、生成新旧指标对比，可选拉起后端。

背景
----
``web/backend/main.py`` 加载主模型时按 ``['cnn_bilstm.pth', 'final_model.pth']`` 的顺序尝试，
而 ``src/training/run_training_all.py`` 里 CNN_BiLSTM 只写 ``checkpoints/final_model.pth``。
两者若是不同一次训练/不同数据划分的产物，网页显示的指标就与屏幕上跑的模型对不上，
所以整轮训练结束后必须把新权重同步成 Web 实际读取的那个文件名。

触发条件
--------
默认等训练日志出现结束语（``🎉 所有模型均已成功分开训练并保存至 checkpoints 文件夹！``），
即 6 个模型全部训完，而不是 CNN_BiLSTM 一收尾就动手。

安全约束
--------
1. 只有 ``final_model.pth`` 的 mtime **晚于本脚本启动时刻**时才同步；
   训练中途崩溃或尚未开始时直接跳过，绝不用旧权重覆盖现有 cnn_bilstm.pth。
2. 要求文件大小连续两次采样一致，避免复制到正在写入的半成品。
3. 同步前把现有 cnn_bilstm.pth 备份为 ``checkpoints/cnn_bilstm.pth.bak``。
4. 后端只在 ``--restart-backend`` 且 8000 端口空闲时拉起；脚本本身不会停止任何进程。

用法
----
    # 默认：等训练完成 -> 同步权重 -> 写对比报告
    python -u tools/sync_web_checkpoint.py --wait-seconds 28800 --poll 30

    # 额外在同步后拉起后端
    python -u tools/sync_web_checkpoint.py --wait-seconds 28800 --restart-backend
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
FINAL_CKPT = CKPT_DIR / "final_model.pth"
WEB_CKPT = CKPT_DIR / "cnn_bilstm.pth"
BACKUP_CKPT = CKPT_DIR / "cnn_bilstm.pth.bak"
TRAIN_LOG = ROOT / "logs" / "train_ustc_retrain.log"
COMPARE_SCRIPT = ROOT / "tools" / "compare_metrics.py"
COMPARE_REPORT = ROOT / "logs" / "metrics_compare_after_retrain.txt"
BACKEND_LOG = ROOT / "logs" / "backend_after_retrain.log"
RESULT_FILE = ROOT / "logs" / "finalize_result.txt"

TRAIN_DONE_MARKER = "所有模型均已成功分开训练并保存"
STABLE_SAMPLES = 2
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000


def log(message: str, echo: bool = True) -> None:
    """写一行带时间戳的进度：既打印到作业输出，也追加到结果文件。"""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    if echo:
        print(line, flush=True)
    try:
        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with RESULT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def read_log_tail(path: Path, window: int = 65536) -> str:
    """读日志尾部文本；文件不存在或不可读时返回空串。

    注意编码：训练日志是用 PowerShell 的 ``Tee-Object`` 写的，Windows PowerShell 5.1
    下默认落成 **UTF-16LE(带 BOM)**，而普通 ``>`` 重定向又会写成 UTF-8。
    早先这里写死 ``encoding='utf-8'``，读 UTF-16 日志时整篇解不出来，
    结束语标记永远匹配不到，守候就会一直空转 —— 所以这里要按 BOM 判断再解码。
    """
    if not path.exists():
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        # UTF-16：按字节截尾会切断码元，整篇解码后取尾部字符更稳
        return raw.decode("utf-16", errors="ignore")[-window:]

    chunk = raw[-window:] if len(raw) > window else raw
    for encoding in ("utf-8", "gbk"):
        try:
            return chunk.decode(encoding)
        except UnicodeDecodeError:
            continue
    return chunk.decode("utf-8", errors="ignore")


def training_progress() -> str:
    """返回一句进度描述，便于日志里看清等到哪一步了。"""
    text = read_log_tail(TRAIN_LOG)
    val_count = text.count("➜ [VAL]")
    done_models = text.count("训练完成，正在保存模型权重")
    return f"最近窗口 val 记录 {val_count}，已完成模型 {done_models}"


def wait_for_training_done(wait_seconds: float, poll_seconds: float) -> bool:
    """阻塞等待训练结束语出现。"""
    deadline = time.monotonic() + wait_seconds
    last_report = 0.0
    while time.monotonic() < deadline:
        if TRAIN_DONE_MARKER in read_log_tail(TRAIN_LOG):
            return True
        now = time.monotonic()
        if now - last_report > 1800:  # 每半小时记一次心跳，避免结果文件一片空白
            last_report = now
            log(f"守候中… {training_progress()}", echo=False)
        time.sleep(poll_seconds)
    return False


def stat_signature(path: Path) -> tuple[int, int] | None:
    """返回 (mtime_ns, size)，文件不存在时返回 None。"""
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def sync_weights(started_at_ns: int) -> bool:
    """把新的 final_model.pth 同步为 cnn_bilstm.pth。"""
    signature = stat_signature(FINAL_CKPT)
    if signature is None:
        log("final_model.pth 不存在，跳过同步")
        return False
    if signature[0] <= started_at_ns:
        log(
            "final_model.pth 的 mtime 早于本次守候启动时刻，说明没有新的训练产物，跳过同步"
        )
        return False

    for _ in range(10):
        first = stat_signature(FINAL_CKPT)
        time.sleep(5)
        if stat_signature(FINAL_CKPT) == first:
            break
    else:
        log("警告：final_model.pth 大小持续变化，仍按当前内容同步")

    if WEB_CKPT.exists():
        try:
            shutil.copyfile(WEB_CKPT, BACKUP_CKPT)
            log(f"已备份原 {WEB_CKPT.name} -> {BACKUP_CKPT.name}")
        except OSError as exc:
            log(f"备份失败，中止同步以免丢失现有权重：{exc}")
            return False

    try:
        shutil.copyfile(FINAL_CKPT, WEB_CKPT)
    except OSError as exc:
        log(f"同步失败：{exc}")
        return False

    source_size = stat_signature(FINAL_CKPT)
    target_size = stat_signature(WEB_CKPT)
    same = source_size is not None and target_size is not None and source_size[1] == target_size[1]
    log(
        "同步完成：{0} -> {1}（{2:.2f}MB，大小{3}）".format(
            FINAL_CKPT.name,
            WEB_CKPT.name,
            (target_size[1] if target_size else 0) / 1024 / 1024,
            "一致" if same else "不一致，请人工检查",
        )
    )
    return True


def run_metrics_compare() -> None:
    """跑一次新旧指标对比，结果写入日志文件。"""
    if not COMPARE_SCRIPT.exists():
        log("未找到 tools/compare_metrics.py，跳过指标对比")
        return
    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(COMPARE_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"指标对比执行失败：{exc}")
        return

    output = (completed.stdout or "") + (completed.stderr or "")
    try:
        COMPARE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with COMPARE_REPORT.open("w", encoding="utf-8") as handle:
            handle.write(output)
    except OSError as exc:
        log(f"对比报告写入失败：{exc}")
        return

    summary_lines = [line for line in output.splitlines() if line.strip().startswith(("测试准确率", "恶意检出率", "漏报率"))]
    log(f"指标对比已写入 {COMPARE_REPORT.relative_to(ROOT)}")
    for line in summary_lines:
        log("  " + line.strip())


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.6)
        return sock.connect_ex((host, port)) == 0


def start_backend() -> None:
    """在 8000 端口空闲时拉起后端服务（独立进程，输出写入日志文件）。"""
    if port_in_use(BACKEND_HOST, BACKEND_PORT):
        log(f"{BACKEND_HOST}:{BACKEND_PORT} 已有服务在监听，跳过启动")
        return

    BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        handle = BACKEND_LOG.open("a", encoding="utf-8")
    except OSError as exc:
        log(f"无法打开后端日志文件：{exc}")
        return

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "uvicorn",
                "web.backend.main:app",
                "--host",
                BACKEND_HOST,
                "--port",
                str(BACKEND_PORT),
            ],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError as exc:
        log(f"后端启动失败：{exc}")
        handle.close()
        return

    log(f"后端已拉起 pid={process.pid}，日志 {BACKEND_LOG.relative_to(ROOT)}")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if port_in_use(BACKEND_HOST, BACKEND_PORT):
            log(f"后端就绪：http://{BACKEND_HOST}:{BACKEND_PORT}/login")
            return
        if process.poll() is not None:
            log(f"后端进程提前退出（exit={process.returncode}），请查看 {BACKEND_LOG.name}")
            return
        time.sleep(2)
    log("后端 60 秒内未监听端口，请查看日志确认")


def main() -> int:
    parser = argparse.ArgumentParser(description="等待训练完成并同步 Web 端模型权重")
    parser.add_argument("--wait-seconds", type=float, default=28800.0, help="最长等待秒数，默认 8 小时")
    parser.add_argument("--poll", type=float, default=30.0, help="轮询间隔秒数，默认 30 秒")
    parser.add_argument("--restart-backend", action="store_true", help="同步完成后拉起后端服务（端口空闲时）")
    args = parser.parse_args()

    started_at_ns = time.time_ns()
    log("守候启动：等待训练日志出现全部模型完成标记")
    log(f"训练日志：{TRAIN_LOG}")

    if not wait_for_training_done(args.wait_seconds, args.poll):
        log("等待超时，训练尚未出现结束语，本次不做后续处理")
        return 1

    log("检测到训练结束语，开始收尾")

    synced = sync_weights(started_at_ns)
    run_metrics_compare()

    if args.restart_backend:
        start_backend()
    else:
        log("未指定 --restart-backend，后端保持现状（需重启才会加载新权重）")

    # 附带打印新旧核心指标的 JSON 摘要，便于直接核对
    try:
        with (CKPT_DIR / "model_metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for name, values in metrics.items():
            if isinstance(values, dict):
                log(
                    "  {0}: test_accuracy={1} detection_rate={2} fnr={3}".format(
                        name,
                        values.get("test_accuracy"),
                        values.get("detection_rate"),
                        values.get("fnr"),
                    )
                )
    except (OSError, json.JSONDecodeError):
        pass

    return 0 if synced else 2


if __name__ == "__main__":
    sys.exit(main())
