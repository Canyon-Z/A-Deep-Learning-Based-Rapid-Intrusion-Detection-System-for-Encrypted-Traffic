"""
加密流量入侵检测系统 - 主入口

启动方式：
    python run.py                 → 启动 Web 界面（默认）
    python run.py --reload        → 启动 Web 界面并开启 uvicorn 热重载（仅开发用）
    python run.py --mode train    → 命令行训练（把其余参数原样转发给 run_training_all.py）

Web 服务启动后自动打开浏览器访问 http://127.0.0.1:8000/login
主机与端口可用环境变量 DLIDS_HOST / DLIDS_PORT 覆盖（与 web/backend/main.py 同一套变量）。
"""
import argparse
import os
import sys
import webbrowser
from threading import Timer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _env_port(default=8000):
    """解析 DLIDS_PORT，非法值回退默认端口（避免直接 int() 抛错）。"""
    raw = os.environ.get('DLIDS_PORT')
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        print(f"Warning: DLIDS_PORT={raw!r} is not an integer, falling back to {default}.")
        return default


WEB_HOST = os.environ.get('DLIDS_HOST', '127.0.0.1')
WEB_PORT = _env_port()


def open_browser():
    """自动打开浏览器到登录页面"""
    webbrowser.open(f"http://{WEB_HOST}:{WEB_PORT}/login")


def run_web(reload=False):
    """启动 FastAPI Web 服务，1.5 秒后自动打开浏览器。

    reload 默认关闭：热重载会在后台常驻一个文件监视进程并重启应用，
    只适合开发时使用，日常启动不需要。
    """
    import uvicorn

    Timer(1.5, open_browser).start()
    uvicorn.run("web.backend.main:app", host=WEB_HOST, port=WEB_PORT, reload=reload)


def run_train(train_argv):
    """命令行训练：把 --mode train 之后的参数原样交给 run_training_all.py 的 main()。"""
    sys.argv = [sys.argv[0]] + list(train_argv)
    from src.training.run_training_all import main as train_main

    train_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密流量入侵检测系统启动脚本")
    parser.add_argument(
        '--mode',
        choices=['web', 'train'],
        default='web',
        help="web=启动 Web 界面（默认）；train=命令行训练",
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help="开发用：开启 uvicorn 热重载（默认关闭）",
    )
    # 训练相关的参数（--epochs/--dataset/--device 等）由 run_training_all.py 自己解析，
    # 这里用 parse_known_args 把它们原样透传，避免在入口重复维护一份参数表。
    args, passthrough = parser.parse_known_args()
    if args.mode == 'train':
        run_train(passthrough)
    else:
        run_web(reload=args.reload)
