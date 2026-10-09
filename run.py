"""
加密流量入侵检测系统 - 主入口

启动方式：
    python run.py          → 启动 Web 界面（默认）
    python run.py --mode train  → 命令行训练

Web 服务启动后自动打开浏览器访问 http://127.0.0.1:8000/login
"""
import argparse
import os
import sys
import webbrowser
from threading import Timer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    """自动打开浏览器到登录页面"""
    webbrowser.open("http://127.0.0.1:8000/login")


def run_web():
    """启动 FastAPI Web 服务，1.5 秒后自动打开浏览器"""
    import uvicorn

    Timer(1.5, open_browser).start()
    uvicorn.run("web.backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密流量入侵检测系统启动脚本")
    parser.parse_args()
    run_web()
