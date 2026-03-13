import argparse
import uvicorn
import os
import sys
import webbrowser
from threading import Timer

# Ensure the current directory is added to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

def run_web():
    # Open browser after 1.5 seconds to give server time to start
    Timer(1.5, open_browser).start()

    # Run the FastAPI app using uvicorn
    # reload=True enables auto-reload on code changes
    uvicorn.run("web.backend.main:app", host="127.0.0.1", port=8000, reload=True)

def run_train():
    # TODO (组长): 调用训练脚本
    print("启动训练模式...")
    # from src.training.train import train_model
    # train_model(...)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密流量入侵检测系统启动脚本")
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'train'], help='运行模式: web (界面) 或 train (训练)')
    args = parser.parse_args()

    if args.mode == 'web':
        run_web()
    elif args.mode == 'train':
        run_train()
