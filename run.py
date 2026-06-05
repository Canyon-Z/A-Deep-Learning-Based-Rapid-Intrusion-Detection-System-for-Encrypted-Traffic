import argparse
import os
import sys
import webbrowser
from threading import Timer

# Ensure the current directory is added to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    webbrowser.open("http://127.0.0.1:8000/login")

def run_web():
    import uvicorn

    # Open browser after 1.5 seconds to give server time to start
    Timer(1.5, open_browser).start()

    # Run the FastAPI app using uvicorn
    # reload=True enables auto-reload on code changes
    uvicorn.run("web.backend.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密流量入侵检测系统启动脚本")
    parser.parse_args()
    run_web()
