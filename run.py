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
    # Call the training script
    print("启动训练模式...")
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.preprocessing.data_loader import get_dataloaders
    from src.training.train import train_model
    # Switch to CNN_BiLSTM
    from src.models.cnn_bilstm import CNN_BiLSTM

    # Configuration
    DATA_ROOT = "data/processed"  # Path to your processed images folder
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 1. Load Data
    print(f"正在从目录 {DATA_ROOT} 加载数据...")
    train_loader, val_loader, test_loader, class_idx = get_dataloaders(
        data_root=DATA_ROOT, 
        batch_size=BATCH_SIZE
    )
    
    if train_loader is None:
        print("错误: 未能在 data_root 中找到有效数据，请检查路径结构。")
        return

    num_classes = len(class_idx)
    class_names = list(class_idx.keys())
    print(f"检测到类别: {class_names} (共 {num_classes} 类)")

    # 2. Initialize Model (Using CNN_BiLSTM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # Instantiate CNN_BiLSTM
    model = CNN_BiLSTM(num_classes=num_classes, hidden_dim=64)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Config for logging
    config = {
        "model": "CNN_BiLSTM",
        "input_shape": "Seq(784)",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "classes": class_names,
        "preprocessing": "Session + All Layers + Truncate(784)",
        "num_epochs": NUM_EPOCHS
    }

    # 5. Start Training
    print("开始模型训练...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=str(device),
        config=config
    )
    
    # Save the final model (with metadata)
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'config': config
    }, save_path)
    print(f"训练完成，模型及元数据已保存至 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密流量入侵检测系统启动脚本")
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'train'], help='运行模式: web (界面) 或 train (训练)')
    args = parser.parse_args()

    if args.mode == 'web':
        run_web()
    elif args.mode == 'train':
        run_train()
