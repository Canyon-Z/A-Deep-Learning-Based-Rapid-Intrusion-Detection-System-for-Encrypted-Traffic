import wandb
import random
import time

# 1. 初始化一次新的运行 (Run)
#    - project: 指定项目名称，所有相关的实验都会归在这个项目下
#    - config: 记录这次实验的超参数，方便以后对比
wandb.init(
    project="my-first-project",  # 你可以改成你喜欢的项目名
    name="test-run-1",           # 这次实验的名字，方便识别
    config={
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 32,
        "model_type": "SimpleCNN"
    }
)
print("开始记录实验...")

# 2. 模拟训练循环
for epoch in range(wandb.config.epochs):
    # 模拟训练指标：真实训练中这里会是计算出的loss和accuracy
    # 这里我们用随机数来模拟指标的变化趋势
    loss = 0.5 * (1.0 / (epoch + 1)) + random.uniform(-0.05, 0.05)
    accuracy = 0.5 + 0.4 * (epoch / wandb.config.epochs) + random.uniform(-0.02, 0.02)

    # 3. 记录指标！这是 wandb 最核心的操作
    #    wandb.log() 会将数据实时同步到云端仪表盘
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

    print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={accuracy:.4f}")
    time.sleep(1)  # 暂停1秒，让数据上传更明显

# 4. 标记本次运行结束 (在脚本中不是必须的，但是个好习惯)
wandb.finish()
print("实验记录完成！快去浏览器看看你的第一个仪表盘吧！")