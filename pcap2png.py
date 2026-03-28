import os
import numpy as np
from PIL import Image
import dpkt

# ------------ 关键：和你真实目录完全匹配的路径 ------------
PCAP_DIR = "data_USTC-TK2016/USTC-TFC2016-master"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 28

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pcap_to_png_dpkt(pcap_path, save_path):
    try:
        flow_bytes = []
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                            payload = bytes(ip.data.data)
                            flow_bytes.extend(payload)
                except Exception:
                    continue
        # 固定长度 784 字节（28*28）
        flow_bytes = flow_bytes[:784]
        if len(flow_bytes) < 784:
            flow_bytes += [0] * (784 - len(flow_bytes))
        # 转成灰度图
        img_array = np.array(flow_bytes, dtype=np.uint8).reshape((IMG_SIZE, IMG_SIZE))
        img = Image.fromarray(img_array, 'L')
        img.save(save_path)
        print(f"✅ 成功生成: {save_path}")
    except Exception as e:
        print(f"❌ 失败 {pcap_path}: {e}")

def batch_convert():
    count = 0
    print(f"🔍 正在扫描 pcap 目录: {PCAP_DIR}")
    for root, dirs, files in os.walk(PCAP_DIR):
        print(f"📂 遍历子目录: {root}")
        
        # 获取相对路径，用来在 OUTPUT_DIR 中创建对应的包含类别名的子文件夹
        rel_path = os.path.relpath(root, PCAP_DIR)
        target_dir = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            if file.endswith(".pcap"):
                print(f"📄 找到 pcap: {file}")
                pcap_path = os.path.join(root, file)
                # 使用相同的层级结构保存 PNG，这样按文件夹名就可以作为类别标签
                png_name = os.path.splitext(file)[0] + ".png"
                save_path = os.path.join(target_dir, png_name)
                pcap_to_png_dpkt(pcap_path, save_path)
                count += 1
    print(f"\n📊 总计处理 {count} 个 pcap 文件")

if __name__ == "__main__":
    print("🚀 开始批量转换 PCAP → PNG...")
    batch_convert()
    print("🎉 转换完成！PNG 已保存到 data/processed")