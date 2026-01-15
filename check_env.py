import os
import sys
import numpy as np
import tensorflow as tf

print("1. 检查 TensorFlow...")
print(f"   TF Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"   GPU: {gpus}")

print("\n2. 检查路径...")
cwd = os.getcwd()
print(f"   当前目录: {cwd}")
data_path = os.path.join(cwd, 'data', 'rl_data_final.csv')
if os.path.exists(data_path):
    print(f"   ✅ 数据存在 ({os.path.getsize(data_path)} bytes)")
else:
    print(f"   ❌ 数据缺失: {data_path}")

print("\n3. 尝试加载环境...")
try:
    sys.path.insert(0, cwd)
    from envs.simsche import CinemaGym


    # 创建一个哑巴 logger
    class DummyLogger:
        def info(self, msg): print(f"   [EnvLog] {msg}")


    env = CinemaGym(logger=DummyLogger(), enable_logging=True)
    state = env.reset()
    print(f"   ✅ 环境 Reset 成功，State Shape: {state.shape}")
    print(f"   State 示例: {state[0, -1, :7]}")  # 打印第一部电影最新一天的前7维特征
except Exception as e:
    print(f"   ❌ 环境加载失败: {e}")
    import traceback

    traceback.print_exc()

print("\n诊断结束。")