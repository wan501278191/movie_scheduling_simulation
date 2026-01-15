# 影院排片强化学习项目 (SAC + CinemaGym)

## 1. 项目环境说明
本项目基于 **TensorFlow 2.x** 开发。
- 开发语言：Python 3.8+
- 核心框架：TensorFlow, Gymnasium, Pandas

## 2. 安装依赖
请在项目根目录下，运行以下命令安装所需库：
pip install -r requirements.txt

## 3. 文件结构说明
- `envs/simsche.py`: 自定义 Gym 仿真环境 (核心环境逻辑)
- `policys/`: SAC 算法与网络模型实现
- `data/`: 包含 `rl_data_final.csv` 
- `configs/`: 训练与评估的配置文件
- `train/train_sac.py`: 模型训练脚本
- `main_all.py`: 模型评估与策略对比主脚本

## 4. 运行指南

### (1) 快速测试/评估 (建议先运行此步)
使用已有的权重文件进行评估，并生成排片结果：
python main_all.py

### (2) 重新训练模型
如需从头训练 SAC 智能体：
python train_sac.py --config configs/11_sac_new_env.json