# 文件: tools/memory.py

import numpy as np
import random
from collections import deque

# 这是旧的、可能导致其他项目出错的 Memory 类
# 我们把它留在这里，以防您的其他项目需要它
# 如果您确定所有项目都可以使用下面的 MemoryDict，也可以删除这个
class Memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 这个方法可能返回 list，与 SAC Agent 不兼容
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ===================================================================
# 【请确保您的 SAC 训练脚本使用的是下面这个新的类】
# 它会输出 SAC Agent 需要的字典（dict）格式
# ===================================================================
class MemoryDict:
    def __init__(self, capacity):
        """
        使用一个固定容量的双端队列来高效存储经验。
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info):
        """
        将一个经验转换元组存入缓冲区。
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次的经验，并将其组织成代理所期望的字典格式。
        """
        sample_indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[i] for i in sample_indices]
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 返回 SAC Agent 需要的字典格式
        return {
            'state': np.array(states, dtype=np.float32),
            'action': np.array(actions, dtype=np.float32),
            'reward': np.array(rewards, dtype=np.float32),
            'next_state': np.array(next_states, dtype=np.float32),
            'done': np.array(dones, dtype=np.float32)
        }

    def __len__(self):
        """
        返回缓冲区当前的存储数量。
        """
        return len(self.buffer)