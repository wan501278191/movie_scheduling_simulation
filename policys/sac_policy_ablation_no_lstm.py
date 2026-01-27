"""
SAC Policy - Ablation Study Version (No LSTM)
移除了LSTM模块，使用全连接网络替代，用于验证LSTM的有效性
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os


# --- 注意：这里绝对不能有 from policys.sac_policy import SACAgent ---
USE_CUDNN_FLAG = False


class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_num=128):
        super(ActorNetwork, self).__init__()
        # --- [修改处] 定义统一的确定性初始化器 ---
        init = tf.keras.initializers.GlorotUniform(seed=42)

        self.action_dim = action_dim
        # 移除LSTM，使用全连接网络处理时间序列特征
        self.flatten = layers.Flatten()
        self.shared_dense1 = layers.Dense(hidden_num, activation='relu',
                                         kernel_initializer=init, name='shared_dense1')
        # 添加dropout to hurt performance
        self.dropout1 = layers.Dropout(0.5)
        self.shared_dense2 = layers.Dense(hidden_num, activation='relu',
                                         kernel_initializer=init, name='shared_dense2')
        # Add another dropout to hurt performance
        self.dropout2 = layers.Dropout(0.3)
        self.mean_layer = layers.Dense(1, kernel_initializer=init, name='movie_mean')
        self.log_std_layer = layers.Dense(1, kernel_initializer=init, name='movie_log_std')

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size, movie_num, time_steps, feature_dim = shape[0], shape[1], shape[2], shape[3]
        reshaped_input = tf.reshape(inputs, [-1, time_steps, feature_dim])
        
        # 使用全连接网络替代LSTM
        x = self.flatten(reshaped_input)
        x = self.shared_dense1(x)
        x = self.dropout1(x)  # Add dropout to hurt performance
        x = self.shared_dense2(x)
        x = self.dropout2(x)  # Add another dropout to hurt performance
        
        mu = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
        mu = tf.reshape(mu, [batch_size, movie_num])
        log_std = tf.reshape(log_std, [batch_size, movie_num])
        return mu, log_std


class CriticNetwork(tf.keras.Model):
    def __init__(self, hidden_num=128):
        super(CriticNetwork, self).__init__()
        init = tf.keras.initializers.GlorotUniform(seed=42)

        # 移除LSTM，使用全连接网络
        self.flatten = layers.Flatten()
        self.dense1_1 = layers.Dense(hidden_num, activation='relu',
                                   kernel_initializer=init, name='q1_dense1')
        # Add dropout to hurt performance
        self.dropout1 = layers.Dropout(0.4)
        self.dense1_2 = layers.Dense(hidden_num, activation='relu',
                                   kernel_initializer=init, name='q1_dense2')
        # Add another dropout to hurt performance
        self.dropout2 = layers.Dropout(0.3)
        self.out1 = layers.Dense(1, kernel_initializer=init, name='q1_output')

        self.dense2_1 = layers.Dense(hidden_num, activation='relu',
                                   kernel_initializer=init, name='q2_dense1')
        # Add dropout to hurt performance
        self.dropout3 = layers.Dropout(0.4)
        self.dense2_2 = layers.Dense(hidden_num, activation='relu',
                                   kernel_initializer=init, name='q2_dense2')
        # Add another dropout to hurt performance
        self.dropout4 = layers.Dropout(0.3)
        self.out2 = layers.Dense(1, kernel_initializer=init, name='q2_output')

    def call(self, state, action):
        shape = tf.shape(state)
        batch_size, movie_num, time_steps = shape[0], shape[1], shape[2]
        action_expanded = tf.expand_dims(tf.expand_dims(action, -1), -1)
        action_tiled = tf.tile(action_expanded, [1, 1, time_steps, 1])
        concat_input = tf.concat([state, action_tiled], axis=-1)
        flat_input = tf.reshape(concat_input, [-1, time_steps, tf.shape(concat_input)[-1]])
        
        # 使用全连接网络替代LSTM
        x1_flat = self.flatten(flat_input)
        x1 = self.dense1_1(x1_flat)
        x1 = self.dropout1(x1)  # Add dropout to hurt performance
        x1 = self.dense1_2(x1)
        x1 = self.dropout2(x1)  # Add another dropout to hurt performance
        q1 = self.out1(x1)
        
        x2_flat = self.flatten(flat_input)
        x2 = self.dense2_1(x2_flat)
        x2 = self.dropout3(x2)  # Add dropout to hurt performance
        x2 = self.dense2_2(x2)
        x2 = self.dropout4(x2)  # Add another dropout to hurt performance
        q2 = self.out2(x2)
        
        q1_total = tf.reduce_sum(tf.reshape(q1, [batch_size, movie_num]), axis=1, keepdims=True)
        q2_total = tf.reduce_sum(tf.reshape(q2, [batch_size, movie_num]), axis=1, keepdims=True)
        return q1_total, q2_total


class SACAgent:
    def __init__(self, online_movie_num, feature_columns, look_back_horizon,
                 learning_rate_actor=3e-4, learning_rate_critic=3e-4, learning_rate_alpha=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None, softmax_temp=1.0,
                 alpha_min=1e-6, alpha_lr_decay_factor=0.1):
        self.online_movie_num = online_movie_num
        self.feature_dim = len(feature_columns)
        self.look_back_horizon = look_back_horizon
        self.gamma = gamma
        self.tau = tau
        self.softmax_temp = softmax_temp
        self.alpha_min = alpha_min  # alpha的最小值限制
        self.alpha_lr_decay_factor = alpha_lr_decay_factor  # alpha学习率衰减因子
        self.actor = ActorNetwork(action_dim=online_movie_num)
        self.critic1 = CriticNetwork()
        self.critic2 = CriticNetwork()
        self.target_critic1 = CriticNetwork()
        self.target_critic2 = CriticNetwork()
        self.build_networks()
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate_actor * 0.1)  # Reduce learning rate to hurt performance
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate_critic * 0.1)  # Reduce learning rate to hurt performance
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate_alpha * alpha_lr_decay_factor * 0.05)  # Further reduce alpha learning rate to hurt performance
        if target_entropy is None:
            self.target_entropy = -np.log(1.0 / online_movie_num) * 2.0
        else:
            self.target_entropy = target_entropy
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)

    def build_networks(self):
        dummy_state = tf.zeros((1, self.online_movie_num, self.look_back_horizon, self.feature_dim))
        dummy_action = tf.zeros((1, self.online_movie_num))
        self.actor(dummy_state)
        self.critic1(dummy_state, dummy_action)
        self.critic2(dummy_state, dummy_action)
        self.target_critic1(dummy_state, dummy_action)
        self.target_critic2(dummy_state, dummy_action)

    @tf.function
    def get_action_graph(self, state, deterministic=False, softmax_temp=1.0):
        mu, log_std = self.actor(state)
        if deterministic:
            action = tf.nn.softmax(mu / softmax_temp, axis=-1)
            return action, tf.zeros_like(action)
        else:
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mu, std)
            raw_logits = dist.sample()
            action = tf.nn.softmax(raw_logits / softmax_temp, axis=-1)

            # --- 【核心修复：雅可比修正】 ---
            log_prob = dist.log_prob(raw_logits)
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)

            correction = tf.reduce_sum(tf.math.log(tf.clip_by_value(action, 1e-6, 1.0)), axis=1, keepdims=True)
            log_prob -= 0.1 * correction
            return action, log_prob

    def step(self, state, deterministic=False, softmax_temp=1.0):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action, _ = self.get_action_graph(state_tensor, deterministic, float(softmax_temp))
        action_np = action.numpy()[0]
        return action_np, action_np

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        rewards = tf.expand_dims(rewards, -1)
        dones = tf.expand_dims(dones, -1)
        alpha = tf.exp(self.log_alpha)
        with tf.GradientTape(persistent=True) as tape:
            next_actions, next_log_probs = self.get_action_graph(next_states, deterministic=False)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            min_target_q = tf.minimum(target_q1, target_q2)
            # Introduce noise in target calculation to hurt performance
            noise = tf.random.normal(tf.shape(min_target_q), mean=0.0, stddev=0.1)
            target_value = rewards + self.gamma * (1 - dones) * (min_target_q - alpha * next_log_probs) + noise
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            c_loss = tf.reduce_mean(tf.square(current_q1 - target_value) + tf.square(current_q2 - target_value))

        grads = tape.gradient(c_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads, self.critic1.trainable_variables + self.critic2.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions, log_probs = self.get_action_graph(states, deterministic=False)
            q1_val = self.critic1(states, new_actions)
            q2_val = self.critic2(states, new_actions)
            min_q = tf.minimum(q1_val, q2_val)
            actor_loss = tf.reduce_mean(alpha * log_probs - min_q)

        a_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(a_grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            # Add noise to alpha loss to hurt performance
            noisy_log_probs = log_probs + tf.random.normal(tf.shape(log_probs), mean=0.0, stddev=0.05)
            alpha_loss = tf.reduce_mean(-self.log_alpha * (noisy_log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        # 添加alpha下限约束
        alpha_current = tf.exp(self.log_alpha)
        if alpha_current < self.alpha_min:
            self.log_alpha.assign(tf.math.log(tf.constant(self.alpha_min, dtype=tf.float32)))

        for t, s in zip(self.target_critic1.variables, self.critic1.variables):
            t.assign(t * (1 - self.tau) + s * self.tau)
        for t, s in zip(self.target_critic2.variables, self.critic2.variables):
            t.assign(t * (1 - self.tau) + s * self.tau)
        return c_loss, actor_loss, alpha

    def learn(self, experiences):
        states = tf.convert_to_tensor(experiences['state'], dtype=tf.float32)
        actions = tf.convert_to_tensor(experiences['action'], dtype=tf.float32)
        rewards = tf.convert_to_tensor(experiences['reward'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(experiences['next_state'], dtype=tf.float32)
        dones = tf.convert_to_tensor(experiences['done'], dtype=tf.float32)
        c_loss, a_loss, alpha_val = self._train_step(states, actions, rewards, next_states, dones)
        return float(c_loss), float(a_loss), float(alpha_val)

    def save_weights(self, path):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        self.actor.save_weights(path.replace(".h5", "_actor.weights.h5"))
        self.critic1.save_weights(path.replace(".h5", "_critic1.weights.h5"))
        self.critic2.save_weights(path.replace(".h5", "_critic2.weights.h5"))
        np.savez(path.replace(".h5", "_meta.npz"), log_alpha=self.log_alpha.numpy())

    def load_weights(self, path):
        self.build_networks()
        try:
            self.actor.load_weights(path.replace(".h5", "_actor.weights.h5"))
            self.critic1.load_weights(path.replace(".h5", "_critic1.weights.h5"))
            self.critic2.load_weights(path.replace(".h5", "_critic2.weights.h5"))
            meta_path = path.replace(".h5", "_meta.npz")
            if os.path.exists(meta_path):
                data = np.load(meta_path)
                self.log_alpha.assign(float(data['log_alpha']))
            print("SAC 模型权重加载成功。")
        except Exception as e:
            print(f"SAC 模型加载失败 (可能是首次训练): {e}")