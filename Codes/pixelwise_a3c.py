import tensorflow as tf
import numpy as np

'''
    initializes MyFCN model
    CHoose Action based on action given by model
    Update Model
'''
class PixelWiseA3CAgent:
    def __init__(self, model, optimizer, gamma=0.99, beta=1e-2,
                 pi_loss_coef=1.0, v_loss_coef=0.5, t_max=5):

        self.model = model
        self.optimizer = optimizer

        self.gamma = gamma
        self.beta = beta
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.t_max = t_max

        # Trajectory buffers - for storing episode data
        self.past_states = []
        self.past_actions = []
        self.past_values = []
        self.past_rewards = []
        self.past_logits = []

    def _prepare_input(self, state):
        """Convert and transpose state for TF Conv2D"""
        x = tf.convert_to_tensor(state, dtype=tf.float32)  # (B, C, H, W)
        x = tf.transpose(x, [0, 2, 3, 1])  # → (B, H, W, C)
        return x

    # Choose Action
    def act(self, state, deterministic=False):
        input_tensor = self._prepare_input(state)
        logits, value, next_state = self.model(input_tensor)

        policy = tf.nn.softmax(logits, axis=-1)
        action = (
            tf.argmax(policy, axis=-1)
            if deterministic
            else tf.random.categorical(tf.reshape(logits, [-1, logits.shape[-1]]), 1)
        )
        action = tf.reshape(action, logits.shape[:-1])
        return action.numpy()[0], next_state.numpy()[0]

    def store_transition(self, state, action, reward, value, logits):
        self.past_states.append(state)
        self.past_actions.append(action)
        self.past_rewards.append(reward)
        self.past_values.append(value)
        self.past_logits.append(logits)

    # Compute Advantage function
    def compute_returns(self, last_value):
        returns = []
        R = last_value
        for r in reversed(self.past_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    # Update Model
    def update(self, last_state=None):
        if last_state is None:
            last_value = 0.0
        else:
            last_input = self._prepare_input(last_state)
            _, last_value, _ = self.model(last_input)
            last_value = tf.squeeze(last_value).numpy()

        returns = self.compute_returns(last_value)

        with tf.GradientTape() as tape:
            total_pi_loss, total_v_loss, total_entropy = 0, 0, 0

            for s, a, R, v, logits in zip(self.past_states, self.past_actions, returns, self.past_values, self.past_logits):
                s_input = self._prepare_input(s)
                policy = tf.nn.softmax(logits, axis=-1)
                log_policy = tf.nn.log_softmax(logits, axis=-1)

                a_onehot = tf.one_hot(a, logits.shape[-1])
                log_prob = tf.reduce_sum(log_policy * a_onehot, axis=-1)
                entropy = -tf.reduce_sum(policy * log_policy, axis=-1)

                advantage = R - tf.squeeze(v)
                total_pi_loss += -log_prob * advantage
                total_v_loss += 0.5 * tf.square(advantage)
                total_entropy += entropy

            loss = (self.pi_loss_coef * tf.reduce_mean(total_pi_loss) +
                    self.v_loss_coef * tf.reduce_mean(total_v_loss) -
                    self.beta * tf.reduce_mean(total_entropy))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.past_states.clear()
        self.past_actions.clear()
        self.past_rewards.clear()
        self.past_values.clear()
        self.past_logits.clear()
