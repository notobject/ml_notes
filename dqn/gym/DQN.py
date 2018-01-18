# encoding=utf-8
# Created by Mr.Long on 2017/12/29 0029.
# 这是文件的概括
import tensorflow as tf
import numpy as np
import random
from collections import deque

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size


class DQN():
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.state_dim()
        self.action_dim = env.action_dim()
        self.batch_size = 29

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("./model/")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("加载模型...")
        else:
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.matmul(self.Q_value, tf.reshape(self.action_input, shape=[-1, self.batch_size])),
                                 reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        if self.time_step > 0 and self.time_step % 10000 == 0:
            self.saver.save(self.session, "./model/game-model", global_step=self.time_step)

    def egreedy_action(self, state):

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            Q_value = self.Q_value.eval(feed_dict={
                self.state_input: [state]
            })[0]
            action = np.argmax(Q_value)
            return action
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
