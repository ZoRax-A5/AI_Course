from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
import os, sys, time, datetime, json, random, copy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt

valid_action = ['u', 'r', 'd', 'l']

# Actions dictionary
actions_dict = {
    'u': 0,
    'r': 1,
    'd': 2,
    'l': 3
}

num_actions = len(actions_dict)

"""
题目要求:  编程实现 DQN 算法在机器人自动走迷宫中的应用
输入: 由 Maze 类实例化的对象 maze
必须完成的成员方法：train_update()、test_update()
补充：如果想要自定义的参数变量，在 \_\_init\_\_() 中以 `self.xxx = xxx` 创建即可
"""


class DQNmodel:
    def __init__(self, maze, lr=0.01):
        self.model = Sequential()
        self.model.add(Dense(2, input_shape=(2,)))
        self.model.add(PReLU())
        self.model.add(Dense(2))
        self.model.add(PReLU())
        self.model.add(Dense(num_actions))
        self.model.compile(optimizer='adam', loss='mse')

    def getModel(self):
        return self.model


class Robot(QRobot):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon=0.5, max_memory=100, data_size=32):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        self.maze = maze

        maze.set_reward(reward={
            "hit_wall": -0.75,
            "destination": -1.0,
            "default": -0.04,
        })

        self.state = None
        self.action = None

        self.discount = gamma
        self.epsilon = epsilon

        self.model = DQNmodel(maze).getModel()
        self.memory = ReplayDataSet(max_size=max_memory)

        self.data_size = data_size

        h, w, _ = maze.maze_data.shape
        self.is_visit_m = np.zeros((h, w), dtype=np.int)

    def train_update(self):
        """ 
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """

        # -----------------请实现你的算法代码--------------------------------------
        # get current state
        self.state = self.sense_state()
        self.is_visit_m[self.state] = 1
        # get next action
        print('epsilon=', self.epsilon)
        action = random.choice(valid_action) if random.random() < self.epsilon else \
            valid_action[np.argmax(self.model.predict(np.asarray(self.sense_state()).reshape((1, -1))))]
        # apply action and get reward
        reward = self.maze.move_robot(action)
        # get next state after applying action
        next_state = self.sense_state()
        if reward == self.maze.reward['default'] and self.is_visit_m[self.state]:
            reward = -0.25
        # get if game is over
        game_over = True if self.maze.reward['destination'] == reward else False
        # Store episode (experience)
        print(self.state, action, reward, next_state, game_over)
        self.memory.add(self.state, valid_action.index(action), reward, next_state, game_over)

        # Train neural network model
        inputs, targets = self.get_data(data_size=self.data_size)

        h = self.model.fit(
            inputs,
            targets,
            epochs=10,
            batch_size=25,
            verbose=0,
        )
        loss = self.model.evaluate(inputs, targets, verbose=0)
        print('loss=', loss)
        self.epsilon = self.update_parameter()
        # -----------------------------------------------------------------------

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """

        # -----------------请实现你的算法代码--------------------------------------
        self.state = self.sense_state()
        self.state = np.asarray(self.state).reshape((1, -1))
        action = valid_action[np.argmax(self.experience.predict(self.state))]
        reward = self.maze.move_robot(action)
        # -----------------------------------------------------------------------

        return action, reward

    def pesudo_random_choice(self):
        print('pesudo-random-choice')
        valid_move = []
        for move in valid_action:
            tmaze = copy.copy(self.maze)
            tmaze.move_robot(move)
            current_state = tmaze.sense_robot()
            if not self.is_visit_m[current_state]:
                valid_move.append(move)

        if len(valid_move) == 0:
            valid_move = valid_action
        return random.choice(valid_move)

    def get_data(self, data_size):
        mem_size = len(self.memory.Experience)
        env_size = 2
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, num_actions))
        if mem_size < 2:
            return inputs, targets
        sample = self.memory.random_sample(data_size)

        for i in range(data_size):
            envstate, action, reward, envstate_next, game_over = np.asarray(sample[0][i]).reshape((1, -1)), \
                                                                 sample[1][i][0], sample[2][i][0], \
                                                                 np.asarray(sample[3][i]).reshape((1, -1)), \
                                                                 sample[4][i][0]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.model.predict(envstate)
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * np.max(self.model.predict(envstate_next))
        return inputs, targets

    def update_parameter(self):
        """
        衰减随机选择动作的可能性
        """

        self.t += 1
        if self.epsilon < 0.1:
            self.epsilon = 0.1
        else:
            self.epsilon -= 0.04

        return self.epsilon
