import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from Maze import Maze
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from Runner import Runner
from torch_py.QNetwork import QNetwork


class Robot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    ''' QLearning parameters'''
    epsilon0 = 0.5  # 初始贪心算法探索概率
    gamma = 0.9  # 公式中的 γ

    EveryUpdate = 1  # the interval of target model's updating

    """some parameters of neural network"""
    target_model = None
    eval_model = None
    batch_size = 32
    learning_rate = 1e-2
    TAU = 1e-3
    step = 1  # 记录训练的步数

    """setting the device to train network"""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.,
            "destination": -50.,
            "default": 1.,
        })
        self.maze = maze
        self.maze_size = maze.maze_size

        """build network"""
        self.target_model = None
        self.eval_model = None
        self._build_network()

        """create the memory to store data"""
        max_size = max(self.maze_size ** 2 * 3, 1e4)
        self.memory = ReplayDataSet(max_size=max_size)

    def _build_network(self):
        seed = 0
        random.seed(seed)

        """build target model"""
        self.target_model = QNetwork(state_size=2, action_size=4, seed=seed).to(self.device)

        """build eval model"""
        self.eval_model = QNetwork(state_size=2, action_size=4, seed=seed).to(self.device)

        """build the optimizer"""
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=self.learning_rate)

    def target_replace_op(self):
        """
            Soft update the target model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        # for target_param, eval_param in zip(self.target_model.parameters(), self.eval_model.parameters()):
        #     target_param.data.copy_(self.TAU * eval_param.data + (1.0 - self.TAU) * target_param.data)

        """ replace the whole parameters"""
        self.target_model.load_state_dict(self.eval_model.state_dict())

    def _choose_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().to(self.device)
        if random.random() < self.epsilon:
            action = random.choice(self.valid_action)
        else:
            self.eval_model.eval()
            with torch.no_grad():
                q_next = self.eval_model(state).cpu().data.numpy()  # use target model choose action
            self.eval_model.train()

            action = self.valid_action[np.argmin(q_next).item()]
        return action

    def _learn(self, batch: int = 16):
        if len(self.memory) < batch:
            print("the memory data is not enough")
            return
        state, action_index, reward, next_state, is_terminal = self.memory.random_sample(batch)

        """ convert the data to tensor type"""
        state = torch.from_numpy(state).float().to(self.device)
        action_index = torch.from_numpy(action_index).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        is_terminal = torch.from_numpy(is_terminal).int().to(self.device)

        self.eval_model.train()
        self.target_model.eval()

        """Get max predicted Q values (for next states) from target model"""
        Q_targets_next = self.target_model(next_state).detach().min(1)[0].unsqueeze(1)

        """Compute Q targets for current states"""
        Q_targets = reward + self.gamma * Q_targets_next * (torch.ones_like(is_terminal) - is_terminal)

        """Get expected Q values from local model"""
        self.optimizer.zero_grad()
        Q_expected = self.eval_model(state).gather(dim=1, index=action_index)

        """Compute loss"""
        loss = F.mse_loss(Q_expected, Q_targets)
        loss_item = loss.item()

        """ Minimize the loss"""
        loss.backward()
        self.optimizer.step()

        """copy the weights of eval_model to the target_model"""
        self.target_replace_op()
        return loss_item

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        next_state = self.sense_state()
        is_terminal = 1 if next_state == self.maze.destination or next_state == state else 0

        self.memory.add(state, self.valid_action.index(action), reward, next_state, is_terminal)

        """--间隔一段时间更新target network权重--"""
        if self.step % self.EveryUpdate == 0:
            self._learn(batch=32)

        """---update the step and epsilon---"""
        self.step += 1
        self.epsilon = max(0.01, self.epsilon * 0.995)

        return action, reward

    def test_update(self):
        state = np.array(self.sense_state(), dtype=np.int16)
        state = torch.from_numpy(state).float().to(self.device)

        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()

        action = self.valid_action[np.argmin(q_value).item()]
        reward = self.maze.move_robot(action)
        return action, reward

    # def train_update(self):
    #     self.state = self.sense_state()  # 获取机器人当初所处迷宫位置

    #     self.create_Qtable_line(self.state)  # 对当前状态，检索Q表，如果不存在则添加进入Q表

    #     action = self.pesudo_random_choice() if random.random() < self.epsilon else max(
    #         self.q_table[self.state], key=self.q_table[self.state].get)  # 选择动作

    #     reward = self.maze.move_robot(action)  # 以给定的动作（移动方向）移动机器人

    #     next_state = self.sense_state()  # 获取机器人执行动作后所处的位置

    #     if self.is_visit_m[next_state]:
    #         reward = -5

    #     self.create_Qtable_line(next_state)  # 对当前 next_state ，检索Q表，如果不存在则添加进入Q表

    #     self.update_Qtable(reward, action, next_state)  # 更新 Q 表 中 Q 值
    #     # self.update_parameter()  # 更新其它参数
    #     self.epsilon = max(0.05, self.epsilon * 0.7)

    #     return action, reward

    def pesudo_random_choice(self):
        valid_move = []
        for move in self.valid_action:
            tmaze = copy.copy(self.maze)
            last_state = tmaze.sense_robot()
            tmaze.move_robot(move)
            current_state = tmaze.sense_robot()
            if not self.is_visit_m[current_state]:
                valid_move.append(move)

        if len(valid_move) == 0:
            valid_move = self.valid_action
        return random.choice(valid_move)


if __name__ == "__main__":
    """ create maze"""

    training_epoch = 20  # 训练轮数
    maze_size = 11  # 迷宫size
    training_per_epoch = 300

    maze = Maze(maze_size=maze_size)
    print(maze)

    robot = Robot(maze)
    robot.memory.build_full_view(maze=maze)
    print(robot.maze.reward)  # 输出最小值选择策略的reward值
    runner = Runner(robot)
    runner.run_training(training_epoch, training_per_epoch)

    # 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
    runner.generate_gif(filename="results/dqn_size10.gif")

    """Test Robot"""
    robot.reset()
    for _ in range(25):
        a, r = robot.test_update()
        print("action:", a, "reward:", r)
        if r == maze.reward["destination"]:
            print("success")
            break
