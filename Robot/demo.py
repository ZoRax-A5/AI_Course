# 导入相关包
import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
from torch import optim

from QRobot import QRobot
from Maze import Maze
from ReplayDataSet import ReplayDataSet
from Runner import Runner
from torch_py.QNetwork import QNetwork

from tensorflow import keras

# 机器人移动方向
move_map = {
    'u': (-1, 0),  # up
    'r': (0, +1),  # right
    'd': (+1, 0),  # down
    'l': (0, -1),  # left
}


class SearchTree(object):

    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
    stack = [root]
    while True:
        current_node = stack.pop()
        is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问
        if current_node.loc == maze.destination:  # 到达目标点
            path = back_propagation(current_node)
            break

        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 入队
        for child in current_node.children:
            stack.append(child)
    # -----------------------------------------------------------------------
    return path


"""
题目要求:  编程实现 DQN 算法在机器人自动走迷宫中的应用
输入: 由 Maze 类实例化的对象 maze
必须完成的成员方法：train_update()、test_update()
补充：如果想要自定义的参数变量，在 \_\_init\_\_() 中以 `self.xxx = xxx` 创建即可
"""

is_visit_m = None


def init_visit(h, w):
    global is_visit_m
    if is_visit_m:
        return
    else:
        is_visit_m = np.zeros((h, w), dtype=np.int)


class Robot(QRobot):
    global is_visit_m
    valid_action = ['u', 'r', 'd', 'l']

    ''' QLearning parameters'''
    epsilon0 = 0.5  # 初始贪心算法探索概率
    final_epsilon = 0.01  # 最终贪心算法探索概率
    gamma = 0.9  # 公式中的 γ

    EveryUpdate = 1  # the interval of target model's updating

    """some parameters of neural network"""
    target_model = None
    eval_model = None
    batch_size = 32
    learning_rate = 0.01
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
        # dqn reward
        maze.set_reward(reward={
            "hit_wall": 60.,
            "destination": -80.,
            "default": 1.,
        })
        # qn reward
        # maze.set_reward(reward={
        #     "hit_wall": -12.,
        #     "destination": 60.,
        #     "default": -0.2,
        # })
        self.maze = maze
        self.maze_size = maze.maze_size
        if self.maze_size > 10:
            self.epsilon = 0.75

        """build network"""
        self.target_model = None
        self.eval_model = None
        self._build_network()

        """create the memory to store data"""
        max_size = max(self.maze_size ** 2 * 3, 1e4)
        self.memory = ReplayDataSet(max_size=max_size)
        self.memory.build_full_view(maze=self.maze)

        self.observe_step = self.maze_size ** 2

        h, w, _ = maze.maze_data.shape
        init_visit(h, w)

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
            # action = random.choice(self.valid_action)
            action = self.pesudo_random_choice()
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
        is_visit_m[state] += 1
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        # print(state, action)
        next_state = self.sense_state()
        # if is_visit_m[next_state] and reward == self.maze.reward["default"]:
        #     reward = 2

        is_terminal = 1 if next_state == self.maze.destination else 0

        self.memory.add(state, self.valid_action.index(action), reward, next_state, is_terminal)

        """--间隔一段时间更新target network权重--"""
        if self.step % self.EveryUpdate == 0:
            self._learn(batch=self.batch_size)

        """---update the step and epsilon---"""
        self.step += 1
        self.epsilon = max(self.final_epsilon, self.epsilon * 0.995)
        # if self.step > self.observe_step:
        #     self.epsilon -= (self.epsilon0 - self.final_epsilon)/300

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
    #     state = self.sense_state()  # 获取机器人当初所处迷宫位置

    #     is_visit_m[state] = 1 # 标记当前位置已经访问

    #     self.create_Qtable_line(state)  # 对当前状态，检索Q表，如果不存在则添加进入Q表

    #     action = random.choice(self.valid_action) if random.random() < self.epsilon else max(
    #         self.q_table[state], key=self.q_table[state].get)  # 选择动作

    #     reward = self.maze.move_robot(action)  # 以给定的动作（移动方向）移动机器人

    #     next_state = self.sense_state()  # 获取机器人执行动作后所处的位置

    #     # if is_visit_m[next_state] and reward == self.maze.reward['default']:
    #     #     reward = -2

    #     self.create_Qtable_line(next_state)  # 对当前 next_state ，检索Q表，如果不存在则添加进入Q表

    #     self.update_Qtable(reward, action, next_state)  # 更新 Q 表 中 Q 值
    #     # self.update_parameter()  # 更新其它参数
    #     # self.epsilon = max(0.05, self.epsilon * 0.7)
    #     self.epsilon = max(self.final_epsilon, self.epsilon0 - self.step * 0.02)

    #     return action, reward

    def pesudo_random_choice(self):
        # print('pesudo')

        valid_move = []
        visit_lst = []
        for move in self.valid_action:
            if not self.maze.is_hit_wall(self.sense_state(), move):
                valid_move.append(move)
                x, y = self.sense_state()[0] + move_map[move][0], self.sense_state()[1] + move_map[move][1]
                visit_lst.append(is_visit_m[(x, y)])

        target_action = valid_move[visit_lst.index(min(visit_lst))]
        # x, y = self.sense_state()[0] + move_map[target_action][0], self.sense_state()[1] + move_map[target_action][1]
        # is_visit_m[(x, y)] += 1
        return target_action
        # return random.choice(valid_move)


if __name__ == "__main__":
    """ create maze"""

    training_epoch = 20  # 训练轮数
    maze_size = 4  # 迷宫size
    training_per_epoch = 300

    maze = Maze(maze_size=maze_size)
    print(maze)

    robot = Robot(maze)
    robot.memory.build_full_view(maze=maze)
    print(robot.maze.reward)  # 输出最小值选择策略的reward值
    runner = Runner(robot)
    runner.run_training(training_epoch, training_per_epoch)



    """Test Robot"""
    robot.reset()
    for _ in range(85):
        a, r = robot.test_update()
        print("action:", a, "reward:", r)
        if r == maze.reward["destination"]:
            print("success")
            break

    # # 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
    # runner.generate_gif(filename="results/dqn_size10.gif")