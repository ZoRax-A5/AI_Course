import random
import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0),  # up
    'r': (0, +1),  # right
    'd': (+1, 0),  # down
    'l': (0, -1),  # left
}

is_visit_m = None


def init_visit(h, w):
    global is_visit_m
    if is_visit_m:
        return
    else:
        is_visit_m = np.zeros((h, w), dtype=np.int)


class QRobot(object):
    global is_visit_m
    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):
        # maze.set_reward(reward={
        #     "hit_wall": -30.,
        #     "destination": 80.,
        #     "default": -0.1,
        # })
        self.maze = maze
        if maze.maze_size > 10:
            epsilon0 = 0.7

        self.state = None
        self.action = None

        """ 设置QLearning 的相关参数 """

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0  # 随机选择动作的概率
        self.epsilon = epsilon0

        self.t = 0  # 用于衰减epsilon

        self.q_table = {}
        self.reset()

        h, w, _ = maze.maze_data.shape
        init_visit(h, w)

    def current_state_valid_actions(self):

        return self.maze.can_move_actions(self.sense_state())

    def reset(self):
        """
        重置机器人在迷宫中的状态
        """

        self.maze.reset_robot()
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def update_parameter(self):
        """
        衰减随机选择动作的可能性
        """

        self.t += 1
        if self.epsilon < 0.01:
            self.epsilon = 0.01
        else:
            self.epsilon -= self.t * 0.1

        return self.epsilon

    def sense_state(self):
        """
        获取机器人在迷宫中的实时位置
        """

        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        以当前机器人的状态创建 Q 表；
        如果当前状态不存在，则为 Q 表添加新列，如：Qtable[state] ={'u':xx, 'd':xx, ...}
        如果当前状态已存在，则不做任何改动
        :param state: 机器人当前状态
        """

        if state not in self.q_table:
            self.q_table[state] = {
                a: 0.0 for a in self.valid_action}

    def update_Qtable(self, r, action, next_state):
        """
        更新 Q 表中的 Q 值
        :param r: 迷宫返回的奖励值
        :param action: 机器人选择的动作
        :next_state: 机器人执行动作后的状态
        """

        current_r = self.q_table[self.state][action]

        update_r = r + self.gamma * float(max(self.q_table[next_state].values()))

        self.q_table[self.state][action] += self.alpha * (update_r - current_r)

    def train_update(self):
        """
        以训练状态选择动作，并更新相关参数
        :return :action, reward 如："u", -1
        """

        self.state = self.sense_state()  # 获取机器人当初所处迷宫位置
        is_visit_m[self.state] += 1
        self.create_Qtable_line(self.state)  # 对当前状态，检索Q表，如果不存在则添加进入Q表

        action = self.pesudo_random_choice() if random.random() < self.epsilon else max(
            self.q_table[self.state], key=self.q_table[self.state].get)  # 选择动作

        reward = self.maze.move_robot(action)  # 以给定的动作（移动方向）移动机器人

        next_state = self.sense_state()  # 获取机器人执行动作后所处的位置

        self.create_Qtable_line(next_state)  # 对当前 next_state ，检索Q表，如果不存在则添加进入Q表

        self.update_Qtable(reward, action, next_state)  # 更新 Q 表 中 Q 值
        # self.update_parameter()  # 更新其它参数
        if self.maze.maze_size < 10:
            self.epsilon = max(0.01, self.epsilon * 0.95)

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作，并更新相关参数
        :return :action, reward 如："u", -1
        """

        self.state = self.sense_state()  # 获取机器人当初所处迷宫位置

        self.create_Qtable_line(self.state)  # 对当前状态，检索Q表，如果不存在则添加进入Q表

        action = max(self.q_table[self.state],
                     key=self.q_table[self.state].get)  # 选择动作

        reward = self.maze.move_robot(action)  # 以给定的动作（移动方向）移动机器人

        return action, reward

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
    from QRobot import QRobot
    from Maze import Maze
    from Runner import Runner

    training_epoch = 20  # 训练轮数
    maze_size = 5  # 迷宫size
    training_per_epoch = 300

    maze = Maze(maze_size=maze_size)
    print(maze)

    robot = QRobot(maze)
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