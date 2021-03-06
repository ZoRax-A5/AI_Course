import numpy as np  # 提供维度数组与矩阵运算
import copy  # 从copy模块导入深度拷贝方法
from board import Chessboard


# 基于棋盘类，设计搜索策略
class Game:
    def __init__(self, show=True):
        """
        初始化游戏状态.
        """

        self.chessBoard = Chessboard(show)
        self.solves = []
        self.gameInit()

    # 重置游戏
    def gameInit(self, show=True):
        """
        重置棋盘.
        """

        self.Queen_setRow = [-1] * 8
        self.chessBoard.boardInit(False)

    ##############################################################################
    ####                请在以下区域中作答(可自由添加自定义函数)                 ####
    ####              输出：self.solves = 八皇后所有序列解的list                ####
    ####             如:[[0,6,4,7,1,3,5,2],]代表八皇后的一个解为                ####
    ####           (0,0),(1,6),(2,4),(3,7),(4,1),(5,3),(6,5),(7,2)            ####
    ##############################################################################
    #                                                                            #

    permute_lst = []

    def Permutations(self, lst, begin, end):
        if begin == end:
            self.permute_lst.append(list(lst))
        else:
            for i in range(begin, end):
                lst[i], lst[begin] = lst[begin], lst[i]
                self.Permutations(lst, begin + 1, end)
                lst[i], lst[begin] = lst[begin], lst[i]

    def check(self, lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if abs(lst[j] - lst[i]) == abs(j - i):
                    return False
        return True

    def run(self, row=0):
        self.Permutations([i for i in range(8)], 0, 8)
        for lst in self.permute_lst:
            if self.check(lst):
                self.solves.append(lst)

    #                                                                            #
    ##############################################################################
    #################             完成后请记得提交作业             #################
    ##############################################################################

    def showResults(self, result):
        """
        结果展示.
        """

        self.chessBoard.boardInit(False)
        for i, item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i, item, False)

        self.chessBoard.printChessboard(False)

    def get_results(self):
        """
        输出结果(请勿修改此函数).
        return: 八皇后的序列解的list.
        """

        self.run()
        print(self.solves == [[0, 4, 7, 5, 2, 6, 1, 3], [0, 5, 7, 2, 6, 3, 1, 4], [0, 6, 3, 5, 7, 1, 4, 2],
                              [0, 6, 4, 7, 1, 3, 5, 2], [1, 3, 5, 7, 2, 0, 6, 4], [1, 4, 6, 3, 0, 7, 5, 2],
                              [1, 4, 6, 0, 2, 7, 5, 3], [1, 5, 0, 6, 3, 7, 2, 4], [1, 5, 7, 2, 0, 3, 6, 4],
                              [1, 6, 2, 5, 7, 4, 0, 3], [1, 6, 4, 7, 0, 3, 5, 2], [1, 7, 5, 0, 2, 4, 6, 3],
                              [2, 0, 6, 4, 7, 1, 3, 5], [2, 4, 1, 7, 0, 6, 3, 5], [2, 4, 1, 7, 5, 3, 6, 0],
                              [2, 4, 6, 0, 3, 1, 7, 5], [2, 4, 7, 3, 0, 6, 1, 5], [2, 5, 3, 0, 7, 4, 6, 1],
                              [2, 5, 3, 1, 7, 4, 6, 0], [2, 5, 1, 4, 7, 0, 6, 3], [2, 5, 1, 6, 4, 0, 7, 3],
                              [2, 5, 1, 6, 0, 3, 7, 4], [2, 5, 7, 1, 3, 0, 6, 4], [2, 5, 7, 0, 4, 6, 1, 3],
                              [2, 5, 7, 0, 3, 6, 4, 1], [2, 6, 1, 7, 4, 0, 3, 5], [2, 6, 1, 7, 5, 3, 0, 4],
                              [2, 7, 3, 6, 0, 5, 1, 4], [3, 1, 4, 7, 5, 0, 2, 6], [3, 1, 6, 4, 0, 7, 5, 2],
                              [3, 1, 6, 2, 5, 7, 0, 4], [3, 1, 6, 2, 5, 7, 4, 0], [3, 1, 7, 4, 6, 0, 2, 5],
                              [3, 1, 7, 5, 0, 2, 4, 6], [3, 0, 4, 7, 5, 2, 6, 1], [3, 0, 4, 7, 1, 6, 2, 5],
                              [3, 5, 0, 4, 1, 7, 2, 6], [3, 5, 7, 1, 6, 0, 2, 4], [3, 5, 7, 2, 0, 6, 4, 1],
                              [3, 6, 2, 7, 1, 4, 0, 5], [3, 6, 0, 7, 4, 1, 5, 2], [3, 6, 4, 2, 0, 5, 7, 1],
                              [3, 6, 4, 1, 5, 0, 2, 7], [3, 7, 0, 2, 5, 1, 6, 4], [3, 7, 0, 4, 6, 1, 5, 2],
                              [3, 7, 4, 2, 0, 6, 1, 5], [4, 1, 3, 5, 7, 2, 0, 6], [4, 1, 3, 6, 2, 7, 5, 0],
                              [4, 1, 5, 0, 6, 3, 7, 2], [4, 1, 7, 0, 3, 6, 2, 5], [4, 2, 0, 5, 7, 1, 3, 6],
                              [4, 2, 0, 6, 1, 7, 5, 3], [4, 2, 7, 3, 6, 0, 5, 1], [4, 0, 3, 5, 7, 1, 6, 2],
                              [4, 0, 7, 3, 1, 6, 2, 5], [4, 0, 7, 5, 2, 6, 1, 3], [4, 6, 3, 0, 2, 7, 5, 1],
                              [4, 6, 0, 3, 1, 7, 5, 2], [4, 6, 0, 2, 7, 5, 3, 1], [4, 6, 1, 3, 7, 0, 2, 5],
                              [4, 6, 1, 5, 2, 0, 3, 7], [4, 6, 1, 5, 2, 0, 7, 3], [4, 7, 3, 0, 2, 5, 1, 6],
                              [4, 7, 3, 0, 6, 1, 5, 2], [5, 1, 6, 0, 3, 7, 4, 2], [5, 1, 6, 0, 2, 4, 7, 3],
                              [5, 2, 4, 6, 0, 3, 1, 7], [5, 2, 4, 7, 0, 3, 1, 6], [5, 2, 0, 6, 4, 7, 1, 3],
                              [5, 2, 0, 7, 4, 1, 3, 6], [5, 2, 0, 7, 3, 1, 6, 4], [5, 2, 6, 3, 0, 7, 1, 4],
                              [5, 2, 6, 1, 3, 7, 0, 4], [5, 2, 6, 1, 7, 4, 0, 3], [5, 3, 1, 7, 4, 6, 0, 2],
                              [5, 3, 0, 4, 7, 1, 6, 2], [5, 3, 6, 0, 2, 4, 1, 7], [5, 3, 6, 0, 7, 1, 4, 2],
                              [5, 0, 4, 1, 7, 2, 6, 3], [5, 7, 1, 3, 0, 6, 4, 2], [6, 1, 3, 0, 7, 4, 2, 5],
                              [6, 1, 5, 2, 0, 3, 7, 4], [6, 2, 0, 5, 7, 4, 1, 3], [6, 2, 7, 1, 4, 0, 5, 3],
                              [6, 3, 1, 4, 7, 0, 2, 5], [6, 3, 1, 7, 5, 0, 2, 4], [6, 4, 2, 0, 5, 7, 1, 3],
                              [6, 0, 2, 7, 5, 3, 1, 4], [7, 1, 3, 0, 6, 4, 2, 5], [7, 1, 4, 2, 0, 6, 3, 5],
                              [7, 2, 0, 5, 1, 4, 6, 3], [7, 3, 0, 2, 5, 1, 6, 4]])
        return self.solves


game = Game()
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
game.showResults(solutions[0])