"""
找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

输入: k = 3, n = 7
输出: [[1,2,4]]
"""
from typing import List
 
class Solution:
    def __init__(self):
        self.res = []
        # 已经收集的元素总和，也就是 path 里元素的总和
        self.sum_now = 0
        self.path = []

    def combinationSum3(self, k: int, n: int):
        self.backtracking(k, n, 1)
        return self.res

    def backtracking(self, k: int, n: int, start_num: int):
        # 剪枝：和 > target，无意义了
        if self.sum_now > n:  
            return
        # len(path)==k 时不管 sum 是否等于n都会返回
        if len(self.path) == k:  
            if self.sum_now == n:
                self.res.append(self.path[:])
            # 如果 len(path)==k 但是 和不等于 target，直接返回
            return
        # 集合固定为 9 个数
        for i in range(start_num, 10 - (k - len(self.path)) + 1):
            self.path.append(i)
            self.sum_now += i
            self.backtracking(k, n, i + 1)
            self.path.pop()
            self.sum_now -= i





        