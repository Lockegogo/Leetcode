"""
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
"""

from typing import List


class Solution:
    def __init__(self) -> None:
        self.res = [[]]
        self.path = []

    def subsets(self, nums: List[int]) -> List[List[int]]:
        startindex = 0
        self.backtracking(nums, startindex)
        return self.res

    def backtracking(self, nums, startindex):
        # 回溯结束条件
        if startindex == len(nums):
            # self.res.append(self.path[:])
            return

        # 单层递归逻辑
        for i in range(startindex, len(nums)):
            self.path.append(nums[i])
            self.res.append(self.path[:])
            self.backtracking(nums, i+1)
            self.path.pop()


nums = [0]
sol = Solution()
print(sol.subsets(nums))