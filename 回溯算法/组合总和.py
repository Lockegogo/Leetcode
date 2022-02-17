"""
给你一个无重复元素的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的所有不同组合 ，并以列表形式返回。你可以按任意顺序返回这些组合。

candidates 中的 同一个数字可以无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
对于给定的输入，保证和为 target 的不同组合数少于 150 个。

输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/combination-sum
"""
from typing import List





class Solution:
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        self.res = self.trackbacking(startindex, candidates, target)
        return self.res

    def trackbacking(self, startindex, candidates, target):
        # 确定终止条件
        if self.sum == target:
            self.res.append(self.result[:])
            return
        if self.sum > target:
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        for i in range(startindex, len(candidates)):
            self.result.append(candidates[i])
            self.sum += candidates[i]
            # 因为可以无限制选取同一个数字，所以是 i
            self.trackbacking(i, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
        return self.res

class Solution:
    """剪枝策略"""
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        # 为了剪枝提前进行排序
        candidates.sort()
        self.res = self.trackbacking(startindex, candidates, target)
        return self.res

    def trackbacking(self, startindex, candidates, target):
        # 确定终止条件
        if self.sum == target:
            # 因为是 shallow copy，所以不能直接传入self.result
            self.res.append(self.result[:])
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        # 如果本层 sum + condidates[i] > target，就提前结束遍历，剪枝
        for i in range(startindex, len(candidates)):
            if self.sum + candidates[i] > target:
                return
            self.result.append(candidates[i])
            self.sum += candidates[i]
            # 因为可以无限制选取同一个数字，所以是 i
            self.trackbacking(i, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
        return self.res


candidates = [2,3,6,7]
target = 7
sol = Solution()
print(sol.combinationSum(candidates, target))