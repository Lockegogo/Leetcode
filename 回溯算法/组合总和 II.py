"""
给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中只能使用 一次 。
注意：解集不能包含重复的组合。 

输入: candidates = [2,5,2,1,2], target = 5,
输出:
[
[1,2,2],
[5]
]
来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/combination-sum-ii
"""
from typing import List

class Solution:
    """剪枝策略"""
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0
        # 存储用过的元素值
        self.used = []

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        # 为了剪枝提前进行排序
        candidates.sort()
        self.used = [0]*len(candidates)
        self.trackbacking(startindex, candidates, target)
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
            # 注意这里，list[-1] 代表的不是 list[0] 的前一位而是列表的最后一位，这是不符合比较逻辑的，所以要从 i=1 开始取值
            if i >= 1 and candidates[i] == candidates[i-1] and self.used[i-1] == 0:
                continue

            self.result.append(candidates[i])
            self.sum += candidates[i]
            self.used[i] = 1
            # 这是在同一树层上去重
            self.trackbacking(i+1, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
            self.used[i] = 0
        return self.res


candidates = [1]
target = 1
sol = Solution()
print(sol.combinationSum2(candidates, target))