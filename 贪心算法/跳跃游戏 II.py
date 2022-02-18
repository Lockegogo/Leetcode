"""
给你一个非负整数数组 nums ，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。假设你总是可以到达数组的最后一个位置。

输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
"""
from typing import List
class Solution:
    def jump(self, nums: List[int]) -> int:
        """以最小的步数增加最大的覆盖范围"""
        # 当前覆盖的最远距离下标
        curDistance = 0
        # 下一步覆盖的最远距离下标
        nextDistance = 0
        # 记录走的最大步数
        step = 0
        if len(nums) == 1: return 0
        for i in range(len(nums)):
            nextDistance = max(i + nums[i], nextDistance)
            # 指针遍历当前范围内的所有元素，看从哪个位置出发可以最大程度地扩展自己的覆盖范围
            # 如果指针走到当前势力范围的最后一个元素，但是当前范围没到最后一个位置，step + 1，往后走
            if i == curDistance:
                if curDistance != len(nums) - 1:
                    step += 1
                    curDistance = nextDistance
                    if nextDistance >= len(nums) - 1: break
        return step

nums = [7,0,9,6,9,6,1,7,9,0,1,2,9,0,3]
sol = Solution()
print(sol.jump(nums))