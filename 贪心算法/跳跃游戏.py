"""
给定一个非负整数数组 nums ，你最初位于数组的第一个下标 。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个下标。

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/jump-game
"""
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        maxscale = 0
        # 要在最大范围内取值
        for i in range(len(nums) - 1):
            maxscale = max(maxscale, i + nums[i])
            # 如果此时最大范围还没有超过 i，以后也不可能超过了
            if maxscale <= i:
                return False
        return maxscale >= len(nums)-1


nums = [0, 1]
sol = Solution()
print(sol.canJump(nums))
