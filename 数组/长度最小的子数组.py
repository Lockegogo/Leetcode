"""
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

示例：
输入：s = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
"""


from sympy import re


class Solution:
    """
    暴力算法，太暴力了，超时了
    """

    def minSubArrayLen(self, target, nums):
        length = 0
        result = 0
        while length <= len(nums):
            length += 1
            for i in range(len(nums) - length + 1):
                if sum(nums[i:i+length]) >= target:
                    result = length
                    return result
        return result


s = 7
nums = [2, 3, 1, 2, 4, 3, 7]
sol = Solution()
print(sol.minSubArrayLen(s, nums))


class Solution2:
    """
    滑动窗口
    """
    def minSubArrayLen(self, target, nums):
        # 定义一个无限大的数
        result = float("inf")
        start = 0
        sublength = 0
        for end in range(len(nums)):
            while sum(nums[start:end+1]) >= target:
                # 不能直接赋值 result, 要体现最小长度
                sublength = end-start + 1
                result = min(result, sublength)
                if sublength <= result:
                    result = sublength
                start += 1
        return 0 if result == float("inf") else result


s = 100
nums = [2, 3, 1, 2, 4, 3, 1, 1, 9, 1]
sol = Solution2()
print(sol.minSubArrayLen(s, nums))
