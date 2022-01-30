"""
给你一个按非递减顺序排序的整数数组 nums，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。

输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]

请设计时间复杂度为 O(n) 的算法解决本问题
"""


class Solution:
    """
    暴力算法
    """
    def sortedSquares(self, nums):
        for i in range(len(nums)):
            nums[i] = nums[i] ** 2
        # 排序
        nums.sort()
        return nums


nums = [-4, -1, 0, 3, 10]
a = Solution()
print(a.sortedSquares(nums))


# --------------------------------------- #
class Solution2:
    """
    双指针法
    """
    def sortedSquares(self, nums):
        length = len(nums)
        i, j, k = 0, length-1, length-1
        arr = [0]*len(nums)
        while k >= 0:
            rm = nums[j] ** 2
            lm = nums[i] ** 2
            if rm >= lm:
                arr[k] = rm
                j -= 1
            else:
                arr[k] = lm
                i += 1
            k -= 1
        return arr

nums = [-4, -1, 0, 3, 10]
a = Solution2()
print(a.sortedSquares(nums))
