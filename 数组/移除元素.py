"""
给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

说明：
1. 为什么返回数值是整数，但输出的答案是数组呢？
答：输入数组是以引用方式传递的，这意味着在函数里修改数组对于调用者是可见的。我们会根据你的函数返回的长度打印出数组中该长度范围内的所有元素。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-element
"""


class Solution:
    def removeElement(self, nums, val):
        """
        暴力解法：发现需要移除的元素，就将数组集体向前移动一位
        """
        length = len(nums)
        i = 0
        while i < length:
            # 发现目标，长度减一，指针向前移动一位
            if nums[i] == val:
                # range 前闭后开；数组下标比长度少 1 
                # 注意 j 只循环到倒数第二位元素，是 length - 1
                for j in range(i, length-1):
                    nums[j] = nums[j+1]
                length -= 1
                nums = nums[:length]
                i -= 1
            i += 1
        return length, nums

nums = [0, 1, 2, 3, 6, 3, 2, 2, 2, 3, 2]
val = 2
sol = Solution()
length, arr = sol.removeElement(nums, val)
print(length, 'nums =', arr)

# ------------------------------------ #
class Solution2:
    def removeElement(self, nums, val):
        """
        双指针法
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        slowIndex = 0
        fastIndex = 0
        while fastIndex < len(nums):
            if val != nums[fastIndex]:
                nums[slowIndex] = nums[fastIndex]
                slowIndex += 1
                fastIndex += 1
            else:
                # 当快指针遇到要删除的元素时停止赋值
                # 慢指针停止移动，快指针继续前进
                # 也可以把 fastIndex += 1 写在外面
                fastIndex += 1
        return slowIndex, nums[:slowIndex]

nums = [0, 1, 2, 3, 6, 3, 2, 2, 2, 3, 2]
val = 2
sol2 = Solution2()
len, arr = sol2.removeElement(nums, val)
print(len, 'nums =', arr)