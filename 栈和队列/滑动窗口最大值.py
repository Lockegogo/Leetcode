"""
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/sliding-window-maximum
"""

from collections import deque
class MyQueue():
    """
    定义单调队列类: 从大到小
    """

    def __init__(self) -> None:
        self.queue = deque()

    def pop(self, value):
        """
        每次弹出时比较当前弹出的数值是否等于队列出口元素的数值，如果相等则直接弹出
        """
        if self.queue and value == self.queue[0]:
            # 弹出队首元素
            self.queue.popleft()

    def push(self, value):
        """
        如果 push 的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到 push 的数值小于等于队列入口元素的数值为止。保证队列由大到小。
        """
        # 如果队列非空并且队尾元素的值小于新加进来的值
        while self.queue and self.queue[-1] < value:
            # 弹出队尾元素，相当于让 value 一直往前插队超过比它小的人
            self.queue.pop()
        self.queue.append(value)

    def front(self):
        """
        查询当前队列里的最大值，直接返回 front
        """
        return self.queue[0]

class Solution:
    def maxSlidingWindow(self, nums, k):
        que = MyQueue()
        result = []
        # 先将前 k 的元素放进队列
        for i in range(k):
            que.push(nums[i])
        result.append(que.front())
        for i in range(k, len(nums)):
            # 滑动窗口移除最前面元素
            que.pop(nums[i - k])
            # 滑动窗口前加入最后面的元素
            que.push(nums[i])
            # 记录对应的最大值
            result.append(que.front())
        return result

sol = Solution()
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(sol.maxSlidingWindow(nums,k))