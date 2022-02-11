"""
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。
"""

from typing import List
# 时间复杂度：O(nlogk)
# 空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 要统计元素出现频率
        map_ = {}  # nums[i]:对应出现的次数
        for i in range(len(nums)):
            # get(key, default) 函数返回指定键的值，如果不存在返回 default
            map_[nums[i]] = map_.get(nums[i], 0) + 1

        # 对频率排序
        # 定义一个小顶堆，大小为 k
        pri_que = []  # 小顶堆
        # 用固定大小为 k 的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            # heapq: 只能构建小根堆，也是一种优先队列，它能以任意顺序增加对象，并且能在任意时间找到或移除最小的元素
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k:  # 如果堆的大小大于了 K，则队列弹出，保证堆的大小一直为 k
                heapq.heappop(pri_que)

        # 找出前 K 个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result


sol = Solution()
nums = [1, 1, 1, 2, 2, 3, 1, 1]
k = 2
print(sol.topKFrequent(nums, k))
