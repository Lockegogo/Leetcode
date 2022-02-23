from typing import List


class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        # 将 A 按绝对值从大到小排列
        A = sorted(A, key=abs, reverse=True)
        for i in range(len(A)):
            if K > 0 and A[i] < 0:
                A[i] *= -1
                K -= 1
        if K > 0:
            A[-1] *= (-1)**K  # 取A最后一个数只需要写-1
        return sum(A)
