from typing import List


class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        # sort 直接在原列表上操作，无返回值
        # sorted 返回新的排好序 list，原来的列表保持不变
        # 将 A 按绝对值从大到小排列
        A = sorted(A, key=abs, reverse=True)
        for i in range(len(A)):
            # 如果 K 不够，先把绝对值大的负数取反
            if K > 0 and A[i] < 0:
                A[i] *= -1
                K -= 1
        # 如果 K 有富裕，在绝对值最小的数上反复取反操作
        if K > 0:
            # 取 A 最后一个数只需要写 -1
            A[-1] *= (-1)**K
        return sum(A)
