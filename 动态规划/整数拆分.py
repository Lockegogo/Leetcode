"""
给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。返回你可以获得的最大乘积。

示例 1: 输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
"""

class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[2] = 1
        for i in range(3, n+1):
            for j in range(i):
                dp[i] = max(dp[i],max((i-j)*j,(dp[i-j]*j)))
        return dp[n]

n = 58
sol = Solution()
print(sol.integerBreak(n))
