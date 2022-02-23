import numpy as np
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 初始化二维数组
        dp = np.zeros([m, n])
        dp[m-2, n-1] = 1
        # 在最右边
        for j in range(n):
            dp[m-1, j] = 1
        # 在最下边
        for i in range(m):
            dp[i, n-1] = 1

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i, j] = dp[i+1, j] + dp[i, j+1]
        return int(dp[0, 0])


m = 3
n = 7
sol =Solution()
print(sol.uniquePaths(m,n))
