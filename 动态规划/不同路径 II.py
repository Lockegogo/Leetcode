import numpy as np
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        # 初始化二维数组
        obstacleGrid = np.array(obstacleGrid)
        m,n = obstacleGrid.shape
        dp = np.zeros([m, n])
        # 如果障碍出现在终点
        if obstacleGrid[m-1, n-1] == 1:
            return 0
        for j in range(n):
            dp[m-1, j] = 1
            # 如果障碍出现在边沿，边沿上和边沿左的全部为 0
            if obstacleGrid[m-1, j] == 1:
                for k in range(j+1):
                    dp[m-1,k] = 0

        for i in range(m):
            dp[i, n-1] = 1
            if obstacleGrid[i, n-1] == 1:
                for k in range(i+1):
                    dp[k, n-1] = 0

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                # 如果障碍出现在中间
                if obstacleGrid[i,j] == 1:
                    dp[i, j] = 0
                else:
                    dp[i, j] = dp[i+1, j] + dp[i, j+1]
        return int(dp[0, 0])

import numpy as np
obstacleGrid = [[0,0,0],[1,1,0],[0,0,0]]
sol =Solution()
print(sol.uniquePathsWithObstacles(obstacleGrid))