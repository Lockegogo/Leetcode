class Solution:
    def minCostClimbingStairs(self, cost):
        dp = [0] * (len(cost)+1)
        dp[0] = 0
        dp[1] = 0
        for i in range(2,len(cost)+1):
            dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
        return dp[-1]


cost = [1,100,1,1,1,100,1,1,100,1]
sol = Solution()
print(sol.minCostClimbingStairs(cost))