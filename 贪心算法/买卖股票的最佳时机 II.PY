"""
给定一个数组 prices ，其中 prices[i] 表示股票第 i 天的价格。

在每一天，你可能会决定购买和 / 或出售股票。你在任何时候最多只能持有一股股票。你也可以购买它，然后在同一天出售。
返回你能获得的最大利润和。

输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
"""

from typing import List
class Solution:
    """贪心算法"""
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(len(prices)-1):
            temp =prices[i+1] - prices[i]
            if temp > 0:
                profit += temp
        return profit


prices = [7,1,5,3,6,4]
sol = Solution()
print(sol.maxProfit(prices))



