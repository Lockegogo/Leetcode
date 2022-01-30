"""
给你一个正整数 n，生成一个包含 1 到 n^2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix。

输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
"""


class Solution:
    def generateMatrix(self, n):
        left, right, up, down = 0, n-1, 0, n-1
        # 初始化 matrix
        matrix = [[0]*n for _ in range(n)]
        num = 1
        while left <= right and up <= down:
            # 填充左到右：左闭右开
            for i in range(left, right+1):
                matrix[up][i] = num
                num += 1
            up += 1
            # 填充上到下
            for i in range(up, down+1):
                matrix[i][right] = num
                num += 1
            right -= 1
            # 填充右到左：步距 -1
            for i in range(right, left-1, -1):
                matrix[down][i] = num
                num += 1
            down -= 1
            # 填充下到上
            for i in range(down, up-1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
        return matrix

sol = Solution()
print(sol.generateMatrix(3))
