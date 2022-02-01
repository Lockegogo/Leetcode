"""
编写一个算法来判断一个数 n 是不是快乐数。

快乐数定义为：
1. 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
2. 然后重复这个过程直到这个数变为 1，也可能是无限循环 但始终变不到 1。
3. 如果这个过程结果为 1，那么这个数就是快乐数。
4. 如果 n 是快乐数就返回 true ；不是，则返回 false 。
"""

class Solution:
    def isHappy(self, n: int) -> bool:
        def calculate_happy(num):
            sum_ = 0
            
            # 从个位开始依次取，平方求和
            while num:
                # %: 取模，返回除法的余数
                sum_ += (num % 10) ** 2
                # //: 取整除，返回商的整数部分（向下取整）
                num = num // 10
            return sum_

        # 记录中间结果
        record = set()

        while True:
            n = calculate_happy(n)
            if n == 1:
                return True
            
            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

sol = Solution()
n = 1985
print(sol.isHappy(n))