"""
有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

逆波兰表达式是一种后缀表达式，算符写在后面。其优点是：
1. 去掉括号后表达式无歧义
2. 适合用栈操作运算：遇到数字则入栈，遇到算符则取出栈顶两个元素进行计算，并将结果压入栈中。

注意两个整数之间的除法只保留整数部分。

可以保证给定的逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/evaluate-reverse-polish-notation
"""

from ast import List
from re import S


class Solution:
    def evalRPN(self, tokens):
        stack = list()
        cal = ["+", "-", "*", "/"]
        for i in range(len(tokens)):
            if tokens[i] not in cal:
                stack.append(tokens[i])
            else:
                # 第一个出来的数应该在运算符的后面
                a = int(stack.pop())
                # 第二个出来的数应该在运算符的前面
                b = int(stack.pop())
                # 注意 a 和 b 与运算符的位置
                result = int(eval(f'{b}{tokens[i]}{a}'))
                stack.append(result)
        return int(stack.pop())


sol = Solution()
tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
print(sol.evalRPN(tokens))
