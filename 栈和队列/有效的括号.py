"""
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

输入：s = "(]"
输出：false

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/valid-parentheses
"""

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(')')
            elif s[i] == '[':
                stack.append(']')
            elif s[i] == '{':
                stack.append('}')
            elif stack and s[i] == stack[-1]:
                stack.pop()
            else:
                return False
        # 如果 stack 为空，返回 True
        return stack == []


sol = Solution()
s = "["
print(sol.isValid(s))


