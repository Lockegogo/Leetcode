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
        a = []
        