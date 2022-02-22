"""
给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

输入："abbaca"
输出："ca"

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string
"""

# 开心消消乐？

class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        if s == []:
            return None
        for i in range(len(s)):
            # 消除后可能会有新的相同元素又碰到一起
            if stack == [] or s[i] != stack[-1]:
                stack.append(s[i])
            else:
                stack.pop()
        return "".join(stack)

sol = Solution()
s = "abbacctmt"
print(sol.removeDuplicates(s))