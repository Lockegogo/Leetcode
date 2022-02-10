"""
给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。

输入: s = "abab"
输出: true
解释: 可由子串 "ab" 重复两次构成。
"""

class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != -1 and len(s) % (len(s) - (nxt[-1] + 1)) == 0:
            return True
        return False

    def getNext(self, nxt, s):
        """
        前缀表统一减一得到 next 数组
        """
        nxt[0] = -1
        j = -1
        # i 在 j 的后面
        for i in range(1, len(s)):
            # 如果 i 和 j+1 指向的字母不对，j 往前跳
            # 直到跳到和 i 指向相同的位置或者初始位置 -1
            while j >= 0 and s[i] != s[j+1]:
                j = nxt[j]
            if s[i] == s[j+1]:
                j += 1
            nxt[i] = j
        return nxt