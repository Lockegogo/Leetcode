"""
给定两个字符串 s  和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
注意：如果 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

输入: s = "anagram", t = "nagaram"
输出: true

进阶：如果输入字符串包含 `unicode` 字符怎么办？你能否调整你的解法来应对这种情况？
"""

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in range(len(s)):
            # 并不需要记住字符 a 的 ASCII，只要求出一个相对数值就可以了
            # ord(): 是 chr () 函数（对于 8 位的 ASCII 字符串）或 unichr () 函数（对于 Unicode 对象）的配对函数，它以一个字符（长度为 1 的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值，如果所给的 Unicode 字符超出了你的 Python 定义范围，则会引发一个 TypeError 的异常。
            record[ord(s[i]) - ord("a")] += 1
        # print(record)
        for i in range(len(t)):
            record[ord(t[i]) - ord("a")] -= 1
        for i in range(26):
            if record[i] != 0:
                # record数组如果有的元素不为零，说明字符串 s 和 t 一定是谁多了字符或者谁少了字符。
                return False
                #如果有一个元素不为零，则可以判断字符串 s 和 t 不是字母异位词
                break
        return True

# sol = Solution()
# s = "anagram"
# t = "nagaram"
# print(sol.isAnagram(s,t))




# 介绍 defaultdict 的解题思路
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import defaultdict
        
        s_dict = defaultdict(int)
        t_dict = defaultdict(int)

        for x in s:
            s_dict[x] += 1
        
        for x in t:
            t_dict[x] += 1

        return s_dict == t_dict

sol = Solution()
s = "anagram"
t = "nagaram"
print(sol.isAnagram(s,t))