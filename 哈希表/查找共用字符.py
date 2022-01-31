"""
给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按任意顺序返回答案。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。

输入：words = ["bella","label","roller"]
输出：["e","l","l"]
"""
from typing import List

class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        if not words: return []
        result = []
        # 用来统计所有字符串里字符出现的最小频率
        hash = [0] * 26
        # 用第一个字符给 hash 初始化
        for i,c in enumerate(words[0]):
            hash[ord(c) - ord('a')] += 1
        # 统计除第一个字符串外字符的出现频率
        for i in range(1,len(words)):
            hashOtherStr = [0] * 26
            for j in range(len(words[i])):
                hashOtherStr[ord(words[i][j]) - ord('a')] += 1
            # 更新 hash, 保证 hash 里统计 26 个字符在所有字符串里出现的最小
            for k in range(26):
                hash[k] = min(hash[k], hashOtherStr[k])
        # 将 hash 统计的字符次数，转换成输出形式
        for i in range(26):
            # 注意这里是 while，多个重复字符
            while hash[i] != 0:
                result.extend(chr(i+ord('a')))
                hash[i] -= 1
        return result

# words = ["bella","label","roller"]
# sol = Solution()
# print(sol.commonChars(words))


# ---------------------------------------------- #
import collections
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        tmp = collections.Counter(words[0])
        result = []
        for i in range(1,len(words)):
            # 使用 & 取交集: Counter({'l':2, 'e': 1})
            tmp = tmp & collections.Counter(words[i])

        # 剩下的就是每个单词都出现的字符（键），个数（值）
        for j in tmp:
            v = tmp[j]
            while(v):
                result.append(j)
                v -= 1
        return result

words = ["bella","label","roller"]
sol = Solution()
print(sol.commonChars(words))