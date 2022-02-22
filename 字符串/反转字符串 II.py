"""
给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

输入：s = "abcdefg", k = 2
输出："bacdfeg"

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/reverse-string-ii
"""


class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        # 将字符串变为列表
        s = list(s)

        # 反转全部字符串的函数
        # 其实我们也可以用 s[::-1]
        def reverse(s):
            left, right = 0, len(s) - 1
            while(left < right):
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
            return s

        for i in range(0, len(s), 2*k):
            s[i:(i+k)] = reverse(s[i:(i+k)])

        # 将列表转换为字符串
        return ''.join(s)


s = "abcdefg"
k = 2
sol = Solution()
result = sol.reverseStr(s, k)
print(result)

# ---------------------- #


class Solution:
    # 1.去除多余的空格
    def trim_spaces(self, s):
        n = len(s)
        left = 0
        right = n-1

        while left <= right and s[left] == ' ':  # 去除开头的空格
            left += 1
        while left <= right and s[right] == ' ':  # 去除结尾的空格
            right = right-1
        tmp = []
        while left <= right:  # 去除单词中间多余的空格
            if s[left] != ' ':
                tmp.append(s[left])
            elif tmp[-1] != ' ':  # 当前位置是空格，但是相邻的上一个位置不是空格，则该空格是合理的
                tmp.append(s[left])
            left += 1
        return tmp

    # 2.翻转字符数组
    def reverse_string(self, nums, left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        return None
        
    # 3.翻转每个单词
    def reverse_each_word(self, nums):
        start = 0
        end = 0
        n = len(nums)
        while start < n:
            while end < n and nums[end] != ' ':
                end += 1
            self.reverse_string(nums, start, end-1)
            start = end+1
            end += 1
        return None

    # 4.翻转字符串里的单词
    def reverseWords(self, s):
        # 测试用例："the sky is blue"
        # 输出：['t', 'h', 'e', ' ', 's', 'k', 'y', ' ', 'i', 's', ' ', 'b', 'l', 'u', 'e']
        l = self.trim_spaces(s)
        # 输出：['e', 'u', 'l', 'b', ' ', 's', 'i', ' ', 'y', 'k', 's', ' ', 'e', 'h', 't']
        self.reverse_string(l,  0, len(l) - 1)
        # 输出：['b', 'l', 'u', 'e', ' ', 'i', 's', ' ', 's', 'k', 'y', ' ', 't', 'h', 'e']
        self.reverse_each_word(l)
        return ''.join(l)
