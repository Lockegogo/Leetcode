"""
给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

说明：当 needle 是空字符串时，我们应当返回什么值呢？0

输入：haystack = "hello", needle = "ll"
输出：2
"""

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a=len(needle)
        b=len(haystack)
        if a==0:
            return 0
        next=self.getnext(a,needle)
        p=-1
        for j in range(b):
            while p>=0 and needle[p+1]!=haystack[j]:
                p=next[p]
            if needle[p+1]==haystack[j]:
                p+=1
            if p==a-1:
                return j-a+1
        return -1

    def getnext(self,a,needle):
        next=['' for i in range(a)]
        # 1. 初始化
        # j 指针指向前缀末尾的位置，同时也代表 i 之前子串的最长相等前后缀的长度
        # i 指针指向后缀末尾的位置
        j=-1
        next[0]=j
        for i in range(1,len(needle)):
            # 3. 处理前后缀不相同的情况
            # 如果不相等，j 指针就要回退
            while (j>-1 and needle[j+1]!=needle[i]):
                j=next[j]
            # 3. 处理前后缀相同的情况
            # 如果相等，j 指针继续前进，同时还要将 j 赋给 next[i]
            if needle[j+1]==needle[i]:
                j+=1
            # next[i] 存储了 i 之前子串的最长相等前后缀的长度
            # next[i] 表相对于前缀表全部减一
            next[i]=j
        return next

haystack = 'aabaabaaf'
needle = 'aabaaf'
sol = Solution()
print(sol.strStr(haystack,needle))




# def getnext(needle):
#     a = len(needle)
#     next = ['' for i in range(a)]
#     # 1. 初始化
#     # j 指针指向前缀末尾的位置，同时也代表 i 之前子串的最长相等前后缀的长度
#     # i 指针指向后缀末尾的位置
#     i, j = 0, -1
#     next[0] = j
#     while(i < a-1):
#         # 2. 处理前后缀相同的情况
#         # 如果相等，j 指针继续前进，同时还要将 j 赋给 next[i]
#         if j == -1 or needle[j] == needle[i]:
#             j += 1
#             i += 1
#             # next[i] 存储了 i 之前子串的最长相等前后缀的长度
#             next[i] = j
#         # 3. 处理前后缀不相同的情况
#         # 如果不相等，j 指针就要回退
#         else:
#             j = next[j]
#     return next

# needle = 'aabaabaaf'
# result = getnext(needle)
