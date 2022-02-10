"""
给你一个字符串 s ，逐个翻转字符串中的所有单词 。
请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

说明：
输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
翻转后单词间应当仅用一个空格分隔。
翻转后的字符串中不应包含额外的空格。

输入：s = "the sky is blue"
输出："blue is sky the"

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/reverse-words-in-a-string
"""
import re
class Solution:
    def reverseWords(self, s: str) -> str:
        result = re.split(r'\s+', s.strip())
        left, right = 0, len(result) - 1
        while(left < right):
            result[left], result[right] = result[right], result[left]
            left += 1
            right -= 1
        return ' '.join(result)

s = "  hello world  "
sol = Solution()
print(sol.reverseWords(s))


# ------------------------------ #
class Solution:
    # 1.去除多余的空格
    def trim_spaces(self,s):     
        n=len(s)
        left=0
        right=n-1
        # 去除开头的空格
        while left<=right and s[left]==' ':       
            left+=1
        # 去除结尾的空格
        while left<=right and s[right]==' ':        
            right=right-1
        tmp=[]
        # 去除单词中间多余的空格
        while left<=right:                                    
            if s[left]!=' ':
                tmp.append(s[left])
            elif tmp[-1]!=' ':  
                # 当前位置是空格，但是相邻的上一个位置不是空格，则该空格是合理的                                
                tmp.append(s[left])
            left+=1
        return tmp
    # 2.翻转字符数组
    def reverse_string(self,nums,left,right):
        while left<right:
            nums[left], nums[right]=nums[right],nums[left]
            left+=1
            right-=1
        return None
    # 3.翻转每个单词
    def reverse_each_word(self, nums):
        start=0
        end=0
        n=len(nums)
        while start<n:
            while end<n and nums[end]!=' ':
                end+=1
            self.reverse_string(nums,start,end-1)
            start=end+1
            end+=1
        return None

    # 4.翻转字符串里的单词
    # 测试用例："the sky is blue"
    def reverseWords(self, s): 
        # 输出：['t', 'h', 'e', ' ', 's', 'k', 'y', ' ', 'i', 's', ' ', 'b', 'l', 'u', 'e'
        l = self.trim_spaces(s)        
        # 输出：['e', 'u', 'l', 'b', ' ', 's', 'i', ' ', 'y', 'k', 's', ' ', 'e', 'h', 't']             
        self.reverse_string( l,  0, len(l) - 1)   
        # 输出：['b', 'l', 'u', 'e', ' ', 'i', 's', ' ', 's', 'k', 'y', ' ', 't', 'h', 'e']
        self.reverse_each_word(l)               
        return ''.join(l) 