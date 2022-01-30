"""
题目：给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

提示：
1. 你可以假设 nums 中的所有元素是不重复的。
2. n 将在 [1, 10000] 之间。
3. nums 的每个元素都将在 [-9999, 9999] 之间。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-search
"""
# 注意二分查找需要数组有序
arr = [-1, 0, 3, 5, 9, 12]

# 左闭右闭 [left, right]
def search(x, arr):
    start, end = 0, len(arr) - 1
    # 如果循环结束还没有找到，返回 -1
    # 循环结束条件：start > end
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            start = mid + 1
        elif arr[mid] > x:
            end = mid - 1
    return -1


print(search(12, arr))

# 左闭右开
def search2(x, arr):
    start, end = 0, len(arr)
    # 如果循环结束还没有找到，返回 -1
    # 循环结束条件：start > end
    while start < end:
        mid = (start + end) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            start = mid + 1
        elif arr[mid] > x:
            end = mid
    return -1


print(search2(12, arr))


# -------------------------------------- #
# 递归法：左闭右开 or 左闭右闭
# 有点问题：x 超出范围后不返回 -1 而是报错
def binary_search(x, start, end):
    if start > end:  # 结束循环
        return -1
    mid = (end + start)//2
    if arr[mid] == x:
        return mid
    elif arr[mid] < x:
        return binary_search(x, mid + 1, end)
    elif arr[mid] > x:
        return binary_search(x, start, mid - 1)

# print(binary_search(12, 0, len(arr)))
