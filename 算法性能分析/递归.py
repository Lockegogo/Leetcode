# 递归算法的时间复杂度：递归次数 * 每次递归的时间复杂度
import time


def function(x, n):
    if n == 0:
        return 1
    else:
        # 1. 递归算法必须有一个结束的点
        # 2. 满二叉树
        # 3. 把递归的结果存起来，存一次，用多次
        a = function(x, n // 2)
        if n % 2 == 0:
            # 这样做递归调用的时候做乘法，是常数项操作
            return a * a
        else:
            return a * a * x


# print(function(2,10))
# 斐波拉契求和


def fibonacci(i):
    if i == 1:
        return 1
    elif i <= 0:
        return 0
    else:
        return fibonacci(i-1) + fibonacci(i-2)


# time_start = time.time()
# print(fibonacci(35))
# time_end = time.time()
# print(f"time: {time_end-time_start}")


def HalfSearch(OrderedList, key, left, right):
    if left > right:
        return None
    mid = (left + right) // 2
    if key == OrderedList[mid]:
        return mid
    elif key > OrderedList[mid]:
        return HalfSearch(OrderedList, key, mid + 1, right)
    else:
        return HalfSearch(OrderedList, key, left, mid - 1)


time_start = time.time()
ls = [i for i in range(100)]
print(HalfSearch(ls, 8, 1, 100))
time_end = time.time()
print(f"binary_search_time: {time_end-time_start}")
