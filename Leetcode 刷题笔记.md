Leetcode 刷题笔记

[TOC]

## 算法性能分析

### 1. 大 $O$ 的定义

==大 $O$ 的定义==：大 $O$ 就是数据量级突破一个点且数据量级非常大的情况下所表现出的时间复杂度，这个数据量也就是常数项系数已经不起决定性作用的数据量。

> 所以我们说的时间复杂度都是忽略常数项系数的，因为一般情况下都是默认数据规模足够大。

### 2. 时间和空间复杂度：递归

==递归算法的注意事项==：

1. 递归是在过程或函数中调用自身的过程
2. 递归必须有一个明确的递归结束条件，成为递归出口
3. 递归算法比较简洁，但运行效率较低
4. 递归调用过程，系统用==栈==来存储每一层的返回点和局部量，如果递归次数过多，容易造成==栈==溢出

==时间复杂度（递归）==：**递归的次数 * 每次递归的时间复杂度**

==空间复杂度（递归）==：**递归深度 * 每次递归的空间复杂度**

> 空间复杂度：一个算法在运行过程中占用内存空间大小的量度，利用程序的 空间复杂度，可以对程序运行中需要多少内存有个预先估计。

```python
# 斐波拉契求和
def fibonacci(i):
    if i == 1:
        return 1
    elif i <= 0:
        return 0
    else:
        return fibonacci(i-1) + fibonacci(i-2)
```

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291316373.webp" alt="图片" style="zoom:67%;" />

如果把递归过程抽象成一颗递归树，在这棵二叉树中每一个节点都是一次递归，而一棵深度为 k 的二叉树最多可以有 $2^k -1$ 个护节点。所以该递归算法的时间复杂度为 $O(2^n)$ 。

减少复杂度的方法：==把递归的结果存起来==。

==递归深度==如下：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291330993.webp" alt="图片" style="zoom:67%;" />

递归第 $n$ 个斐波那契数的话，递归调用栈的深度就是 $n$。那么每次递归的空间复杂度是 $O (1)$， 调用栈深度为 $n$，所以这段递归代码的空间复杂度就是 $O (n)$。

| 求斐波拉契数 | 时间复杂度 | 空间复杂度 |
| ------------ | ---------- | ---------- |
| 非递归       | $O(n)$     | $O(1)$     |
| 递归算法     | $O(2^n)$     | $O(n)$     |
| 优化递归算法 | $O(n)$     | $O(n)$     |

可以看出，有斐波拉契数的时候，使用递归算法并不一定是在性能上最优的，但递归算法确实简化了代码层面的复杂度。

---

==二分查找递归实现==：

```python
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
```

二分查找的时间复杂度是 $O(logn)$，那么递归二分查找的空间复杂度是多少呢？

我们依然看 **每次递归的空间复杂度和递归的深度**

首先我们先明确这里的空间复杂度里面的 n 是什么？二分查找的时候 n 就是指查找数组的长度，也就是代码中的 arr 数组。

每次递归的空间复杂度可以看出主要就是参数里传入的这个 arr 数组，即：$O (n)$。

再来看递归的深度，二分查找的递归深度是 $logn$ ，递归深度就是调用栈的长度，那么这段代码的空间复杂度为 $n * logn = O (nlogn)$。

如果希望递归二分查找的空间复杂度是 $O(logn)$，可以把这个数组放在外面而不是放在递归函数参数里，将数组定义为**全局变量**。

### 3. 代码的内存消耗

不同的编程语言各自的内存管理方式。

- C/C++ 这种内存堆空间的申请和释放完全靠自己管理
- Java 依赖 JVM 来做内存管理，不了解 jvm 内存管理的机制，很可能会因一些错误的代码写法而导致内存泄漏或内存溢出
- Python 内存管理是由私有堆空间管理的，所有的 python 对象和数据结构都存储在私有堆空间中。程序员没有访问堆的权限，只有解释器才能操作。

例如 Python 万物皆对象，并且将内存操作封装的很好，**所以 python 的基本数据类型所用的内存会要远大于存放纯数据类型所占的内存**，例如，我们都知道存储 int 型数据需要四个字节，但是使用 Python 申请一个对象来存放数据的话，所用空间要远大于四个字节。

==内存对齐==：为什么会有内存对齐？

- 平台原因：不是所有硬件平台都能访问任意内存地址上的任意数据，某些硬件平台只能在某些地址处取某些特定类型的数据，否则抛出硬件异常。为了同一个程序可以在多平台运行，需要内存对齐。
- 硬件原因：经过内存对齐后，CPU 访问内存的速度大大提升

## 数组

**数组是存放在连续内存空间上的相同类型数据的集合。**

数组可以方便的通过下标索引的方式获取到下标下对应的数据。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202201291415754.webp)

需要两点注意的是

- **数组下标都是从 0 开始的。**
- **数组内存空间的地址是连续的**

正是**因为数组的在内存空间的地址是连续的，所以我们在删除或者增添元素的时候，就难免要移动其他元素的地址。**

例如删除下标为 3 的元素，需要对下标为 3 的元素后面的所有元素都要做移动操作，如图所示：

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202201291418145.webp)

 Java 是没有指针的，同时也不对程序员暴漏其元素的地址，寻址操作完全交给虚拟机。 Java 的二维数组可能是如下排列的方式：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291423798.webp" alt="图片" style="zoom:67%;" />

> assert 用于判断一个表达式，在表达式条件为 false 的时候触发异常，断言可以在条件不满足程序运行的情况下直接返回错误，而不必等程度运行后出现崩溃的情况。

###  1. 二分查找

> 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

二分法的关键是：区间的定义。区间的定义是==不变量==。要在二分查找的过程中，保持不变量，就是在 while 寻找中每一次边界都要坚持根据区间的定义来操作，这就是==循环不变量规则==。

写二分法，区间的定义一般分为两种，左闭右闭即 [left, right]，或者左闭右开即 [left, right)。

#### 1.1 左闭右闭

- 循环结束条件：`while start <= end`，因为 `left == right` 在区间 [start, end] 是有意义的
- 当 `arr[mid] > x`时 end 要赋值为 `mid-1`，因为当前这个 `arr[mid]` 一定不是 target；另一种情况同理

```python
# 注意二分查找需要数组有序
arr = [-1, 0, 3, 5, 9, 12]

# 二分查找：左闭右闭 [left, right]
def search(x, arr):
    # 注意这里的 end = len(arr) - 1
    start, end = 0, len(arr) - 1
    # 如果循环结束还没有找到，返回 -1
    # 循环结束条件：start <= end
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
```

> 这种解法感觉比较直观一点。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291557553.webp" alt="图片" style="zoom: 80%;" />

#### 1.2 左闭右开

- 循环结束条件：`while start < end`，这里使用 < , 因为 `start = end` 在区间 [left, right) 是没有意义的
- 当 `arr[mid] > x`时 end 要赋值为 `mid`，因为当前这个 `arr[mid]`不等于 target，去左区间继续寻找，而寻找区间是左闭右开区间，所以 end 更新为 mid

```python
# 左闭右开
def search2(x, arr):
    # 注意这里的 end = len(arr)
    start, end = 0, len(arr)
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
```

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291557705.webp" alt="图片" style="zoom: 80%;" />

### 2. 原地移除元素

> 原地移除元素，并返回移除后数组的新长度。要求不使用额外的数组空间，必须使用 $O(1)$ 额外空间并原地修改输入数组。你不需要考虑数组中超出新长度后面的元素。

**要知道数组的元素在内存地址中是连续的，不能单独删除数组中的某个元素，只能覆盖。**

#### 2.1 暴力解法

暴力解法：两层 for 循环，一个 for 循环遍历数组元素，第二个 for 循环更新数组（用后面的替换前面的）。

```python
class Solution:
    def removeElement(self, nums, val):
        """
        暴力解法：发现需要移除的元素，就将数组集体向前移动一位
        """
        length = len(nums)
        i = 0
        while i < length:
            # 发现目标，长度减一，指针向前移动一位
            if nums[i] == val:
                # range 前闭后开；数组下标比长度少 1 
                # 注意 j 只循环到倒数第二位元素，是 length - 1
                for j in range(i, length-1):
                    nums[j] = nums[j+1]
                length -= 1
                nums = nums[:length]
                i -= 1
            i += 1
        return length, nums

nums = [0, 1, 2, 3, 2, 9, 6, 3, 2, 2, 2, 3, 2]
val = 2
sol = Solution()
len, arr = sol.removeElement(nums, val)
print(len, 'nums =', arr)
```

#### 2.2 双指针法

双指针法（快慢指针法）：通过一个快指针和慢指针在一个 for 循环下完成两个 for 循环的工作。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv43W2OFzic9tNsB9dGwCaYbQ1Td0CliauEJ9O31cb7bLxS9AsXN6Of0icic2rBuNBrwP9ibPsqHQEI2BTA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

双指针法（快慢指针法）在数组和链表的操作中是非常常见的，很多考察数组、链表、字符串等操作的面试题，都使用双指针法。

```python
class Solution2:
    def removeElement(self, nums, val):
        """
        双指针法
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        slowIndex = 0
        fastIndex = 0
        while fastIndex < len(nums):
            if val != nums[fastIndex]:
                nums[slowIndex] = nums[fastIndex]
                slowIndex += 1
                fastIndex += 1
            else:
                # 当快指针遇到要删除的元素时停止赋值
                # 慢指针停止移动，快指针继续前进
                fastIndex += 1
        return slowIndex, nums[:slowIndex]
```

### 3. 有序数组的平方

> 给你一个按==非递减顺序==排序的整数数组 nums，返回 每个数字的平方组成的新数组，要求也按非递减顺序排序。
>
> 输入：nums = [-4,-1,0,3,10]
> 输出：[0,1,9,16,100]
>
> 请设计时间复杂度为 O(n) 的算法解决本问题.

#### 3.1 暴力解法

最直观的算法莫过于：每个数平方之后进行排序。

```python
class Solution:
    def sortedSquares(self, nums):
        for i in range(len(nums)):
            nums[i] = nums[i] ** 2
        # 排序
        nums.sort()
        return nums
            
nums = [-4,-1,0,3,10,2]
a = Solution()
print(a.sortedSquares(nums))
```

#### 3.2 双指针法

数组其实是有序的，只不过负数平方之后可能成为最大数了。

那么数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。此时可以考虑双指针法，$i$ 指向起始位置，$j$ 指向终止位置。

定义一个新数组 result，和 $A$ 数组一样的大小，让 $k$ 指向 result 数组的终止位置。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv49PTAcQFBFFQtyH6RIEERSMIU4yk8AYZ3XI8cF1wJszznjJ1etuFxu4ibvvndawdu3nxfwUpibp9kA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
class Solution2:
    """
    双指针法
    """
    def sortedSquares(self, nums):
        length = len(nums)
        i, j, k = 0, length-1, length-1
        arr = [0]*len(nums)
        while k >= 0:
            rm = nums[j] ** 2
            lm = nums[i] ** 2
            if rm >= lm:
                arr[k] = rm
                j -= 1
            else:
                arr[k] = lm
                i += 1
            k -= 1
        return arr

nums = [-4, -1, 0, 3, 10]
a = Solution2()
print(a.sortedSquares(nums))
```

### ==4. 长度最小的子数组==

> 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。
>
> 示例：
> 输入：s = 7, nums = [2,3,1,2,4,3]
> 输出：2
> 解释：子数组 [4,3] 是该条件下的长度最小的子数组。

#### 4.1 暴力解法

从长度为 1 开始试，长度加 1，计算连续长度的子数组和，判断是否满足条件，如果满足返回该长度，不满足继续加 1。

该解法超出时间限制了：

```python
class Solution:
    """
    暴力算法
    """
    def minSubArrayLen(self, target, nums):
        length = 0
        result = 0
        while length <= len(nums):
            length += 1
            for i in range(len(nums) - length + 1):
                if sum(nums[i:i+length]) >= target:
                    result = length
                    return result
        return result


s = 100
nums = [2, 3, 1, 2, 4, 3, 7]
sol = Solution()
print(sol.minSubArrayLen(s, nums))
```

#### 4.2 滑动窗口

所谓滑动窗口，就是==不断调节子序列的起始位置和终止位置==，从而得出我们想要的结果。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6Dh6cLPrfmXOctLPWibfcWd4gzEh6DCeqpTNtAzEBtzpxf4JZfBOMnt7xEWYj5QJp22uzfUMYkxaQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

在本题中实现滑动窗口，主要确定如下三点：

- 窗口内是什么？
- 如何移动窗口的起始位置？
- 如何移动窗口的结束位置？

窗口就是满足其和 ≥ s 的长度最小的连续子数组。

窗口的结束位置如何移动：窗口的结束位置就是遍历数组的指针，窗口的起始位置为数组的数组的起始位置。

解题的关键在于窗口的起始位置如何移动？

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201292038812.webp" alt="图片" style="zoom:80%;" />

可以发现**滑动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。从而将 O (n^2) 的暴力解法降为 O (n)。**

```python
class Solution2:
    """
    滑动窗口
    """
    def minSubArrayLen(self, target, nums):
        # 定义一个无限大的数
        result = float("inf")
        start = 0
        sublength = 0
        for end in range(len(nums)):
            while sum(nums[start:end+1]) >= target:
                # 不能直接赋值 result, 要体现最小长度
                sublength = end-start + 1
                result = min(result, sublength)
                if sublength <= result:
                    result = sublength
                start += 1
        return 0 if result == float("inf") else result
```

==为什么时间复杂度是 $O(n)$==？

不要以为 for 里面放一个 while 就以为是 $O(n^2)$， 主要是看每一个元素被操作的次数，每个元素在滑动窗后进来操作一次，出去操作一次，每个元素都是被操作两次，所以时间复杂度是 2*n 也就是 $O(n)$

### ==5. 螺旋矩阵==

> 给你一个正整数 n，生成一个包含 1 到 $n^2$ 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix。
>
> ![img](https://gitee.com/lockegogo/markdown_photo/raw/master/202201292221425.jpeg)
>
> ```
> 输入：n = 3
> 输出：[[1,2,3],[8,9,4],[7,6,5]]
> ```

这道题目可以说在面试中出现频率较高的题目，不涉及什么算法，就是模拟过程，但十分考察对代码的掌控能力。

==循环不变量原则==，模拟顺时针画矩阵的过程：

- 填充上行从左到右
- 填充右列从上到下
- 填充下行从右到左
- 填充左列从下到上

由外向内一圈一圈这么画下去。

可以发现这里的边界条件非常多，在一个循环中，如此多的边界条件，如果不按照固定规则来遍历，那就是一进循环深似海，从此 offer 是路人。

这里一圈画下来，我们要画每四条边，这四条边怎么画，每画一条边都要坚持一致的左闭右开，或者左开右闭的原则，这样这一圈才能按照统一的规则画下来。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202201292234667.webp)

这里每一种颜色，代表一条边，我们遍历的长度，可以看出每一个拐角处的处理规则，拐角处让给新的一条边来继续画。这也是坚持了每条边左闭右开的原则。

但是下面的 python 代码是把一行画满再画下一步。

```python
class Solution:
    def generateMatrix(self, n):
        left, right, up, down = 0, n-1, 0, n-1
        # 初始化 matrix
        matrix = [[0]*n for _ in range(n)]
        num = 1
        while left <= right and up <= down:
            # 填充左到右：左闭右开
            for i in range(left, right+1):
                matrix[up][i] = num
                num += 1
            up += 1
            # 填充上到下
            for i in range(up, down+1):
                matrix[i][right] = num
                num += 1
            right -= 1
            # 填充右到左
            for i in range(right, left-1, -1):
                matrix[down][i] = num
                num += 1
            down -= 1
            # 填充下到上
            for i in range(down, up-1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
        return matrix

sol = Solution()
print(sol.generateMatrix(3))
```

## 链表

链表是一种通过指针串联在一起的线性结构，每一个节点是又两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向 null。链接的入口点称为列表的头节点 head。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301840625.png" alt="1-sll.png" style="zoom: 80%;" />

- 单链表：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301601892.webp" alt="图片" style="zoom:67%;" />

- 双链表：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv7pic3gapgzE7Xvlpj5vX9xeEMZiafETnkXzZfEqvEvIVp94gfXg6ic6POb1cWVia7h0kViarObN17AFZQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

- 循环链表：可以用来解决约瑟夫环问题

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301602336.webp" alt="图片" style="zoom:67%;" />

==链表的特点==：

1. 因为节点的内存地址不需要连续，所以相比顺序表，对于内存的利用更加高效
2. 同时管理器只需要存储第一个节点的地址即可，对于后续节点，也只需要前一个节点有指针即可
3. 根据下标的查询操作只能从第一个节点依次往后进行
4. 越靠近头部的操作时间复杂度越低，越靠近尾部的时间复杂度越高

==链表的存储方式==：数组在内存中是连续分布的，但是链表在内存中不是连续分布的，链表是通过指针域的指针链接在内存中各个节点，链表中的节点散乱分布在内存中的某地址上，分配机制取决于操作系统的内存管理。

==链表的定义==：

```c++
// 单链表
struct ListNode {
    int val;  // 节点上存储的元素
    ListNode *next;  // 指向下一个节点的指针
    ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数
};

// 通过自己定义构造函数初始化节点
ListNode* head = new ListNode(5);

// 使用默认构造函数初始化节点
ListNode* head = new ListNode();
head->val = 5;
// 如果不定义构造函数使用默认构造函数的话，在初始化的时候就不能直接给变量赋值
```

==链表的操作==：

1. **删除节点**：只要将 C 节点的 next 指针 指向 E 节点就可以了。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301608093.webp" alt="图片" style="zoom: 67%;" />

2. **添加节点**：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301609546.webp" alt="图片" style="zoom:67%;" />

> 数组在定义的时候，长度就是固定的，如果想改动数组的长度，就需要重新定义一个新的数组。
>
> 链表的长度可以是不固定的，并且可以动态增删，适合数据量不固定，频繁增删，较少查询的场景。

### 1. 移除链表元素

> 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回新的头节点。
>
> 输入：head = [1,2,6,3,4,5,6], val = 6
> 输出：[1,2,3,4,5]

这里以链表 1 4 2 4 来举例，移除元素 4。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201301615941.webp" alt="图片" style="zoom:67%;" />

如果使用 C | C++ 编程语言的话，不要忘了还要从内存中删除这两个移除的节点。如果使用 java ，python 的话就不用手动管理内存了。

==如果需要删除头节点怎么办？==

- 直接使用原来的链表来进行删除操作：将头结点向后移动一位
- 设置一个虚拟头节点再进行删除操作：设置虚拟头节点，按照统一方式进行移除

```python
class ListNode:
    """
    单个节点只需要存储两个值，在构造函数中赋值即可。默认情况下一个节点的地址放 None，等有需要时再进行赋值
    """

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_2_listnode(array):
    tem_node = ListNode()
    node = ListNode()
    for i in array:
        # 记得是判定val是否有值，并且用一个node记住头节点，然后返回的是头节点
        if not tem_node.val:
            tem_node.val = i
            node = tem_node
        else:
            tem_node.next = ListNode(i)
            tem_node = tem_node.next
    return node

class Solution:
    def removeElements(self, head, val):
        # 新建虚拟头节点 dummy
        dummy = ListNode(None)
        dummy.next = head
        cur = dummy
        while cur.next:
            if cur.next.val != val:
                # 指针向后移动一位
                cur = cur.next
            else:
                # 边重新连接
                cur.next = cur.next.next
        # 返回头节点
        return dummy.next

head = [1, 2, 6, 3, 4, 5, 2, 6]
# list --> linkNode
head = list_2_listnode(head)
print(head)
# 需要删除的元素值
val = 6
sol = Solution()
result = sol.removeElements(head, val)
while result:
    print(result.val, end='\t')
    result = result.next

```

### 2. 设计链表

> 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针 / 引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。
>
> 在链表类中实现这些功能：
>
> 1. `get (index)`：获取链表中第 index 个节点的值
> 2. `addAtHead (val)`：在链表的最前面插入一个值为 val 的节点
> 3. `addAtTail (val)`：在链表的最后面插入一个值为 val 的节点
> 4. `addAtIndex (index,val)`：在链表中的第 index 个节点前插入值为 val 的节点
> 5. `deleteAtIndex (index)`：删除链表中的第 index 个节点

```python
# 单链表
class Node:
    # 定义链表节点结构体
    def __init__(self, val):
        self.val = val
        self.next = None


class MyLinkedList:

    def __init__(self):
        self._head = Node(0)  # 虚拟头部节点
        self._count = 0  # 添加的节点数

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if 0 <= index < self._count:
            node = self._head
            for _ in range(index + 1):
                node = node.next
            return node.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self._count, val)

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index < 0:
            index = 0
        elif index > self._count:
            return

        # 计数累加
        self._count += 1

        # 新建节点
        add_node = Node(val)
        prev_node, current_node = None, self._head
        for _ in range(index + 1):
            prev_node, current_node = current_node, current_node.next
        else:
            prev_node.next, add_node.next = add_node, current_node

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if 0 <= index < self._count:
            # 计数-1
            self._count -= 1
            prev_node, current_node = None, self._head
            for _ in range(index + 1):
                prev_node, current_node = current_node, current_node.next
            else:
                prev_node.next, current_node.next = current_node.next, None
```

```python
# 双链表
# 相对于单链表, Node新增了prev属性
class Node:

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None


class MyLinkedList:

    def __init__(self):
        self._head, self._tail = Node(0), Node(0)  # 虚拟节点
        self._head.next, self._tail.prev = self._tail, self._head
        self._count = 0  # 添加的节点数

    def _get_node(self, index: int) -> Node:
        # 当index小于_count//2时, 使用_head查找更快, 反之_tail更快
        if index >= self._count // 2:
            # 使用prev往前找
            node = self._tail
            for _ in range(self._count - index):
                node = node.prev
        else:
            # 使用next往后找
            node = self._head
            for _ in range(index + 1):
                node = node.next
        return node

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if 0 <= index < self._count:
            node = self._get_node(index)
            return node.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self._update(self._head, self._head.next, val)

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self._update(self._tail.prev, self._tail, val)

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index < 0:
            index = 0
        elif index > self._count:
            return
        node = self._get_node(index)
        self._update(node.prev, node, val)

    def _update(self, prev: Node, next: Node, val: int) -> None:
        """
            更新节点
            :param prev: 相对于更新的前一个节点
            :param next: 相对于更新的后一个节点
            :param val:  要添加的节点值
        """
        # 计数累加
        self._count += 1
        node = Node(val)
        prev.next, next.prev = node, node
        node.prev, node.next = prev, next

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if 0 <= index < self._count:
            node = self._get_node(index)
            # 计数-1
            self._count -= 1
            node.prev.next, node.next.prev = node.next, node.prev


# Your MyLinkedList object will be instantiated and called as such:
index = 7
val = 5
obj = MyLinkedList()
param_1 = obj.get(index)
obj.addAtHead(val)
obj.addAtTail(val)
obj.addAtIndex(index, val)
obj.deleteAtIndex(index)
```

### 3. 反转链表

> 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
>
> 输入：head = [1,2,3,4,5]
> 输出：[5,4,3,2,1]
>
> 提示：链表可以选用迭代或递归方式完成反转。

如果再定义一个新的链表，实现链表元素的反转，这是对内存空间的浪费。

其实只需要改变链表的 next 指针的指向，直接将链表反转，而不用重新定义一个新的链表，如下图所示：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302117044.webp" alt="图片" style="zoom:67%;" />

之前链表的头节点是元素 1，反转之后头节点就是元素 5，这里并没有添加或者删除节点，仅仅是改变 next 指针的方向。

#### 3.1 双指针法

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv7ftmCo9j6fqIwpACbibyzDaeAjalAsyVzzxgSYicicuV3TH3vzia4rANEUghDYQPdiajHNJaWvsDTBSLQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

首先定义一个 cur 指针，指向头节点，再定义一个 pre 指针，初始化为 null。

然后开始反转，首先把 `cur.next` 节点用 `tmp` 指针保存一下，也就是保存一下 `cur` 指针指向节点的下一个节点。为什么要保存这个节点呢，因为接下来要改变 `cur.next`的指向，将其指向 `pre`，此时已经反转了第一个节点了。

接下来，就是循环走如下代码逻辑了，继续移动 `pre` 和 `cur` 指针。

最后，`cur` 指针已经指向了 `null`，循环结束，链表也反转完毕。此时我们 `return pre`指针就可以了，`pre` 指针就指向了新的头节点。

```python
# 双指针法
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur = head
        pre = None
        while(cur != None):
            # 保存一下 cur 的下一个节点，因为接下来要改变 cur->next
            temp = cur.next
            # 反转
            cur.next = pre
            # 更新 pre、cur 指针
            pre = cur
            cur = temp
        return pre
```

#### 3.2 递归法

递归法相对抽象，但是其实和双指针法是一样的逻辑，同样是当 `cur` 为空时循环结束，不断将 `cur` 指向 `pre`的过程。关键是初始化的地方。

```python
# 递归法
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        def reverse(pre, cur):
            if not cur:
                return pre
            tmp = cur.next
            cur.next = pre
            return reverse(cur, tmp)

        return reverse(None, head)
```

### 4. 删除链表的倒数第 N 个结点

> 给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。
>
> 输入：head = [1,2,3,4,5], n = 2
> 输出：[1,2,3,5]

#### 4.1 双指针法

双指针法的经典应用，如果要删除倒数第 $n$ 个节点，让 fast 移动 $n$ 步，然后让 fast 和 slow 同时移动，slow 指向被删除节点的上一个节点（此时 fast 指向链表末尾，`fast.next = None`，可以作为循环结束条件），利用 `slow.next = slow.next.next`就可以执行删除操作。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302210663.webp" alt="图片" style="zoom:67%;" />

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6XEYZuxIMibUKGOia3uXPT1QIpXDrVCkiaS4JOmMEquK7Ob6qiby32FefRvyY8fBWA225jRAxBLVT1uQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6XEYZuxIMibUKGOia3uXPT1Q68l06Y58F0g9wl53pwhJJDicmCnbntdUxvbhHARom8RnTvFY9ibZ0Kyg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode()
        dummy.next = head
        fast,slow = dummy, dummy
        # 快指针先走 n 步
        for _ in range(n):
            fast = fast.next
        # 快慢指针同时移动至 slow 指向被删除节点的上一个节点，方便删除
        while fast.next:
            fast = fast.next
            slow = slow.next
        # 删除 slow 所指向的节点
        slow.next = slow.next.next
        return head
```



### ==5. 链表相交==

> 给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null`。请注意相交的定义基于节点的==引用==，而不是基于节点的值。换句话说，如果一个链表的第 $k$ 个节点与另一个链表的第 $j$ 个节点是同一节点（引用完全相同），则这两个链表相交。
>
> <img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302216435.png" alt="img" style="zoom: 80%;" />
>
> 题目数据保证整个链式结构中不存在环。
> 注意：函数返回结果后，链表必须保持其原始结构。
> 你是否能设计一个时间复杂度 `O(n)` 、仅用 `O(1)` 内存的解决方案？
>
> 示例：
> 输入：listA = [4,1,8,4,5], listB = [5,0,1,8,4,5]
> 输出：Reference of the node with value = 8

#### 5.1 末尾对齐

简单来说，就是求两个链表交点节点的指针，交点不是数值相等，而是指针相等。

==算法==：我们求出两个链表的长度，并**求出两个链表长度的差值**，然后让 `curA` 移动到和 `curB` ==末尾对齐==的位置：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302225779.webp" alt="图片" style="zoom:67%;" />

此时我们就可以比较 `curA` 和 `curB` 是否相同，如果不相同，同时向后移动 `curA` 和 `curB`，如果遇到 `curA == curB`，则找到交点。否则循环退出返回空指针。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        countA=0
        countB=0
        curA=headA
        curB=headB
        
        while curA!=None:
            curA=curA.next
            countA+=1

        while curB!=None:
            curB=curB.next
            countB+=1

        # 让 curA 指向长链表
        if countA<countB:
            curA=headB
            curB=headA
        else:
            curA=headA
            curB=headB
            
        gap = abs(countA-countB)
        for i in range(gap):
            # 末尾对齐
            curA=curA.next
        
        while curA!=None:
            if curA==curB:
                return curA
            curA=curA.next
            curB=curB.next
```

#### 5.2 快慢指针法

可以这么理解，两个指针同时从头节点开始移动，有的链表长，有的链表短，如果有交点，说明它们有一段路是共用的（如果相遇，一定在共用的路上相遇），当指针开始走时，短的先到，可以想象成两段路一样长，但是短的链表的指针走的快，如果有交点，他们最终一定会在共用路段的第一个节点（交点）相遇。why?

这还需要用到一点数学知识。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        """
        根据快慢法则，走的快的一定会追上走得慢的。
        在这道题里，有的链表短，他走完了就去走另一条链表，我们可以理解为走的快的指针。
        那么，只要其中一个链表走完了，就去走另一条链表的路。如果有交点，他们最终一定会在同一个位置相遇。
        """
        # 用两个指针代替 a 和 b
        cur_a, cur_b = headA, headB     

        # 如果没有交点，能够走出循环吗？
        while cur_a != cur_b:
            # 如果 a 走完了，那么就切换到 b 走
            cur_a = cur_a.next if cur_a else headB  
            # 同理，b 走完了就切换到 a    
            cur_b = cur_b.next if cur_b else headA      
        
        return cur_a
```

### ==6. 环形链表==

> 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 `pos` 是 `-1`，则在该链表中没有环。
>
> 不允许修改链表。
>
> <img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302332897.png" alt="img" style="zoom:80%;" />
>
> 输入：head = [3,2,0,-4], pos = 1
> 输出：返回索引为 1 的链表节点
> 解释：链表中有一个环，其尾部连接到第二个节点

这道题目，不仅考察对链表的操作，而且还需要一些数学运算。

- 判断链表是否有环
- 如果有环，如何找到这个环的入口

#### 6.1 判断链表是否有环

可以使用==快慢指针法==，  分别定义 fast 和 slow 指针，从头结点出发，fast 指针每次移动两个节点，slow 指针每次移动一个节点，如果 fast 和 slow 指针在途中相遇 ，说明这个链表有环。

为什么 fast 走两个节点，slow 走一个节点，有环的话，一定会在环内相遇呢，而不是永远的错开呢？

首先第一点：**fast 指针一定先进入环中，如果 fast 指针和 slow 指针相遇的话，一定是在环中相遇，这是毋庸置疑的。**

那么来看一下，**为什么 fast 指针和 slow 指针一定会相遇呢？**

可以画一个环，然后让 fast 指针在任意一个节点开始追赶 slow 指针。会发现最终都是这种情况， 如下图：

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202201302340577.webp)

fast 和 slow 各自再走一步， fast 和 slow 就相遇了。这是因为 fast 是走两步，slow 是走一步，**其实相对于 slow 来说，fast 是一个节点一个节点的靠近 slow 的**，所以 fast 一定可以和 slow 重合。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6XEYZuxIMibUKGOia3uXPT1Q4EzD6ODP5hPLhuY6fOPrCffzK1YFLh6XHg7elvCgiaicibrZvy3tlqjGQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

#### 6.2 如何找到环的入口

假设从头结点到环形入口节点 的节点数为 $x$。环形入口节点到 fast 指针与 slow 指针相遇节点节点数为 $y$。从相遇节点再到环形入口节点节点数为 $z$。如图所示：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302348173.webp" alt="图片" style="zoom:80%;" />

那么相遇时：slow 指针走过的节点数为 $x+y$，fast 指针走过的节点数为 $x+y+n(y+z)$，$n$ 为 fast 指针在环内走了 $n$ 圈才遇到 slow 指针。

> 为什么第一次在环中相遇，slow 的 步数 是 x+y 而不是 x + 若干环的长度 + y 呢？
>
> 因为 slow 进环的时候，fast 一定是先进来了，而且在环的任意一个位置：
>
> <img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201310007789.webp" alt="图片" style="zoom:80%;" />
>
> 那么 fast 指针走到环入口 3 的时候，已经走了 $k+n$ 个节点，slow 相应走了 $(k+n)/2$ 个节点，因为 $k$ 小于 $n$，所以 $(k+n)/2$ 一定小于 $n$，这说明 slow  一定没有走到环入口 3，而 fast 已经到环入口 3 了，也就是**在 slow 开始走的那一环已经和 fast 相遇了**。精彩的分析！！

因为 fast 指针是一步走两个节点，slow 指针一步走一个节点， 所以 fast 指针走过的节点数 = slow 指针走过的节点数 * 2：
$$
\begin{aligned}
(x+y) * 2&=x+y+n(y+z)\\
x+y&=n(y+z)
\end{aligned}
$$
因为要找环形的入口，那么要求的是 x，因为 x 表示头结点到环形入口节点的的距离。整理如下：
$$
x=(n-1)(y+z)+z
$$
这就意味着，**==从头结点出发一个指针==，==从相遇节点也出发一个指针==，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是 环形入口的节点**。

操作步骤如下：

1. 找到相遇节点；
2. 在头节点和相遇节点同时定义两个指针，按链表行走，两个指针相遇的地方就是环的入口处。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # 如果相遇，说明有环
            if slow == fast:
                # 重新定义两个指针
                p = head
                q = slow
                while p!=q:
                    p = p.next
                    q = q.next
                #你也可以return q
                return p

        return None
```

> 思路很复杂，但是算法却很简单。两个循环就解决了。

## 哈希表

哈希表是根据关键码的值而直接进行访问的数据结构，直白来讲数组就是一张哈希表。哈希表中关键码就是数组的索引下表，然后通过下表直接访问数组中的元素，如下图所示：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201311522910.webp" alt="图片" style="zoom:67%;" />

**哈希表可以用来快速判断一个元素是否出现在集合里。**例如要查询一个名字是否在这所学校里。要枚举的话时间复杂度是 O (n)，但如果使用哈希表的话， 只需要 O (1) 就可以做到。我们只需要初始化把这所学校里学生的名字都存在哈希表里，在查询的时候通过索引直接就可以知道这位同学在不在这所学校里了。将学生姓名映射到哈希表上就涉及到了 **hash function ，也就是==哈希函数==**。

哈希函数通过 hashCode 把名字转化为数值，一般 hashcode 是通过特定编码方式，可以将其他数据格式转化为不同的数值，这样就把学生名字映射为哈希表上的索引数字了。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201311525968.webp" alt="图片" style="zoom:67%;" />

如果 hashCode 得到的数值大于哈希表的大小怎么办？

为了保证映射出来的索引数值都落在哈希表上，我们会再对数值做一个==取模==的操作。

但如果学生的数量大于哈希表的大小怎么办，此时就算哈希函数计算的再均匀，也避免不了会有几位学生的名字同时映射到哈希表同一个索引下标的位置。

### 1. ==哈希碰撞==

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201311530990.webp" alt="图片" style="zoom:67%;" />

哈希碰撞有两种解决办法，拉链法和线性探测法。

#### 1.1 拉链法

将发生冲突的元素存储再链表中：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201311532491.webp" alt="图片" style="zoom:67%;" />

#### 1.2 线性探测法

使用线性探测法，一定要保证 `tableSize` 大于 `dataSize`。我们需要依靠哈希表中的空位来解决碰撞问题。

例如冲突的位置，放了小李，那么就向下找一个空位放置小王的信息。所以要求 `tableSize` 一定要大于 `dataSize` ，要不然哈希表上就没有空置的位置来存放 冲突的数据了。如图所示：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201311533135.webp" alt="图片" style="zoom:67%;" />



==总结一下==：当我们遇到了要快速判断一个元素是否出现在集合里，就要考虑哈希法，但是哈希法也是牺牲了空间换取时间，因为我们要使用额外的数组，set 或者 map 来存放数据，才能实现快速的查找。

### 2. 有效的字母异位词

> 给定两个字符串 s  和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
>
> 注意：如果 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
>
> 输入: s = "anagram", t = "nagaram"
> 输出: true
>
> 进阶：如果输入字符串包含 `unicode` 字符怎么办？你能否调整你的解法来应对这种情况？

数组其实就是一个简单哈希表，而且这道题目中字符串只有小写字符，那么就可以定义一个数组，来记录字符串 s 里字符出现的次数。

需要定义一个多大的数组呢？定义一个数组 record，大小为 26 就可以了，初始化为 0，因为字符 a 到字符 z 的 ASCII 就是 26 个连续的数值，**字符 a 映射为下表 0，相应的字符 z 映射为下表 25。**遍历第一个字符时，字母出现一次对应位置元素加一；遍历第二个字符时，字母出现一次，对应位置元素减一；最后检查 record 数组如果有的元素不为 0，说明 字符串 s 和 t 一定是谁多了字符或者谁少了字符，`return false`。反之 `return true`。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in range(len(s)):
            #并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[ord(s[i]) - ord("a")] += 1
        # print(record)
        for i in range(len(t)):
            record[ord(t[i]) - ord("a")] -= 1
        for i in range(26):
            if record[i] != 0:
                #record数组如果有的元素不为零0，说明字符串s和t 一定是谁多了字符或者谁少了字符。
                return False
                #如果有一个元素不为零，则可以判断字符串s和t不是字母异位词
                break
        return True

sol = Solution()
s = "anagram"
t = "nagaram"
print(sol.isAnagram(s,t))
```

> 函数`ord()`是 `chr()` 函数（对于 8 位的 ASCII 字符串）或 `unichr()` 函数（对于 Unicode 对象）的配对函数，它以一个字符（长度为 1 的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值，如果所给的 Unicode 字符超出了你的 Python 定义范围，则会引发一个 `TypeError` 的异常。

```python
# 使用字典
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s_dict = {}
        for i in range(len(s)):
            if s[i] not in s_dict:
                s_dict[s[i]] = 1
            else:
                s_dict[s[i]] += 1
        for i in range(len(t)):
            if t[i] not in s_dict:
                return False
            else:
                s_dict[t[i]] -= 1
        for value in s_dict:
            if s_dict[value] != 0:
                return False
        return True
```

### 3. 查找共用字符

> 给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按任意顺序返回答案。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。
>
> 输入：words = ["bella","label","roller"]
> 输出：["e","l","l"]

这道题目一眼看上去，就是用哈希法，**“小写字符”，“出现频率”， 这些关键字都是为哈希法量身定做的啊**。

可以使用==暴力解法==，一个字符串一个字符串去搜，时间复杂度为 $O(n^m)$，$n$ 是字符串长度，$m$ 是有几个字符串。可以看出这是指数级别的时间复杂度，非常高，而且代码实现也不容易，因为要统计重复的字符，还要适当的替换或去重。

==哈希法==：整体思路就是统计出搜索字符串里 26 个字符的出现的频率，然后取每个字符频率最小值，最后转成输出格式就可以了。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201312112275.webp" alt="图片" style="zoom:67%;" />

```python
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

words = ["bella","label","roller"]
sol = Solution()
print(sol.commonChars(words))
```

```python
# 另一种解法
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
```

### 4. 两个数组的交集

> 给定两个数组`nums1`和`nums2`，返回它们的交集。输出结果中的每个元素一定是 **唯一** 的。我们可以 **不考虑输出结果的顺序** 。
>
> 输入：nums1 = [1,2,2,1], nums2 = [2,2]
> 输出：[2]

这道题目我们要学会使用一种哈希数据结构：`unordered_set`，这个数据结构可以解决很多类似的问题。注意题目特意说明：**输出结果中的每个元素一定是唯一的，也就是说输出的结果的去重的， 同时可以不考虑输出结果的顺序**。

使用数组来做哈希的题目，是因为题目都限制了数值的大小。而这道题目没有限制数值的大小，就无法使用数组来做哈希表了。而且如果哈希值比较少，特别分散或者跨度非常大，使用数组就造成空间的极大浪费，此时就需要使用另一种结构体：`set`。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201312215331.webp" alt="图片" style="zoom: 67%;" />

```python
from typing import List
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        setA = set(nums1)
        setB = set(nums2)
        result = list(setA & setB)
        return result

nums1 = [1,2,2,1]
nums2 = [2,2]
sol = Solution()
print(sol.intersection(nums1, nums2))
```

为什么我们遇到哈希问题不直接用 set，用什么数组？

因为直接使用 set 不仅占用空间比数组大，而且速度要比数组慢，set 把数值映射到 key 上都要做 hash 计算。

### ==5. 快乐数==

> 编写一个算法来判断一个数 n 是不是快乐数。
>
> ==快乐数== 定义为：
>
> - 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
> - 然后重复这个过程直到这个数变为 1，也可能是无限循环 但始终变不到 1。
> - 如果这个过程结果为 1，那么这个数就是快乐数。
> - 如果 n 是快乐数就返回 true ；不是，则返回 false 。

题目说了会无限循环，那么也就是说**求和的过程中，sum 会重复出现，这对解题很重要！所以这道题目使用哈希法，来判断这个 sum 是否重复出现，如果重复了就是 return false， 否则一直找到 sum 为 1 为止。**

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        def calculate_happy(num):
            sum_ = 0
            
            # 从个位开始依次取，平方求和
            while num:
                # %: 取模，返回除法的余数
                sum_ += (num % 10) ** 2
                # //: 取整除，返回商的整数部分（向下取整）
                num = num // 10
            return sum_

        # 记录中间结果
        record = set()

        while True:
            n = calculate_happy(n)
            if n == 1:
                return True
            
            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

sol = Solution()
n = 1985
print(sol.isHappy(n))
```

### 6. 两数之和

> 给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出和为目标值 `target` 的那两个整数，并返回它们的数组下标。
>
> 输入：`nums = [2,7,11,15], target = 9`
> 输出：`[0,1]`
> 解释：因为 `nums[0] + nums[1] == 9` ，返回 `[0, 1]`。

很明显暴力的解法是两层 for 循环查找，时间复杂度是 $O(n^2)$。

```python
from typing import List
class Solution:
    """
    暴力算法
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    sum = nums[i] + nums[j]
                    if sum == target:
                        return list([i,j])

nums = [2,7,11,15]
target = 18
sol = Solution()
print(sol.twoSum(nums,target))
```

本题我们使用 map，先看下使用数组和 set 来做哈希法的局限：

- 数组的大小是受限制的，而且如果元素很少，而哈希值太大会造成内存空间的浪费；
- set 是一个集合，里面放的元素只能是一个 key，而两数之和这道题目，不仅要判断 $y$ 是否存在而且还要记录 $y$ 的下标位置，所以 set 也不能用。

此时就要选择另一种数据结构：map ，map 是一种 key-value 的存储结构，可以用 key 保存数值，用 value 在保存数值所在的下标。

```python
# 更好的解法
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        records = dict()
        # 用枚举更方便，就不需要通过索引再去取当前位置的值
        for idx, val in enumerate(nums):
            # 寻找 target - val 是否在 map 中
            if target - val not in records:
                records[val] = idx
            else:
                return [records[target - val], idx] # 如果存在就返回字典记录索引和当前索引

nums = [2,7,11,15]
target = 18
sol = Solution()
print(sol.twoSum(nums,target))
```

### 7. 四数相加

> 给你四个整数数组 `nums1`、`nums2`、`nums3` 和 `nums4`，数组长度都是 $n$ ，请你计算有多少个元组 $(i, j, k, l)$ 能满足：
>
> - $0 <= i, j, k, l < n$
> - 有`nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`
>
> 输入：`nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]`
> 输出：2

==本题解题思路==：

1. 首先定义一个字典，key 放 a 和 b 两数之和，value 放 a 和 b 两数之和出现的次数
2. 遍历 A 和 B 数组，统计两个数组之和以及出现的次数，放在字典中
3. 定义变量 count，用来统计 $a+b+c+d = 0$ 出现的次数
4. 继续遍历 C 和 D 数组，找到如果 0-(c+d) 在字典中出现过的话，就用 count 把字典中 key 对应的 value 也就是出现次数统计出来
5. 最后返回统计值 count 就可以了

```python
from typing import List

class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        result = {}
        count = 0
        for i in nums1:
            for j in nums2:
                temp = i+j
                if temp not in result:
                    result[temp] = 1
                else:
                    result[temp] += 1
        for k in nums3:
            for t in nums4:
                temp = -(k+t)
                if temp in result:
                    count += result[temp]
        return count
```

### 8. 赎金信

> 给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。如果可以，返回 true ；否则返回 false 。magazine 中的每个字符只能在 ransomNote 中使用一次。
>
> 输入：`ransomNote = "a", magazine = "b"`
> 输出：false
>
> 输入：`ransomNote = "aa", magazine = "aab"`
> 输出：true

本题需要注意两点：

1. 为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思，说明杂志里面的字母不可重复使用
2. 你可以假设两个字符串均只含有小写字母

因为题目只有小写字母，那可以采用空间换取时间的哈希策略，用一个长度为 26 的数组去记录 magazine 里字母出现的次数。

然后再用 `ransomNote` 去验证这个数组是否包含了 `ransomNote` 所需要的所有字母：依然是数组在哈希法中的应用。

为什么不用 map 呢？其实在本题，使用 map 的空间消耗要比数组大一些，因为 map 需要维护红黑树或者哈希表，而且还要做哈希函数，是费时的，数据量大的话就能体现出差别来了。

下面给出用==字典==做赎金信的代码：

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        def str2dict(x):
            result = {}
            for i in x:
                if i not in result:
                    result[i] = 1
                else:
                    result[i] += 1
            return result
        ransomNote = str2dict(ransomNote)

        # 去杂志中找，找到就减一
        for t in magazine:
            if t in ransomNote:
                ransomNote[t] -= 1

        # 遍历字典，如果还有 value 值大于 0，说明赎金信中还有字母没有在杂志中找到，返回 False
        for key in ransomNote:
            if ransomNote[key] > 0:
                return False
        
        return True
```

### ==9. 三数之和==：去重

> 给你一个包含 n 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
>
> 注意：答案中不可以包含重复的三元组。
>
> 输入：`nums = [-1,0,1,2,-1,-4]`
> 输出：`[[-1,-1,2],[-1,0,1]]`

两层 for 循环就可以确定 a 和 b 的数值了，可以使用哈希法来确定 $0-(a+b)$ 是否在数组里出现过，但是题目要求不可以包含重复的三元组，把符合条件的三元组放进 vector 中，然后再去重，这样是非常费时的，很容易超时，去重的过程不好处理，有很多小细节，如果在面试中很难想到位。时间复杂度可以做到 $O (n^2)$，但还是比较费时的，因为不好做剪枝操作。

所以，这道题使用哈希法并不合适，因为在去重的操作中有很多细节需要注意，==双指针法==比哈希法高效一些。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202202011742067.gif)

1. 将数组排序，一层 for 循环，$i$ 从下标 0 的地方开始，同时定一个下标 left 定义在 $i+1$ 的位置上，定义下标 right 在数组结尾的位置上
2. 依然还是在数组中找到 `abc` 使得 $a + b +c =0$，我们这里相当于  `a = nums [i], b = nums [left], c = nums [right]`
3. 如果 `nums [i] + nums [left] + nums [right] > 0`  就说明此时三数之和大了，因为数组是==排序后==的，所以 right 就应该向左移动，这样才能让三数之和小一些
4. 如果 `nums [i] + nums [left] + nums [right] < 0` 说明此时三数之和小了，left 就向右移动，才能让三数之和大一些，直到 left 与 right 相遇为止
4. 注意，由于结果不能包含重复的三元组，所有遇到相同的元素指针不要停留，继续移动。

三数之和的==双指针解法==是一层 for 循环 `num[i]` 为确定值，然后循环内有 left 和 right 作为双指针，找到  `nums [i] + nums [left] + nums [right] == 0`。

```python
from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)):
            left = i + 1
            right = len(nums) - 1
            if nums[i] > 0:
                break
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    # 将满足条件的数组存起来
                    # 因为已经排序过了，所以一定有 nums[i] <= nums[left] <= nums[right]
                    result.append((nums[i], nums[left], nums[right]))
                    # 因为结果不能有重复的三元组，所以遇到相同的元素指针继续移动
                    while left != right and nums[left] == nums[left + 1]:
                        left += 1
                    while left != right and nums[right] == nums[right - 1]:
                        right -= 1
                    # 如果左边连续的两个不相同，右边也不相同，那么左右指针可以同时移动
                    # 因为如果只移动一个只变动一个值，三元组的和一定不再等于 0，逻辑推断可以帮助我们减少一次判断
                    left += 1
                    right -= 1
                elif total > 0:
                    right -= 1
                elif total < 0:
                    left += 1
        return result

nums = [-1, 0, 1, 2, -1, -4]
sol = Solution()
print(sol.threeSum(nums))
```

### ==10. 四数之和==：去重

> 给你一个由 n 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且==不重复==的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：
>
> 1. `0 <= a, b, c, d < n`
> 2. a、b、c 和 d 互不相同
> 3. `nums[a] + nums[b] + nums[c] + nums[d] == target`
>
> 输入：`nums = [1,0,-1,0,-2,2], target = 0`
> 输出：`[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]`

四数之和的==双指针解法==是两层 for 循环 `nums [k] + nums [i]`为确定值，依然是循环内有 left 和 right 作为双指针，找出 `nums [k] + nums [i] + nums [left] + nums [right] == target`的情况，三数之和的时间复杂度是 $O(n^2)$，四数之和的时间复杂度是 $O(n^3)$。

> 和==四数相加==不同，四数相加是四个独立的数组，只要找到 `A [i] + B [j] + C [k] + D [l] = 0`就可以，不用考虑有重复的四个元素相加等于 0 的情况；而本题==四数之和==要求在一个集合中找出四个数相加等于 target，同时四元组不能重复。还是使用==双指针法==。

```python
from typing import List
class Solution:
    """
    双指针法
    """
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        result = []
        nums.sort()
        for j in range(len(nums)):
            # 去重
            if j > 0 and nums[j] == nums[j-1]:
                # continue: 跳出本次循环，继续进行下一轮循环
                # break: 结束所有循环
                continue
            for i in range(j+1,len(nums)):
                # 去重
                if i > j + 1 and nums[i] == nums[i-1]:
                    continue
                left = i + 1
                right = len(nums) - 1
                while left < right:
                    total = nums[j] + nums[i] + nums[left] + nums[right]
                    if total == target:
                        # 将满足条件的数组存起来
                        result.append((nums[j], nums[i], nums[left], nums[right]))
                        # 因为结果不能有重复的三元组，所以遇到相同的元素指针继续移动
                        while left != right and nums[left] == nums[left + 1]:
                            left += 1
                        while left != right and nums[right] == nums[right - 1]:
                            right -= 1
                        # 如果左边连续的两个不相同，右边也不相同，那么左右指针可以同时移动
                        # 因为如果只移动一个只变动一个值，三元组的和一定不再等于 0，逻辑推断可以帮助我们减少一次判断
                        left += 1
                        right -= 1
                    elif total > target:
                        right -= 1
                    elif total < target:
                        left += 1
        return result

nums = [2,2,2,2,2,2]
target = 8
sol = Solution()
print(sol.fourSum(nums, target))
```

> 知识点：==跳出本次循环，继续进行下一轮循环==
>
> - for - continue
> - while - i++

> 三数之和和四数之和都是使用的双指针法；而两数之和使用的是字典法。

## 字符串

### 1. 反转字符串

> 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
>
> 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O (1) 的额外空间解决这一问题。
>
> 输入：s = ["h","e","l","l","o"]
> 输出：["o","l","l","e","h"]

在反转链表中，使用了双指针的方法，那么反转字符串依然是使用双指针的方法，只不过对于字符串的反转更加简单一些。因为字符串也是一种数组，所以元素在内存中是连续分布，这就决定了反转链表和反转字符串方式上还是有所差异的。

对于字符串，我们定义两个指针（索引下标），一个从字符串前面，一个从字符串后面，两个指针同时向中间移动你，并交换元素。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202202021045240.gif)

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while(left < right):
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```



### 2. 反转字符串 II

> 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
>
> 如果剩余字符少于 k 个，则将剩余字符全部反转。
>
> 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
>
> 输入：`s = "abcdefg", k = 2`
> 输出：`"bacdfeg"`

在遍历字符串的过程中，只要让 `i += (2*k)`, `i` 每次移动 $2*k$ ，然后判断是否需要有反转的区间。

==当需要固定规律一段一段去处理字符串的时候，要想想在 for 循环的表达式上做做文章==

```python
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
result = sol.reverseStr(s,k)
print(result)
```



### 3. 替换空格

> 请实现一个函数，把字符串 s 中的每个空格替换成 "%20"。
>
> 输入：`s = "We are happy."`
> 输出：`"We%20are%20happy."`

首先扩充数组到每个空格替换成 "%20" 之后的大小。然后从后向前替换空格，也就是双指针法，过程如下：$i$ 指向新长度的末尾，$j$ 指向旧长度的末尾。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv4rrwjMJIUoIpiaHQ4Iiae3S5yuecBvUEic1pKiagE7VLAHnSC4iawGXibgicT4Igb9ib4QTLWAofWYibJhxCw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

> **从前向后填充可以吗？**
>
> 从前向后填充就是 $O(n^2)$ 的算法，因为每次添加元素都要将添加元素之后的左右元素向后移动。

其实很多==数组填充类==的问题，都可以先预先给数组==扩容==带填充后的大小，然后在==从后向前==进行操作。

这么做有两个好处：

1. 不用申请新数组；
2. 从后向前填充元素，避免了从前向后填充元素要将添加元素之后的所有元素向后移动。

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        # 数空格
        counter = s.count(' ')
        res = list(s)
        # 每碰到一个空格就多拓展两个格子(空格)，1 + 2 = 3个位置存'%20'
        res.extend([' '] * counter * 2)
        # 原始字符串的末尾，拓展后的末尾
        left, right = len(s) - 1, len(res) - 1

        while left >= 0:
            # 如果不是空格
            if res[left] != ' ':
                res[right] = res[left]
                right -= 1
            else:
                # [right - 2, right), 左闭右开
                res[right - 2: right + 1] = '%20'
                right -= 3
            left -= 1
        return ''.join(res)
```

### 4. 翻转字符串里的单词

> 给你一个字符串 s ，逐个翻转字符串中的所有单词 。
> 请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。
>
> 说明：
> 输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
> 翻转后单词间应当仅用一个空格分隔。
> 翻转后的字符串中不应包含额外的空格。
>
> 输入：`s = "the sky is blue"`
> 输出：`"blue is sky the"`

```python
# 我的解法
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
```

提高下本题的难度：**不要使用辅助空间，空间复杂度要求为 O (1)。**

想一下，我们将整个字符串都反转过来，那么单词的顺序指定是倒序了，只不过单词本身也倒叙了，那么再把单词反转一下，单词不就正过来了。

1. 移除多余空格
2. 将整个字符串反转
3. 将每个单词反转

要对一句话里的单词顺序进行反转，==先整体反转再局部反转==是一个很妙的思路。

举个例子，源字符串为："the sky is blue"

- 移除多余空格 : "the sky is blue"
- 字符串反转："eulb si yks eht"
- 单词反转："blue is sky the"

### 5. 左旋转字符串

> 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。要左旋转字符串，可以==先局部反转再整体反转==。
>
> 输入: `s = "abcdefg", k = 2`
> 输出: `"cdefgab"`

```python
# 方法一：可以使用切片方法
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[0:n]
```

```python
# 方法二：也可以使用上文描述的方法，有些面试中不允许使用切片，那就使用上文作者提到的方法
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        s = list(s)
        s[0:n] = list(reversed(s[0:n]))
        s[n:] = list(reversed(s[n:]))
        s.reverse()
        
        return "".join(s)
```

### 6. 实现 strStr() 函数：KMP

> 给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。
>
> 说明：当 `needle` 是空字符串时，我们应当返回什么值呢？==0==
>
> 输入：`haystack = "hello", needle = "ll"`
> 输出：`2`

#### 6.1 KMP

==什么是 KMP？==  Knuth，Morris 和 Pratt

==KMP 有什么用？==

KMP 的经典思想是: **当出现字符串不匹配时，可以记录一部分之前已经匹配的文本内容，利用这些信息避免从头再去做匹配。**

所以如何记录已经匹配的文本内容，是 KMP 的重点，也是 `next` 数组肩负的重任。

==什么是前缀表？==

前缀表是用来**回退**的，它记录了**模式串**（短）与**主串**（文本串，长）不匹配的时候，模式串应该从哪里开始重新匹配

本质上`next` 数组就是一个前缀表（prefix table）。

举个栗子：要在文本串：`aabaabaafa` 中查找是否出现过一个模式串：`aabaaf`。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202202021344894.gif)

可以看出，文本串中第六个字符 b 和模式串中第六个字符 f 不匹配。如果暴力匹配，此时就要**从头匹配**了。但是如果使用前缀表，就不会从头匹配，而是从上次已经匹配的内容开始匹配，找到了模式串中第三个字符 b 继续开始匹配。

**那前缀表是如何记录的呢？**

首先要知道前缀表的任务是当前位置匹配失败，找到之前已经匹配上的位置，再重新匹配，也意味着在某个字符失配时，前缀表会告诉你下一步匹配中，模式串应该跳到哪个位置。

**前缀表**：**记录下标 $i$ 之前（包括 $i$）的字符串中，有多大长度的相同前缀后缀。**

==最长公共前后缀：==

文章中字符串的**前缀是指不包含最后一个字符的所有以第一个字符开头的连续子串**。**后缀是指不包含第一个字符的所有以最后一个字符结尾的连续子串**。

前缀表要求的就是相同前后缀的长度。

而“最长公共前后缀”里面的 “公共”，更像是说前缀和后缀公共的长度。这其实并不是前缀表所需要的。

所以字符串 a 的最长相等前后缀为 0。字符串 aa 的最长相等前后缀为 1。字符串 aaa 的最长相等前后缀为 2。

==为什么一定要用前缀表？==

回顾一下，刚刚匹配的过程在下标 5 的地方遇到不匹配，模式串是指向 $f$，如图：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202202021354494.webp" alt="图片" style="zoom:80%;" />

然后就找到了下标 2，指向 b，继续匹配：如图：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202202021355038.webp" alt="图片" style="zoom:80%;" />

**下标 5 之前这部分的字符串（也就是字符串 aabaa）的最长相等的前缀 和 后缀字符串是 子字符串 aa ，因为找到了最长相等的前缀和后缀，匹配失败的位置是后缀==子串的====后面==，那么我们找到与其相同的==前缀的后面==从新匹配就可以了。**==而前缀的后面这个位置的下标正好是前缀的长度，也就是前缀表中存储的值。==

所以前缀表具有告诉我们当前位置匹配失败，跳到之前已经匹配过的地方的能力。

==如何计算前缀表？==

长度为前 1 个字符的子串 `a`，最长相同前后缀的长度为 0。（注意字符串的**前缀是指不包含最后一个字符的所有以第一个字符开头的连续子串**；**后缀是指不包含第一个字符的所有以最后一个字符结尾的连续子串**。）

长度为前 2 个字符的子串 `aa`，最长相同前后缀的长度为 1。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202202021446973.webp" alt="图片" style="zoom:80%;" />

长度为前 3 个字符的子串 `aab`，最长相同前后缀的长度为 0。

以此类推：长度为前 4 个字符的子串 `aaba`，最长相同前后缀的长度为 1。长度为前 5 个字符的子串 `aabaa`，最长相同前后缀的长度为 2。长度为前 6 个字符的子串 `aabaaf`，最长相同前后缀的长度为 0。

那么把求得的最长相同前后缀的长度就是对应前缀表的元素，如图：

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202202021452316.webp" alt="图片" style="zoom:80%;" />

可以看出模式串与前缀表对应位置的数字表示的就是：**下标 $i$ 之前（包括 $i$）的字符串中，有多大长度的相同前缀后缀。**

![pic](D:\Dropbox\工作计划\Leetcode\pics\640.gif)

找到的不匹配的位置， 那么此时我们要看它的前一个字符的前缀表的数值是多少。为什么要前一个字符的前缀表的数值呢，因为要找前面字符串的最长相同的前缀和后缀。前一个字符的前缀表的数值是 2， 所有把下标移动到下标 2 的位置继续比配。最后就在文本串中找到了和模式串匹配的子串了。

==前缀表与 next 数组==

很多 KMP 算法的时间都是使用 `next` 数组来做回退操作，那么 next 数组与前缀表有什么关系呢？

`next` 数组就可以是前缀表，但是很多实现都是把前缀表统一减一（右移一位，初始位置为 - 1）之后作为 `next` 数组。

右移操作之后，比较遇到不匹配时，直接看 f 这个位置(用前缀表要比较 f 的前一个位置的值)对应的值去跳就可以了。

==使用 next 数组来匹配==

有了 next 数组，就可以根据 next 数组来 匹配文本串 s，和模式串 t 了。注意 next 数组是新前缀表（旧前缀表统一减一了）。

![](D:\Dropbox\工作计划\Leetcode\pics\2.gif)

==时间复杂度分析==

其中 $n$ 为文本串长度，$m$ 为模式串长度，因为在匹配的过程中，根据前缀表不断调整匹配的位置，可以看出匹配的过程是 $O (n)$，之前还要单独生成 next 数组，时间复杂度是 $O (m)$。所以整个 KMP 算法的时间复杂度是 $O (n+m)$ 的。

暴力的解法显而易见是 $O (n * m)$，所以 **KMP 在字符串匹配中极大的提高的搜索的效率。**

都知道使用 KMP 算法，一定要构造 next 数组。

==构造 next 数组==

我们定义一个函数 `getNext` 来构建 next 数组，函数参数为指向 next 数组的指针，和一个字符串。

**构造 next 数组其实就是计算模式串 s，前缀表的过程。** 主要有如下三步：

1. **初始化**：定义两个指针 $i$ 和 $j$，$j$ 指向前缀起始位置，$i$ 指向后缀起始位置，然后还要对 next 数组进行初始化赋值，`next[i]` 表示 $i$ （包括 $i$）之前最长相等的前后缀长度（其实就是 $j$）
2. **处理前后缀不相同的情况**：$i$ 从 1 开始，进行 `s[i]` 与 `s[j+1]` 的比较，所以遍历模式串 s 的循环下标 $i$ 需要从 1 开始，如果遇到 `s[i]` 与 `s[j+1]` 不相同的情况，就要向前回退。怎么回退呢？`next[j]` 就是记录着 $j$（包括 $j$）之前的子串的相同前后缀的长度，那么  `s[i]` 与 `s[j+1]` 不相同，就要找 $j+1$ 前一个元素在 next 数组里的值（就是 `next[j]` ）
3. **处理前后缀相同的情况**：如果 `s[i]` 和 `s[j+1]`相同，那么就同时向后移动 $i$ 和 $j$ 说明找到了相同的前后缀，同时还要将 $j$ （前缀的长度）赋给 `next[i]`，因为 `next[i]` 要记录相同前后缀的长度。

![](D:\Dropbox\工作计划\Leetcode\pics\3.gif)

```python
def getnext(needle):
    a = len(needle)
    next = ['' for i in range(a)]
    # 1. 初始化
    # j 指针指向前缀末尾的位置，同时也代表 i 之前子串的最长相等前后缀的长度
    # i 指针指向后缀末尾的位置
    i, j = 0, -1
    next[0] = j
    while(i < a-1):
        # 2. 处理前后缀相同的情况
        # 如果相等，j 指针继续前进，同时还要将 j 赋给 next[i]
        if j == -1 or needle[j] == needle[i]:
            j += 1
            i += 1
            # next[i] 存储了 i 之前子串的最长相等前后缀的
            next[i] = j
        # 3. 处理前后缀不相同的情况
        # 如果不相等，j 指针就要回退
        else:
            j = next[j]
    return next
```

得到了 next 数组之后，就要用这个来做匹配了。

==使用 next 数组来做匹配==

在文本串 s 里 找是否出现过模式串 t。

定义两个下标 $j$ 指向模式串起始位置，$i$ 指向文本串起始位置。

那么 $j$ 初始值依然为 - 1，为什么呢？**依然因为 next 数组里记录的起始位置为 - 1。**$i$ 就从 0 开始，遍历文本串。接下来就是 s[i] 与 t[j + 1] （因为 $j$ 从 - 1 开始的） 进行比较。

- 如果 s[i] 与 t[j + 1] 不相同，$j$ 就要从 next 数组里寻找下一个匹配的位置
- 如果 s [i] 与 t [j + 1] 相同，那么 $i$ 和 $j$ 同时向后移动

如何判断在文本串 s 里出现了模式串 t 呢，如果 $j$ 指向了模式串 t 的末尾，那么就说明模式串 t 完全匹配文本串 s 里的某个子串了。



### ==7. 重复的子字符串==：KMP

> 给定一个非空的字符串 `s` ，检查是否可以通过由它的一个子串重复多次构成。
>
> 输入: s = "abab"
> 输出: true
> 解释: 可由子串 "ab" 重复两次构成

又是一道标准的 KMP 题目。

在一个串中查找是否出现过另一个串，这是 KMP 的看家本领。那么寻找重复子串怎么也涉及到 KMP 算法了呢？

我们知道`next` 数组记录的最长相同前后缀，如果 `next [len - 1] != -1`，则说明字符串有最长相同的前后缀，且最长相等前后缀的长度为：`next [len - 1] + 1`

如果可以由子串重复构成，最后一位的 next 值一定不为 -1。

如果 `len % (len - (next [len - 1] + 1)) == 0` ，则说明 (数组长度 - 最长相等前后缀的长度) 正好可以被 数组的长度整除，说明有该字符串有重复的子字符串。

数组长度减去最长相同前后缀的长度相当于是第一个周期的长度，也就是一个周期的长度，如果这个周期可以被整除，就说明整个数组就是这个周期的循环。

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202202102223512.webp)

`next [len - 1] = 7，next [len - 1] + 1 = 8`，8 就是此时字符串 `asdfasdfasdf` 的最长相同前后缀的长度。

`(len - (next [len - 1] + 1))` 也就是：12 (字符串的长度) - 8 (最长公共前后缀的长度) = 4， 4 正好可以被 12 (字符串的长度) 整除，所以说明有重复的子字符串（`asdf`）。

```python
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
```

## 栈与队列

- ==队列==：先进先出
- ==栈==：先进后出

![图片](https://gitee.com/lockegogo/markdown_photo/raw/master/202202102303567.webp)

栈提供 push 和 pop 等接口，所有元素必须符合先进后出规则，所有栈不提供走访功能，也不提供迭代器。

### 1.  用栈实现队列

> 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：
>
> 实现 MyQueue 类：
>
> `void push(int x)` 将元素 x 推到队列的末尾
> `int pop()` 从队列的开头移除并返回元素
> `int peek()` 返回队列开头的元素
> `boolean empty()` 如果队列为空，返回 true ；否则，返回 false

使用栈来模拟队列的行为，如果仅仅用一个栈，是一定不行的，所以需要两个栈：一个输入栈，一个输出栈。

![图片](D:\Dropbox\工作计划\Leetcode\pics\4.gif)

在 push 数据的时候，只要数据放进输入栈就好，但是在 pop 的时候，操作就复杂一些：

1. 输出栈如果为空，就把进栈数据==全部导入==，再从出栈弹出数据；
2. 如果输出栈不为空，则直接从出栈弹出数据就可以了。

最后如何判断队列为空呢？如果进栈和出栈都为空的话，说明模拟的队列为空了。

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # python 内置数据结构 list 可以用来实现栈
        # append() 向栈顶添加元素
        # pop() 可以以后进先出的顺序删除元素（从尾巴删除）
        # 列表的问题是：列表是动态数组，当列表扩大却没有新空间保存新的元素时，会自动重新分配内存块，并将原来的内存中的值复制到新的内存块中，导致 append() 操作会消耗更多的时间
        self.stack1 = list()
        self.stack2 = list()

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        # self.stack1用于接受元素
        self.stack1.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        # self.stack2 用于弹出元素，如果 self.stack2 为 [],则将 self.stack1 中元素全部弹出给 self.stack2
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2[-1]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.stack1 == [] and self.stack2 == []
```

### 2. 用队列实现栈

> 请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（`push`、`top`、`pop`和`empty`）。

每次需要弹出操作时，元素从第一个队列全部输入第二个队，除了最后一个元素，弹出最后一个元素后，再把元素从第二个队放回第一个队。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6QYtsghRUccTciaMXjtJDFHib6dxnuIvt6j9OGPpJo9Bib7Rh67lCANhpykcDQ9aVdp4GbGAvhnHMwA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

不过这道题目用一个队列就够了：一个队列在模拟栈弹出元素的时候只要将队列头部的元素（除了最后一个元素外）重新添加进队列尾部，此时在再去弹出元素就是栈的顺序了。

### 3. 有效的括号

> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
>
> 有效字符串需满足：
> 左括号必须用相同类型的右括号闭合。
> 左括号必须以正确的顺序闭合。
>
> 输入：s = "(]"
> 输出：false

括号匹配时使用栈解决的经典问题。首先要弄清楚，字符串里的括号不匹配==有几种情况==：

1. 字符串里左方向的括号多余了
2. 括号没有多余，但是括号的类型没有匹配上
3. 字符串里右方向的括号多余了

第一种情况，已经遍历完了字符串，但是栈不为空，说明有相应的左括号没有右括号来匹配，所以 return false；

第二种情况：遍历字符串匹配的过程中，发现栈里没有要匹配的字符。所以 return false；

第三种情况：遍历字符串匹配的过程中，栈已经为空了，没有匹配的字符了，说明右括号没有找到对应的左括号 return false。

那什么时候说明左括号和右括号全都匹配了呢，就是字符串遍历完之后，栈是空的，就说明全都匹配了。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(')')
            elif s[i] == '[':
                stack.append(']')
            elif s[i] == '{':
                stack.append('}')
            elif stack and s[i] == stack[-1]:
                stack.pop()
            else:
                return False
        # 如果 stack 为空，返回 True
        return stack == []
```



### 4. 逆波兰表达式求值

> 有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
>
> 逆波兰表达式是一种==后缀表达式==，算符写在后面。其优点是：
>
> 1. 去掉括号后表达式无歧义
> 2. 适合用栈操作运算：遇到数字则入栈，遇到算符则取出栈顶两个元素进行计算，并将结果压入栈中。
>
> 注意两个整数之间的除法只保留整数部分。
> 可以保证给定的逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
>
> 输入：tokens = ["2","1","+","3","*"]
> 输出：9

其实波兰表达式相当于是二叉树中的后序遍历，大家可以把运算符作为中间节点，按照后序遍历的规则画出一个二叉树。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv4mCxur8W49qtZmumwtiax6RxExibQQUD4byjhqjr42XrV2K5zRFiaoRf2KBhdFpMPibx8PQuLzOamiaQA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
class Solution:
    def evalRPN(self, tokens):
        stack = list()
        cal = ["+", "-", "*", "/"]
        for i in range(len(tokens)):
            if tokens[i] not in cal:
                stack.append(tokens[i])
            else:
                # 第一个出来的数应该在运算符的后面
                a = int(stack.pop())
                # 第二个出来的数应该在运算符的前面
                b = int(stack.pop())
                # 注意 a 和 b 与运算符的位置
                # eval: 好新颖的表达方式
                result = int(eval(f'{b}{tokens[i]}{a}'))
                stack.append(result)
        return int(stack.pop())
```

### ==5. 滑动窗口最大值==：单调队列

> 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
>
> 返回 滑动窗口中的最大值 。
>
> 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
> 输出：[3,3,5,5,6,7]
> 滑动窗口的位置         最大值
> [1  3  -1] -3  5  3  6  7    3
> 1 [3  -1  -3] 5  3  6  7    3
>  1  3 [-1  -3  5] 3  6  7    5
>  1  3  -1 [-3  5  3] 6  7    5
>  1  3  -1  -3 [5  3  6] 7    6
>  1  3  -1  -3  5 [3  6  7]    7

这是使用==单调队列==的经典题目。==难点==是如何求一个区间的最大值。笔记最后面有单调栈的题目。

暴力方法，遍历一遍的过程中每次从窗口中在找到最大的数值，这样很明显是 $O (n * k)$ 的算法。

有的同学可能会想用一个**大顶堆（优先级队列）**来存放这个窗口里的 k 个数字，这样就可以知道最大的最大值是多少了， 但是问题是这个窗口是移动的，而大顶堆每次只能弹出最大值，我们无法移除其他数值，这就造成大顶堆维护的不是滑动窗口里面的数值了。

此时我们需要一个队列，随着窗口的移动，队列也一进一出，每次移动之后，队列告诉我们里面的最大值是什么。

**每次窗口移动的时候，调用 `que.pop` (滑动窗口中移除元素的数值)，`que.push` (滑动窗口添加元素的数值)，然后 `que.front ()` 就返回我们要的最大值。**

为实现这一点，队列里面的元素一定需要排序，而且最大值放在出口，但如果把窗口里的元素都放进队列里，窗口移动的时候，队列需要弹出元素。

那么问题来了，已经排序之后的队列怎么能把窗口要移除的元素（这个元素可不一定是最大值）弹出呢。

其实队列没有必要维护窗口里的所有元素，只需要维护**有可能**成为窗口里最大值的元素就可以了，同时保证队列里面的元素数值是从大到小的

那么这个维护元素单调递减的队列就叫做**==单调队列==，即单调递减或单调递增的队列。**

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv4mCxur8W49qtZmumwtiax6R0axb2Svoib5fzy1ibMlLRFslLlq9TSG84soSCoicvH5jmlQUpKwHiaXZ6A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

对于窗口里的元素 {2, 3, 5, 1 ,4}，单调队列里只维护 {5, 4} 就够了，保持单调队列里单调递减，此时队列出口元素就是窗口里最大元素。

设计单调队列的时候，pop 和 push 操作要保持如下规则：

1. pop (value)：如果窗口移除的元素 value 等于单调队列的出口元素，那么队列弹出元素，否则不用任何操作
2. push (value)：如果 push 的元素 value 大于入口元素的数值，那么就将队列入口的元素弹出，直到 push 元素的数值小于等于队列入口元素的数值为止

保持如上规则，每次窗口移动的时候，只要问 `que.front ()` 就可以返回当前窗口的最大值。

为了更直观的感受到单调队列的工作过程，以题目示例为例，输入: `nums = [1,3,-1,-3,5,3,6,7]`, 和 `k = 3`，动画如下：

一定要保证 push 之后队列永远保持由大到小，如果不是，就把不满足条件的弹出去。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv4mCxur8W49qtZmumwtiax6RTqVY4F0yIyztfaEjM6VMst2jUgoMZA3UUpsib0ZF3jPS907uLpSia42w/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
from collections import deque
class MyQueue():
    """
    定义单调队列类: 从大到小
    """
    def __init__(self) -> None:
        self.queue = deque()

    def pop(self, value):
        """
        每次弹出时比较当前弹出的数值是否等于队列出口元素的数值，如果相等则直接弹出；
        如果不相等，说明什么？说明那个应该被弹出的元素已经在维持队列从大到小的顺序时就被弹出了，此时不用做任何操作
        """
        if self.queue and value == self.queue[0]:
            # 弹出队首元素
            self.queue.popleft()

    def push(self, value):
        """
        如果 push 的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到 push 的数值小于等于队列入口元素的数值为止。保证队列由大到小。
        """
        # 如果队列非空并且队尾元素的值小于新加进来的值
        while self.queue and self.queue[-1] < value:
            # 弹出队尾元素，相当于让 value 一直往前插队超过比它小的人
            self.queue.pop()
        self.queue.append(value)

    def front(self):
        """
        查询当前队列里的最大值，直接返回 front
        """
        return self.queue[0]

class Solution:
    def maxSlidingWindow(self, nums, k):
        que = MyQueue()
        result = []
        # 先将前 k 的元素放进队列，一个滑动窗口
        for i in range(k):
            que.push(nums[i])
        result.append(que.front())
        for i in range(k, len(nums)):
            # 滑动窗口移除最前面元素：减一个
            que.pop(nums[i - k])
            # 滑动窗口前加入最后面的元素：加一个
            que.push(nums[i])
            # 记录对应的最大值
            result.append(que.front())
        return result

sol = Solution()
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(sol.maxSlidingWindow(nums,k))
```

### ==6. [前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)==：优先级队列
> 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。
> 
> 输入: `nums = [1,1,1,2,2,3], k = 2`
输出: [1,2]

> 进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。

这个题目主要涉及如下三块内容：

1. 要统计元素出现频率：字典
2. 对频率排序：==优先级队列==
3. 找出前 k 个高频元素

==什么是优先级队列？==

其实就是一个披着队列外衣的堆，因为优先级队列对外接口只是从队列取元素，从队尾添加元素，再无其他取元素的方式，看起来就是一个队列。

而优先级队列内部元素是自动依照元素的权值排列的，怎么做到的？

缺省情况下 priority_queue 利用 max-heap（大顶堆）完成对元素的排序，这个大顶堆是以 vector 为表现形式的 complete binary tree（完全二叉树）。

**堆是一颗完全二叉树，树中每个结点的值都不小于（或不大于）其左右孩子的值。** 如果父亲结点是大于等于左右孩子就是大顶堆，小于等于左右孩子就是小顶堆。

本题我们就要使用==优先级队列==来对部分频率进行==排序==。为什么不用快排，使用快排需要将 map 转换为 vector 结构，然后对整个数组进行排序，而在这种场景下，我们其实只需要维护 k 个有序的序列就可以了，所以使用优先级队列是最优的。

==**所以我们要用小顶堆，因为要统计最大前 k 个元素，只有小顶堆每次将最小的元素弹出，最后小顶堆里积累的才是前 k 个最大元素。**==

寻找前 k 个最大元素流程如图所示：（图中的频率只有三个，所以正好构成一个大小为 3 的小顶堆，如果频率更多一些，则用这个小顶堆进行扫描）

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7UUhQ00bAh5bicicl1ia840WOm4WC58QMfMH1dmEcCEBCfyRbKmj2j7xbMbPYIl0A13RqBeSNMuvCdg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
from typing import List
# 时间复杂度：O(nlogk)
# 空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 要统计元素出现频率
        map_ = {}  # nums[i]:对应出现的次数
        for i in range(len(nums)):
            # get(key, default) 函数返回指定键的值，如果不存在返回 default
            map_[nums[i]] = map_.get(nums[i], 0) + 1

        # 对频率排序
        # 定义一个小顶堆，大小为 k
        pri_que = []  # 小顶堆
        # 用固定大小为 k 的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            # heapq: 只能构建小根堆，也是一种优先队列，它能以任意顺序增加对象，并且能在任意时间找到或移除最小的元素
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k:
                # 如果堆的大小大于了 K，则队列弹出，保证堆的大小一直为 k
                heapq.heappop(pri_que)

        # 找出前 K 个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result


sol = Solution()
nums = [1, 1, 1, 2, 2, 3, 1, 1]
k = 2
print(sol.topKFrequent(nums, k))
```



## ==二叉树==

### 1. 二叉树的基本知识

#### 1.1 二叉树的类型

在我们解题过程中二叉树有两种主要的形式：满二叉树和完全二叉树。

==满二叉树==：如果一棵二叉树只有度为 0 的节点和度为 2 的节点，并且度为 0 的节点在同一层上，则这棵二叉树为满二叉树。也可以说深度为 k，有 $2^k-1$ 个节点的二叉树。

==完全二叉树==：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到了最大值，并且最下面一层的节点都集中在该层最左边的若干位置上。若最底层为第 $h$ 层，则该层包含了 1 ~ $(2^{h -1})$  个节点。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJYI02wXUJPyRR5McrVkj8jx9uNRU5Ymf828Jm0niaoSZibgrOILfLOSNCw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

> 优先级队列其实是一个堆，堆就是一颗完全二叉树，同时保证父子节点的顺序关系。

前面介绍的树，都是没有数值的，而==二叉搜索树==是有数值的，二叉搜索树是一个有序树。

- 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
- 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
- 它的左、右子树也分别为二叉排序树

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJYhmUrf1vZhSEYfDic3s3se9rSDU29b3giaZqPDh07LLoA9rpQXfQZt9Tw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />

==平衡二叉搜索树==：又被称为 AVL（Adelson-Velsky and Landis）树，且具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过 1，并且左右两个子树都是一棵平衡二叉树。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJYa8wHlYkDjNr9A1MIIq6CqvmRSyTA86mUoGJGSD6EZYcB0rE0OSDNvg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

> 最后一棵不是平衡二叉树，因为它的左右两个子树的高度差的绝对值超过了 1。

#### 1.2 二叉树的存储方式

二叉树可以==链式存储==，也可以==顺序存储==。链式存储用指针，顺序存储用数组。顺序存储的元素在内存是连续分布的，而链式存储则是通过指针把分布在散落在各个地址的节点串联在一起。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJYhOtOVEfpN4JAITOwA0iaspU5KbRKJvibP50RwGz3ULWGK3jtVBQZttWg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

**顺序存储的方式如图：**

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJYziaEFwVGibLU3xqIut1Ab70okYkAPPK1kaKW4nkkxVvmmjnO6r7xsB1Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

用数组来存储二叉树如何遍历呢？

如果父节点的数组下表是 $i$，那么它的左孩子就是 $2^i+1$，右孩子就是 $2^i+2$。但是用链式表示的二叉树更有利于我们理解，所以我们一般用链式存储二叉树。

#### 1.3 二叉树的遍历方式

二叉树主要有两种遍历方式：

1. 深度优先遍历：先往深走，遇到叶子节点再往回走；
   -  前序遍历（递归法，迭代法）：中左右
   -  中序遍历（递归法，迭代法）：左中右
   -  后续遍历（递归法，迭代法）：左右中
2. 广度优先遍历：一层一层的去遍历。
   - 层次遍历（迭代法）

深度优先遍历中的==前中后==，其实指的就是==中间节点的遍历顺序==。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4VQjjPNDEmZ3AEPHaA7FJY2VYcOOG9mmJI64TLoqVTb7eYPib27Dks3a8Z1MdEaSz4VBlF5Xicibygw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

和二叉树相关的题目，经常会使用递归的方式来实现深度优先遍历，也就是前中后序遍历。

之前我们讲栈和队列的时候，就说过栈其实是递归的一种实现结构，也就是说==前中后序遍历==的逻辑其实都还是可以借助==栈==使用非递归的方式来实现的。而广度优先遍历的实现一般使用队列来实现，这也是队列先进先出的特点所决定的，因为需要先进先出的结构，才能一层一层的来遍历二叉树。

二叉树的定义和链表差不多，相对于链表，二叉树的节点里多了一个指针，有两个指针，指向左右孩子。

> 在现场面试的时候面试官可能要求手写代码，所以数据结构的定义以及简单逻辑的代码一定要锻炼白纸写出来。



### ==2. 二叉树的递归遍历==

==递归的三要素==：

1. **确定递归函数的参数和返回值**：确定哪些参数是递归的过程中需要处理的，那么就在递归函数中加上这个参数，并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。
2. **确定终止条件**：写完了递归算法，运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，操作系统也是一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。
3. **确定单层递归的逻辑**：确定每一层递归需要处理的信息，在这里也就会重复调用自己来实现递归的过程。

```python
# 二叉树的构建
class Node:
    def __init__(self, data):
        # data 是传入的值
        # 下面三个都是 Node 类的属性，初始化这些属性
        # 函数内新引入的变量均为局部变量，故无论是设置还是使用它的属性都得利用 self. 的方式
        # 如果不加 self. 这个变量就无法在 init 函数之外被使用
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
    # 将新值与父节点进行比较
        if self.data:  # 非空
            if data < self.data:            #新值较小，放左边
                if self.left is None:       #若空，则新建插入节点
                    self.left = Node(data)
                else:                       #否则，递归往下查找
                    self.left.insert(data)
            elif data > self.data:          #新值较大，放右边
                if self.right is None:      #若空，则新建插入节点
                    self.right = Node(data)
                else:                       #否则，递归往下查找
                    self.right.insert(data)
        else:
            self.data = data                

    # 打印这棵树，中序遍历
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(self.data),
        if self.right:
            self.right.PrintTree()

# 使用insert方法添加节点
root = Node(12)
root.insert(6)
root.insert(14)
root.insert(3)

root.PrintTree()


# 前序遍历-递归-LC144_二叉树的前序遍历
class Solution:
    def preorderTraversal(self, root):
        # 确定递归函数的参数和返回值
        result = []
        
        def traversal(root):
            # 确定终止条件：什么时候递归结束？
            # 当前遍历的节点为空，那么本层递归结束
            if root == None:
                return
            # 确定单层递归的逻辑
            # 递归函数在使用时，会沿着那个方向一直走到底，撞到南墙之后再回头一步
            result.append(root.data)  # 前序
            traversal(root.left)    # 左
            traversal(root.right)   # 右

        traversal(root)
        return result

# 中序遍历-递归-LC94_二叉树的中序遍历
class Solution:
    def inorderTraversal(self, root):
        result = []

        def traversal(root):
            if root == None:
                return
            traversal(root.left)    # 左
            result.append(root.data)  # 中序
            traversal(root.right)   # 右

        traversal(root)
        return result

# 后序遍历-递归-LC145_二叉树的后序遍历
class Solution:
    def postorderTraversal(self, root):
        result = []

        def traversal(root):
            if root == None:
                return
            traversal(root.left)    # 左
            traversal(root.right)   # 右
            result.append(root.data)  # 后序

        traversal(root)
        return result

sol = Solution()
print(sol.inorderTraversal(root))
```

### ==3. 二叉树的迭代遍历==

递归的实现就是：每一次递归调用都会把函数的局部变量、参数值和返回地址等压入栈中，然后递归返回的时候，从栈顶弹出上一次递归的各项参数，所以这就是递归为什么可以返回上一层位置的原因。

#### 3.1 前序遍历（迭代法）

前序遍历是中左右，每次先处理的是中间节点，那么先将根节点放入栈中，然后将右孩子加入栈，再加入左孩子。

==为什么先加入右孩子，再加入左孩子？==

因为这样出栈的时候才是中左右的顺序：

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6iaia8CS4C89jib6Vibw1icFhEzZe7WRTdPbhZcSLVGdWItQ8SxEGw6SYu1ib7mKYbtX6XDr31DbSAId2w/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
# 前序遍历-迭代法
class Solution:
    def preorderTraversal(self, root):
        # 根节点为空则返回空列表
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            # 中节点先处理
            node = stack.pop()
            result.append(node.data)
            # 右孩子先入栈
            if node.right:
                stack.append(node.right)
            # 左孩子后入栈
            if node.left:
                stack.append(node.left)

        return result
```

#### 3.2 中序遍历（迭代法）

在刚才的迭代过程中其实我们有两个操作：

1. 处理：将元素放进 result 数组中
2. 访问：遍历节点

前序遍历的代码不能和中序遍历通用，因为前序遍历的顺序是中左右，先访问的元素的==中间节点==，要处理的元素也是==中间节点==；但是中序遍历顺序是左中右，先访问的是二叉树顶部的节点，然后一层一层向下访问，直到达到树左面的==最底部==，再开始处理节点（也就是把节点的数值放进 result 数组中），**这就造成了处理顺序和访问顺序是不一致的。**

在使用迭代法写==中序遍历==，就需要借助==指针的遍历==来帮助访问节点，栈则用来处理节点上的元素。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6iaia8CS4C89jib6Vibw1icFhEzfibSLW18Y3JricktoAP7PKVwTpgyKLicuE3tMAMEC9VJLb7h05UzGT5Mw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
# 中序遍历-迭代法: 左中右
class Solution:
    def inorderTraversal(self, root):
        # 根节点为空则返回空列表
        if not root:
            return []
        # 不能提前将节点加入 stack
        stack = []
        result = []
        # 补充一个指针
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树节点
            if cur:
                stack.append(cur)
                cur = cur.left
            # 到达最左节点后处理栈顶节点
            else:
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右节点
                cur = cur.right
        return result
```



#### 3.3 后序遍历（迭代法）

先序遍历是==中左右==，后序遍历是左右中，我们只需要调整一下先序遍历的代码顺序，就变成==中右左==，然后反转 result 数组，输出的结果就是左右中了

==总结==：迭代法的前序遍历和中序遍历完全是两种代码风格，并不像递归法那样对代码稍作调整，就可以实现，==这是因为前序遍历中访问节点和处理节点可以同步处理，但是中序无法做到同步。==

```python
# 后序遍历-迭代-LC145_二叉树的后序遍历
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 左孩子先入栈
            if node.left:
                stack.append(node.left)
            # 右孩子后入栈
            if node.right:
                stack.append(node.right)
        # 将最终的数组翻转
        return result[::-1]
```



### ==4. 二叉树的层序遍历==

> 给你一个二叉树，请你返回其按层序遍历得到的节点值。（即逐层地，从左到右访问所有节点）。
>
> ![img](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)
>
> 输入：`root = [3,9,20,null,null,15,7]`
> 输出：`[[3],[9,20],[15,7]]`

二叉树的层序遍历需要借助一个辅助数据结构即==队列==来实现，队列先进先出，符合一层一层遍历的逻辑，而栈先进后出适合模拟深度优先遍历也就是递归的逻辑。

而层序遍历就是图论中的广度优先遍历，只不过我们应用在二叉树上。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv46LWl4MUIjcWpVpicNJm3DzxtKvgrxiaEqhntgf8JFQaNxvxygnqRJv6LIxmy4ibYbmRXWddn9ibnvrw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

```python
from collections import deque
class Solution:
    """二叉树层序遍历迭代解法"""
    def levelOrder(self, root):
        results = []
        if not root:
            return results
        que = deque([root])
        while que:
            size = len(que)
            result = []
            for _ in range(size):
                cur = que.popleft()
                result.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            results.append(result)

        return results
```



### 5. 翻转二叉树

> 给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。
>
> ![img](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)
>
> ```
> 输入：root = [4,2,7,1,3,6,9]
> 输出：[4,7,2,9,6,3,1]
> ```

想要翻转二叉树，其实就是把每一个节点的左右孩子交换一下就可以了，关键在于==遍历顺序==。**前中后序应该选哪一种遍历顺序？**

遍历的过程中去翻转每一个节点的左右孩子就可以达到整体翻转的效果。

这道题目使用前序遍历和后序遍历都可以，唯独中序遍历不行，因为中序遍历会把左子树的左右孩子翻转两次。

层序遍历也是可以的，只要把每一个节点的左右孩子翻转一下的遍历方式都是可以的。

```python
class Solution:
    """
    递归法翻转二叉树
    """
    def invertTree(self, root):
        # 1. 确定递归函数的参数和返回值
        # 2. 确定终止条件
        if not root:
            return None
        # 3. 确定单层递归的逻辑：先交换左右孩子节点，然后反转左子树，再反转右子树
        root.left, root.right = root.right, root.left #中
        self.invertTree(root.left) #左
        self.invertTree(root.right) #右
        return root


class Solution:
    """
    迭代法：深度优先遍历（前序遍历）
    """
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        st = []
        st.append(root)
        while st:
            node = st.pop()
            node.left, node.right = node.right, node.left #中
            if node.right:
                st.append(node.right) #右
            if node.left:
                st.append(node.left) #左
        return root


import collections
class Solution:
    """
    迭代法：广度优先遍历（层序遍历）
    """
    def invertTree(self, root):
        queue = collections.deque() #使用 deque()
        if root:
            queue.append(root)
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                node.left, node.right = node.right, node.left #节点处理
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
```

### 6. 对称二叉树

> 给你一个二叉树的根节点 `root` ， 检查它是否轴对称。
>
> ![img](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)
>
> ```
> 输入：root = [1,2,2,3,4,4,3]
> 输出：true
> ```

首先要想清楚，判断对称二叉树要比较的是哪两个节点，可不是左右节点！而是要比较根节点的 左右子树，所以在递归遍历的过程中，也是要==同时遍历两棵树==。   

==那应该如何选择遍历顺序？==

本题遍历顺序只能是”==后序遍历==“，因为我们要通过递归函数的返回值来判断两个子树的内侧节点是否相等。正因为要遍历两棵树而且要比较内侧和外侧节点，所以准确的来说是一个树的遍历顺序是==左右中==，一个树的遍历顺序是==右左中==。

#### 6.1 递归法

1. 确定递归函数的参数和返回值

因为我们要比较的是根节点的两个子树是否是相互翻转的，进而判断这个树是不是对称树，所以要比较的是两个树，参数自然也是左子树节点和右子树节点；

返回值自然是 bool 类型

2. 确定终止条件

要比较两个节点数值相不相同，首先要把两个节点为空的情况弄清楚！否则后面比较数值的时候就会操作空指针了。

节点为空的情况有：

- 左节点为空，右节点不为空，不对称
- 左不为空，右为空，不对称
- 左右都为空，对称

此时已经排除了节点为空的情况，那么剩下就是左右节点不为空：

- 左右都不为空，比较节点数值，不相同就 return false

此时左右节点都不为空，且数值也不相同的情况我们也处理了

3. 确定单层递归的逻辑

此时才进入单层递归的逻辑，单层递归的逻辑就是处理左右节点都不为空，且数值相同的情况：

- 比较二叉树外侧是否对称：传入的是左节点的左孩子，右节点的右孩子
- 比较内侧是否对称，传入左节点的右孩子，右节点的左孩子
- 如果左右都对称就返回 true，有一侧不对称就返回 false

```python
class Solution:
    """
    递归法
    """
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.compare(root.left, root.right)
        
    def compare(self, left, right):
        # 首先排除空节点的情况
        if left == None and right != None:
            return False
        elif left != None and right == None:
            return False
        elif left == None and right == None:
            return True
        # 排除了空节点，再排除数值不相同的情况
        elif left.val != right.val:
            return False
        
        # 此时就是：左右节点都不为空，且数值相同的情况
        # 此时才做递归，做下一层的判断
        outside = self.compare(left.left, right.right) #左子树：左、 右子树：右
        inside = self.compare(left.right, right.left) #左子树：右、 右子树：左
        isSame = outside and inside # 左子树：中、 右子树：中 （逻辑处理）
        return isSame

```

#### 6.2 迭代法

通过队列来判断根节点的左子树和右子树的内侧和外侧是否相等，如动画所示：

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6wEswj6eibFksDvAa4qKiaqSSzmaHHgdWyepdicDtnMxqXOlOz5jtmN1haIYvh82g6dOZT2JlYKcT5Q/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)



### 7. 二叉树的最大深度

> 给定二叉树，找出其最大深度。
>
> 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数量。
>
> 给定二叉树 `[3,9,20,null,null,15,7]`
>
>     3
>    / \
>   9  20
>        /  \
>      15   7

#### 7.1 递归法

本题可以使用前序（中左右），也可以使用后序遍历（左右中），==使用前序求的就是深度，使用后序求的是高度==。

> 1. 树的高度和深度是相同的
> 2. 同一层的节点的深度是相同的，但是高度不一定相同！

**而根节点的高度就是二叉树的最大深度**，所以本题中我们通过后序求的根节点高度来求的二叉树最大深度。

我先用后序遍历（左右中）来计算树的高度。

1. ==确定递归函数的参数和返回值==：参数就是传入树的根节点，返回就返回这棵树的深度，所以返回值为 int 类型。
2. ==确定终止条件==：如果为空节点的话，就返回 0，表示高度为 0。
3. ==确定单层递归的逻辑==：先求它的左子树的深度，再求的右子树的深度，最后取左右深度最大的数值 再 + 1 （加 1 是因为算上当前中间节点）就是目前节点为根节点的树的深度。

```python
class solution:
    def maxdepth(self, root: treenode) -> int:
        return self.getdepth(root)
        
    def getdepth(self, node):
        if not node:
            return 0
        leftdepth = self.getdepth(node.left) #左
        rightdepth = self.getdepth(node.right) #右
        depth = 1 + max(leftdepth, rightdepth) #中
        return depth
```

```python
# n 叉树的深度
class solution:
    def maxdepth(self, root: 'node') -> int:
        if not root:
            return 0
        depth = 0
        for i in range(len(root.children)):
            depth = max(depth, self.maxdepth(root.children[i]))
        return depth + 1
```



#### 7.2 迭代法

使用迭代法，最适合用层序遍历，因为最大的深度就是二叉树的层数，和层序遍历的方式一样。

```python
import collections
class solution:
    def maxdepth(self, root: treenode) -> int:
        if not root:
            return 0
        depth = 0 #记录深度
        queue = collections.deque()
        queue.append(root)
        while queue:
            # 记录每一层的节点数量
            size = len(queue)
            depth += 1
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth
```

```python
# n 叉树的深度
import collections
class solution:
    def maxdepth(self, root: 'node') -> int:
        queue = collections.deque()
        if root:
            queue.append(root)
        depth = 0 #记录深度
        while queue:
            size = len(queue)
            depth += 1
            for i in range(size):
                node = queue.popleft()
                for j in range(len(node.children)):
                    if node.children[j]:
                        queue.append(node.children[j])
        return depth
```

### 8. 二叉树的最小深度

> 给定一个二叉树，找出其最小深度。
>
> 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
>
> **说明：**叶子节点是指没有子节点的节点。
>
> ![img](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：2
> ```

遍历顺序依然是后序遍历，因为要比较递归返回之后的结果，但在处理中间节点的逻辑上，最大深度很容易理解，最小深度可有一个误区，如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv47wZWmbVLj4Qsp9ezEEJRe3YNKueX3iagA9lmT1ibe5ib7DMGIp4S5gn94EqhA9rtUjta8Kg3eeiaiclg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

这就重新审题了，题目中说的是：**最小深度是从根节点到最近叶子节点的最短路径上的节点数量。**注意是**叶子节点**。

什么是叶子节点，左右孩子都为空的节点才是叶子节点！

#### 8.1 递归法

1. ==确定递归函数的参数和返回值==：参数为要传入的二叉树根节点，返回的是 int 类型的深度。
2. ==确定终止条件==：终止条件也是遇到空节点返回 0，表示当前节点的高度为 0。
3. ==确定单层递归的逻辑==：这块和求最大深度可就不一样了，如果直接把求最大深度的代码由 max 改为 min，就会进入上面的误区：没有左孩子的分支会算为最短深度，所以，如果左子树为空，右子树不为空，说明最小深度是 1 + 右子树的深度；反之，右子树为空，左子树不为空，最小深度是 1 + 左子树的深度；最后如果左右子树都不为空，返回左右子树深度最小值 + 1。

遍历的顺序为后序（左右中），可以看出：**求二叉树的最小深度和求二叉树的最大深度的差别主要在于处理左右孩子不为空的逻辑。**

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        min_depth = float('inf')
        left_depth = self.minDepth(root.left)
        right_depth = self.minDepth(root.right)
        if not root.left and root.right:
            return right_depth + 1
        if not root.right and root.left:
            return left_depth + 1
        # 左右子树都存在的情况下
        return 1 + min(left_depth, right_depth)
```



#### 8.2 迭代法

依然使用层序遍历，需要注意的是，只有当左右孩子都为空的时候，才说明遍历到最低点了，如果其中一个孩子为空则不是最低点。

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        que = deque()
        que.append(root)
        res = 1

        while que:
            for _ in range(len(que)):
                node = que.popleft()
                # 当左右孩子都为空的时候，说明是最低点的一层了，退出
                if not node.left and not node.right:
                    return res
                if node.left is not None:
                    que.append(node.left)
                if node.right is not None:
                    que.append(node.right)
            res += 1
        return res
```



### 9. 完全二叉树的节点个数

> 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
>
> 完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
>
> ![img](https://assets.leetcode.com/uploads/2021/01/14/complete.jpg)
>
> ```
> 输入：root = [1,2,3,4,5,6]
> 输出：6
> ```

```python
class Solution:
    """
    递归法：后序遍历
    """
    def countNodes(self, root: TreeNode) -> int:
        return self.getNodesNum(root)
        
    def getNodesNum(self, cur):
        if not cur:
            return 0
        leftNum = self.getNodesNum(cur.left) # 左
        rightNum = self.getNodesNum(cur.right) # 右
        treeNum = leftNum + rightNum + 1 # 中
        return treeNum
```

```python
import collections
class Solution:
    """
    迭代法
    """
    def countNodes(self, root: TreeNode) -> int:
        queue = collections.deque()
        if root:
            queue.append(root)
        result = 0
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                result += 1 #记录节点数量
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result
```



完全二叉树只有两种情况，情况一：满二叉树；情况二：最后一层叶子节点没有满。

对于情况一，可以直接用 $2^ {树深度} - 1$ 来计算，注意这里根节点深度为 1。

对于情况二，分别递归左孩子，和右孩子，递归到某一深度一定会有左孩子或者右孩子为满二叉树，然后依然可以按照情况 1 来计算。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5m863Pdu5kXzODQvpUAmvVlft2xzXkcxiaLAibo0LckVibIP9AwP9micnHtRxpUhIxTIjN0jlllXE5sw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

完全二叉树（二）如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5m863Pdu5kXzODQvpUAmvV2kuXDs0PcYr0eX3MuRUOlnMStzehiahYP0HruJWKVRhFB4qOSayZxBg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

可以看出如果整个树不是满二叉树，就递归其左右孩子，直到遇到满二叉树为止，用公式计算这个子树（满二叉树）的节点数量。

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = root.left
        right = root.right
        # 这里初始为0是有目的的，为了下面求指数方便
        leftHeight = 0 
        rightHeight = 0
        # 求左子树深度
        while left: 
            left = left.left
            leftHeight += 1
        # 求右子树深度
        while right: 
            right = right.right
            rightHeight += 1
        if leftHeight == rightHeight:
            # 注意(2<<1) 相当于2^2，所以 leftHeight 初始为0
            return (2 << leftHeight) - 1 
        return self.countNodes(root.left) + self.countNodes(root.right) + 1
```

### 10. 平衡二叉树

> 给定一个二叉树，判断它是否是高度平衡的二叉树。
>
> 本题中，一棵高度平衡二叉树定义为：
>
> > 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。
>
> ![img](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：true
> ```

这道题和二叉树的最大深度很像，但是有很大区别。

- 二叉树节点的==深度==：指从根节点到该节点的最长简单路径边的条数；==前序遍历求深度==
- 二叉树节点的==高度==：指从该节点到叶子节点的最长简单路径边的条数；==后序遍历求高度==

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5AicEE6uQAWBwnjfCibSMLWEdRj9uCiaFicvNT8x4EKAl0ItTicLDMXicojhTaY92uodFoib57KO4dziciblg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

关于根节点的深度究竟是 1 还是 0，不同的地方有不一样的标准，leetcode 的题目中都是以节点为一度，即根节点深度是 1。但维基百科上定义用边为一度，即根节点的深度是 0，我们暂时以 leetcode 为准（毕竟要在这上面刷题）。

因为求深度可以从上到下去查 所以需要前序遍历（中左右），而高度只能从下到上去查，所以只能后序遍历（左右中）

那么有一个问题，为什么在求二叉树的最大深度中求的是二叉树的最大深度，也用的是后序遍历。那是因为代码的逻辑其实是求的根节点的高度，而根节点的高度就是这颗树的最大深度，所以才可以使用后序遍历。

==递归三部曲==：

1. ==明确递归函数的参数和返回值==：参数为传入的节点指针，返回值要返回传入节点为根节点树的深度。

那么如何标记左右子树是否差值大于 1 呢？如果当前传入节点为根节点的二叉树已经不是二叉平衡树了，还返回高度的话就没有意义了。

所以如果已经不是二叉平衡树了，可以返回 - 1 来标记已经不符合平衡树的规则了。

2. ==明确终止条件==：递归的过程中依然是遇到空节点了为终止，返回 0，表示当前节点为根节点的树高度为 0
3. ==明确单层递归的逻辑==：如何判断当前传入节点为根节点的二叉树是否是平衡二叉树呢，当然是左子树高度和右子树高度相差 1。分别求出左右子树的高度，然后如果差值小于等于 1，则返回当前二叉树的高度，否则则返回 - 1，表示已经不是二叉树了。

此时递归的函数就已经写出来了，这个递归的函数传入节点指针，返回以该节点为根节点的二叉树的高度，如果不是二叉平衡树，则返回 - 1。

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if self.get_height(root) != -1:
            return True
        else:
            return False
    
    def get_height(self, root: TreeNode) -> int:
        # Base Case
        if not root:
            return 0
        # 左
        left_height = self.get_height(root.left)
        if left_height == -1:
            return -1
        # 右
        right_height = self.get_height(root.right)
        if right_height == -1:
            return -1
        # 中
        if abs(left_height - right_height) > 1:
            return -1
        else:
            return 1 + max(left_height, right_height)

```

### 11. 二叉树的所有路径

> 给你一个二叉树的根节点 `root` ，按 **任意顺序** ，返回所有从根节点到叶子节点的路径。
>
> **叶子节点** 是指没有子节点的节点。
>
> ![img](https://assets.leetcode.com/uploads/2021/03/12/paths-tree.jpg)
>
> ```
> 输入：root = [1,2,3,null,5]
> 输出：["1->2->5","1->3"]
> ```

这道题目要求从根节点到叶子的路径，所以需要前序遍历，这样才方便让父节点指向孩子节点，找到对应的路径。

在这道题目中将第一次涉及到回溯，因为我们要把路径记录下来，需要回溯来回退一一个路径在进入另一个路径。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4G44A2PAPYlRHvZgTJ2ic6mYr5s71k3kavk4bFkZia9J2xLMLia9gBKXd25Lsuf0B0uAnBb98npPePw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

1. ==递归函数函数参数以及返回值==：要传入根节点，记录每一条路径的 path，和存放结果集的 result，这里递归不需要返回值
2. ==确定递归终止条件==：**那么什么时候算是找到了叶子节点？** 是当 cur 不为空，其左右孩子都为空的时候，就找到叶子节点。为什么没有判断 cur 是否为空呢，因为下面的逻辑可以控制空节点不入循环。再来看一下终止处理的逻辑。这里使用 vector 结构 path 来记录路径，所以要把 vector 结构的 path 转为 string 格式，在把这个 string 放进 result 里。**那么为什么使用了 vector 结构来记录路径呢？** 因为在下面处理单层递归逻辑的时候，要做回溯，使用 vector 方便来做回溯。
3. ==确定单层递归逻辑==：因为是前序遍历，需要先处理中间节点，中间节点就是我们要记录路径上的节点，先放进 path 中。然后是递归和回溯的过程，上面说过没有判断 cur 是否为空，那么在这里递归的时候，如果为空就不进行下一层递归了。所以递归前要加上判断语句，下面要递归的节点是否为空。此时还没完，递归完，要做回溯啊，因为 path 不能一直加入节点，它还要删节点，然后才能加入新的节点。

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        path=[]
        res=[]
        def backtrace(root, path):
            if not root:return 
            path.append(root.val)
            if (not root.left)and (not root.right):
               res.append(path[:])
            ways=[]
            if root.left:ways.append(root.left)
            if root.right:ways.append(root.right)
            for way in ways:
                backtrace(way,path)
                path.pop()
        backtrace(root,path)
        return ["->".join(list(map(str,i))) for i in res]
```



## ==回溯算法==

回溯算法，也叫回溯搜索法，是一种搜索方式。

回溯是递归的副产品，只要有递归就会有回溯。回溯函数就是递归函数。

### 1. 回溯法的基本知识

回溯法并不是高效的算法，因为回溯法的本质是==穷举==，穷举所有可能，然后选出我们想要的答案，如果想让回溯法高效一些，可以加一些剪枝的操作，但也改变不了回溯法就是穷举的本质。

==回溯法解决的问题==：

- 组合问题：N 个数里面按照一定规则找出 k 个数的集合
- 切割问题：一个字符串按照一定规则有几种切割方式
- 子集问题：一个 N 个数的集合里有多少符合条件的子集
- 排列问题：N 个数按一定规则全排列，有几种排列方式
- 棋盘问题：N 皇后，解数独等等

> 组合无序，排列有序。

==如何理解回溯法：==

回溯法解决的问题都可以抽象为树形结构，因为回溯法解决的都是在集合中递归查找子集：

- **集合的大小就构成了数的宽度**，for 循环横向遍历
- **递归的深度就构成了树的深度**，递归纵向遍历

递归就要有终止条件，所以必然是一棵高度有限的树（N 叉树）。

==回溯法模板==：

1. 回溯函数模板返回值以及参数

回溯法需要的参数可不像二叉树递归的时候那么容易一次性确定下来，所以一般都是先写逻辑，然后需要什么参数就填什么参数。

2. 回溯函数终止条件

搜索到叶子节点，也就找到了满足条件的一条答案，把这个答案存放起来，并结束本层递归。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv7eH4u8vicR1FRwEE7zdfeGgbR6xCibegiamdWibnnmDlVA71ibOsDWKSdwCymdfJ6xrIlbl9QOnnEoUxQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片"  />

```c++
for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
    处理节点;
    backtracking(路径，选择列表); // 递归
    回溯，撤销处理结果
}
```

for 循环就是遍历集合区间，可以理解一个节点有多少个孩子，这个 for 循环就执行多少次。

backtracking 这里自己调用自己，实现递归。

大家可以从图中看出 **for 循环可以理解是横向遍历，backtracking（递归）就是纵向遍历**，这样就把这棵树全遍历完了，一般来说，搜索叶子节点就是找的其中一个结果了。

```c++
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}
```

### 2. 组合

> 给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。
>
> ```
> 输入：n = 4, k = 2
> 输出：
> [
>   [2,4],
>   [3,4],
>   [2,3],
>   [1,2],
>   [1,3],
>   [1,4],
> ]
> ```

直接的解法当然是使用 for 循环，例如示例中 k 为 2，很容易想到 用两个 for 循环，这样就可以输出和示例中一样的结果。

**如果 n 为 100，k 为 50 呢，那就 50 层 for 循环，你写得出来吗**？

**此时就会发现虽然想暴力搜索，但是用 for 循环嵌套连暴力都写不出来！**

回溯搜索法来了，虽然回溯法也是暴力，但至少能写出来，不像 for 循环嵌套 k 层让人绝望。

==那么回溯法怎么暴力搜呢？==

上面我们说了**要解决 n 为 100，k 为 50 的情况，暴力写法需要嵌套 50 层 for 循环，那么回溯法就用递归来解决嵌套层数的问题**。

递归来做层叠嵌套（可以理解是开 k 层 for 循环），每一次的递归中嵌套一个 for 循环，那么递归就可以用于解决==多层嵌套循环==的问题了。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5SBsnFmqibXvBHKstibeks0Yn5pKSm1aE6f6ckJ2bjxn32w41eWibiaZficLOwRhAibxnxvTiaIWlTyFQicg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**每次从集合中选取元素，可选择的范围随着选择的进行而收缩，调整可选择的范围**。

**图中可以发现 n 相当于树的宽度，k 相当于树的深度**。

那么如何在这个树上遍历，然后收集到我们要的结果集呢？

图中每次搜索到了叶子节点，我们就找到一个结果。

相当于只需要把到达叶子节点的结果搜集起来，就可以求得 n 个数中 k 个数的组合集合。

==回溯法三部曲==：

1. **递归函数的返回值以及参数**

这里主要定义两个局部变量，一个用来存放符合条件的==单一结果== result，一个用来存放符合条件==结果的集合== path。

函数里一定有两个参数，既然是集合 n 里面取 k 的数，那么 n 和 k 是两个 int 型的参数。

然后还需要一个参数，为 int 型变量 `startIndex`，这个参数用来记录本层递归中，集合从哪里开始遍历（集合就是 [1,...,n] ），也就是下一层递归搜索的起始位置（如果可以重复的，应该就不需要这个参数）。

> **为什么要有这个 startIndex 呢？**
>
> 每次从集合中选取元素，可选择的范围随着选择的进行而收缩，调整可选择的范围，就要靠`startIndex`

2. **回溯函数终止条件**

什么时候到达所谓的叶子节点了呢？path 这个数组的大小如果达到了 k，说明我们找到了一个子集大小为 k 的组合了，这个数组存的就是根节点到叶子节点的路径。

此时用 result 二维数组，把 path 保存起来，并终止本层递归。

3. **单层搜索的过程**

回溯法的搜索过程就是一个树型结构的遍历过程，如下图，可以看出 for 循环用来横向遍历，递归的过程是纵向遍历。for 循环每次从 `startIndex` 开始遍历，然后用 path 保存取到得节点 $i$。递归函数通过不断调用自己一直往深处遍历，总会遇到叶子节点，遇到叶子节点就要返回。

==剪枝优化：==

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5SBsnFmqibXvBHKstibeks0YjDGIsuDiaIcQgaexwCnu7rG7EB9NyPQtgNB7thYmzVhNByVICNTydMQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片"  />

**如果 for 循环选择得起始位置之后得元素个数已经不足我们需要的元素个数了，那么就没有必要搜索了**

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []
        ## 1. 确定回溯函数的参数
        def backtrack(n, k, StartIndex):
            ## 2. 确定回溯函数的终止条件
            if len(path) == k:
                res.append(path[:])
                return
            ## 3. 进入单层循环逻辑
            # for：横向遍历
            for i in range(StartIndex, n + 1):
                path.append(i)
                # 回溯函数：纵向遍历
                backtrack(n, k, i+1)
                # 回溯完成后要把之前的元素 pop 出去
                path.pop()
        backtrack(n, k, 1)
        return res
    
    
class Solution:
    """
    剪枝
    """
    def combine(self, n: int, k: int) -> List[List[int]]:
        res=[]  #存放符合条件结果的集合
        path=[]  #用来存放符合条件结果
        def backtrack(n,k,startIndex):
            if len(path) == k:
                res.append(path[:])
                return
            # k - len(path) 是我们还需要选取的元素的个数
            for i in range(startIndex,n - (k - len(path)) + 2):  #优化的地方
                path.append(i)  #处理节点 
                backtrack(n,k,i+1)  #递归
                path.pop()  #回溯，撤销处理的节点
        backtrack(n,k,1)
        return res
```

### 3. 组合总和 III

> 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
>
> ```
> 输入: k = 3, n = 7
> 输出: [[1,2,4]]
> ```

本题就是在 [1,2,3,4,5,6,7,8,9] 这个集合中找到和为 n 的 k 个数的组合。

和上一题相比，无非就是多了一个限制，本题是要找到和为 n 的 k 个数的组合，而整个集合已经是固定的 [1,...,9]。

本题 k 相当于了树的深度，9（因为整个集合就是 9 个数）就是树的宽度。

例如 k = 2，n = 4 的话，就是在集合 [1,2,3,4,5,6,7,8,9] 中求 k（个数） = 2, n（和） = 4 的组合。

选取过程如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6vKujia98Cyl8icF4GEOLJxQ1htrhHXMRicN3S2U3ClLGAia2X7g5nFdZNAFGec8gJ5kP0K7S6bCmung/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片"  />

```python
from typing import List
 
class Solution:
    def __init__(self):
        self.res = []
        # 已经收集的元素总和，也就是 path 里元素的总和
        self.sum_now = 0
        self.path = []

    def combinationSum3(self, k: int, n: int):
        self.backtracking(k, n, 1)
        return self.res

    def backtracking(self, k: int, n: int, start_num: int):
        # 剪枝：和 > target，无意义了
        if self.sum_now > n:  
            return
        # len(path)==k 时不管 sum 是否等于 n 都会返回
        if len(self.path) == k:  
            if self.sum_now == n:
                self.res.append(self.path[:])
            # 如果 len(path)==k 但是 和不等于 target，直接返回
            return
        # 集合固定为 9 个数
        for i in range(start_num, 10 - (k - len(self.path)) + 1):
            # 单层搜索时操作了两个元素 path 和 sum_now，这两个元素在函数返回时都要进行判断
            self.path.append(i)
            self.sum_now += i
            self.backtracking(k, n, i + 1)
            self.path.pop()
            self.sum_now -= i
```

### 4. 电话号码的字母组合

> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
>
> 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
>
> ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)
>
> ```
> 输入：digits = "23"
> 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
> ```

本题需要解决三个问题：

1. 数字和字母如何映射？
2. 两个字母就两个 for 循环，三个字母就三个 for 循环，这样代码根本写不出来
3. 输入 1 * # 案件的异常情况

==数字和字母如何映射==：可以使用 map 或者定义一个二维数组来做映射

==回溯法来解决 n 个 for 循环的问题==：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4g9ialwsB98zmuWnyLlpiaohoHHDTWSd9h1PQ9ibjVtIlibWldpCleITDILBVEGeuEruaa3KYU1K96tg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图中可以看出遍历的深度，就是输入 "23" 的长度，而叶子节点就是我们要收集的结果，输出 ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]。

```python
class Solution:
    def __init__(self):
        self.answers: List[str] = []
        self.answer: str = ''
        self.letter_map = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

    def letterCombinations(self, digits: str) -> List[str]:
        self.answers.clear()
        if not digits: return []
        self.backtracking(digits, 0)
        return self.answers
    
    def backtracking(self, digits: str, index: int) -> None:
        # 回溯函数没有返回值
        # Base Case
        if index == len(digits):    # 当遍历穷尽后的下一层时
            self.answers.append(self.answer)
            return 
        # 单层递归逻辑  
        letters = self.letter_map[digits[index]]
        # 注意这里不是从 startindex 开始遍历的，因为本题每一个数字代表不同的集合，求不同集合之间的组合
        # 之前的组合题目求的是相同集合中的组合
        for letter in letters:
            self.answer += letter   # 处理
            self.backtracking(digits, index + 1)    # 递归至下一层
            self.answer = self.answer[:-1]  # 回溯
```

### 5. 组合总和

> 给你一个==无重复元素==的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的所有不同组合 ，并以列表形式返回。你可以按任意顺序返回这些组合。
>
> candidates 中的**同一个数字可以无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
>
> 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
>
> 输入：`candidates = [2,3,6,7], target = 7`
> 输出：`[[2,2,3],[7]]`

自己写出来了一个版本，但是结果有重复。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5HTwmlN4eYQibVjpl0McN9OLwetk9yD2yCsdpcHYAuMfibQJ7ROHibZbKdZ0SFQQwdZPJS9pTO8m4Ew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

注意叶子节点的返回条件，因为本题没有组合数量要求，仅仅是总和的限制，所以递归没有层数的限制，只要选取的元素总和超过 target，就 return ！（这一点我也想到了）

==本题还需要 startindex 来控制 for 循环的起始位置，对于组合问题，什么时候需要 startindex 呢？==

- 如果是==一个集合==求组合，就需要 startindex
- 如果是==多个集合==取组合，各个集合之间相互不影响，就不用 startindex（例如电话号码的字母组合）

本题单层 for 循环依然是从 startIndex 开始，搜索 candidates 集合。

```python
class Solution:
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        self.res = self.trackbacking(startindex, candidates, target)
        return self.res

    def trackbacking(self, startindex, candidates, target):
        # 确定终止条件：用和来结束遍历，而不是路径长度
        if self.sum == target:
            self.res.append(self.result[:])
            return
        if self.sum > target:
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        # 如果在同一个集合中求组合，一定要加上 startindex ！！
        for i in range(startindex, len(candidates)):
            self.result.append(candidates[i])
            self.sum += candidates[i]
            # 因为可以无限制选取同一个数字，所以是 i
            # 如果不能重复选同一个数字，应该是 i + 1
            self.trackbacking(i, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
        return self.res
```

==剪枝优化：==这个优化一般不容易想到，但是在求和问题中，==排序后加剪枝==是常见的套路！

以及上面的版本一的代码大家可以看到，对于 sum 已经大于 target 的情况，其实是依然进入了下一层递归，只是下一层递归结束判断的时候，会判断 sum > target 的话就返回。其实如果已经知道下一层的 sum 会大于 target，就没有必要进入下一层递归了。那么可以在 for 循环的搜索范围上做做文章了。

**对总集合排序之后，如果下一层的 sum（就是本层的 sum + candidates [i]）已经大于 target，就可以结束本轮 for 循环的遍历**。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5HTwmlN4eYQibVjpl0McN9O2ZPUF3xSn9tJeH4iciacibIjFNichTVmOgVdkkJhcXDVgApgl4BNayYHpg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```python
class Solution:
    """剪枝策略"""
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        # 为了剪枝提前进行排序
        candidates.sort()
        self.res = self.trackbacking(startindex, candidates, target)
        return self.res

    def trackbacking(self, startindex, candidates, target):
        # 确定终止条件
        if self.sum == target:
            # 因为是 shallow copy，所以不能直接传入self.result
            self.res.append(self.result[:])
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        # 如果本层 sum + condidates[i] > target，就提前结束遍历，剪枝
        for i in range(startindex, len(candidates)):
            if self.sum + candidates[i] > target:
                return
            self.result.append(candidates[i])
            self.sum += candidates[i]
            # 因为可以无限制选取同一个数字，所以是 i
            self.trackbacking(i, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
        return self.res
```

### 6. 组合总和 II

> 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
>
> candidates 中的每个数字在每个组合中只能使用 一次 。
>
> 注意：解集不能包含重复的组合。
>
> ```
> 输入: candidates = [2,5,2,1,2], target = 5,
> 输出:
> [
> [1,2,2],
> [5]
> ]
> ```

这道题目和上一题组合总和的区别：

1. 本题 candidates 中的每个数字在每个组合中只能使用一次。
2. 本题数组 candidates 的元素是有重复的，而 [39. 组合总和 ](https://mp.weixin.qq.com/s?__biz=MzUxNjY5NTYxNA==&mid=2247494682&idx=2&sn=45c7642a4b42f589be18cd087d0c3388&scene=21#wechat_redirect)是无重复元素的数组 candidates

这个题目的关键在==去重==！题目给的集合里**有重复的元素**，但是不能有重复的组合！！！

如果把所有的组合全部求出来，再用 set 或者 map 去重，这么做很容易**超时**！所以我们需要在搜索的过程中就去掉重复组合。

所谓去重，==其实就是使用过的元素不能重复选取==。

都知道组合问题可以抽象成树型结构，那么使用过在这个树形结构上是有两个维度的，一个维度是==同一树枝==上使用过，一个维度是==同一树层==上使用过，没有理解这两个层面的“使用过”是造成大家没有彻底理解去重的根本原因。

题目要求：元素在同一组合是就可以重复使用的，只要给的集合有重复的数字，但是这两个组合不能相同，所以我们==要去重的是同一树层上的“使用过”，因为同一树枝上的都是一个组合里的元素，不用去重==。以 `candidates = [1, 1, 2], target = 3`为例：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv48aCU4UTGAaibHh1UFayia1yBvRvuXqu2Z4jnaY2fEhUoL3Ggr0zxN7vgzKBRHO7QmeBy5BO1BqeFg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

==回溯三部曲：==

1. **递归函数参数**：与上一题相同，此时还需要加一个 ==bool 型数组 used==，用来记录同一树枝上的元素是否使用过，这个集合去重的重任就是 used 完成的。

2. **递归终止条件**：终止条件为 `sum > target` 和 `sum == target`。
3. **单层搜索的逻辑**：如何判断同一树层上元素是否使用过了呢？

**如果 `candidates[i] == candidates[i - 1]` 并且 `used[i - 1] == false`，就说明：前一个树枝，使用了 `candidates [i - 1]`，也就是说同一树层使用过 candidates [i - 1]**。此时 for 循环里应该做 continue 的操作。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv48aCU4UTGAaibHh1UFayia1yFn6HgwBDohL8uc9icx9afAMLSQKaibWwItd8bZHaL9WYvmTTX7IwAg9A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我在图中将 used 的变化用橘黄色标注上，可以看出在 `candidates [i] == candidates [i - 1]` 相同的情况下：

- `used [i - 1] == true`，说明同一树支 `candidates [i - 1]` 使用过
- `used [i - 1] == false`，说明同一树层 `candidates [i - 1]` 使用过

和上一题相比，同样是求组合总和，但就是因为**其数组 candidates 有重复元素，而要求不能有重复的组合**，难度提升了不少。

```python
class Solution:
    """剪枝策略"""
    def __init__(self) -> None:
        self.res = []
        self.result = []
        self.sum = 0
        # 存储用过的元素值
        self.used = []

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        startindex = 0
        # 为了剪枝提前进行排序
        candidates.sort()
        self.used = [0]*len(candidates)
        self.trackbacking(startindex, candidates, target)
        return self.res

    def trackbacking(self, startindex, candidates, target):
        # 确定终止条件
        if self.sum == target:
            # 因为是 shallow copy，所以不能直接传入self.result
            self.res.append(self.result[:])
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        # 如果本层 sum + condidates[i] > target，就提前结束遍历，剪枝
        for i in range(startindex, len(candidates)):
            # 剪枝
            if self.sum + candidates[i] > target:
                    return
            # 检查同一树层是否出现曾经使用过的相同元素
            # 若数组中前后元素值相同，但前者却未被使用(used == False)，说明是for loop中的同一树层的相同元素情况
            # 注意这里，list[-1] 代表的不是 list[0] 的前一位而是列表的最后一位，这是不符合比较逻辑的，所以要从 i=1 开始取值
            # 为什么只用比较前一个呢？因为已经对数组进行排序了
            if i >= 1 and candidates[i] == candidates[i-1] and self.used[i-1] == 0:
                continue

            self.result.append(candidates[i])
            self.sum += candidates[i]
            self.used[i] = 1
            self.trackbacking(i+1, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
            self.used[i] = 0
        return self.res
```

### 7. 分割回文字符串

> 给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。
>
> **回文串** 是正着读和反着读都一样的字符串。
>
> ```
> 输入：s = "aab"
> 输出：[["a","a","b"],["aa","b"]]
> ```

本题涉及两个关键问题：

1. 切割问题，有不同的切割方式
2. 判断回文

切割问题类似于组合问题，例如对于字符串 abcdef：

- 组合问题：选取一个 a 之后，在 bcdef 中再去选取第二个，选取 b 之后在 cdef 中在选组第三个.....。
- 切割问题：切割一个 a 之后，在 bcdef 中再去切割第二段，切割 b 之后在 cdef 中在切割第三段.....。

所以切割问题，也可以抽象为一棵树形结构：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv752l07A1icibBf67wY0GN5cOWDabGLaaOOJKXX23gIU966mkvzD94MOl6TAUAvuCl509osqRRbpfYw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

递归用来纵向遍历，for 循环用来横向遍历，切割线（就是图中的红线）切割到字符串的结尾位置，说明找到了一个切割方法。

此时可以发现，切割问题的回溯搜索的过程和组合问题的回溯搜索的过程是差不多的。

==回溯三部曲==：

1. 递归函数参数：全局变量数组 path 存放切割后回文的子串，二维数组 result 存放结果集。还需要 startindex，因为切割过的地方，不能重复切割，和组合问题保持一致。

2. 递归函数终止条件：从树形结构图中可以看出，切割线找到了字符串的后面，说明找到了一种切割方法，此时就是本层递归的终止条件，==在代码中==，递归参数需要传入 startindex 表示下一轮递归遍历的起始位置，这个 startindex 就是切割线。
3. 单层搜索的逻辑：在递归循环中，如何截取子串？我们 定义了起始位置 startIndex，那么 [startIndex, i] 就是要截取的子串。首先判断这个子串是不是回文，如果是回文，就加入 path，path 用来记录切割过的回文子串。**注意切割过的位置，不能重复切割，所以，backtracking (s, i + 1); 传入下一层的起始位置为 i + 1**。

==判断回文子串==：

我们可以使用双指针法，一个指针从前向后，一个指针从后向前，如果前后指针所指向的元素是相等的，就是回文字符串了

==难点剖析==：

- 切割问题可以抽象为组合问题
- 如何模拟切割线
- 切割问题中递归如何终止
- 在递归循环中如何截取子串
- 如何判断回文

> 我们在做难题的时候，总结出来难究竟难在哪里也是一种需要锻炼的能力。

又是学习思路后独自完成的一题，真棒！

```python
class Solution:
    def __init__(self) -> None:
        self.path = []
        self.res = []

    def partition(self, s: str) -> List[List[str]]:
        startindex = 0
        self.backtracking(s, startindex)
        return self.path

    def backtracking(self, s, startindex):
        # 递归结束条件
        # 当指针走到最后，回溯结束
        if startindex == len(s):
            self.path.append(self.res[:])
            return

        # 进入单层循环
        for i in range(startindex, len(s)):
            # 这里就应该判断这个子字符串是不是回文串了，如果不是就 continue
            if not self.isPalindrome(s[startindex:i+1]):
                continue
            self.res.append(s[startindex:i+1])
            self.backtracking(s, i+1)
            self.res.pop()

    def isPalindrome(self, s):
        """判断某个字符串是否为回文字符串"""
        i, j = 0, len(s)-1
        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return False
        return True
```

### 8. 复制 ip 地址：难

> 有效 IP 地址正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
>
> 例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
> 给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按任何顺序返回答案。
>
> ```
> 输入：s = "25525511135"
> 输出：["255.255.11.135","255.255.111.35"]
> ```

本题和上一题比较类似，其实都是切割问题，切割问题就可以使用回溯法把所有可能性搜出来：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6mT5fLDGWGROx8nmCACI1veyyNRst5JoiclawgZesPZWNO9H195g9kRFICHpvw9gMVyXqoNMltnXQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

==回溯三部曲==：

1. 递归参数：startindex 一定是需要的，因为不能重复分割，记录下一层递归分割的起始位置；本题我们还需要一个变量，记录添加逗号的数量
2. 递归终止条件：本题明确要求只会分成 4 段，所以不能用切割线切到最后作为终止条件，而是分割的段数作为终止条件。pointNum 表示逗点数量，pointNum 为 3 说明字符串分成了 4 段了。然后验证一下第四段是否合法，如果合法就加入到结果集里
3. 单层搜索的逻辑：在 for 循环中 `[startIndex, i]` 这个区间就是截取的子串，需要判断这个子串是否合法，如果合法就在字符串后面加上符号`.` 表示已经分割，如果不合法就结束本层循环。

然后就是递归和回溯的过程：

- 递归调用时，下一层递归的 startindex 要从 ==i + 2== 开始，因为需要在字符串中加入分隔符，同时记录分隔符的数量 pointNum 要加 1。
- 回溯的时候，就将刚刚加入的分隔符删掉，同时 pointNum 减 1

```python
class Solution:
    def __init__(self):
        self.result = []

    def restoreIpAddresses(self, s: str) -> List[str]:
        '''
        本质切割问题使用回溯搜索法，本题只能切割三次，所以纵向递归总共四层
        因为不能重复分割，所以需要start_index来记录下一层递归分割的起始位置
        添加变量 point_num 来记录逗号的数量[0,3]
        '''
        self.result.clear()
        if len(s) > 12: return []
        self.backtracking(s, 0, 0)
        return self.result

    def backtracking(self, s: str, start_index: int, point_num: int) -> None:
        # Base Case
        if point_num == 3:
            if self.is_valid(s, start_index, len(s)-1):
                self.result.append(s[:])
            # 注意 return 是纵向结束，但是 break 是横向
            return
        # 单层递归逻辑
        for i in range(start_index, len(s)):
            # [start_index, i]就是被截取的子串
            if self.is_valid(s, start_index, i):
                s = s[:i+1] + '.' + s[i+1:]
                self.backtracking(s, i+2, point_num+1)  # 在填入.后，下一子串起始后移2位
                s = s[:i+1] + s[i+2:]    # 回溯
            else:
                # 若当前被截取的子串大于255或者大于三位数，直接结束本层循环，横向
                break

    def is_valid(self, s: str, start: int, end: int) -> bool:
        if start > end: return False
        # 若数字是0开头，不合法
        if s[start] == '0' and start != end:
            return False
        if not 0 <= int(s[start:end+1]) <= 255:
            return False
        return True
```

### 9. 子集

> 给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。
>
> 解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。
>
> ```
> 输入：nums = [1,2,3]
> 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
> ```

自己写出来的 ！！
```python
class Solution:
    def __init__(self) -> None:
        self.res = [[]]
        self.path = []

    def subsets(self, nums: List[int]) -> List[List[int]]:
        startindex = 0
        self.backtracking(nums, startindex)
        return self.res

    def backtracking(self, nums, startindex):
        # 回溯结束条件
        if startindex == len(nums):
            # self.res.append(self.path[:])
            return

        # 单层递归逻辑
        for i in range(startindex, len(nums)):
            self.path.append(nums[i])
            self.res.append(self.path[:])
            self.backtracking(nums, i+1)
            self.path.pop()
```

如果把子集问题、组合问题、分割问题都抽象为一棵树的话，那么==组合问题和分割问题==都是收集树的==叶子节点==，而==子集问题==是找==树的所有节点==！

其实子集也是一种组合问题，因为它的集合是无序的，子集 {1,2} 和 子集 {2,1} 是一样的。既然无序，取过的元素不会重复取，写回溯算法的时候，for 就要从 startindex 开始，而不是从 0 开始！

> 什么时候可以从 0 开始呢？要么是求在不同的集合中取元素，要么是在解决排列问题。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv7icZnSOhUqwR4ibqNP3nHyktNROmSHwzzNwsWCBrtBH5tHuhg5YKSPl77r8OiapekZ77Dn8NchSoMBw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

从图中红线部分，可以看出**遍历这个树的时候，把所有节点都记录下来，就是要求的子集集合**。



## ==贪心算法==

### 1. 贪心算法的基础知识

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5NV9jjxYN7qSW0cae98Ogorh6MlYWDpoxqlSmBbwAjzdpqqxCicz92qbK1lnQJW98YyEN9wib3aibpw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

==贪心的本质==：这种方法模式一般将求解过程分成若干个步骤，但是每个步骤都应用贪心原则，选取当前状态下最好 / 最优的选择（局部最有利的选择），并以此希望最后堆叠出的结果也是最好最优的解。

> 看着这个名字，贪心，贪婪这两字的内在含义最为关键。这就好像一个贪婪的人，他事事都想要眼前看到最好的那个，看不到长远的东西，也不为最终的结果和将来着想，贪图眼前局部的利益最大化，有点走一步看一步的感觉。

这么说有点抽象，来举一个例子：

例如，有一堆钞票，你可以拿走十张，如果想达到最大的金额，你要怎么拿？

指定每次拿最大的，最终结果就是拿走最大数额的钱。

每次拿最大的就是局部最优，最后拿走最大数额的钱就是推出全局最优。

再举一个例子如果是 有一堆盒子，你有一个背包体积为 n，如何把背包尽可能装满，如果还每次选最大的盒子，就不行了。这时候就需要动态规划。动态规划的问题在下一个系列会详细讲解。

==贪心的套路==：什么时候用贪心？如何通过局部最优，推出整体最优？

如何验证可否用贪心算法呢？**最好用的策略就是举反例，如果想不到反例，那么就试一试贪心吧**。

==贪心算法的一般解题步骤：==

1. 将问题分解为若干个子问题
2. 找出适合的贪心策略
3. 求解每一个子问题的最优解
4. 将局部最优解堆叠成全局最优解

------------

1. 从某个初始解出发
2. 采用迭代的过程，当可以向目标前进一步时，就根据局部最优策略，得到一部分解，缩小问题规模
3. 将所有解综合起来

> 思考找硬币过程，案例学习



### 2. 分发饼干

> 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
> 对每个孩子 $i$，都有一个胃口值 $g[i]$，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 $j$，都有一个尺寸 $s[j]$ 。如果 $s[j] >= g[i]$，我们可以将这个饼干 $j$ 分配给孩子 $i$ ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
>
> ```
> 输入: g = [1,2,3], s = [1,1]
> 输出: 1
> ```

为了了满足更多的小孩，就不要造成饼干尺寸的浪费。

大尺寸的饼干既可以满足胃口大的孩子也可以满足胃口小的孩子（小尺寸的饼干只能满足小的），那么就应该优先满足胃口大的。

**这里的局部最优就是大饼干喂给胃口大的，充分利用饼干尺寸喂饱一个，全局最优就是喂饱尽可能多的小孩**。

可以尝试使用贪心策略，先将饼干数组和小孩数组排序。

然后==从后向前==遍历小孩数组，用大饼干优先满足胃口大的，并统计满足小孩数量。如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4jZrEXMwopFkYlhVRTajjg47daCuGygmrJU4a0gJ5NMB6sqdrGehpzgagzMNx3vT7EVxGUHUSfSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

> 试想一下如果你把 9 喂给了 孩子 2 或孩子 1，最终只能满足两个小孩，显然不是最优的。

文中详细介绍了思考的过程，**想清楚局部最优，想清楚全局最优，感觉局部最优是可以推出全局最优，并想不出反例，那么就试一试贪心**。

```python
class Solution:
    # 思路 1：优先考虑饼干
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        res = 0
        for i in range(len(s)):
            # 小饼干先喂饱小胃口
            if res <len(g) and s[i] >= g[res]:  
                # 只有得到饼干了小孩指针才往前走
                res += 1
        return res
```



### ==3. 摆动序列==

> 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
>
> - 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。
>
> - 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
>
> 子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
>
> 给你一个整数数组 `nums` ，返回 `nums` 中作为摆动序列的最长子序列的长度 。
>
> ```
> 输入：nums = [1,17,5,10,13,15,10,5,16,8]
> 输出：7
> 解释：这个序列包含几个长度为 7 摆动序列。
> 其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
> ```

#### 3.1 贪心解法

本题要求通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。

来分析一下，要求删除元素使其达到最大摆动序列，应该删除什么元素呢？

用示例二来举例，如图所示：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4gWEbz8UpRSdmUP9LKOia17Ec2w3rxzzJjnjP62nUpgQZMeELGbyZOShuGaJ3WvcNEy30fKKCK9Og/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

==局部最优==：删除单调坡度上的节点（不包括单调坡度两端的节点），那么这个坡度就可以有两个局部峰值。

==整体最优==：整个序列有最多的局部峰值，从而达到最长摆动序列。

局部最优推出全局最优，并举不出反例，那么试试贪心！

实际操作中，其实连删除的操作都不用做，因为题目要求的是最长摆动子序列的长度，所以只需要统计数组的峰值数量就可以了（相当于是删除单一坡度上的节点，然后统计长度）

==这就是贪心所贪心的地方，让峰值尽可能的保持峰值，然后删除单一坡度上的节点。==

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        # 题目里 nums 长度大于等于 1，当长度为 1 时，其实到不了 for 循环里去，所以不用考虑 nums 长度
        preC, curC, res = 0, 0, 1
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            # 差值为 0 时，不算摆动
            if curC * preC <= 0 and curC != 0:
                res += 1
                # 如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
                preC = curC
        return res
```

#### 3.2 动态规划

对于我们当前考虑的这个数，要么作为山峰（即 `nums [i] > nums [i-1]`），要么是作为山谷（即 `nums [i] < nums [i - 1]`）。

- 设 dp 状态 `dp[i][0]`，表示考虑前 $i$ 个数，第 $i$ 个数作为山峰的摆动子序列的最长长度
- 设 dp 状态 `dp[i][1]`，表示考虑前 $i$ 个数，第 $i$ 个数作为山谷的摆动子序列的最长长度

则转移方程为：

- `dp[i][0] = max(dp[i][0], dp[j][1] + 1)`，其中 `0 < j < i` 且 `nums[j] < nums[i]`，表示将 `nums [i]` 接到前面某个山谷后面，作为山峰。
- `dp[i][1] = max(dp[i][1], dp[j][0] + 1)`，其中 `0 < j < i` 且 `nums[j] > nums[i]`，表示将 `nums [i]` 接到前面某个山峰后面，作为山谷。

#### 3.3 进阶

可以用两棵线段树来维护区间的最大值：

- 每次更新 `dp[i][0]`，则在 `tree1` 的 `nums[i]` 位置值更新为 `dp[i][0]`
- 每次更新 `dp[i][1]`，则在 `tree2` 的 `nums[i]` 位置值更新为 `dp[i][1]`
- 则 dp 转移方程中就没有必要 j 从 0 遍历到 i-1，可以直接在线段树中查询指定区间的值即可



### 4. 最大子数组和：DP

> 给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
>
> **子数组** 是数组中的一个连续部分。
>
> ```
> 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
> 输出：6
> 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
> ```

#### 4.1 暴力解法

暴力解法的思路，第一层 for 就是设置起始位置，第二层 for 循环遍历数组寻找最大值：

#### 4.2 贪心解法

==如何贪心呢？==

如果 -2 和 1 在一起，计算起点的时候，一定是从 1 开始计算，因为负数只会拉低总和，这就是贪心贪的地方！

==局部最优==：当前连续和为负数的时候立刻放弃，从下一个元素重新计算连续和，因为负数加上下一个元素连续和只会越来越小。

==全局最优==：选取最大的“连续和”

**局部最优的情况下，并记录最大的连续和，可以推出全局最优。**

从代码角度上来讲：遍历 `nums`，从头开始用 `count` 累积，如果 `count` 一旦加上 `nums [i]` 变为负数，那么就应该从 `nums [i+1]` 开始==从 0 累积 count== 了，因为已经变为负数的 `count`，只会==拖累总和==。（很有道理）

> 这相当于是暴力解法中的不断调整最大子序和和区间的起始位置。

而区间的终止位置，起始就是如果 count 取到最大值了，及时记录下来。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv6iby4YcbthvSiavmrQz0Vof28oUibXcZ460BFBvrD6nquPACzd6OmfJfsicVIVL0fbdf9Z8L4Mas08cQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

红色的起始位置就是贪心每次取 count 为正数的时候，开始一个区间的统计。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        result = -float('inf')
        count = 0
        for i in range(len(nums)):
            count += nums[i]
            if count > result:
                result = count
            if count <= 0:
                count = 0
        return result
```

#### 4.3 动态规划

==动态规划==更好理解：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i] 表示 i 之前包含 i 的连续子数组最大和
        dp = [0]*len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
        return max(dp)
```



### 5. 买卖股票的最佳时机 II

> 给定一个数组 `prices` ，其中 `prices[i]` 表示股票第 i 天的价格。
>
> 在每一天，你可能会决定购买和 / 或出售股票。你在任何时候最多只能持有一股股票。你也可以购买它，然后在同一天出售。
> 返回你能获得的最大利润和。
>
> 输入: `prices = [7,1,5,3,6,4]`
> 输出: `7`
> 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

本题要清楚两点：

1. 只有一只股票
2. 当前只有买股票或者卖股票的操作
3. 想要获得利润至少要两天为一个交易单元

==贪心算法==：

如果想到最终利润是可以==分解==的，那么本题就很容易了！

如何分解呢？

假如第 0 天买入，第 3 天卖出，那么利润为：prices [3] - prices [0]。

相当于 (prices [3] - prices [2]) + (prices [2] - prices [1]) + (prices [1] - prices [0])。如果三天的利润和里面某一天的利润是负数，例如第 1 天，那么显然（第 0 天买入，第 3 天卖出）的利润就低于（第一天买入，第 3 天卖出）的利润。

此时就是把利润分解为每天为单位的维度，而不是从 0 天到第 3 天整体去考虑，那么根据 prices 可以得到每天的利润序列：`(prices [i] - prices [i - 1]).....(prices [1] - prices [0])`

> 其实就相当于一位有预知能力的股民，她可以知道今天和第二天股票的价格（贪心算法的一个核心是短视，题目说每一天的价格都知道，但我们把主人公限定为一个”目光短浅“的人），显然，如果第二天的价格比今天高，那就今天买明天卖肯定可以赚钱。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4ibkWqQ584eOMJ75r8rPctCW19I8Bu6jkGyuJOolYazIM28NR2oQr4ykmgfVxv1TnLiaLH23hQ9Jxg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

第一天是没有利润的，至少第二天才会有利润，所以利润的序列比股票序列少一天！

==从图中可以发现，其实我们需要收集每天的正利润就可以，收集正利润的区间，就是股票买卖的区间，而我们只需要关注最终利润，不需要记录区间。==

那么只收集正利润就是贪心算法贪心的地方！

==局部最优==：收集每天的正利润；

==全局最优==：求得最大利润

```python
class Solution:
    """贪心算法"""
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(len(prices)-1):
            temp =prices[i+1] - prices[i]
            if temp > 0:
                profit += temp
        return profit
```

### ==6. 跳跃游戏==

> 给定一个非负整数数组 nums ，你最初位于数组的第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。
>
> ```
> 输入：nums = [3,2,1,0,4]
> 输出：false
> 解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ，所以永远不可能到达最后一个下标。
> ```

这个问题其实跳几步无所谓，关键在于可跳的覆盖范围！不一定非要明确一次究竟跳几步，每次取最大的跳跃步数，这个就是可以跳跃的覆盖范围。这个范围内，别管怎么跳的，反正一定可以跳过来。

==那么问题就转化为跳跃覆盖范围究竟可不可以覆盖到终点！==

每次移动取最大跳跃步数（得到最大的覆盖范围），每移动一个单位，就更新最大覆盖范围。

- ==贪心算法局部最优解==：每次取最大跳跃步数（取最大覆盖范围），

- ==整体最优解==：最后得到整体最大覆盖范围，看是否能到终点。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6pDDdhseTveobLyAvuu6uQwibzic4VXfNYibgqI3mcoE8AbK2ObecsZibWjNmG6kBaqhzQ9NK0XBRJFQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        maxscale = 0
        # python 不支持动态修改 for 循环中变量，可以使用 while 循环代替
        # while i <= maxscale
        for i in range(len(nums) - 1):
            maxscale = max(maxscale, i + nums[i])
            # 如果此时最大范围还没有超过 i，以后也不可能超过了
            if maxscale <= i:
                return False
        return maxscale >= len(nums)-1
```

### ==7. 跳跃游戏 II==

> 给你一个非负整数数组 nums ，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。
>
> 你的目标是使用==最少==的跳跃次数到达数组的最后一个位置。假设你总是可以到达数组的最后一个位置。
>
> 输入: nums = [2,3,1,1,4]
> 输出: 2
> 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

本题需要计算最小步数，那么要想清楚什么时候步数才一定要加一呢？

==贪心的思路==：

- ==局部最优==：当前可移动距离尽可能多走，如果还没到终点，步数再加一
- ==整体最优==：一步尽可能多走，从而能达到最小步数

思路虽然是这样的，但是在写代码的时候不能真的能跳多远就跳多远，那样就不知道下一步最远能跳到哪里了。

**所以真正解题的时候，要从覆盖范围出发，不管怎么跳，覆盖范围内一定是可以跳到的，==以最小的步数增加覆盖范围==，覆盖范围一旦覆盖了终点，得到的就是最小步数。**

这里需要统计两个覆盖范围，==当前这一步的最大覆盖==和==下一步最大覆盖==。

如果移动下标达到了当前这一步的最大覆盖最远距离了，还没有到终点的话，那么就必须再走一步来增加覆盖范围，直到覆盖范围覆盖了终点。

> 不管你从哪里起跳，步数加一之后指针移到最大的范围处。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv40YfANYrjM5rTqA3zx25n8EpcMf0hHuYEx5qrHXAy6buLJgicibda2zwqicVoYbH6icUymN0fHYA1zxg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

图中覆盖范围的意义在于，只要红色的区域，最多两步一定可以到！不管具体怎么跳，反正一定可以跳到。

> 尝试自己写，错了三次，认真学习。

#### 7.1 方法一

移动下标达到了当前覆盖的==最远==距离下标时，步数就要加一，来增加覆盖距离。最后的步数就是最少步数。

这里还有个特殊情况需要考虑，当移动下标达到了当前覆盖的最远距离下标时：

- 如果当前覆盖最远距离下标不是集合终点，步数就加一，还需要继续走
- 如果当前覆盖最远距离下标就是集合终点，步数不用加一，因为不能再往后走了。

#### 7.2 方法二

针对方法一的特殊情况，可以统一处理，即：移动下标只要遇到当前覆盖最远距离的下标，直接步数加一，不考虑是不是终点的情况。

想要达到这样的效果，只要让移动下标，最大只能移动到 `len(nums) - 2` 的地方就可以了。

因为当移动下标指向 `len(nums) - 2`时（倒数第一位）：

- 如果移动下标等于当前覆盖最大距离下标，需要再走一步，因为最后一步一定是可以到的终点

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv40YfANYrjM5rTqA3zx25n8ZHicEPQJwd8zHuj91oAkxbzFhicq0uIDnqrB5dlOauU09drhGhoicKk6Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- 如果移动下标不等于当前覆盖最大距离下标，说明当前覆盖最远距离就可以直接达到终点了，不需要再走一步：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv40YfANYrjM5rTqA3zx25n8TqrZFIcJSzqNicnibtCLvdz4NgA2v5kdJ3Mmy0ReWP7wXUNkhwUPKDMw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

关键在于控制移动下标只移动到倒数第二个位置，所以移动下标只要遇到当前覆盖最远距离的下标，直接步数加 1，不用考虑别的。

==理解本题的关键在于==：以==最小的步数增加最大的覆盖范围==，直到覆盖范围覆盖了终点，这个范围内最小步数一定可以跳到，不用管具体是怎么跳的，不纠结于一步究竟跳一个单位还是两个单位。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        """以最小的步数增加最大的覆盖范围"""
        # 当前覆盖的最远距离下标
        curDistance = 0
        # 下一步覆盖的最远距离下标
        nextDistance = 0
        # 记录走的最大步数
        step = 0
        if len(nums) == 1: return 0
        for i in range(len(nums)):
            nextDistance = max(i + nums[i], nextDistance)
            # 指针遍历当前范围内的所有元素，看从哪个位置出发可以最大程度地扩展自己的覆盖范围
            # 如果指针走到当前势力范围的最后一个元素，但是当前范围没到最后一个位置，step + 1，往后走
            if i == curDistance:
                if curDistance != len(nums) - 1:
                    step += 1
                    curDistance = nextDistance
                    if nextDistance >= len(nums) - 1: break
        return step
```

#### ==7.3 动态规划==

dp[i] 表示跳到 i 的最小次数，递推公式是：`dp[i] = min(dp[i], dp[j]+1) if j + num[j] >= i`

感觉可以尝试下 dp，贪心算法有点难以理解。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        """以最小的步数增加最大的覆盖范围"""
        dp = [float('inf')]* len(nums)
        dp[0] = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if j + nums[j] >= i:
                    dp[i] = min(dp[i], dp[j] + 1) 
        return dp[-1]
```

但是上述方法超时了。**105 / 106** 个通过测试用例。

### 8. [K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

> 给你一个整数数组 nums 和一个整数 k ，按以下方法修改该数组：
>
> - 选择某个下标 i 并将 `nums[i]` 替换为 `-nums[i]` 。
> - 重复这个过程恰好 k 次。可以多次选择同一个下标 i 。
>
> 以这种方式修改数组后，返回数组可能的最大和 。

```
输入：nums = [4,2,3], k = 1
输出：5
解释：选择下标 1 ，nums 变为 [4,-2,3] 。
```

==贪心算法：==

- ==局部最优==：让绝对值大的负数变为正数，当前数值达到最大
- ==整体最优==：整个数组和达到最大

那么如果将负数都转变为正数后，K 依然大于 0，此时的问题就是一个有序正整数序列，如何转变 K 次正负，让数组和达到最大？

==继续贪心==：

- ==局部最优==：只找到数值最小的正整数进行翻转，当前数值可以达到最大
- ==全局最优==：整个数组和达到最大

本题的解题步骤是：

1. 将数组按照==绝对值==大小从大到小进行排序
2. 从前向后遍历，遇到负数将其变为正数，同时 `K = K-1`，这里包含的思想是先抓紧把绝对值大的负数取反
3. 如果 K 还大于 0（此时全部都是正数了），那么反复转变绝对值最小的元素（也就是排在末尾的元素），将 K 用完
4. 求和

```python
class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        A = sorted(A, key=abs, reverse=True) # 将A按绝对值从大到小排列
        for i in range(len(A)):
            if K > 0 and A[i] < 0:
                A[i] *= -1
                K -= 1
        if K > 0:
            A[-1] *= (-1)**K #取A最后一个数只需要写 -1
        return sum(A)
```



### ==9. 加油站==

> 在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
>
> 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
>
> 给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则保证它是唯一的。
>
> ```
> 输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
> 输出: 3
> ```

#### ==9.1 暴力解法==

暴力的方法就是遍历每一个加油站为起点的情况，模拟一圈，如果跑了一圈，中途没有断油，并且最后油量大于等于 0，说明这个起点是 ok 的，for 循环适合模拟从头到尾的遍历，而 while 循环适合模拟环形遍历，要善于使用 while！

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # 从 i 出发
        for i in range(len(cost)):
            # rest 表示从 i 走到 i+1 后车里还剩下多少油
            rest = gas[i] - cost[i]
            # 取余，表示 i 的下一个
            index = (i+1) % len(cost)
            # 模拟以 i 为起点行驶一圈
            # 如果 rest > 0 表明你已经走到了下一个节点
            while rest > 0 and index != i:
                rest += gas[index] - cost[index]
                index = (index + 1) % len(cost)
            if rest >= 0 and index == i:
                return i
        return -1
```

#### 9.2 贪心算法

> 不太好理解

换一个思路：首先如果总油量减去总消耗大于等于零那么一定可以跑完一圈，说明各个站点的加油站剩油量 rest[i] 相加一定是大于等于零的。每个加油站的剩余量 rest [i] 为 gas [i] - cost [i]。i 从0 开始累加 rest[i] 和记为 curSum，一旦 curSum 小于零，说明 [0,i] 区间都不能作为起始位置，起始位置从 i+1 算起，再从 0 计算 curSum。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4BCMNa6HRCJD3LIFkh8fwl5onEw1GSAWTEahpzjSRErQRrVlvFA0mI42X3Oj3uDgjib3exTjicVbgA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

==局部最优==：当前累加 rest[j] 的和 curSum 一旦小于 0，起始位置至少要是 j+1，因为从 j 开始一定不行

==全局最优==：找到可以跑一圈的起始位置

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start = 0
        curSum = 0
        totalSum = 0
        for i in range(len(gas)):
            curSum += gas[i] - cost[i]
            totalSum += gas[i] - cost[i]
            if curSum < 0:
                curSum = 0
                start = i + 1
        if totalSum < 0: return -1
        return start
```

### 10. 分发糖果

> n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
>
> 你需要按照以下要求，给这些孩子分发糖果：
>
> 每个孩子至少分配到 1 个糖果。
> 相邻两个孩子评分更高的孩子会获得更多的糖果。
> 请你给每个孩子分发糖果，计算并返回需要准备的最少糖果数目。
>
> ```
> 输入：ratings = [1,0,2]
> 输出：5
> 解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
> ```

这道题目一定要确定一边之后，再确定另一边，例如比较每一个孩子的左边，然后再比较右边，==如果两边一起考虑一定会顾此失彼==。

==先确定右边评分大于左边的情况（也就是从前向后遍历）：==

- 局部最优：只要右边评分比左边大，右边的孩子就多一个糖果；

- 全局最优：相邻的孩子中，评分高得右孩子获得比左边孩子更多的糖果

==再确定左孩子大于右孩子的情况（从后向前遍历）：因为要根据 ratings[i+1] 来确定 ratings[i] 的值==

- 如果  ratings [i] > ratings [i + 1]，此时 candyVec [i]（第 i 个小孩的糖果数量）就有两个选择了，一个是 candyVec [i + 1] + 1（从右边这个加 1 得到的糖果数量），一个是 candyVec [i]（之前比较右孩子大于左孩子得到的糖果数量），两者取最大值即可

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        nums = [1]*len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1]:
                nums[i] = nums[i-1] + 1
        for j in range(len(ratings)-2,-1,-1):
            if ratings[j] > ratings[j+1]:
                nums[j] = max(nums[j], nums[j+1] + 1)
        return sum(nums)
```



### 11. 柠檬水找零

> 在柠檬水摊上，每一杯柠檬水的售价为 5 美元。顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
>
> 每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。
>
> 注意，一开始你手头没有任何零钱。
>
> 给你一个整数数组 bills ，其中 bills[i] 是第 i 位顾客付的账。如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
>
> ```
> 输入：bills = [5,5,5,10,20]
> 输出：true
> ```

如何找零才能保证完整全部账单的找零呢？

我们需要维护三种金额的数量：5，10 和 20，有如下三种情况：

1. 账单是 5，直接收下
2. 账单是 10，消耗一个 5，增加一个 10
3. 账单是 20，优先消耗一个 10 和一个 5，如果不够，再消耗三个 5

情况一和情况二都是固定策略，唯一的不确定性来自于情况三（贪心）:

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        res = {key: 0 for key in ['5', '10', '20']}
        if bills[0] != 5:
            return False
        else:
            res['5'] = 1
        for i in range(1, len(bills)):
            if bills[i] == 5:
                res['5'] += 1
            elif bills[i] == 10:
                if res['5'] > 0:
                    res['5'] -= 1
                else:
                    return False
                res['10'] += 1
            elif bills[i] == 20:
                if res['10'] > 0 and res['5'] > 0:
                    res['10'] -= 1
                    res['5'] -= 1
                elif res['10'] == 0 and res['5'] > 2:
                    res['5'] -= 3
                else:
                    return False
        return True
```

> 上面的代码是自己写的，其实不需要用字典的，定义三个变量 five，ten 和 twenty 也可以。

### ==12. 根据身高重建队列==

> 假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
>
> 请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
>

本题有两个维度，h 和 k，如何确定一个维度，然后再按照另一个维度重新排列，和分发糖果题目有点像，如果两个维度一起考虑一定会顾此失彼！

我们按身高去排列，使得前面的节点一定都比本节点高，那么只需要按照 k 为下标重新插入队列就可以了，为什么呢？

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6juc01BZlQtlZpy2LVOWFKrjs0KOIGUnCMY2qQy2WNa6Xec6oPDVXibnPSmzliatYAlkYoPeeOfCNg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />

按照身高排序之后，优先按照身高高的 people 的 k 来插入，后序插入节点也不影响前面已经插入的节点，最终按照 k 的规则完成队列，所以在按照身高从大到小排序后：

==局部最优==：优先按身高高的 people 的 k 来插入，插入操作过后的 people 满足队列属性

==全局最优==：最后都做完插入操作，整个队列满足题目队列属性

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
    	# 先按照h维度的身高顺序从高到低排序。确定第一个维度
        # lambda返回的是一个元组：当-x[0](维度h）相同时，再根据x[1]（维度k）从小到大排序
        people.sort(key=lambda x: (-x[0], x[1]))
        que = []
	
	    # 根据每个元素的第二个维度k，贪心算法，进行插入
        # people已经排序过了：同一高度时k值小的排前面。
        for p in people:
            # insert 函数插入时，位置上的旧元素往后移动腾出位置
            que.insert(p[1], p)
        return que
```

### 13. 用最少数量的箭引爆气球

> 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
>
> 一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
>
> 给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
>
> 输入：points = [[10,16],[2,8],[1,6],[7,12]]
> 输出：2
> 解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球

直觉上看，貌似只射重叠最多的气球，用的弓箭一定最少，那么有没有当前重叠了三个气球，我射两个，留下一个和后面的一起射这样弓箭用的更少的情况呢？没有

==局部最优==：当气球出现重叠时，一起射，所用弓箭数最少

==全局最优==：把所有气球射爆所用弓箭数最少

那如何模拟气球射爆的过程呢？是在数组中移除元素还是做标记？

如果把气球排序之后，从前到后遍历气球，被射过的气球仅仅跳过就行了，没有必要让气球数组 remove 气球，只要记录一下箭的数量就可以了。

==为了让气球尽可能重叠，我们需要对数组进行排序==。如果按照起始位置排序，那么就从前往后遍历数组，靠左尽可能让气球重复。

从前向后遍历遇到重叠的气球了怎么办？

**如果气球重叠了，重叠气球中右边边界的最小值 之前的区间一定需要一个弓箭**。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5jic4icLjE03g34yP8iaaUQqyrCveWgFggR16h4myicJ5JQ2pvxXxaf3FR4icXvzxr800TQrQrhb6Dn6Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

> 可以看出第一组重叠气球，一定需要一个箭，但是气球 3 的左边界大于第一组重叠气球的最小右边界，所以再需要一支箭来射气球 3。

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0: return 0
        points.sort(key=lambda x: x[0])
        result = 1
        for i in range(1, len(points)):
            if points[i][0] > points[i - 1][1]: # 气球i和气球i-1不挨着，注意这里不是>=
                result += 1     
            else:
                points[i][1] = min(points[i - 1][1], points[i][1]) # 更新重叠气球最小右边界
        return result
```

### ==14. 划分字母区间==

> 字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
>
> 输入：S = "ababcbacadefegdehijhklij"
> 输出：[9,7,8]
> 解释：
> 划分结果为 "ababcbaca", "defegde", "hijhklij"。
> 每个字母最多出现在一个片段中。
> 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。

题目要求同一字母最多出现在一个片段中，那么如何把同一个字母都圈在同一个区间里呢？

在遍历的过程中相当于是要找每一个字母的边界，如果找到之前遍历过的所有字母的最远边界，说明这个边界就是分割点了。

1. 统计每一个字符最后出现的位置
2. 从头遍历字符，并更新字符的最远出现下标，如果找到字符最远出现位置下标和当前下标相等了，则找到了分割点

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6hB37qeLpLNqKNqibibn4TyePRIoSQ3rLUfR8s9rEdvE4hcrfB7Nkw1T2e4ymiceibDTp6wrYLymczicw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        hash = [0] * 26
        for i in range(len(s)):
            hash[ord(s[i]) - ord('a')] = i
        result = []
        left = 0
        right = 0
        for i in range(len(s)):
            # 不断更新最大边界
            right = max(right, hash[ord(s[i]) - ord('a')])
            if i == right:
                result.append(right - left + 1)
                left = i + 1
        return result
```

### ==15. 合并区间==

> 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
>
> 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
> 输出：[[1,6],[8,10],[15,18]]
> 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

注意，此题一定要排序，如果按照左边界排序，排序之后局部最优，每次合并都取最大的右边界，这样就可以合并更多的区间了

整体最优：合并所有重叠的区间。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6MhxFViaiczr8HCw7SN51icjaj9wIWucDKTfHeSNDYeTwj1zfia2ibHUKVib7EpuTpRmCu8LIPfe3huqOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

知道如何判断重复之后，剩下的就是合并了，如何去模拟合并区间呢？

其实就是用合并区间后左边界和右边界，作为一个新的区间，加入到 result 数组里就可以了。如果没有合并就把原区间加入到 result 数组里。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0: return intervals
        # 排序
        intervals.sort(key=lambda x: x[0])
        result = []
        result.append(intervals[0])
        for i in range(1, len(intervals)):
            # 取结果列表里最后一个值
            last = result[-1]
            if last[1] >= intervals[i][0]:
                result[-1] = [last[0], max(last[1], intervals[i][1])]
            else:
                result.append(intervals[i])
        return result
```



### 16. 单调递增的数字

> 当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。
>
> 给定一个整数 n ，返回 小于或等于 n 的最大数字，且数字呈单调递增 。
>
> ```
> 输入: n = 10
> 输出: 9
> ```

题目要求小于等于 N 的最大单调递增的整数，那么拿一个两位的数字来举例：

例如：98，一旦出现 strNum [i - 1] > strNum [i] 的情况（非单调递增），首先想让 strNum [i - 1]---，然后 strNum [i] 设为9，这样这个整数就是 89，即小于 98 的最大的单调递增整数。

==局部最优==：遇到 strNum [i - 1] > strNum [i] 的情况，让 strNum [i - 1]--，然后 strNum [i] 给为 9，可以保证这两位变成最大单调递增整数。

==全局最优==：得到小于等于 N 的最大单调递增的整数

此外，还要考虑遍历顺序，只有从后向前遍历才能重复利用上次比较的结果。

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        a = list(str(n))
        for i in range(len(a)-1,0,-1):
            if int(a[i-1]) > int(a[i]):
                a[i-1] = str(int(a[i-1]) - 1)
                # 把后面所有位都设置为 9
                a[i:] = '9' * (len(a) - i)  # python不需要设置flag值，直接按长度给9就好了
        return int("".join(a))
```

### 17. ==贪心算法：买卖股票的最佳时机含手续费==

> 给定一个整数数组 prices，其中 prices[i] 表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。
>
> 你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
>
> 返回获得利润的最大值。
>
> 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
>
> 输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
> 输出：8
> 解释：能够达到的最大利润: 
> 在此处买入 prices[0] = 1
> 在此处卖出 prices[3] = 8
> 在此处买入 prices[4] = 4
> 在此处卖出 prices[5] = 9
> 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8

如果没有手续费的情况下，贪心策略不用关心具体什么时候买卖，只要收集每天的正利润，最后稳稳的就是最大利润了，而本题有了手续费，就要关心什么时候买卖了，因为计算所获得利润，需要考虑买卖利润可能不足以手续费的情况。

如果使用贪心策略，就是最低值买，最高值（如果算上手续费还盈利）就卖。

此时无非就是要找两个点，买入日期和卖出日期。

- 买入日期：遇到更低点就记录一下
- 卖出日期：这个就不好算了，但也没必要算出准确的卖出日期，只要当前价格大于（最低价格+手续费），就可以收回利润，至于准确的卖出日期，就是连续收获利润区间里的最后一天（并不需要计算是具体哪一天）。

所以我们在做收获利润操作的时候其实有三种情况：

- 情况一：收获利润的这一天并不是收获利润区间里的最后一天（不是真正的卖出，相当于持有股票），所以后面要继续收获利润
- 情况二：前一天是收获利润区间里的最后一天（相当于真正卖出了），今天要重新记录最小价格了
- 情况三：不做操作，保持原有状态（买入、卖出、不买不卖）

```python
class Solution: # 贪心思路
    def maxProfit(self, prices: List[int], fee: int) -> int:
        result = 0
        minPrice = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < minPrice:
                minPrice = prices[i]
            elif prices[i] >= minPrice and prices[i] <= minPrice + fee: 
                continue
            else: 
                result += prices[i] - minPrice - fee
                # 这行代码看下面的解释
                minPrice = prices[i] - fee
        return result
```

贪心中为啥是 minPrice = prices [i] - fee; 举例：[1,3,5], fee=1, 连续上涨两天卖出

1. 第 i 天：res += prices [i]- minPrice-fee，表示利润
2. 第 i+1 天：res += prices [i+1] - prices [i]，不需要手续费。相当于第 i 天并没有卖出，而是在第 i+1 天卖出。 等价于 prices [i+1] - prices [i] == prices [i+1] - ( prices [i] - fee) -fee 所以 minPrice 在 prices [i] > minPrice+fee 的时候要更新为 prices [i] - fee

也就是说，当连续获利时，手续费只算在某一次交易中即可，所以当某一天获利 prince[i] - minPrice - fee > 0 时，minPrice 也要更新为 `今天的价格 - fee`，这样下一轮如果继续获利，将不会多余减去 fee 这个费用，如果不继续获利，minPrice 也要进行变换了，对结果毫无影响。

其实感觉这里时间久了肯定想不起来，还是不太好理解！如果遇到股票买卖含手续费的问题，优先使用动态规划解题！！！一定记住！！！

### 18. 贪心算法：监控二叉树

> 给定一个二叉树，我们在树的节点上安装摄像头。
>
> 节点上的每个摄影头都可以监视**其父对象、自身及其直接子对象。**
>
> 计算监控树的所有节点所需的最小摄像头数量。
>
> ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/29/bst_cameras_01.png)
>
> ```
> 输入：[0,0,null,0,0]
> 输出：1
> 解释：如图所示，一台摄像头足以监控所有节点。
> ```

这道题目首先要想，如何放置才能让摄像头最小的呢？

从题目示例中，可以得到启发，我们发现题目实例中的摄像头都没有放在叶子节点上！

这是很重要的线索，摄像头可以覆盖上中下三层，如果把摄像头放在叶子节点上，就浪费了一层的覆盖。所以把摄像头放在叶子节点的父节点位置，才能充分利用摄像头的覆盖面积。

那有同学可能会问：为什么不从头节点开始看起，为啥要用叶子节点看？

因为头节点放不放摄像头也就省下一个摄像头，叶子节点放不放摄像头省下的摄像头数量是指数级别的。

所以我们要从下往上看：

==局部最优==：让叶子节点的父节点安摄像头，所用摄像头最少

==整体最优==：全部摄像头数量所用最少

此时，大体思路就是从低到上，先给叶子节点父节点放个摄像头，然后隔两个节点放一个摄像头，直到二叉树头节点

此时这道题目还有两个难点：

1. 二叉树的遍历：可以使用后序遍历也就是左右中的顺序，这样就可以在回溯的过程中从下到上进行推导了
2. 如何隔两个节点放一个摄像头：此时需要状态转移公式，节点有三种状态：
   1. 该节点无覆盖：0
   2. 本节点有摄像头：1
   3. 本节点有覆盖：2

在遍历数的过程中，遇到空节点，空姐点究竟是哪一种状态呢？

回归本质，为了让摄像头数量最少，我们要尽量让叶子节点的父节点安装摄像头，这样才能摄像头的数量最少，那么空节点不能是无覆盖的状态，这样叶子节点就要放摄像头了，空节点也不能是由摄像头的状态，这样叶子节点的父节点就没有必要放摄像头了，而是可以把摄像头放在叶子节点的爷爷节点上。

**所以空节点的状态只能是有覆盖，这样就可以在叶子节点的父节点放摄像头了**。

==有点过于复杂了，先放一放。==

## ==动态规划==

### 1. 动态规划的基础知识

动态规划，英文：Dynamic Programming，简称 DP，如果某一问题有很多==重叠子问题==，使用动态规划是最有效的。事实上，一个问题能否用动态规划解决，并不仅仅是因为它可以被拆解为很多小问题，而在于这些小问题会不会被重复调用。

[举个例子，有 n 个阶梯，一个人每一步只能跨一个台阶或是两个台阶，问这个人一共有多少种走法？](https://www.zhihu.com/question/39948290)

首先对这个问题进行抽象，n 个阶梯，每个阶梯都代表一个位置，就像是图论中的一个点，然后这些 n 个不同位置之间会有一些桥梁把它们连起来：

<img src="https://pic1.zhimg.com/v2-9191c5c55acfdebd6adeae22b6d7c00f_r.jpg?source=1940ef5c" alt="img" style="zoom: 50%;" />

这个图，就是该问题的抽象表达形式，那么这个问题就转化为了从 Node 0 到 Node 10 有几种不同的路可以走？

==这就是问题的本质。== 这不就是统计学中的==递推公式==嘛？？

那么如果我在计算出了从 5 到 10 的路径数，这个路径数是不是可以保存下来？

为什么要保存？因为这个信息一会儿还要被再次用到！

因为不管我是从 3 走过来的，还是 4 走过来的，走到 5 之后，存在的路径就是第一次计算出的结果，你无需重复计算。

如果暴力遍历的话，从 3 到 10 之后，你肯定会把 5-10 的可能路径都算一遍，然后从 4 到 10 的时候，你又会把 5-10 的路径算一遍，也就是重复计算了，那么既然这样，我们创建一个数组 `a[]`，专门来存放位点 x 到 10 的所有可能路径数，初始值记为 0，然后每当要计算 x 到 10 的路径数时，先检测一下该路径数的值是不是大于 0，如果大于，就说明它之前已经被计算过了，并存在于 `a[x]` 中，那我们马上可以得到一个==递推关系==：`a[x] = a[x+1] + a[x+2]` (因为 x 可以一步到 x+1，也可以一步到 x+2 )

`a[6] = a[7] + a[8];`

`a[7] = a[8] + a[9];`

我们发现，在计算 `a[6]` 和 `a[7]` 的时候，我们都用了 `a[8]`，也就是重复利用了结果！==这就是动态规划！==

---

所以动态规划中**每一个状态一定是由上一个状态推导出来的**，这一点就区分与贪心，贪心没有状态推导，而是从局部直接选最优的。

例如：有 $N$ 件物品和一个最多能背重量为 $W$ 的背包。第 i 件物品的重量是 `weight [i]`，得到的价值是 `value [i]` 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

动态规划中 `dp [j]` 是由 `dp [j-weight [i]]` 推导出来的，然后取 `max (dp [j], dp [j - weight [i]] + value [i])`。

但如果是贪心呢，每次拿物品选一个最大的或者最小的就完事了，和上一个状态没有关系。

所以贪心解决不了动态规划的问题。

==动态规划的解题步骤：==

- 确定 dp 数组以及下表的含义
- 确定递推公式
- dp 数组如何初始化
- 确定遍历顺序
- 举例推导 dp 数组

==动态规划应该如何 debug？==

**找问题的最好方式就是把 dp 数组打印出来，看看究竟是不是按照自己思路推导的！**

做动规的题目，写代码之前一定要把状态转移在 dp 数组上的具体情况模拟一遍，心中有数，确定最后推出的是想要的结果。然后再写代码，如果没通过就打印 dp 数组，看看是不是和子集预先推导的不一样，如果 dp 数组一样，那就是自己的递推公式、初始化或者遍历顺序有问题，如果和自己预先推导的不一样，那么就是代码实现细节有问题。

> 在大厂，问问题是一个专业活，要体现自己的专业性。

### 2. 动态规划和贪心算法的区别

==共同点==：两者都具有最优子结构性质

==不同点==：在动态规划算法中，每步所做出的选择往往依赖于相关子问题的解，因而只有在解出相关子问题时才能做出选择；而贪心算法，仅在当前状态下做出最好选择，即局部最优选择，然后再去解做出这个选择后产生的相应子问题

另外，动态规划算法通常以==自顶向上==的方式解各子问题，而贪心算法通常==自顶向下==的方式进行。

### 3. 斐波那契数

> **斐波那契数** （通常用 `F(n)` 表示）形成的序列称为 **斐波那契数列** 。该数列由 `0` 和 `1` 开始，后面的每一项数字都是前面两项数字的和。也就是：
>
> ```
> F(0) = 0，F(1) = 1
> F(n) = F(n - 1) + F(n - 2)，其中 n > 1
> ```
> 给定 n，计算 F(n)：
>
> ```
> 输入：n = 2
> 输出：1
> 解释：F(2) = F(1) + F(0) = 1 + 0 = 1
> ```

动态规划的经典例子：斐波那契数

==动规五部曲==：

1. 确定 dp 数组以及下标的含义：我们要用一个一维 dp 数组来保存递归的结果

> `dp[i]` 的定义为：第 i 个数的斐波那契数值是 `dp[i]`

2. 确定递推公式：本题递推公式已经直接给出，**状态转移方程 `dp [i] = dp [i - 1] + dp [i - 2]`;**
3. dp 数组如何初始化：本题也直接给出了初始方式
4. 确定遍历顺序：从递归公式 `dp [i] = dp [i - 1] + dp [i - 2]`中可以看出，`dp[i]` 是依赖 `dp [i - 1]` 和 `dp [i - 2]`，那么遍历顺序一定是从前到后遍历的
5. 举例推导 dp 数组：当 N 为 10 的时候，dp 数组应该是如下的数列：0 1 1 2 3 5 8 13 21 34 55

> 如果代码写出来，发现结果不对，就把 dp 数组打印出来看看和我们推导的数列是否一致。

```python
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        a, b, c = 0, 1, 0
        for i in range(1, n):
            c = a + b
            a, b = b, c
        return c

# 递归实现
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)
```

### 4. 爬楼梯

> 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
>
> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

1. ==确定 dp 数组以及下标的含义==：`dp[i]`：爬到第 i 层楼梯，有 dp [i] 种方法

2. ==确定递推公式==：这个例子在“动态规划的基础知识”中介绍过了（反向），递推公式就是：`dp[i] = dp[i - 1] + dp[i - 2]`

3. ==dp 数组如何初始化==，回顾一下数组的定义，爬到第 $i$ 层楼梯，有 `dp[i]` 种方法，那么 i 为 0，`dp[i]` 应该是多少呢？

> 需要注意的是：题目中说了 n 是一个正整数，题目根本就没说 n 有为 0 的情况，所以本题根本就不应该讨论 dp[0] 的初始化。

4. ==确定遍历顺序==：从递推公式可以看出遍历顺序一定是从前向后遍历的
5. ==举例推导 dp 数组==

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5n0Xo3DC2KTibuE4Lhj7jhhLoGBoEEoPVviad5RQQsrTjFyoT1lazH6CFV5XSb7kmPuibf3et9wRxkg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

> 此时应该发现，这就是斐波那契数列！

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        # dp[i] 数组存储爬到第 i 层楼梯时有 dp[i] 种方法
        if n <= 1:
            return n
        dp = [0] * (n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
```

### 5. 使用最小花费爬楼梯

> 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。
>
> 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
>
> 请你计算并返回达到楼梯顶部的最低花费。
>
> 输入：cost = [10,15,20]
> 输出：15
> 解释：你将从下标为 1 的台阶开始。

这是我自己写出来的呢！可棒了！

```python
class Solution:
    def minCostClimbingStairs(self, cost):
        # dp[i] 代表走到第 i 个台阶的最小花费
        dp = [0] * (len(cost)+1)
        dp[0] = 0
        dp[1] = 0
        for i in range(2,len(cost)+1):
            dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
        return dp[-1]
```

### 6. 不同路径

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
>
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
>
> 问总共有多少条不同的路径？
>
> ![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
>
> ```
> 输入：m = 3, n = 7
> 输出：28
> ```

#### 6.1 动态规划

是自己推导出来的，超棒！

dp 数组里存储节点到终点的总路径：初始化 + 递推公式就搞定了！

```python
import numpy as np
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 初始化二维数组
        dp = np.zeros([m, n])
        dp[m-2, n-1] = 1
        for j in range(n):
            dp[m-1, j] = 1
        for i in range(m):
            dp[i, n-1] = 1
        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i, j] = dp[i+1, j] + dp[i, j+1]
        return int(dp[0, 0])

```

#### 6.2 数论

在这个图中，可以看出一共 m，n 的话，无论怎么走，走到终点都需要 m + n - 2 （m - 1 + n - 1）步。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6swmiaLRIjBO9gUN4RfMTfFeGjAcICTXoABd6LRk0MnNPvgMVTzKwQrrR8SDTPWxBIPCvwCTzPlZQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

在这 m + n - 2 步中，一定有 m - 1 步是要向下走的，不用管什么时候向下走。那么有几种走法呢？可以转化为，给你 m + n - 2 个不同的数，随便取 m - 1 个数，有几种取法。这其实是一个组合问题了：$C_{m+n-2}^{m-1}$，求组合的时候，要防止==两个 int 相乘溢出==，所以不能把算式的分子都算出来，分母都算出来再做除法，而是应该在计算分子的时候，不断除以分母。

#### 6.3 深搜

这一题也可以使用图论里的深搜，来枚举有多少种路径。

注意题目中说机器人每次只能向下或者向右移动一步，那么其实**机器人走过的路径可以抽象为一颗二叉树，而叶子节点就是终点！**

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6swmiaLRIjBO9gUN4RfMTfFeGjAcICTXoABd6LRk0MnNPvgMVTzKwQrrR8SDTPWxBIPCvwCTzPlZQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

此时问题就可以转化为求二叉树叶子节点的个数。但是写出来会发现代码超时，我们分析一下时间复杂度，这个深搜的算法，其实是要遍历整个二叉树，这棵树的深度就是 $m+n-1$（深度按从 1 开始算），那二叉树的节点个数就是 $2^{m+n-1} -1$，可以理解深搜的算法就是遍历了整个二叉树，其实没有遍历整个二叉树，只是近似而已，所以上面深搜代码的时间复杂度为 $O(2^{m+n-1} -1)$，可以看出这指数级别的复杂度是非常大的。

### 7. 不同路径 II：障碍

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
>
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
>
> 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
>
> 网格中的障碍物和空位置分别用 1 和 0 来表示。
>
> ![img](https://gitee.com/lockegogo/markdown_photo/raw/master/202202182348368.jpeg)
>
> 输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
> 输出：2
> 解释：3x3 网格的正中间有一个障碍物。从左上角到右下角一共有 2 条不同的路径：
>
> 1. 向右 -> 向右 -> 向下 -> 向下
> 2. 向下 -> 向下 -> 向右 -> 向右
>

==主要思路就是==：初始化.

1. 如果障碍物出现在终点，直接返回 0，永远也走不到
2. 如果障碍物在最右边或者最下边，该位置以上（包括该位置）或该位置以左（包括该位置）全部设置为 0
3. 如果障碍物出现在中间，将该位置的 dp 值设置为 0

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4ACEvjQUThueyLtEmtKZh1Oiak3icibK9TgKzrkoMpQVKQn5GbVLNYpVbwYfoIsoiaSniaKBibibJHYBkkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

自己独立思考完成！很棒！虽然提交了 4 次才成功，只是一些边界条件和初始化没有好好审查。

```python
import numpy as np
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        # 初始化二维数组
        obstacleGrid = np.array(obstacleGrid)
        m,n = obstacleGrid.shape
        dp = np.zeros([m, n])
        # 如果障碍出现在终点
        if obstacleGrid[m-1, n-1] == 1:
            return 0
        for j in range(n):
            dp[m-1, j] = 1
            # 如果障碍出现在边沿，边沿上和边沿左的全部为 0
            if obstacleGrid[m-1, j] == 1:
                for k in range(j+1):
                    dp[m-1,k] = 0

        for i in range(m):
            dp[i, n-1] = 1
            if obstacleGrid[i, n-1] == 1:
                for k in range(i+1):
                    dp[k, n-1] = 0

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                # 如果障碍出现在中间
                if obstacleGrid[i,j] == 1:
                    dp[i, j] = 0
                else:
                    dp[i, j] = dp[i+1, j] + dp[i, j+1]
        return int(dp[0, 0])
```

### 8. 整数拆分

> 给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。返回你可以获得的最大乘积。
>
> 示例 1: 输入: 2
> 输出: 1
> 解释: 2 = 1 + 1, 1 × 1 = 1。

==动规五部曲==：卡在递推公式那里没有想出来

1. 确定 dp 数组以及下标的含义：
   - dp[i]：分拆数字 i，可以得到最大乘积 dp[i]  
2. ==确定递推公式==：想一想 dp[i] 最大乘积是怎么得到的？其实可以从 1 遍历 j，然后有两种渠道得到 dp[i]
   - 一种是 $j\times(i-j)$  直接相乘
   - 一种是 $j\times dp[(i-j)]$，相当于是拆分 $(i-j)$
   - 从 1 开始遍历 $j$，比较 $j\times(i-j)$ 和 $j\times dp[(i-j)]$ 取最大的：$d p[i]=\max \left(d p[i], \max \left((i-j) \times j, d p[i-j] \times j\right)\right)$ ，==这一步真的不太好想==
3. dp 数组的初始化：dp[2] = 1
3. 确定遍历顺序：从递推公式来看，先有 dp[i - j]，后有 dp[i]，所以遍历 i 一定是从前往后的顺序，i 从 3 开始

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[2] = 1
        for i in range(3, n+1):
            for j in range(i):
                dp[i] = max(dp[i],max((i-j)*j,(dp[i-j]*j)))
        return dp[n]
```

### 9. 不同的二叉搜索树：难

> 给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。
>
> ![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)
>
> ```
> 输入：n = 3
> 输出：5
> ```

给我 n 个节点，我能知道可以组成多少个不同的二叉搜索树！

可以从 1 开始尝试找规律：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5Eev9at7TiapAd6lv3wXnuJw1pZsUWUwo1SffO3hCZ3mFY6ibzg5oQZ7Fc9wOEzmQMMkzaFMUAbqPQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5Eev9at7TiapAd6lv3wXnuJhKGR6avWRCuLISuYicV1tYDHGycg5h1q588EuoO08rdrDp9Z7cica8Ig/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

来看 n 为 3 时有哪几种情况：

1. 当 1 为头节点时，其右子树有两个节点，看这两个节点的布局，是不是和 n 为 2 时两棵树的布局是一样的！
2. 当 3 为头节点时，其左子树有两个节点，看这两个节点的布局，是不是和 n 为 2 的时候两棵树的布局也是一样的啊！
3. 当 2 为头节点时，其左右子树都只有一个节点，布局是不是和 n 为 1 的时候只有一棵树的布局也是一样的啊！

发现到这里，其实我们就找到重叠子问题了，其实也就是发现可以通过 dp[1] 和 dp[2] 来推导出 dp[3] 的某种方式。

- dp [3]，就是 元素 1 为头结点搜索树的数量 + 元素 2 为头结点搜索树的数量 + 元素 3 为头结点搜索树的数量
- 元素 1 为头结点搜索树的数量 = 右子树有 2 个元素的搜索树数量 * 左子树有 0 个元素的搜索树数量
- 元素 2 为头结点搜索树的数量 = 右子树有 1 个元素的搜索树数量 * 左子树有 1 个元素的搜索树数量
- 元素 3 为头结点搜索树的数量 = 右子树有 0 个元素的搜索树数量 * 左子树有 2 个元素的搜索树数量
- 有 2 个元素的搜索树数量就是 dp [2]
- 有 1 个元素的搜索树数量就是 dp [1]
- 有 0 个元素的搜索树数量就是 dp [0]
- dp [3] = dp [2] * dp [0] + dp [1] * dp [1] + dp [0] * dp [2]

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5Eev9at7TiapAd6lv3wXnuJE5mJLfLu9gx2kItp8PBpt7vB0rNYwBnRicmw8f6gHImmFv4RwBSHuMg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

==动规五部曲==：

1. 确定 dp 数组：**dp [i] 表示 1 到 i 为节点组成的二叉搜索树的个数为 dp [i]**。
2. 确定递推公式：p [i] += dp [以 j 为头结点左子树节点数量] * dp [以 j 为头结点右子树节点数量] ，j 相当于是头结点的元素，从 1 遍历到 i 为止。所以递推公式：`dp [i] += dp [j - 1] * dp [i - j]`
3. dp 数组如何初始化：`dp[0] = 1`，从定义上讲，空节点也是一棵二叉树
4. 确定遍历顺序：从递归公式可以看出，节点数为 i 的状态是依靠 i 之前节点数的状态。那么遍历 i 里面每一个数作为头结点的状态，用 j 来遍历。
5. 举例推导 dp 数组：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5Eev9at7TiapAd6lv3wXnuJvgCSmmkVuU7xbK82cHl1X26iaD6ULLWI3eJTiaIo0yTj58YnIsnXuPxA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            # 1 到 i 分别做根节点时，不同左子树的个数乘以不同右子树的个数
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[-1]
```



---

==子序列系列==

### 10. 最长递增子序列

> 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
>
> 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
>
> ```
> 输入：nums = [10,9,2,5,3,7,101,18]
> 输出：4
> 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
> ```

最长上升子序列是动规的经典题目，这里的 dp [i] 是可以根据 dp [j] （j < i）推导出来的：

1. dp[i] 的定义：dp[i] 表示 i 之前==包括 i== 的最长上升子序列。
2. 状态转移方程：位置 i 的最长升序子序列等于 j 从 0 到 i-1 各个位置的最长升序子序列 + 1 的最大值。

$$
\text {if (nums [i] > nums [j]) dp [i] = max (dp [i], dp [j] + 1); }
$$

> 注意这里不是要比较 **dp[i] 与 dp[j] + 1**，而是要取两者的最大值。

3. dp[i] 的初始化：每一个 i，对应的 dp[i]（即最长上升子序列）起始大小至少都是是 1
4. 确定遍历顺序：dp[i] 是有 0 到 i-1 各个位置的最长升序子序列推导而来，那么遍历 i 一定是从前向后遍历

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 初始化
        dp = [1 for i in range(len(nums))]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
 
        return max(dp)
```



### 11. 最长连续递增序列

> 给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
>
> 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
>
> ```
> 输入：nums = [1,3,5,4,7]
> 输出：3
> 解释：最长连续递增序列是 [1,3,5], 长度为 3
> ```

 ```python
 class Solution:
     def findLengthOfLCIS(self, nums: List[int]) -> int:
         # 初始化
         dp = [1 for i in range(len(nums))]
         for i in range(1, len(nums)):
                 if nums[i] > nums[i-1]:
                     dp[i] = dp[i-1] + 1
  
         return max(dp)
 ```

注意这一题的递推公式和上一题的区别：`dp[i] = dp[i-1] + 1`，要体现连续，只能和前一个比大小。



### ==12. 最长重复子数组==：二维数组

> 给两个整数数组 `nums1` 和 `nums2` ，返回两个数组中**公共的** 、长度最长的子数组的长度。
>
> ```
> 输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
> 输出：3
> 解释：长度最长的公共子数组是 [3,2,1] 。
> ```

注意题目中说的子数组，其实就是连续子序列，这种问题动规最拿手，注意是==二维数组==：

1. ==确定 dp 数组以及下标的含义==：`dp [i][j]` ：以下标 i - 1 为**结尾**的 A，和以下标 j - 1 为**结尾**的 B，最长重复子数组长度为 `dp [i][j]`。注意 i 和 j 的遍历都要从 1 开始。那有同学问了，我就定义 `dp[i][j]` 为 以下标 i 为结尾的 A，和以下标 j 为结尾的 B，最长重复子数组长度。不行么？是啊，为什么不行？
2. ==确定递推公式==：当 `A[i-1]` 和 `B[j-1]` 相等的时候，`dp[i][j] = dp[i - 1][j - 1] + 1`

3. ==dp  数组如何初始化==：根据 `dp [i][j]` 的定义，`dp [i][0]` 和 `dp [0][j]` 其实都是没有意义的！但是它们都需要初始化，因为为了方便递推公式 `dp[i][j] = dp[i - 1][j - 1] + 1`，所以都初始化为 0。
4. ==确定遍历顺序==：外层 for 循环遍历 A，内层 for 循环遍历 B
5. 同时题目要求长度最长的子数组的长度，所以在遍历的时候顺便把最大值记录下来

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [[0] * (len(nums2)+1) for _ in range(len(nums1)+1)]
        result = 0
        for k in range(1,len(nums1)+1):
            for t in range(1,len(nums2)+1):
                if nums1[k-1] == nums2[t-1]:
                    dp[k][t] = dp[k-1][t-1] + 1
                result = max(result, dp[k][t])
        return result
```



### 13. 最长公共子序列

> 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列的长度。如果不存在公共子序列 ，返回 0 。一个字符串的 子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
>
> 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
> 两个字符串的公共子序列是这两个字符串所共同拥有的子序列。
>
> ```
> 输入：text1 = "abcde", text2 = "ace" 
> 输出：3  
> 解释：最长公共子序列是 "ace" ，它的长度为 3 。
> ```

本题和“最长重复子数组”的区别在于这里==不要求是连续==的了，但是要有==相对顺序==，即："ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。

1. ==确定 dp 数组以及下标的含义==：`dp[i][j]`表示长度为 [0, i - 1] 的字符串 text1 与长度为 [0, j - 1] 的字符串 text2 的最长公共子序列
2. ==确定递推公式==：
   -  text1 [i - 1] 与 text2 [j - 1] 相同：如果相同，则找到了一个公共元素，有 `dp[i][j] = dp[i - 1][j - 1] + 1`
   -  text1 [i - 1] 与 text2 [j - 1] 不相同：如果不相同，那就看 text1 [0, i - 2] 与 text2 [0, j - 1] 的最长公共子序列 和 text1 [0, i - 1] 与 text2 [0, j - 2] 的最长公共子序列，取最大的。即：`dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])`，多加了一位元素，公共子序列不会更小了，所以两者取最大值。

3. ==dp 数组如何初始化==？test1 [0, i-1] 和空串的最长公共子序列自然是 0，所以 `dp [i][0]` = 0；同理 `dp [0][j]` 也是 0。
4. ==确定遍历顺序==：从递推公式可以看到有三个方向可以推出 `dp[i][j]`，那么为了在递推的过程中，这三个方向都是记过计算的数值，所以要从前向后，从上到下来遍历这个矩阵。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv4Stw2Ra0d202BvyjCibf9goW1zdgxZickuvh2GwiaQgTc0acFndZMybydSgezj1aS8Nmb5pC9DETuPg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0]*(len(text1)+1) for _ in range(len(text2)+1)]
        for i in range(1,len(text1)+1):
            for j in range(1,len(text2)+1):
                if text1[i-1] == text2[j-1]:
                    dp[j][i] = dp[j-1][i-1] + 1
                else:
                    dp[j][i] = max(dp[j-1][i], dp[j][i-1])
        return dp[-1][-1]
```



### 14. 不相交的线

> 在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。
>
> 现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：
>
> 1. nums1[i] == nums2[j]
> 2. 且绘制的直线不与任何其他连线（非水平线）相交。
>
> 请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。
>
> 以这种方法绘制线条，并返回可以绘制的最大连线数。
>
> <img src="https://assets.leetcode.com/uploads/2019/04/26/142.png" alt="img" style="zoom: 25%;" />
>
> 输入：nums1 = [1,4,2], nums2 = [1,2,4]
> 输出：2
> 解释：可以画出两条不交叉的线，如上图所示。 
> 但无法画出第三条不相交的直线，因为从 nums1[1]=4 到 nums2[2]=4 的直线将与从 nums1[2]=2 到 nums2[1]=2 的直线相交。

本题说是求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度。

### 15. 最大子序和

> 给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
>
> **子数组**是数组中的一个连续部分。
>
> ```
> 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
> 输出：6
> 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
> ```

可以用==贪心算法==，也可以用==动态规划==。

1. ==确定 dp 数组以及下标的含义==：**dp[i] 包括==下标 i== 之前的最大连续子数组和为 dp [i]**。
2. ==确定递推公式==：dp[i] 有两个方向可以推导出来：
  - dp [i - 1] + nums [i]，即：nums [i] 加入当前连续子序列和
  - nums [i]，即：**从头开始**计算当前连续子数组和
  - 一定要取最大的，所以 dp [i] = max (dp [i - 1] + nums [i], nums [i]) （选择从自己开始还是带着前面的开始，反正都要有自己参与，不然怎么体现连续？）
3. ==dp 数组如何初始化==？dp [0] = nums [0]
4. ==确定遍历顺序==：从前向后遍历

这里注意最后的结果，不是取 dp 数组最后的值，我们要找最大的连续子序列，就应该找每一个 i 为终点的连续最大子序列，要选 dp 数组中值最大的那一个，永远扣住 dp 数组的定义！！！

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0]*len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
        return max(dp)
```



### 16. 判断子序列

> 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
>
> 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace" 是 "abcde" 的一个子序列，而 "aec" 不是）。
>
> ```
> 输入：s = "abc", t = "ahbgdc"
> 输出：true
> ```

这道题目是编辑距离的入门题目，因为从题意中我们也可以发现，只需要计算删除的情况，不用考虑增加和替换的情况。

所以掌握本题也是对后面要讲解的编辑距离的题目打下基础。

1. ==确定 dp 数组以及下标的含义==：`dp[i][j]`表示以下标 i-1 为结尾的字符串 s，和以下标 j-1 为结尾的字符串 t，相同子序列的长度，注意这里是判断 s 是否为 t 的子序列，即 t 的长度是大于等于 s 的
2. ==确定递推公式==：在确定递推公式时，首先要考虑如下两种操作：
  - `if (s[i - 1] == t[j - 1])`：t 中的一个字符在 s 中也出现了，`dp[i][j] = dp[i - 1][j - 1] + 1`
  - `if (s[i - 1] != t[j - 1])`：相当于 t 要删除元素 `t[j-1]`，继续匹配，那么 `dp[i][j]` 的数值就是看 `s[i-1]`和 `t[j-2]` 的比较结果，即 `dp[i][j] = dp[i][j - 1]`
3. ==dp 数组如何初始化==：从递推公式可以看出 `dp [i][j]` 都是依赖于 `dp [i - 1][j - 1]` 和 `dp [i][j - 1]`，所以 `dp [0][0]` 和 `dp [1][0]` 是一定要初始化的。这里大家已经可以发现，在定义 `dp [i][j]` 含义的时候为什么要**表示以下标 i-1 为结尾的字符串 s，和以下标 j-1 为结尾的字符串 t，相同子序列的长度为 `dp [i][j]`**。因为这样的定义在 dp 二维矩阵中可以留出初始化的区间，如下图所示：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv666gZL7UYkejZQiaFObJOfqQxzeEXbLiazbzQj6nszxccDpxHlHdvicyQsV4LwLdBVxXdQNVYn9xQnA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

4. 确定遍历顺序： `dp [i][j]` 都是依赖于 `dp [i - 1][j - 1]` 和 `dp [i][j - 1]`，那么遍历顺序也应该是从上到下，从左到右
5. 举例推导 dp 数组：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv666gZL7UYkejZQiaFObJOfqJJB9iaKdUmdgVQu4YNE8p1EuJJQicmfmshFjxpdibsvibnibt7dhsZgegkQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

明确 dp 数组的定义后，用一个例子就可以推导出递推公式！！！注意学会这种案例学习法！！

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0]*(len(t)+1) for _ in range(len(s)+1)]
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
        return dp[len(s)][len(t)] == len(s)
```

另一种更直观的写法：不用计数，直接继承 True or False

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[False]*(len(t)+1) for _ in range(len(s)+1)]
        for k in range(len(t)+1):
            dp[0][k] = True
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i][j-1]
        return dp[-1][-1]
```



### ==17. 不同的子序列==

> 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
>
> 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
>
> 题目数据保证答案符合 32 位带符号整数范围。
>
> 输入：s = "rabbbit", t = "rabbit"
> 输出：3

这道题目相对于编辑距离，简单了不少，因为本题相当于只有删除操作，不同考虑替换增加之类的。

1. ==确定 dp 数组以及下标的含义==：`dp [i][j]`：以 i-1 为结尾的 s 子序列中出现以 j-1 为结尾的 t 的个数为 `dp [i][j]`。
2. ==确定递推公式==：可以自己举一个例子推出来
   1. 当 s [i - 1] 与 t [j - 1] 相等时，`dp [i][j]` = `dp [i - 1][j - 1] + dp [i - 1][j]`;
   2. 当 s [i - 1] 与 t [j - 1] 不相等时，`dp [i][j]` 只有一部分组成，不用 `s [i - 1]` 来匹配，即：`dp [i][j] = dp [i - 1][j]`
3. ==dp 数组如何初始化==：从递推公式可以看出 `dp [i][0] 和 dp [0][j]` 是一定要初始化的。每次初始化的时候都要回顾以下 `dp[i][j]` 的定义，不要凭感觉初始化。`dp [i][0]` 表示：以 i-1 为结尾的 s 可以随便删除元素，出现空字符串的个数。那么 `dp [i][0]` 一定都是 1，因为也就是把以 i-1 为结尾的 s，删除所有元素，出现空字符串的个数就是 1。`dp [0][j]`：空字符串 s 可以随便删除元素，出现以 j-1 为结尾的字符串 t 的个数。那么 `dp [0][j]` 一定都是 0，s 如论如何也变成不了 t。最后就要看一个特殊位置了，即：`dp [0][0]` 应该是多少。`dp [0][0]` 应该是 1，空字符串 s，可以删除 0 个元素，变成空字符串 t。
4. ==确定遍历顺序==：`dp[i][j]` 是根据左上方和正上方推出来的，所以遍历的时候一定是从上到下，从左到右

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv4Iu9o8wN9hMecE6hr2GqgfCLp98XfsIFzutxEucnGlDIKhoZl5nIuV4INAdG4qcvCE5yGXAEibxbQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # 初始化
        dp = [[0]*(len(t)+1) for _ in range(len(s)+1)]
        for k in range(len(s)+1):
            dp[k][0] = 1
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]
```

### ==18. 两个字符串的删除操作==

> 给定两个单词 `word1` 和 `word2` ，返回使得 `word1` 和 `word2` **相同**所需的**最小步数**。
>
> **每步** 可以删除任意一个字符串中的一个字符。
>
> 输入: word1 = "sea", word2 = "eat"
> 输出: 2
> 解释: 第一步将 "sea" 变为 "ea" ，第二步将 "eat "变为 "ea"

1. ==确定 dp 数组以及下标的含义==：`dp [i][j]`：以 i-1 位结尾的字符串 word1，和以 j-1 位结尾的字符串 word2，想要达到相等，所需要删除元素的最少次数。

2. ==确定递推公式==：
   1. 当 word1 [i - 1] 与 word2 [j - 1] 相同的时候（不用有任何删除操作），`dp [i][j] = dp [i - 1][j - 1]`
   2. 当 word1 [i - 1] 与 word2 [j - 1] 不相同的时候，有三种情况：
      1. 情况一：删 word1 [i - 1]，最少操作次数为 `dp [i - 1][j] + 1`
      2. 情况二：删 word2 [j - 1]，最少操作次数为 `dp [i][j - 1] + 1`
      3. 情况三：同时删 word1 [i - 1] 和 word2 [j - 1]，操作的最少次数为 `dp [i - 1][j - 1] + 2`
      4. 那最后当然是取最小值，所以递推公式为：`dp [i][j] = min ({dp [i - 1][j - 1] + 2, dp [i - 1][j] + 1, dp [i][j - 1] + 1})`
   
3. ==dp 数组如何初始化？==，`dp [i][0] 和 dp [0][j]` 是一定要初始化的，很明显 `dp [i][0] = i`，`dp [0][j] = i`

4. ==确定遍历顺序==：每一个 dp 值都是根据左上方、正上方和正左方推导出来的

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv4Chic8ZModbs0HoodmA2xYLNxwRgLZ1hMQ5WMIibKnuXvZBsvsVsrAr1c4b0L2VE8JbRpdqkRicQGaw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]
        for k in range(len(word2)+1):
            dp[0][k] = k
        for t in range(len(word1)+1):
            dp[t][0] = t
        for i in range(1,len(word1)+1):
            for j in range(1,len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 2, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[-1][-1]
```



### ==19. 编辑距离：动态规划之终极绝杀==

> 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
>
> 你可以对一个单词进行如下三种操作：
>
> 插入一个字符
> 删除一个字符
> 替换一个字符
>
> 输入：word1 = "horse", word2 = "ros"
> 输出：3
> 解释：
> horse -> rorse (将 'h' 替换为 'r')
> rorse -> rose (删除 'r')
> rose -> ros (删除 'e')

编辑距离终于来了，这道题目如果大家没有了解动态规划的话，会感觉超级复杂。编辑距离是用动规来解决的经典题目：

1. ==确定 dp 数组以及下标的含义==：`dp [i][j] `表示以下标 i-1 为结尾的字符串 word1，和以下标 j-1 为结尾的字符串 word2，最近编辑距离为 `dp [i][j]`。
2. ==确定递推公式==：
   1. `if (word1[i - 1] == word2[j - 1])`：不操作 `dp [i][j] = dp [i - 1][j - 1]`
   2. `if (word1[i - 1] != word2[j - 1])`：
      1. ==增==：word1 增加一个元素，使得  word1 [i - 1] 与 word2 [j - 1] 相同，那么就是以下标 i-2 为结尾的 word1 与 j-1 为结尾的 word2 的最近编辑距离加上一个增加元素的操作：` dp [i][j] = dp [i - 1][j] + 1`；或者 word2 增加一个元素：`dp [i][j] = dp [i][j - 1] + 1`；这时有同学发现了，怎么都是添加元素，删除元素去哪里了？word2 添加一个一个元素，就相当于 word1 删除一个元素
      2. ==删==：见上
      3. ==换==：替换元素，word1 替换 word [i-1]，使其与 word2 [j-1] 相同，此时不用增加元素：`dp [i][j] = dp [i - 1][j - 1] + 1`
      4. 取最小值：`dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)`

3. ==dp 数组如何初始化==：he 上一题的初始化一样
4. ==确定遍历顺序==：显然是从上往下，从左往右

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7wSNhDo4eoV0MLzXyWADsQKicM2cnXGeIiaRAczYTv3iaibkwWMtySGPO1fTrxpy0CMf6KtdXic0Pfbew/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]
        for k in range(len(word1)+1):
            dp[k][0] = k
        for t in range(len(word2)+1):
            dp[0][t] = t
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 1, dp[i][j-1]+1, dp[i-1][j]+1)
        return dp[-1][-1]
```



### ==20. 回文子串==

> 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
>
> 回文字符串 是正着读和倒过来读一样的字符串。
>
> 子字符串是字符串中的由连续字符组成的一个序列。
>
> 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
>
> ```
> 输入：s = "aaa"
> 输出：6
> 解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
> ```

1. ==确定 dp 数组以及下标的含义==：布尔类型的 `dp [i][j]`：**表示区间范围 [i,j] （注意是左闭右闭）的子串是否是回文子串**，如果是 `dp [i][j]` 为 true，否则为 false。
2. ==确定递推公式==：
   1. 当 s [i] 与 s [j] 不相等，那没啥好说的了，`dp [i][j]` 一定是 false。
   2. 当 s [i] 与 s [j] 相等时，这就复杂一些了，有如下三种情况
      1. 下标 i 和 j 相同，同一个字符例如 a，当然是回文子串
      2. 下标 i 和 j 相差 1，例如 aa，也是回文子串
      3. 下标 i 和 j 相差大于 1 的时候，例如 cabac， 此时 s[i] 和 s[j] 已经相同了，我们看 i 到 j 的区间是不是回文子串就看 aba 是不是回文就可以了，abc 的区间就是 i+1 和 j-1 区间，这个区间是不是回文就看 `dp [i + 1][j - 1] ` 是否为 true
3. ==dp 数组如何初始化==：`dp [i][j]` 初始化为 false
4. ==确定遍历顺序==：从递推公式中可以看出情况三根据  `dp [i + 1][j - 1]` 是否为 true，在对 `dp [i][j]` 进行赋值 true 的。`dp [i + 1][j - 1]` 在 `dp [i][j]` 的左下角，如下图所示，所以一定要==从下到上，从左向右==遍历，这样保证`dp [i + 1][j - 1]` 是经过计算的

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6VfgGcjPG2AMuRp9WyCELice18M9BuWDngxmibFSybQmoPWQQ3K91tqa3MUe2ybKJjz1rWLLPvNCdQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />



```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        dp = [[False]*(len(s)) for _ in range(len(s))]
        result = 0
        # 注意遍历顺序：从下往上，从左往右
        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j]:
                    if j - i <= 1:
                        dp[i][j] = True
                        result += 1
                    else:
                        dp[i][j] = dp[i+1][j-1]
                        if dp[i+1][j-1] == True:
                            result += 1
        # 统计 True 的个数
        return result
```

### ==21. 最长回文子串==

> 给你一个字符串 `s`，找到 `s` 中最长的回文子串。
>
> ```
> 输入：s = "cbbd"
> 输出："bb"
> ```

> 注意此题和下一题的区别。

1. ==确定 dp 数组（dp table）以及下标的含义==：布尔类型的 `dp [i][j]`：表示区间范围 [i,j] （注意是左闭右闭）的子串是否是回文子串，如果是 `dp [i][j]` 为 true，否则为 false。
2. ==确定递推公式==：
   1. 当 s [i] 与 s [j] 不相等，那没啥好说的了，`dp [i][j]` 一定是 false。注意这是子串，如果是子序列，不要求连续，就不是这么简单的情况了。
   2. 当 s [i] 与 s [j] 相等时，这就复杂一些了，有如下三种情况
      1. 情况一：下标 i 与 j 相同，同一个字符例如 a，当然是回文子串
      2. 情况二：下标 i 与 j 相差为 1，例如 aa，也是文子串
      3. 情况三：下标 i 与 j 相差大于 1 的时候，例如 cabac，此时 s [i] 与 s [j] 已经相同了，我们看 i 到 j 区间是不是回文子串就看 aba 是不是回文就可以了，那么 aba 的区间就是 i+1 与 j-1 区间，这个区间是不是回文就看 `dp [i + 1][j - 1]` 是否为 true。
   3. 在得到 [i, j]  区间是否是回文字串的时候，直接保存最长回文子串的左边界和右边界
3. ==dp 数组初始化==：全部初始化为 false
4. ==确定遍历顺序==：从下往上，从左到右遍历

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        dp = [[False] * len(s) for _ in range(len(s))]
        maxlenth = 0
        left = 0
        right = 0
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[j] == s[i]:
                    if j - i <= 1 or dp[i + 1][j - 1]:
                        dp[i][j] = True
                if dp[i][j] and j - i + 1 > maxlenth:
                    maxlenth = j - i + 1
                    left = i
                    right = j
        return s[left:right + 1]
```



### 22. 最长回文子序列

> 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
>
> 子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
>
> ```
> 输入：s = "bbbab"
> 输出：4
> 解释：一个可能的最长回文子序列为 "bbbb" 。
> ```

注意回文子串和回文子序列的区别：回文子串需要连续，但是回文子序列可不是连续的！回文子串和回文子序列都是动规经典题目。

> 对于连续的字串而言，如果前一状态不为回文串，会连带着后面的状态也不为回文串，对于不连续的子序列而言，举个例子，字符串 "abbc"，对于其中的字串 "bb" 而言，明显最长的回文串就是 "bb", 而对于 "abbc" 而言，最长的字符串依然为 "bb"，换而言之，这个不连续可以让我们继承它的字串的最长回文序列，我们定义 `dp [i][j]` 为区间 (i,j) 中的最大回文子序列。
>

1. ==确定 dp 数组以及下标的含义==：**`dp [i][j]`：字符串 s 在 [i, j] 范围内最长的回文子序列的长度为 `dp [i][j]`**。
2. ==确定递推公式==：在判断回文子串的题目中，关键逻辑就是看 s [i] 与 s [j] 是否相同。如果 s [i] 与 s [j] 相同，那么 `dp [i][j] = dp [i + 1][j - 1] + 2`：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv63n0Qv1XVrBKFF7SgoSzazwuchq0xJGHkDeUja4ic4vG86BZlOR51ZHTqia8qArPTlj6MM06BOQcqQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

3. 如果 s [i] 与 s [j] 不相同，说明 s [i] 和 s [j] 的同时加入 并不能增加 [i,j] 区间回文子串的长度，那么分别加入 s [i]、s [j] 看看哪一个可以组成最长的回文子序列。加入 s [j] 的回文子序列长度为 `dp [i + 1][j]`；加入 s [i] 的回文子序列长度为 `dp [i][j - 1]`；那么 `dp [i][j]` 一定是取最大的，即：`dp [i][j] = max (dp [i + 1][j], dp [i][j - 1])`; ==区别==

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv63n0Qv1XVrBKFF7SgoSzaz6QbmXpItSCmY14buaBkWRMM3TuTXb1TEf51NwcrusqILDTq1ictQ2wA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

4. ==dp 数组如何初始化==？首先要考虑当 i 和 j 相同的情况，从递推公式：`dp [i][j] = dp [i + 1][j - 1] + 2;` 可以看出 递推公式是计算不到 i 和 j 相同时候的情况。所以需要手动初始化一下，当 i 与 j 相同，那么 `dp [i][j]` 一定是等于 1 的，即：一个字符的回文子序列长度就是 1。其他情况 `dp [i][j]` 初始为 0 就行，这样递推公式：`dp [i][j] = max (dp [i + 1][j], dp [i][j - 1])`; 中 `dp [i][j]` 才不会被初始值覆盖。

5. ==确定遍历顺序==：从矩阵的角度来说，遍历 i 的时候一定要==从下到上==遍历，这样才能保证，下一行的数据是经过计算的。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv63n0Qv1XVrBKFF7SgoSzazjRLMSkvMejGIicicRufKgVNyickGcZYUhibaSGQm0wPqicSLjZEellSKoWQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # dp[i][j] 代表在 i，j 范围内最长回文子序列的长度
        dp = [[0]*len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        for i in range(len(s)-1, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i+1][j])
        return dp[0][-1]
```

这题和最长回文子串的区别就是，它不要求求出的值是连续的，这就意味着当左右两端的字符串无法提高最长回文串的长度时，我们可以继承它的子串的最大回文串长度，而最长回文串不行。

---

## ==背包问题==

### 23. 背包问题基础知识

#### 23.1  01 背包

有 N 件物品和一个最多能被重量为 W 的背包。第 i 件物品的重量是 weight [i]，得到的价值是 value [i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

如果使用暴力的解法，每一件物品其实只有两个状态，取或者不取，所以可以使用回溯法搜索出所有的情况，那么时间复杂度就是 $O (2^n)$，这里的 n 代表物品数量，暴力的解法是指数级别的时间复杂度，所以才需要动态规划的解法来进行优化！

背包最大重量为 4。

物品为：

|        | 重量 | 价值 |
| :----- | :--- | :--- |
| 物品 0 | 1    | 15   |
| 物品 1 | 3    | 20   |
| 物品 2 | 4    | 30   |

问背包能背的物品最大价值是多少？

1. ==确定 dp 数组以及下标的含义==：对于背包问题，有一种写法，是使用二维数组，即 `dp[i][j]` 表示从下标为 [0-i] 的物品里任意取，然后放进容量为 $j$ 的背包，==价值总和最大是多少==。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6xPu8BiaJQNCasvLUeXpIGBE6ZiaHNkJ3wwMicRH5K7Cps0giaa5ynhQnutL7RtJB9mwXZ50erL1jFZA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />

> 要时刻记着这个 dp 数组的含义，下面的一些步骤都围绕着 dp 数组的含义进行。

2. ==确定递推公式==：有两个方向推出来 `dp[i][j]`:

   1. 由 `dp [i - 1][j]` 推出，即背包容量为 j，里面不放物品 i 的最大价值，此时 `dp [i][j]` 就是 `dp [i - 1][j]`
   2. 由 `dp [i - 1][j - weight [i]]` 推出，`dp [i - 1][j - weight [i]]` 为背包容量为 j - weight [i] 的时候不放物品 i 的最大价值，那么 `dp [i - 1][j - weight [i]]` + value [i] （物品 i 的价值），就是背包放物品 i 得到的最大价值
   3. 所以递推公式为：`dp [i][j] = max (dp [i - 1][j], dp [i - 1][j - weight [i]] + value [i]);`

3. ==dp 数组的初始化==：关于初始化，一定要和 dp 数组的定义吻合，否则到递推公式的时候就会越来越乱

   1. 首先从 `dp [i][j]` 的定义触发，如果背包容量 j 为 0 的话，即 `dp [i][0]`，无论是选取哪些物品，背包价值总和一定为 0。
   2. `dp [0][j]`，即：i 为 0，存放编号 0 的物品的时候，各个容量的背包所能存放的最大价值。j 从最大开始，倒叙遍历：`dp[0][j] = dp[0][j-weight[0]] + value[0]`，值得注意的是，这个初始化为什么是倒叙的遍历？正序遍历不行吗？正序遍历不行，`dp [0][j]` 表示容量为 j 的背包存放物品 0 时候的最大价值，物品 0 的价值就是 15，因为题目中说了 每个物品只有一个！所以 `dp [0][j]` 如果不是初始值的话，就应该都是物品 0 的价值，也就是 15。但是一旦正序遍历，那么物品 0 就会被重复加入多次，例如 `dp [0][1]` 是 15，到了 `dp [0][2] = dp [0][2 - 1] + 15`; 也就是 `dp [0][2] = 30` 了，那么就是物品 0 被重复放入了。

   **所以一定要倒叙遍历，保证物品 0 只被放入一次！这一点对 01 背包很重要，后面在讲解滚动数组的时候，还会用到倒叙遍历来保证物品使用一次！**

   > 用不了这么复杂，看下面的代码。

   <img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6xPu8BiaJQNCasvLUeXpIGB65vfGj3952CP20FM0bz71P62cZ12eZsb9XItQmf4FVjBkggK1a92Pg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 80%;" />

   那么现在`dp [0][j]` 和 `dp [i][0]` 都已经初始化了，那么其他下标应该初始化多少呢？`dp [i][j]` 在推导的时候一定是取价值最大的数，如果题目给的价值都是正整数那么非 0 下标都初始化为 0 就可以了，因为 0 就是最小的了，不会影响取最大价值的结果。

   如果题目给的价值有负数，那么非 0 下标就要初始化为负无穷了。例如：一个物品的价值是 - 2，但对应的位置依然初始化为 0，那么取最大值的时候，就会取 0 而不是 - 2 了，所以要初始化为负无穷。这样才能让 dp 数组在递归公式的过程中取最大的价值，而不是被初始值覆盖了。

   <img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6xPu8BiaJQNCasvLUeXpIGB0iaqqakFPjDCnkjkveib2xyic249qmyUV1KVueAHcqJPACsRGNrGkID1A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

4. ==确定遍历顺序==：我们有两个遍历的维度：物品和背包重量

那么问题来了：先遍历物品还是先遍历背包重量呢？其实都可以，但是先遍历物品更好理解。要理解递归的本质和递推的方向。`dp [i][j] = max (dp [i - 1][j], dp [i - 1][j - weight [i]] + value [i])`; 递归公式中可以看出 `dp [i][j]` 是靠 `dp [i-1][j]` 和 `dp [i - 1][j - weight [i]]` 推导出来的。

`dp [i-1][j]` 和 `dp [i - 1][j - weight [i]]` 都在 `dp [i][j]` 的左上角方向（包括正左和正上两个方向），那么先遍历物品，再遍历背包的过程如图所示：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6xPu8BiaJQNCasvLUeXpIGBo4eKE01ZCYzZOBjjmOLqFoukOxQk5mMicy3flylXGtKcaFgsCOJwMmg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />



5. ==举例推导 dp 数组==：来看一下对应的 dp 数组的数值：最终结果就是 `dp[2][4]`

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6xPu8BiaJQNCasvLUeXpIGBpdreelI7fkpCsGM4EuEIqW4YmQmz0tEkRaAiaOt348c4Kj2kfUw8JWQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

做动态规划的题目，最好的过程就是自己在纸上举一个例子把对应的 dp 数组的数值推导一下，然后再动手写代码

```python
def test_2_wei_bag_problem1(bag_size, weight, value) -> int: 
	rows, cols = len(weight), bag_size + 1
	dp = [[0 for _ in range(cols)] for _ in range(rows)]
    
	# 初始化dp数组. 
	for i in range(rows): 
		dp[i][0] = 0
	first_item_weight, first_item_value = weight[0], value[0]
	for j in range(1, cols): 	
		if first_item_weight <= j: 
			dp[0][j] = first_item_value

	# 更新dp数组: 先遍历物品, 再遍历背包. 
	for i in range(1, len(weight)): 
		cur_weight, cur_val = weight[i], value[i]
        # 这里为什么没有倒序？？？
		for j in range(1, cols): 
			if cur_weight > j: # 说明背包装不下当前物品. 
				dp[i][j] = dp[i - 1][j] # 所以不装当前物品. 
			else: 
				# 定义dp数组: dp[i][j] 前i个物品里，放进容量为j的背包，价值总和最大是多少。
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cur_weight]+ cur_val)

	print(dp)


if __name__ == "__main__": 
	bag_size = 4
	weight = [1, 3, 4]
	value = [15, 20, 30]
	test_2_wei_bag_problem1(bag_size, weight, value)
```

#### 23.2 滚动数组

背包问题状态可以压缩的，在使用二维数组的时候，递推公式 `dp [i][j] = max (dp [i - 1][j], dp [i - 1][j - weight [i]] + value [i])`，我们发现如果把 `dp[i-1]`那一层拷贝到 `dp[i]` 上，表达式完全可以是：`dp [i][j] = max (dp [i][j], dp [i][j - weight [i]] + value [i])`，于其把 dp [i - 1] 这一层拷贝到 dp [i] 上，不如只用一个一维数组了，只用 dp [j]（一维数组，也可以理解是一个滚动数组）。这就是滚动数组的由来，需要满足的条件是上一层可以重复利用，直接拷贝到当前层。

注意，`dp[i][j]`里面 i 是物品，j 是背包容量。**`dp [i][j]` 表示从下标为 [0-i] 的物品里任意取，放进容量为 j 的背包，价值总和最大是多少**。

动规五部曲分析如下：

1. ==确定 dp 数组的定义==：在一维 dp 数组中，dp [j] 表示：容量为 j 的背包，所背的物品价值可以最大为 dp [j]。
2. ==一维 dp 数组的递推公式==：dp [j] 为 容量为 j 的背包所背的最大价值，那么如何推导 dp [j] 呢？dp [j] 可以通过 dp [j - weight [j]] 推导出来，dp [j - weight [i]] 表示容量为 j - weight [i] 的背包所背的最大价值。dp [j - weight [i]] + value [i] 表示 容量为 j - 物品 i 重量 的背包 加上 物品 i 的价值。（也就是容量为 j 的背包，放入物品 i 了之后的价值即：dp [j]）。此时 dp [j] 有两个选择，一个是取自己 dp [j]，一个是取 dp [j - weight [i]] + value [i]，指定是取最大的，毕竟是求最大价值，所以递推公式为：`dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);` 需要遍历 i 求解吗？是的。
3. ==一维 dp 数组如何初始化==：dp [j] 表示容量为 j 的背包，所背的物品价值可以最大为 dp [j]，那么 dp [0] 就应该是 0，因为背包容量为 0 所背的物品的最大价值就是 0。那么 dp 数组除了下标 0 的位置，初始为 0，其他下标应该初始化多少呢？看一下递推公式，dp 数组在推导的时候一定是取价值最大的数，如果题目给的价值都是正整数那么非 0 下标都初始化为 0 就可以了，如果题目给的价值有负数，那么非 0 下标就要初始化为负无穷。**这样才能让 dp 数组在递归公式的过程中取的最大的价值，而不是被初始值覆盖了**。
4. ==一维 dp 数组的遍历顺序==：这里和二维 dp 的遍历顺序不太一样，二维 dp 遍历的时候背包容量从小到大，而一维 dp 遍历的时候，背包从大到小。为什么呢？倒序遍历是为了保证物品 i 只被放入一次，举一个例子：

> 物品 0 的重量 weight [0] = 1，价值 value [0] = 15
>
> 如果正序遍历
>
> dp[1] = dp[1 - weight[0]] + value[0] = 15
>
> dp[2] = dp[2 - weight[0]] + value[0] = 30
>
> 此时 dp [2] 就已经是 30 了，意味着物品 0，被放入了两次，所以不能正序遍历。
>
> 为什么倒叙遍历，就可以保证物品只放入一次呢？
>
> 倒叙就是先算 dp [2]
>
> dp [2] = dp [2 - weight [0]] + value [0] = 15  （dp 数组已经都初始化为 0）
>
> dp[1] = dp[1 - weight[0]] + value[0] = 15
>
> 所以从后往前循环，每次取得状态不会和之前取得状态重合，这样每种物品就只取一次了。

再来看两个被嵌套 for 循环的顺序，代码中是先遍历==物品==嵌套遍历==背包容量==，那可不可以先遍历背包容量嵌套遍历物品呢？

不可以！！！

因为一维 dp 的写法，==背包容量一定要倒序遍历==，如果遍历背包容量放在上一层，那么每个 dp [j] 就只会放入一个物品，即：背包里只放入了一个物品。**所以一维 dp 数组的背包在遍历顺序上和二维其实是有很大差异的！**，这一点大家一定要注意。

5. ==举例推导 dp 数组==：一维 dp，用物品 0，物品 1，物品 2 来遍历背包，最终得到结果如下：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv65uJ2vbfGG2z2z5I4x6SbkvGWMRfLXxtNqtKXH1wpsqj21bKevc5xzHrRJiaXXZo2C6ojcKMYW05Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4
    # 初始化: 全为0
    dp = [0] * (bag_weight + 1)

    # 先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        # 注意，背包容量一定要倒序遍历
        for j in range(bag_weight, weight[i] - 1, -1):
            # 递归公式
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)

test_1_wei_bag_problem()
```

可以看到一维 dp 的 01 背包，要比二维的简洁很多，初始化和遍历顺序相对简单了，而且空间复杂度还降了一个数量级。所以下面我们都用滚动数组。

### 24. 分割等和子集

> 给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
>
> ```
> 输入：nums = [1,5,11,5]
> 输出：true
> 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
> ```

这道题目初步看和下面两道题目是一样的，可以用回溯法：

1. 698 划分为 k 个相等的子集
2. 473 火柴拼正方形

这道题目要找是否可以将这个数组分割承两个子集，使得两个子集的元素和相等，那么只要找到集合里能够出现 sum / 2 的子集总和，就算是可以分割成两个相同元素和子集了。

回溯法可以做但是超时了，直接上 ==01 背包==。

> 注意背包问题有多种背包方式，常见的有 01 背包、完全背包、多重背包和分组背包、混合背包等等。
>
> 要注意题目描述中商品是不是可以重复放入。
>
> 如果一个商品可以重复多次放入是完全背包，而智能放入一次是 01 背包，写法是不一样的。

只有确定了如下四点，才能把 01 背包问题套到本题上来：

1. 背包的体积为 sum / 2
2. 背包要放入的商品（集合里的元素）重量为元素的数值，价值也为元素的数值
3. 背包如果正好装满，说明找到了总和为 sum / 2 的子集
4. 背包中每一个元素是不可重复放入

动规五部曲如下：

1. ==确定 dp 数组以及下标的含义==：dp[j] 表示背包容量是 j，最大可以凑成的子集总和为 dp[j]
2. ==确定递推公式==：`dp [j] = max (dp [j], dp [j - nums [i]] + nums [i])`（通俗的理解就是一点都没浪费），本题，相当于背包里放入数值，那么物品 i 的重量是 nums [i]，其价值也是 nums [i]。
3. ==dp 数组如何初始化==：见上
4. ==确定遍历顺序==：如果使用一维 dp 数组，物品遍历的 for 循环放在外层，遍历背包的 for 循环放在内层，且内层 for 循环倒叙遍历！
5. ==举例推导 dp数组==：dp[i] 的数值一定是小于等于 i 的，如果 dp[i] == i （容量和价值相等）说明，集合中的子集总和正好可以凑成总和 i，理解这一点很重要。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv55hHx245gD9gnqSsWX3SX6jr1DBVylMZcfGaZhvZeoqZmKwg1u0ra9oOAicNTqUl26p3s3bwoSW5g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

最后 dp [11] == 11，说明可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums)%2 == 0:
            dp = [0]*int((sum(nums)/2)+1)
        else:
            return False
        for i in range(len(nums)):
            for j in range(len(dp)-1, nums[i]-1, -1):
                dp[j] = max(dp[j], dp[j-nums[i]] + nums[i])
        if dp[-1] == sum(nums)/2:
            return True
        else:
            return False
```

### 25. 最后一块石头的重量

> 有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。
>
> 每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
>
> 如果 x == y，那么两块石头都会被完全粉碎；
> 如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
> 最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
>
> 输入：stones = [2,7,4,1,8,1]
> 输出：1
> 解释：
> 组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
> 组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
> 组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
> 组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。

本题其实就是尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小，这样就化解成 01 背包问题了。

本题物品的重量为 store [i]，物品的价值也为 store [i]。

接下来进行动规五步曲：

1. ==确定 dp 数组以及下标的含义==：**dp [j] 表示容量（这里说容量更形象，其实就是重量）为 j 的背包，最多可以背 dp [j] 这么重的石头**。
2. ==确定递推公式==：01 背包的递推公式为：`dp [j] = max (dp [j], dp [j - weight [i]] + value [i])`
3. ==dp 数组如何初始化==：既然 dp[j] 中的 j 表示容量，那么最大容量是多少呢？就是所有石头的重量和。而我们要求的 target 其实只是最大重量的一半。因为重量都不会是负数，所以 dp [j] 都初始化为 0 就可以了，这样在递归公式 dp [j] = max (dp [j], dp [j - stones [i]] + stones [i]) 中 dp [j] 才不会初始值所覆盖。
4. ==确定遍历顺序==：如果使用一维 dp 数组，物品遍历的 for 循环放在外层，遍历背包的 for 循环放在内层，且内层 for 循环倒序遍历。
5. ==举例推导 dp 数组==：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6fHG15ZFniccFsMniaKiatmJiaIzrcSjzbV1JmpuPmJp2oeia8xbKQ07M8ibMwfsPZs44jzCIlqORZwSnw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

最后 dp [target] 里是容量为 target 的背包所能背的最大重量。那么分成两堆石头，一堆石头的总重量是 dp [target]（target 是容量，dp[target] 是重量），另一堆就是 sum - dp [target]。

**在计算 target 的时候，target = sum / 2 因为是向下取整，所以 sum - dp [target] 一定是大于等于 dp [target] 的**。

那么相撞之后剩下的最小石头重量就是 (sum - dp [target]) - dp [target]。

其实本题和上一题“分割等和子集”几乎是一样的，“分割等和子集”相当于是求背包是否正好装满，而本题是求背包最多能装多少。

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        target = sum(stones) // 2
        dp = [0] * (target+1)
        for i in range(len(stones)):
            for j in range(len(dp)-1, stones[i]-1, -1):
                dp[j] = max(dp[j], dp[j-stones[i]] + stones[i])
        
        return sum(stones) - 2*dp[-1]
```

### 26. 目标和

> 给你一个整数数组 nums 和一个整数 target 。
>
> 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
>
> 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
> 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
>
> 输入：nums = [1,1,1,1,1], target = 3
> 输出：5
> 解释：一共有 5 种方法让最终目标和为 3 。
> -1 + 1 + 1 + 1 + 1 = 3
> +1 - 1 + 1 + 1 + 1 = 3
> +1 + 1 - 1 + 1 + 1 = 3
> +1 + 1 + 1 - 1 + 1 = 3
> +1 + 1 + 1 + 1 - 1 = 3

如何转化为 01 背包问题？

假设加法的总和为 x，那么减法对应的总和就是 sum - x，所以我们要求的是 x - (sum - x) = S，x = (S + sum) / 2，此时问题就转化为，装满容量为 x 的背包，有几种方法。

注意，这次和之前遇到的背包问题不一样了，之前都是要求容量为 j 的背包，最多能装多少。本题则是装满有几种方法，其实这就是一个组合问题了：

1. ==确定 dp 数组以及下标的含义==：dp [j] 表示：填满 j（包括 j）这么大容积的包，有 dp [i] 种方法
2. ==确定递推公式==：有哪些来源可以推出 dp [j] 呢？不考虑 nums[i] 的情况下，填满容量为 j - nums[i] 的背包，有 dp[j - nums[i]] 种方法，那么只要搞到 nums[i] 的话，凑成 dp[j] 就有 dp[j - nums[i]] 种方法。那么只需要把这些方法累加起来就可以了：`dp [i] += dp [j - nums [j]]`，所有求组合类问题的公式，都是类似这种：

3. ==dp数组如何初始化==：从递推公式可以看出，在初始化的时候 dp[0] 一定要初始化为 1，因为它是一个起源，如果它为 0，那么后i面的一切结果都将为 0。dp [j] 其他下标对应的数值应该初始化为 0，从递归公式也可以看出，dp [j] 要保证是 0 的初始值，才能正确的由 dp [j - nums [i]] 推导出来。
> dp [0] = 1，理论上也很好解释，装满容量为 0 的背包，有 1 种方法，就是装 0 件物品。

4. ==确定遍历顺序==：对于 01 背包问题一维 dp 的遍历，nums 放在外循环，target 在内循环，且内循环倒序。

5. ==举例推导 dp 数组==：nums: [1, 1, 1, 1, 1], S: 3； bagSize = (S + sum) / 2 =  (3 + 5) / 2 = 4

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7DDz0s7BN6ribc3sAu6STTm2R8OPhFyfuiaVH357In1uicoe62ozh7CUAxfQuSmYiawbIgiaeT9aj0o3g/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sumValue = sum(nums)
        # 注意这里对边界条件的判断
        if target > sumValue or (sumValue + target) % 2 == 1: return 0
        bagSize = (sumValue + target) // 2
        dp = [0] * (bagSize + 1)
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(bagSize, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[bagSize]
```

### 27. 一和零

> 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
>
> 请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
>
> 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
>
> 输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
> 输出：4
> 解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
> 其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。

m 和 n 相当于是一个==两个维度==的背包，本题依然是 01 背包问题！

但问题是这个背包有两个维度，一个是 m，一个是 n，而不同长度的字符串就是不同大小的待装物品。

1. ==确定 dp 数组以及下标的含义==：最多有 i 个 0 和 j 个 1 的 strs 的最大子集的大小为 `dp [i][j]`。
2. ==确定递推公式==：`dp [i][j]` 可以由前一个 strs 里的字符串推导出来，strs 里的字符串有 zeroNum 个 0，oneNum 个 1。`dp [i][j]` 就可以是 `dp [i - zeroNum][j - oneNum] + 1`。然后我们在遍历的过程中，取 `dp[i][j]` 的最大值。所以递推公式：`dp [i][j] = max (dp [i][j], dp [i - zeroNum][j - oneNum] + 1)`

> 此时大家可以回想一下 01 背包的递推公式：dp [j] = max (dp [j], dp [j - weight [i]] + value [i]);
>
> 对比一下就会发现，字符串的 zeroNum 和 oneNum 相当于物品的重量（weight [i]），字符串本身的个数相当于物品的价值（value [i]）。
>
> **这就是一个典型的 01 背包！** 只不过物品的重量有了两个维度而已。

3. ==dp 数组如何初始化==：01 背包的 dp 数组初始化为 0 就可以。
4. ==确定遍历顺序==：外层 for 循环遍历物品，内层 for 循环遍历背包容量且从后向前遍历！
5. ==举例推导 dp 数组==：以输入：["10","0001","111001","1","0"]，m = 3，n = 3 为例

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7OqPLIszA2uicyHvsHy75y7Mu3rjAqRHhmgUNib71LN1IUu7fYMBFBXqhE6dbiaKdS70QXwUNWsgbXQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(len(strs)):
            zeroNum = strs[i].count('0')
            oneNum = strs[i].count('1')
            for j in range(m, zeroNum-1, -1):
                for k in range(n, oneNum-1, -1):
                    dp[j][k] = max(dp[j][k], dp [j - zeroNum][k - oneNum] + 1)
        return dp[-1][-1]
```

### 28. 完全背包

> 有 N 件物品和一个最多能背重量为 W 的背包。第 i 件物品的重量是 weight [i]，得到的价值是 value [i] 。**每件物品都有无限个（也就是可以放入背包多次）**，求解将哪些物品装入背包里物品价值总和最大。

**完全背包和 01 背包问题唯一不同的地方就是，每种物品有无限件**。

举例如下：

背包最大重量为 4。

物品为：

|        | 重量 | 价值 |
| :----- | :--- | :--- |
| 物品 0 | 1    | 15   |
| 物品 1 | 3    | 20   |
| 物品 2 | 4    | 30   |

**每件商品都有无限个！**

问背包能背的物品最大价值是多少？

01 背包和完全背包==唯一不同==就是体现在遍历顺序上，所以本文就不去做动规五部曲了，我们直接针对遍历顺序进行分析！

首先回顾下 01 背包的核心代码：

```python
# 先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        # 注意，01 背包容量一定要倒序遍历
        for j in range(bag_weight, weight[i] - 1, -1):
            # 递归公式
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
```

我们知道 01 背包内嵌的循环是==从大到小==遍历，为了保证每个物品仅被添加一次。

而完全背包的物品是可以添加多次的，所以一定要==从小到大==去遍历！

```python
# 先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        # 注意，完全背包容量一定要从小到大遍历
        for j in range(weight[i], bag_weight):
            # 递归公式
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
```

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv59GL0I0Z7TCzHPia3z72fNS37PGt7WY6zeIfmX8mS0mxR6icLnRTic1Wiba12Bs4Za7af0l0aQyfwARA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

**其实还有一个很重要的问题，为什么遍历物品在外层循环，遍历背包容量在内层循环？**

这个问题很多题解关于这里都是轻描淡写就略过了，大家都默认 遍历物品在外层，遍历背包容量在内层，好像本应该如此一样，那么为什么呢？

01 背包中二维 dp 数组的两个 for 遍历的先后循序是可以颠倒了，一位 dp 数组的两个 for 循环先后循序一定是先遍历物品，再遍历背包容量。

**在完全背包中，对于一维 dp 数组来说，其实两个 for 循环嵌套顺序同样无所谓！**

因为 dp [j] 是根据 下标 j 之前所对应的 dp [j] 计算出来的。只要保证下标 j 之前的 dp [j] 都是经过计算的就可以了。

遍历物品在外层循环，遍历背包容量在内层循环，状态如图：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv59GL0I0Z7TCzHPia3z72fNShyVzib18fsUBFZviatw4HShSx6S3ADrvACicvwCiaFyQ9yWZxYpybXOA4Q/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

遍历背包容量在外层循环，遍历物品在内层循环，状态如图：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv59GL0I0Z7TCzHPia3z72fNSAKKibtmeYm6xfeRy99FAGvWLtcz3AiaLMvl3wibDBeSjyPluvq475CR2A/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

看了这两个图，大家就会理解，完全背包中，两个 for 循环的先后循序，都不影响计算 dp [j] 所需要的值（这个值就是下标 j 之前所对应的 dp [j]）。

```python
# 先遍历物品，再遍历背包
def test_complete_pack1():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for i in range(len(weight)):
        for j in range(weight[i], bag_weight + 1):
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])

# 先遍历背包，再遍历物品
def test_complete_pack2():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for j in range(bag_weight + 1):
        for i in range(len(weight)):
            if j >= weight[i]: dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])


if __name__ == '__main__':
    test_complete_pack1()
    test_complete_pack2()
```

### 29. 零钱兑换 II

> 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
>
> 请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
>
> 假设每一种面额的硬币有无限个。 
>
> 题目数据保证结果符合 32 位带符号整数。
>
> 输入：amount = 5, coins = [1, 2, 5]
> 输出：4
> 解释：有四种方式可以凑成总金额：
> 5=5
> 5=2+2+1
> 5=2+1+1+1
> 5=1+1+1+1+1

本题和纯完全背包不一样，纯完全背包是能否凑成总金额，而本题是要求凑成总金额的个数！

注意题目描述中是凑成总金额的硬币组合数，为什么强调是组合数呢？

**组合不强调元素之间的顺序，排列强调元素之间的顺序**。

那我为什么要介绍这些呢，因为这和下文讲解遍历顺序息息相关！

回归本题，动规五步曲来分析如下：

1. ==确定 dp 数组以及下标的含义==：dp [j]：凑成总金额 j 的货币组合数为 dp [j]

2. ==确定递推公式==：dp [j] （考虑 coins [i] 的组合总和） 就是所有的 dp [j - coins [i]]（不考虑 coins [i]）相加。所以递推公式：dp [j] += dp [j - coins [i]];

**这个递推公式大家应该不陌生了，我在讲解 01 背包题目的时候在这篇[动态规划：目标和！](https://mp.weixin.qq.com/s?__biz=MzUxNjY5NTYxNA==&mid=2247486709&idx=1&sn=75f1f43d96dbd1c5c3e281b8963e3c50&scene=21#wechat_redirect)中就讲解了，求装满背包有几种方法，一般公式都是：dp [j] += dp [j - nums [i]];**

3. ==dp 数组如何初始化==：首先 dp [0] 一定要为 1，dp [0] = 1 是 递归公式的基础。从 dp [i] 的含义上来讲就是，凑成总金额 0 的货币组合数为 1。下标非 0 的 dp [j] 初始化为 0，这样累计加 dp [j - coins [i]] 的时候才不会影响真正的 dp [j]

4. ==确定遍历顺序==：本题中我们是外层 for 循环遍历物品（钱币），内层 for 遍历背包（金钱总额），还是外层 for 遍历背包（金钱总额），内层 for 循环遍历物品（钱币）呢？

完全背包的两个 for 循环的先后顺序都是可以的。**但本题就不行了！**

因为纯完全背包求得是能否凑成总和，和凑成总和的元素有没有顺序没关系，即：有顺序也行，没有顺序也行！

而本题要求凑成总和的组合数，元素之间要求没有顺序。

所以纯完全背包是能凑成总结就行，不用管怎么凑的。

==本题是求凑出来的方案个数，且每个方案个数是为组合数。==

那么本题，两个 for 循环的先后顺序可就有说法了。

我们先来看 外层 for 循环遍历物品（钱币），内层 for 遍历背包（金钱总额）的情况。

假设：coins [0] = 1，coins [1] = 5。

那么就是先把 1 加入计算，然后再把 5 加入计算，得到的方法数量只有 {1, 5} 这种情况。而不会出现 {5, 1} 的情况。

**所以这种遍历顺序中 dp [j] 里计算的是组合数！**

如果把两个 for 交换顺序，代码如下：

背包容量的每一个值，都是经过 1 和 5 的计算，包含了 {1, 5} 和 {5, 1} 两种情况。

**此时 dp [j] 里算出来的就是排列数！**

可能这里很多同学还不是很理解，**建议动手把这两种方案的 dp 数组数值变化打印出来，对比看一看！（实践出真知）**

5. ==举例推导 dp 数组==：输入: amount = 5, coins = [1, 2, 5] ，dp 状态图如下：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6iaBB3vOiaZKuo7SKo1IMpEm2RoEMHVPWs6RmU1F3X78T9x56c7nwtcpUAoWpjyhviaTYCIibNV4St5w/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

最后红色框 dp [amount] 为最终结果。

在求装满背包有几种方案的时候，认清遍历顺序是非常关键的。

**如果求组合数就是外层 for 循环遍历物品，内层 for 遍历背包**。

**如果求排列数就是外层 for 遍历背包，内层 for 循环遍历物品**。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount + 1)
        dp[0] = 1
        # 遍历物品
        for i in range(len(coins)):
            # 遍历背包
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[amount]
```



## 打家劫舍系列

### 1. 开始打家劫舍

> 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
>
> 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
>
> 输入：[1,2,3,1]
> 输出：4
> 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。偷窃到的最高金额 = 1 + 3 = 4 。

1. ==确定 dp 数组（dp table）以及下标的含义==：**dp [i]：考虑下标 i（包括 i）以内的房屋，最多可以偷窃的金额为 dp [i]**。
2. ==确定递推公式==：决定 dp[i] 的因素就是第 i 个房间偷还是不偷，
   1. 如果偷第 i 房间，那么 dp [i] = dp [i - 2] + nums [i] ，即：第 i-1 房一定是不考虑的，找出 下标 i-2（包括 i-2）以内的房屋，最多可以偷窃的金额为 dp [i-2] 加上第 i 房间偷到的钱；
   2. 如果不偷第 i 房间，那么 dp [i] = dp [i - 1]，即考虑 i-1 房，（**注意这里是考虑，并不是一定要偷 i-1 房，这是很多同学容易混淆的点**）
   3. 然后 dp [i] 取最大值，即 dp [i] = max (dp [i - 2] + nums [i], dp [i - 1]);
3. ==dp 数组如何初始化==：从递推公式可以看出递推公式的基础就是 dp [0] 和 dp [1]，从 dp [i] 的定义上来讲，dp [0] 一定是 nums [0]，dp [1] 就是 nums [0] 和 nums [1] 的最大值即：`dp [1] = max (nums [0], nums [1])`;
4. ==确定遍历顺序==：当然是从前向后
5. ==举例推导==：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6cUozFpI3RjoK7eicTicB9Gc5GvpVvIt7oCXXTPutiaClny6sibK7c3Y7QUuicX4HFkFicqONxffhvq09Q/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[-1]
```

### 2. 打家劫舍 II：环形

> 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
>
> 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
>
> ```
> 输入：nums = [2,3,2]
> 输出：3
> 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
> ```

这和上一题唯一的区别就是成环了，对于一个数组，承欢的话主要有如下三种情况：

1. 考虑不包含首尾元素：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7DroYFRD2ITKia9UcE5BhI72IQicu7UHl2F3sZkv1udHRib8FnBhdUkgSbvvMNY0j4t7XfwIibsCzdLg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

2. 考虑包含首元素，不包含尾元素

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv7DroYFRD2ITKia9UcE5BhI7AMf6iauVEyDz8weCJVpyhvbibss1OOtguh39U1zrK4npnUicaQBBQDL9Q/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

3. 考虑包含尾元素，不包含首元素

**而情况二 和 情况三 都包含了情况一了，所以只考虑情况二和情况三就可以了**。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
      if (n := len(nums)) == 0:
        return 0
      if n == 1:
        return nums[0]
      result1 = self.robRange(nums, 0, n - 2)
      result2 = self.robRange(nums, 1, n - 1)
      return max(result1 , result2)

    def robRange(self, nums: List[int], start: int, end: int) -> int:
      if end == start: return nums[start]
      dp = [0] * len(nums)
      dp[start] = nums[start]
      dp[start + 1] = max(nums[start], nums[start + 1])
      for i in range(start + 2, end + 1):
        dp[i] = max(dp[i -2] + nums[i], dp[i - 1])
      return dp[end]
```

### 3. 打家劫舍 III

> 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。
>
> 除了 root 之外，每栋房子有且只有一个 “父 “房子与之相连。一番侦察之后，聪明的小偷意识到 “这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
>
> 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
>
> ![img](https://assets.leetcode.com/uploads/2021/03/10/rob1-tree.jpg)
>
> ```
> 输入: root = [3,2,3,null,3,null,1]
> 输出: 7 
> 解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
> ```

如果对树的遍历不够熟悉的话，那本题就有难度了。对于树的话，首先就要想到遍历方式，前中后序（深度优先搜索）还是层序遍历（广度优先搜索）。

本题一定要后序遍历，因为通过递归函数的返回值来做下一步计算，关键同样是要讨论当前节点抢还是不抢。如果抢了当前节点，两个孩子就不能动，如果没抢当前节点，就可以考虑抢左右孩子。

暴力搜索方式对节点偷与不偷得到的最大金钱都没有做记录，而是需要实时计算。而动态规划其实就是使用状态转移容器来记录状态的变化，这里可以使用一个长度为 2 的数组，记录当前节点偷与不偷所得到的最大金钱。

1. ==确定递归函数的参数和返回值==：这里我们要求一个节点 偷与不偷的两个状态所得到的金钱，那么返回值就是一个长度为 2 的数组。下标为 0 记录不偷该节点所得到的的最大金钱，下标为 1 记录偷该节点所得到的的最大金钱。所以本题 dp 数组就是一个长度为 2 的数组！

2. ==确定终止条件==：在遍历的过程中，如果遇到空间点的话，很明显无论偷还是不偷都是 0

3. ==确定遍历顺序==：首先明确的是使用后序遍历。因为通过递归函数的返回值来做下一步计算。

   - 通过递归左节点，得到左节点偷与不偷的金钱。

   - 通过递归右节点，得到右节点偷与不偷的金钱。

4. ==确定单层递归的逻辑==：
   1. 如果是偷当前节点，那么左右孩子就不能偷，`val1 = cur->val + left [0] + right [0]`;  （**如果对下标含义不理解就在回顾一下 dp 数组的含义**）
   2. 如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的 (可偷可不偷)，所以：`val2 = max (left [0], left [1]) + max (right [0], right [1])`;
   3. 最后当前节点的状态就是 {val2, val1}; 即：{不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}
5. ==举例推导 dp 数组==：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv5ffdcnYY7AVw7yO3xjiaKdlsr7ptlZAC32FO52dkCnw9heR51tGwmO2ibassrfelfMxVtriaNlq6Kfg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

最后头节点就是取下标 0 和下标 1 的最大值就是偷得的最大金钱

这是树形 DP 的入门题目，树形 DP 就是树上进行递推公式的推导。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        result = self.rob_tree(root)
        return max(result[0], result[1])
    
    def rob_tree(self, node):
        if node is None:
            return (0, 0) # (偷当前节点金额，不偷当前节点金额)
        left = self.rob_tree(node.left)
        right = self.rob_tree(node.right)
        val1 = node.val + left[1] + right[1] # 偷当前节点，不能偷子节点
        val2 = max(left[0], left[1]) + max(right[0], right[1]) # 不偷当前节点，可偷可不偷子节点
        return (val1, val2)
```



## ==股票系列==

### 1. 买卖股票的最佳时机：单次交易

> 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
>
> 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
>
> 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
>
> 输入：[7,1,5,3,6,4]
> 输出：5
> 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票

#### 1.1 贪心算法

因为股票就买卖一次，那么贪心的想法很自然就是取最左最小值，取最右最大值，那么得到的差值就是最大利润。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        low = float("inf")
        result = 0
        for i in range(len(prices)):
            low = min(low, prices[i]) # 取最左最小价格
            result = max(result, prices[i] - low) #直接取最大区间利润
        return result
```

#### 1.2 动态规划

1. ==确定 dp 数组（dp table）以及下标的含义==：`dp [i][0]` 表示第 i 天持有股票所得现金；`dp [i][1]` 表示第 i 天不持有股票所得现金 

2. ==确定递推公式==：如果第 i 天持有股票即 `dp [i][0]`， 那么可以由两个状态推出来：

   1. 第 i-1 天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：`dp [i - 1][0]`
   2. 第 i 天买入股票，所得现金就是买入今天的股票后所得现金即：-prices [i]
   3. 那么 `dp [i][0]` 应该选所得现金最大的，所以 `dp [i][0] = max (dp [i - 1][0], -prices [i])`;

3. ==确定递推公式==：如果第 i 天不持有股票即 `dp [i][1]`， 也可以由两个状态推出来：

   1. 第 i-1 天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：`dp [i - 1][1]`
   2. 第 i 天卖出股票，所得现金就是按照今天股票佳价格卖出后所得现金即：`prices [i] + dp [i - 1][0]`
   3. 同样 `dp [i][1]` 取最大的，`dp [i][1] = max (dp [i - 1][1], prices [i] + dp [i - 1][0])`;

4. ==dp 数组如何初始化==：由递推公式 `dp [i][0] = max (dp [i - 1][0], -prices [i]); 和 dp [i][1] = max (dp [i - 1][1], prices [i] + dp [i - 1][0])`可以看出，其基础都是要从 `dp [0][0]` 和 `dp [0][1]` 推导出来。那么 `dp [0][0]` 表示第 0 天持有股票，此时的持有股票就一定是买入股票了，因为不可能有前一天推出来，所以 `dp [0][0] -= prices [0]`;`dp [0][1]` 表示第 0 天不持有股票，不持有股票那么现金就是 0，所以 `dp [0][1] = 0`;

5. ==确定遍历顺序==：从前向后遍历

6. ==举例推导 dp 数组==：以示例 1，输入：[7,1,5,3,6,4] 为例，dp 数组状态如下：

   ![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6JicibMFqMQf8bIDrDgroNXLhs1aICMCN15USA0W1cwAmADfMYtbRNAbqZ3SFGtK2ibIG3HlDuBlstw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

`dp [5][1]` 就是最终结果。为什么不是 `dp [5][0]` 呢？**因为本题中不持有股票状态所得金钱一定比持有股票状态得到的多！**

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        if len == 0:
            return 0
        dp = [[0] * 2 for _ in range(length)]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0])
        return dp[-1][1]
```

### 2. 买卖股票的最佳时机 II：多次交易

> 给定一个数组 prices ，其中 prices[i] 表示股票第 i 天的价格。
>
> 在每一天，你可能会决定购买和 / 或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以购买它，然后在 同一天 出售。
> 返回 你能获得的 最大 利润 。
>
> 输入: prices = [7,1,5,3,6,4]
> 输出: 7
> 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

贪心解法之前已经讲过，接下来讲动态规划。

本题和上一题的唯一区别就是本题股票可以买卖多次，注意只有一只股票，所以再次购买前要出售掉之前的股票。

这里重申一下 dp 数组的含义：

- `dp [i][0]` 表示第 i 天持有股票所得现金。
- `dp [i][1]` 表示第 i 天不持有股票所得最多现金

如果第 i 天持有股票即 `dp [i][0]`， 那么可以由两个状态推出来：

- 第 i-1 天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：`dp [i - 1][0]`
- 第 i 天买入股票，所得现金就是昨天不持有股票的所得现金减去今天的股票价格，即：`dp [i - 1][1] - prices [i]`

==注意这里和 121. 买卖股票的最佳时机 唯一不同的地方==，就是推导 `dp [i][0]` 的时候，第 i 天买入股票的情况。上一题股票全程只能买卖一次，所以如果买入股票，那么第 i 天持有股票 `dp[i][0]` 一定是 `0 - price[i]`，而 本题，因为一只股票可以买卖多次，所以当第 i 天买入股票的时候，所持有的现金可能有之前买卖过的利润，即：`dp [i - 1][1] - prices [i]`

```python
class Solution:
    """动态规划"""
    def maxProfit(self, prices: List[int]) -> int:
        # (第 i天持有股票所得现金，未持有所得现金)
        dp = [[0]*2 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][-1]
```

### 3. 买卖股票的最佳时机 III：最多交易两次

> 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
>
> 设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。
>
> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
>
> 输入：prices = [3,3,5,0,0,3,1,4]
> 输出：6
> 解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。

这题又比上一题难了不少，关键在于至多买卖两次，这意味着可以买卖一次，可以买卖两次，也可以不买卖。

1. ==确定 dp 数组以及下标的含义==：一天一共就有五个状态， `dp [i][j]` 中 i 表示第 i 天，j 为 [0 - 4] 五个状态，`dp [i][j]` 表示第 i 天状态 j 所剩最大现金。
   - 0. 没有操作
   - 1. 第一次买入
   - 2. 第一次卖出
   - 3. 第二次买入
   - 4. 第二次卖出
2. ==确定递推公式==：`dp [i][1]`，**表示的是第 i 天，==买入股票的状态 (注意理解这句话)==，并不是说一定要第 i 天买入股票**，达到 `dp [i][1]` 状态，有两个具体操作：
   1. 操作一：第 i 天买入股票了，那么 `dp [i][1] = dp [i-1][0] - prices [i]`
   2. 操作二：第 i 天没有操作，而是沿用前一天买入的状态，即：`dp [i][1] = dp [i - 1][1]`
   3. 选择最大的：`dp [i][1] = max (dp [i-1][0] - prices [i], dp [i - 1][1])`
   4. 同理 `dp [i][2] = max (dp [i - 1][1] + prices [i], dp [i - 1][2])`
   5. `dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])`
   6. `dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])`


3. ==dp 数组如何初始化==：
   1. 第 0 天没有操作，即：`dp [0][0] = 0`
   2. 第 0 天做第一次买入的操作，`dp [0][1] = -prices [0]`
   3. 第 0 天做第一次卖出的操作，`dp [0][2] = 0`
   4. 第 0 天第二次买入操作，`dp [0][3] = -prices [0]`
   5. 第 0 天第二次卖出操作，`dp [0][4] = 0`
4. ==确定遍历顺序==：从前向后遍历
5. ==举例推导 dp 数组==：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv7ib5pFuKnCZNic9A3ZT4Jaf2eGL21CWPC25CjgCeGkRiaqXpXvwlk4iaia1eC8xXXN0cqGW7jqK4TYuXg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

大家可以看到红色框为最后两次卖出的状态。现在最大的时候一定是卖出的状态，而两次卖出的状态现金最大一定是最后一次卖出。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0]*5 for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[0][2] = 0
        dp[0][3] = -prices[0]
        dp[0][4] = 0
        for i in range(1, len(prices)):
            dp[i][1] = max(dp [i-1][0] - prices [i], dp [i - 1][1])
            dp[i][2] = max(dp [i-1][1] + prices [i], dp [i - 1][2])
            dp[i][3] = max(dp [i-1][2] - prices [i], dp [i - 1][3])
            dp[i][4] = max(dp [i-1][3] + prices [i], dp [i - 1][4])
        return dp[-1][-1]
```

### 4. 买卖股票的最佳时机 IV：最多 k 笔交易

> 给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
>
> 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
>
> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
>
> 输入：k = 2, prices = [2,4,1]
> 输出：2
> 解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。

其实就是上一题的推广：

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices: return 0 
        dp = [[0]*(2*k+1) for _ in range(len(prices))]
        for t in range(2*k+1):
            if t % 2 == 0:
                dp[0][t] = 0
            else:
                dp[0][t] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(1, 2*k+1):
                dp[i][j] = max(dp[i-1][j-1] + prices[i]*((-1)**j), dp[i-1][j])
        return dp[-1][-1]
```

### 5. 最佳买卖股票时机含冷冻期

> 给定一个整数数组 prices，其中第  prices[i] 表示第 i 天的股票价格 。
>
> 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
>
> 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
>
> ```
> 输入: prices = [1,2,3,0,2]
> 输出: 3 
> 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
> ```

1. ==确定 dp 数组以及下标的含义==：`dp [i][j]`，第 i 天状态为 j，所剩的最多现金为 `dp [i][j]`。有冷冻期后有四个状态：
   1. 状态一：买入股票状态（今天买入股票，或者是之前就买入了股票然后没有操作）
   2. 卖出股票状态，这里就有两种卖出股票状态
   	- 状态二：两天前就卖出了股票，度过了冷冻期，一直没操作，今天保持卖出股票状态
   	- 状态三：今天卖出了股票
   3. 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！ 

> 注意这里的每一个状态，例如状态一，是买入股票状态并不是说今天已经就买入股票，而是说保存买入股票的状态即：可能是前几天买入的，之后一直没操作，所以保持买入股票的状态。

2. ==确定递推公式==：

   1. 达到买入股票状态（状态一）即：`dp [i][0]`，有两个具体操作：

      1. 操作一：前一天就是持有股票状态（状态一），`dp [i][0] = dp [i - 1][0]`
      2. 操作二：今天买入了，有两种情况

   2. - 前一天是冷冻期（状态四）
      - 前一天是保持卖出股票状态（状态二）
      - 所以操作二取最大值，即：`max (dp [i - 1][3], dp [i - 1][1]) - prices [i]`，那么 `dp [i][0] = max (dp [i - 1][0], max (dp [i - 1][3], dp [i - 1][1]) - prices [i]`;

   2. 达到保持卖出股票状态（状态二）即：`dp [i][1]`，有两个具体操作：
      1. 操作一：前一天就是状态二
      2. 操作二：前一天是冷冻期（状态四）
      3. 操作最大值：`dp[i][1] = max(dp[i - 1][1], dp[i - 1][3])`;

   3. 达到今天就卖出股票状态（状态三），即：`dp [i][2]` ，只有一个操作：
      1. 操作一：昨天一定是买入股票状态（状态一），今天卖出，即：`dp [i][2] = dp [i - 1][0] + prices [i]`;

   4. 达到冷冻期状态（状态四），即：`dp [i][3]`，只有一个操作：
      1. 操作一：昨天卖出了股票（状态三）`p[i][3] = dp[i - 1][2]`;

3. ==dp 数组如何初始化==：
   1. 如果是持有股票状态（状态一）那么：`dp [0][0] = -prices [0]`，买入股票所省现金为负数。
   2. 保持卖出股票状态（状态二），第 0 天没有卖出 `dp [0][1]` 初始化为 0 就行，
   3. 今天卖出了股票（状态三），同样 `dp [0][2]` 初始化为 0，因为最少收益就是 0，绝不会是负数。
   4. 同理 `dp [0][3]` 也初始为 0。

4. ==确定遍历顺序==：从前向后遍历
5. ==举例推导 dp 数组==：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv44NQQialrWHkQ24kTw0L2LJo37B8GnMwnlPTJWYMrVq5BrW1vVDFCN15UwJRWBnRP8j99SzlOZNXA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

注意！最后的结果是取状态二、状态三和状态四的最大值！！

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        # (买入，两天前就已经卖出，当天卖出，今天为冷冻期)
        dp = [[0]*4 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        for k in range(1,4):
            dp[0][k] = 0
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], max(dp[i-1][1] - prices[i],dp[i-1][3] - prices[i]))
            dp[i][1] = max(dp[i-1][1], dp[i-1][3])
            dp[i][2] = dp[i-1][0] + prices[i]
            dp[i][3] = dp[i-1][2]
        return max(dp[-1][-3], dp[-1][-2], dp[-1][-1])
```

### 6. 买卖股票的最佳时机含手续费

> 给定一个整数数组 prices，其中 prices[i] 表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。
>
> 你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
>
> 返回获得利润的最大值。
>
> 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
>
> 输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
> 输出：8
> 解释：能够达到的最大利润:  
> 在此处买入 prices[0] = 1
> 在此处卖出 prices[3] = 8
> 在此处买入 prices[4] = 4
> 在此处卖出 prices[5] = 9
> 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8

贪心算法之前已经讲过了，接下俩介绍动态规划！

这题和 买卖股票的最佳时机 II 相比，==只需要在计算卖出操作时减去手续费就可以了==，先自己试着写一写。

==递推公式==：

- `dp [i][0] = max (dp [i - 1][0], dp [i - 1][1] - prices [i])`

- `dp [i][1] = max (dp [i - 1][1], dp [i - 1][0] + prices [i] - fee)`

```python
class Solution: 
    # 动态规划
    def maxProfit(self, prices: List[int], fee: int) -> int:
        if len(prices) == 0: return 0
        # (第 i 天持有股票时的现金，不持有时的现金)
        dp = [[0]*2 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1,len(prices)):
            # 统一为买的时候不花手续费，卖的时候花手续费
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
        return dp[-1][-1]
```



## 单调栈

### 1. 每日温度

> 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指在第 i 天之后，才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 0 来代替。
>
> ```
> 输入: temperatures = [73,74,75,71,69,72,76,73]
> 输出: [1,1,4,2,1,1,0,0]
> ```

==单调栈==：就是栈里的元素保持升序或者降序。

==什么时候用单调栈？==

通常是一维数组，要寻找任意一个元素的右边或者左边第一个比自己大或者小的元素的位置，此时我们就要想到可以用单调栈。

那么单调栈的原理是什么呢？为什么时间复杂度是 O (n) 就可以找到每一个元素的右边第一个比它大的元素位置呢？

单调栈的本质是==空间换时间==，因为在遍历的过程中需要用一个栈来记录右边第一个比当前元素大的元素，优点是只需要遍历一次。

在使用单调栈的时候首先要明确如下几点：

1. 单调栈里存放的元素是什么？

单调栈里只需要存放元素的下标 i 就可以了，如果需要使用对应的元素，直接 T [i] 就可以获取。

2. 单调栈里元素是递增呢？还是递减呢？

注意顺序是从栈头到栈底的顺序，这里我们要使用递增循序，因为只有递增的时候，加入一个元素 i，才知道栈顶元素在数组中右面第一个比栈顶元素大的元素是 i。

使用单调栈主要有三个判断条件。

- 当前遍历的元素 T [i] 小于栈顶元素 T [st.top ()] 的情况
- 当前遍历的元素 T [i] 等于栈顶元素 T [st.top ()] 的情况
- 当前遍历的元素 T [i] 大于栈顶元素 T [st.top ()] 的情况

接下来我们用 temperatures = [73, 74, 75, 71, 71, 72, 76, 73] 为例来逐步分析，输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

首先先将第一个遍历元素加入单调栈：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SKlcyic9Bo7Ujn9WqNF3bexQ07WE9jc8eB5o4nCK5qicBUbgdPzSn5Evw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

加入 T [1] = 74，因为 T [1] > T [0]（当前遍历的元素 T [i] 大于栈顶元素 T [st.top ()] 的情况），而我们要保持一个递增单调栈（从栈头到栈底），所以将 T [0] 弹出，T [1] 加入，此时 result 数组可以记录了，result [0] = 1，即 T [0] 右面第一个比 T [0] 大的元素是 T [1]。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SxO77FcGyShgBkLR2W5v3MK9lZC93seEibzQYyy1KLENJGFNjtO0D8mA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [2]，同理，T [1] 弹出：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9Stiaibf7jhwsMsvUngziaTIQHiaYzfkoNyvm3ZMAphtM7tau6Jwz43Hhb5w/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [3]，T [3] < T [2] （当前遍历的元素 T [i] 小于栈顶元素 T [st.top ()] 的情况），加 T [3] 加入单调栈。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SfE6ia6Uz8kgmSPYiaCiba9A3qbGbEDU1V3iclRQZfPticutLWxLFicGt77Yg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [4]，T [4] == T [3] （当前遍历的元素 T [i] 等于栈顶元素 T [st.top ()] 的情况），此时依然要加入栈，不用计算距离，因为我们要求的是右面第一个大于本元素的位置，而不是大于等于！

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SIIbPiapGicuvKdztdgo1ZzoZsTwAKyCGfibb8RUiabJbW88bKFNsxPNM9Q/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [5]，T [5] > T [4] （当前遍历的元素 T [i] 大于栈顶元素 T [st.top ()] 的情况），将 T [4] 弹出，同时计算距离，更新 result

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SUhXz1cWqvOZrAEsID9F7eviadJYsVsypfnze0SH5jDQz5InazASFISQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

T [4] 弹出之后， T [5] > T [3] （当前遍历的元素 T [i] 大于栈顶元素 T [st.top ()] 的情况），将 T [3] 继续弹出，同时计算距离，更新 result

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SziauZYaJg7pejqu9ibc8JDzUfZKfZSgl5VdZ8Isa7l1wj3mb5sic0XyWw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

直到发现 T [5] 小于 T [st.top ()]，终止弹出，将 T [5] 加入单调栈

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SoZnMyOIC7UtOeN5b3fLBq1dnZmbuIZhEARicrAQO2gWpia5ZQ3x9WoIQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [6]，同理，需要将栈里的 T [5]，T [2] 弹出

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9Sbu3tRo8q1zoibFYyK6fxyxdVflGs6olKvJ7ficic8vXUU26zopwjK0lzQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

同理，继续弹出

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9SAPOlK0UTuOUYqYhuSiaKcDcwq6c4UKx5ftejQt97plFT5N3ADxtSsqg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

此时栈里只剩下了 T [6]

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9ShDJsHpyv8tnQnj1eORbUeo2zlKMXVUNw6BZzvXSfmYfFJtZDtRfKjw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

加入 T [7]， T [7] < T [6] 直接入栈，这就是最后的情况，result 数组也更新完了。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6xQewfHtnef7uqp244fc9S77mdeB7TUBV2JVQianO4cMbfFu78Ua52MDQJXdwkFdCYXoibVzLgsEWg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

此时有同学可能就疑惑了，那 result [6] , result [7] 怎么没更新啊，元素也一直在栈里。

其实定义 result 数组的时候，就应该直接初始化为 0，如果 result 没有更新，说明这个元素右面没有更大的了，也就是为 0。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0]*len(temperatures)
        stack = [0]
        for i in range(1,len(temperatures)):
            # 情况一和情况二
            if temperatures[i]<=temperatures[stack[-1]]:
                stack.append(i)
            # 情况三
            else:
                while len(stack) != 0 and temperatures[i]>temperatures[stack[-1]]:
                    answer[stack[-1]]=i-stack[-1]
                    stack.pop()
                stack.append(i)
            
        return answer
```

### 2. 下一个更大元素

> nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
>
> 给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中 nums1 是 nums2 的子集。
>
> 对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。
>
> 返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。
>
> ```
> 输入：nums1 = [4,1,2], nums2 = [1,3,4,2].
> 输出：[-1,3,-1]
> 解释：nums1 中每个值的下一个更大元素如下所述：
> - 4 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
> - 1 ，用加粗斜体标识，nums2 = [1,3,4,2]。下一个更大元素是 3 。
> - 2 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
> ```

> 注意看清题目是要找更大元素的下标还是更大元素。这也是本题和上一题的区别之一。

在每日温度中是求每个元素下一个比当前元素大的元素的位置，本题则是说 nums1 是 nums2 的子集，找 nums1 中的元素在 nums2 中下一个比当前元素大的元素，那么就要定义一个和 nums1 一样大小的数组 result 来存放结果。

==如何初始化呢？== 题目说如果不存在对应位置就输出 -1，所以 result 数组如果某位置没有被赋值，那么就应该是 -1，所以初始化为 -1。

在遍历 nums2 的过程中，我们要判断 nums2 [i] 是否在 nums1 中出现过，因为最后是要根据 nums1 元素的下标来更新 result 数组。

注意题目中说是两个没有重复元素的数组，没有重复元素，我们就可以用 map 来做映射了，根据数值快速找到下标，还可以判断 nums2[i] 是否在 nums1 中出现过。

使用单调栈，首先要想单调栈是从大到小还是从小到大。

栈头到栈底的顺序，要==从小到大==，也就是保持栈里的元素为递增顺序。只要保持递增，才能找到右边第一个比自己大的元素。

接下来就要分析如下三种情况，一定要分析清楚。

1. 情况一：当前遍历的元素 T [i] 小于栈顶元素 T [st.top ()] 的情况

此时满足递增栈（栈头到栈底的顺序），所以直接入栈。

2. 情况二：当前遍历的元素 T [i] 等于栈顶元素 T [st.top ()] 的情况

如果相等的话，依然直接入栈，因为我们要求的是右边第一个比自己大的元素，而不是大于等于！

3. 情况三：当前遍历的元素 T [i] 大于栈顶元素 T [st.top ()] 的情况

此时如果入栈就不满足递增栈了，这也是找到右边第一个比自己大的元素的时候。

判断栈顶元素是否在 nums1 里出现过，（注意栈里的元素是 nums2 的元素），如果出现过，开始记录结果。

记录结果这块逻辑有一点小绕，要清楚，此时栈顶元素在 nums2 中右面第一个大的元素是 nums2 [i] 即当前遍历元素。

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result = [-1]*len(nums1)
        # stack 中存储的是下标
        stack = [0]
        for i in range(1,len(nums2)):
            # 情况一情况二
            if nums2[i]<=nums2[stack[-1]]:
                stack.append(i)
            # 情况三
            else:
                while len(stack)!=0 and nums2[i]>nums2[stack[-1]]:
                    if nums2[stack[-1]] in nums1:
                        # 根据元素值求下标
                        index = nums1.index(nums2[stack[-1]])
                        result[index]=nums2[i]
                    stack.pop()
                # 把比它小的都弹出去之后，再把它入栈
                stack.append(i)
        return result
```

> 想要加入栈的元素都在栈内元素的右边（因为是按下标顺序入栈的），如果想要入栈的元素更大，势必会引发 pop 操作。

### 3. 下一个更大元素 II

> 给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
>
> 数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。
>
> 输入: nums = [1,2,1]
> 输出: [2,-1,2]
> 解释: 第一个 1 的下一个更大的数是 2；
> 数字 2 找不到下一个更大的数； 
> 第二个 1 的下一个最大的数需要循环搜索，结果也是 2。

==如何处理循环数组？==

一个直观的想法就是直接把两个数组拼接在一起，然后使用单调栈求下一个最大值，最后再把结果集即 result 数组 resize 到原数组大小就可以了。

这种写法很直观，但是做了很多无用操作，例如修改了 nums 数组，而且最后还要把 result 数组 resize 回去。

resize 倒是不费时间，是 O (1) 的操作，但扩充 nums 数组相当于多了一个 O (n) 的操作。

其实也可以不扩充 nums，而是在遍历的过程中模拟走了两边 nums。

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        dp = [-1] * len(nums)
        stack = []
        for i in range(len(nums)*2):
            while(len(stack) != 0 and nums[i%len(nums)] > nums[stack[-1]]):
                    dp[stack[-1]] = nums[i%len(nums)]
                    stack.pop()
            stack.append(i%len(nums))
        return dp
```

### 4. 接雨水

> 给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
>
> ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)
>
> 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
> 输出：6
> 解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

#### 41 双指针法

按照行来计算如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJ1MHkicWYPDupDpFwnbCibRWgRcQh9qCX1NpkwOCmiaezF4VicscCSFTbLg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

按照列来计算如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJaLJzmnM8y0WBvOnwtdfeQhMBCUbA3Aic7lHlzyoiah6rsGk9avsRAcibA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

首先，**如果按照列来计算的话，宽度一定是 1 了，我们再把每一列的雨水的高度求出来就可以了。**

可以看出每一列雨水的高度，取决于，该列 ==左侧最高的柱子== 和 ==右侧最高的柱子== 中 ==最矮的那个柱子== 的高度。

这句话有点绕，例如求列 4 的雨水高度，如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJn8RZOXneJpCAz3o3GT0IMia1GQowW3YGvmsGNcOfoPKibicZbVG9RYlYA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

列 4 左侧最高的柱子是列 3，高度为 2（以下用 lHeight 表示）。

列 4 右侧最高的柱子是列 7，高度为 3（以下用 rHeight 表示）。

列 4 柱子的高度为 1（以下用 height 表示）

那么列 4 的雨水高度为 列 3 和列 7 的高度最小值减列 4 高度，即：min (lHeight, rHeight) - height。

列 4 的雨水高度求出来了，宽度为 1，相乘就是列 4 的雨水体积了。

此时求出了列 4 的雨水体积。

一样的方法，只要从头遍历一遍所有的列，然后求出每一列雨水的体积，相加之后就是总雨水的体积了。

首先从头遍历所有的列，并且**要注意第一个柱子和最后一个柱子不接雨水**，在 for 循环中求左右两边最高柱子，最后计算该列雨水高度。

因为每次遍历列的时候，还要向两边寻找最高的列，所以时间复杂度是 $O(n^2)$ ，空间复杂度是 $O(1)$

```python
class Solution:
    # 按列计算，还是很好理解的
    def trap(self, height: List[int]) -> int:
        res = 0
        for i in range(len(height)):
            if i == 0 or i == len(height)-1: continue
            lHight = height[i-1]
            rHight = height[i+1]
            for j in range(i-1):
                if height[j] > lHight:
                    lHight = height[j]
            for k in range(i+2,len(height)):
                if height[k] > rHight:
                    rHight = height[k]
            res1 = min(lHight,rHight) - height[i]        
            if res1 > 0:
                res += res1
        return res
```

#### 4.2 动态规划

在上一节的双指针解法中，我们可以看到只要记录左边柱子的最高高度和右边柱子的最高高度，就可以计算当前位置的雨水面积，这就是通过列来计算：

当前列雨水面积 = min (左边柱子的最高高度，记录右边柱子的最高高度) - 当前柱子高度。

为了得到两边的最高高度，使用了双指针来遍历，每到一个柱子都向两边遍历一遍，这其实是有重复计算的，我们把==每个位置的左边最高高度记录在一个数组上，右边最高高度记录在一个数组上==，这样就避免了重复计算，这就是动态规划。

当前位置，左边的最高高度是前一个位置的左边最高高度和本高度的最大值。

即从左向右遍历：maxLeft [i] = max (height [i], maxLeft [i - 1]);

从右向左遍历：maxRight [i] = max (height [i], maxRight [i + 1]);

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        leftheight, rightheight = [0]*len(height), [0]*len(height)

        # 初始化
        leftheight[0]=height[0]
        # 递推公式：从前向后遍历
        for i in range(1,len(height)):
            leftheight[i]=max(leftheight[i-1],height[i])
            
        # 初始化
        rightheight[-1]=height[-1]
        # 递推公式：从后向前遍历
        for i in range(len(height)-2,-1,-1):
            rightheight[i]=max(rightheight[i+1],height[i])

        result = 0
        for i in range(0,len(height)):
            summ = min(leftheight[i],rightheight[i])-height[i]
            result += summ
        return result
```

#### 4.3 单调栈

单调栈就是保持栈内元素有序，我们需要自己维持顺序，没有现成的容器可以使用。

##### 4.3.1 准备工作

1. 单调栈是按照行的方向来计算雨水：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJe6WU2ENdjTFAeicrxUiarIdRbialv2GmibAfKXwuteqIIWpGwYiahsrSSfg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

2. 使用单调栈内元素的顺序

从大到小还是从小到大呢？从栈头（元素从栈头弹出）到栈底的顺序应该是==从小到大==的顺序。

因为一旦发现添加的柱子高度大于栈头元素了，此时就出现凹槽了，栈头元素就是凹槽底部的柱子，栈头第二个元素就是凹槽左边的柱子，而添加的元素就是凹槽右边的柱子。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJOuBBE2sCgtTsydibqicwaCq6cNzWYztErjavlmV0CacQcykRnoUuTORw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

3. 遇到相同高度的柱子怎么办？

遇到相同的元素，更新栈内下标，就是将栈里元素（旧下标）弹出，将新元素（新下标）加入栈中。

例如 5 5 1 3 这种情况。如果添加第二个 5 的时候就应该将第一个 5 的下标弹出，把第二个 5 添加到栈中。

**因为我们要求宽度的时候如果遇到相同高度的柱子，需要使用最右边的柱子来计算宽度**。

如图所示：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJdWYRibs75uke0yuDT94HnQJT37icpsJzWDicKasPDduQnsG3FEyZv9QOQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

4. 栈里要保存什么数值？

是用单调栈，其实是通过 长 * 宽 来计算雨水面积的。

==长就是通过柱子的高度来计算==，==宽是通过柱子之间的下标来计算==，那么栈里有没有必要存一个 pair<int, int> 类型的元素，保存柱子的高度和下标呢。

其实不用，栈里就存放 int 类型的元素就行了，表示下标，想要知道对应的高度，通过 height [stack.top ()] 就知道弹出的下标对应的高度了。

所以栈内存放下标，计算的时候用下标对应的柱子高度。

##### 4.3.2 单调栈处理逻辑

先将下标 0 的柱子加入到栈中，`st.push(0);`。

然后开始从下标 1 开始遍历所有的柱子，`for (int i = 1; i < height.size(); i++)`。

如果当前遍历的元素（柱子）高度小于栈顶元素的高度，就把这个元素加入栈中，因为栈里本来就要保持从小到大的顺序（从栈头到栈底）。

如果当前遍历的元素（柱子）高度等于栈顶元素的高度，要更新栈顶元素，因为遇到相同高度的柱子，需要使用最右边的柱子来计算宽度。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv70JQho3UqAvofRwABWKtxJOuBBE2sCgtTsydibqicwaCq6cNzWYztErjavlmV0CacQcykRnoUuTORw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

如果当前遍历的元素（柱子）高度大于栈顶元素的高度，此时就出现凹槽了，取栈顶元素，将栈顶元素弹出，这个就是凹槽的底部，也就是中间位置，下标记为 mid，对应的高度为 height [mid]（就是图中的高度 1）。

此时的栈顶元素 st.top ()，就是凹槽的左边位置，下标为 st.top ()，对应的高度为 height [st.top ()]（就是图中的高度 2）。

当前遍历的元素 i，就是凹槽右边的位置，下标为 i，对应的高度为 height [i]（就是图中的高度 3）。

此时大家应该可以发现其实就是**栈顶和栈顶的下一个元素以及要入栈的三个元素来接水！**

那么==雨水高度==是 min (凹槽左边高度，凹槽右边高度) - 凹槽底部高度，代码为：`int h = min(height[st.top()], height[i]) - height[mid];`

==雨水宽度==是凹槽右边的下标 - 凹槽左边的下标 - 1（因为只求中间宽度），代码为：`int w = i - st.top() - 1 ;`

当前凹槽雨水的体积就是：`h * w`。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 单调栈
        '''
        单调栈是按照 行 的方向来计算雨水
        从栈顶到栈底的顺序：从小到大
        通过三个元素来接水：栈顶，栈顶的下一个元素，以及即将入栈的元素
        雨水高度是 min(凹槽左边高度, 凹槽右边高度) - 凹槽底部高度
        雨水的宽度是 凹槽右边的下标 - 凹槽左边的下标 - 1（因为只求中间宽度）
        '''
        # stack储存index，用于计算对应的柱子高度
        stack = [0]
        result = 0
        for i in range(1, len(height)):
            # 情况一
            if height[i] < height[stack[-1]]:
                stack.append(i)

            # 情况二
            # 当当前柱子高度和栈顶一致时，左边的一个是不可能存放雨水的，所以保留右侧新柱子
            # 需要使用最右边的柱子来计算宽度
            elif height[i] == height[stack[-1]]:
                stack.pop()
                stack.append(i)

            # 情况三
            else:
                # 抛出所有较低的柱子
                while stack and height[i] > height[stack[-1]]:
                    # 栈顶就是中间的柱子：储水槽，就是凹槽的地步
                    mid_height = height[stack[-1]]
                    stack.pop()
                    if stack:
                        right_height = height[i]
                        left_height = height[stack[-1]]
                        # 两侧的较矮一方的高度 - 凹槽底部高度
                        h = min(right_height, left_height) - mid_height
                        # 凹槽右侧下标 - 凹槽左侧下标 - 1: 只求中间宽度
                        w = i - stack[-1] - 1
                        # 体积：高乘宽
                        result += h * w
                stack.append(i)
        return result
        
# 单调栈压缩版        
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = [0]
        result = 0
        for i in range(1, len(height)):
            while stack and height[i] > height[stack[-1]]:
                mid_height = stack.pop()
                if stack:
                    # 雨水高度是 min(凹槽左侧高度, 凹槽右侧高度) - 凹槽底部高度
                    h = min(height[stack[-1]], height[i]) - height[mid_height]
                    # 雨水宽度是 凹槽右侧的下标 - 凹槽左侧的下标 - 1
                    w = i - stack[-1] - 1
                    # 累计总雨水体积
                    result += h * w
            stack.append(i)
        return result
```

### 5. 柱状图中最大的矩形

> 给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
>
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。
>
> ![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)
>
> ```
> 输入：heights = [2,1,5,6,2,3]
> 输出：10
> 解释：最大的矩形为图中红色区域，面积为 10
> ```

本题和==接雨水==是遥相呼应的两道题目，建议都仔细做一做，原理上有很多相同的地方，但细节上又有差异，更可以加深对单调栈的理解。

==接雨水==是找每个柱子左右两边第一个大于该柱子高度的柱子，而本题是找每个柱子左右两边第一个小于该柱子的柱子。

这里就涉及到了单调栈很重要的性质，就是==单调栈里的顺序==，是从小到大还是从大到小。

因为本题是要找每个柱子左右两边第一个小于该柱子的柱子，所以从栈头（元素从栈头弹出）到栈底的顺序应该是==从大到小==的顺序！

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/ciaqDnJprwv6ocGGWsQJmA4o7qRRfZdvO7sic3rSaia6UXXPiacL9YH9x7Gc0SjusxUibN0ItAWficiaqpFMN3uic9RPCw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

只有栈里从大到小的顺序，才能保证栈顶元素找到左右两边第一个小于栈顶元素的柱子。

==所以本题单调栈的顺序正好与接雨水反过来。==

此时大家应该可以发现其实就是栈顶和栈顶的下一个元素以及要入栈的三个元素组成了我们要求的最大面积的高度和宽度。

剩下就是分析清楚如下三种情况：

- 情况一：当前遍历的元素 heights [i] 小于栈顶元素 heights [st.top ()] 的情况
- 情况二：当前遍历的元素 heights [i] 等于栈顶元素 heights [st.top ()] 的情况
- 情况三：当前遍历的元素 heights [i] 大于栈顶元素 heights [st.top ()] 的情况



```python
# 单调栈
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Monotonic Stack
        '''
        找每个柱子左右侧的第一个高度值小于该柱子的柱子
        单调栈：栈顶到栈底：从大到小（每插入一个新的小数值时，都要弹出先前的大数值）
        栈顶，栈顶的下一个元素，即将入栈的元素：这三个元素组成了最大面积的高度和宽度
        情况一：当前遍历的元素heights[i]大于栈顶元素的情况
        情况二：当前遍历的元素heights[i]等于栈顶元素的情况
        情况三：当前遍历的元素heights[i]小于栈顶元素的情况
        '''
        # 输入数组首尾各补上一个0 (与42.接雨水不同的是，本题原首尾的两个柱子可以作为核心柱进行最大面积尝试)
        heights.insert(0, 0)
        heights.append(0)
        stack = [0]
        result = 0
        for i in range(1, len(heights)):
            # 情况一
            if heights[i] > heights[stack[-1]]:
                stack.append(i)
            # 情况二
            elif heights[i] == heights[stack[-1]]:
                stack.pop()
                stack.append(i)
            # 情况三
            else:
                # 抛出所有较高的柱子
                while stack and heights[i] < heights[stack[-1]]:
                    # 栈顶就是中间的柱子，主心骨
                    mid_index = stack[-1]
                    stack.pop()
                    if stack:
                        left_index = stack[-1]
                        right_index = i
                        width = right_index - left_index - 1
                        height = heights[mid_index]
                        result = max(result, width * height)
                stack.append(i)
        return result

# 单调栈精简
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.insert(0, 0)
        heights.append(0)
        stack = [0]
        result = 0
        for i in range(1, len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                mid_height = heights[stack[-1]]
                stack.pop()
                if stack:
                    # area = width * height
                    area = (i - stack[-1] - 1) * mid_height
                    result = max(area, result)
            stack.append(i)
        return result
```





## 补充题目

### 1. 快速排序

> 给你一个整数数组 `nums`，请你将该数组升序排列。
>
> ```
> 输入：nums = [5,2,3,1]
> 输出：[1,2,3,5]
> ```

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，然后再递归调用函数对两部分的序列分别进行快速排序，以此使得整个序列达到有序。

我们定义函数 `randomized_quicksort(nums, l, r)` 为对 nums 数组里 `[l, r]` 的部分进行排序，每次先调用 randomized_partition 函数对 nums 数组里 `[l, r]`的部分进行划分，并返回分界的下标 `pos` ，然后按上述讲的递归调用 `randomized_quicksort(nums, l, pos - 1)` 和 `randomized_quicksort(nums, pos + 1, r)` 即可。

那么核心就是划分函数的实现了，划分函数一开始需要确定一个分界值，我们称之为主元 pivot，然后再进行划分。整个划分函数 partition 主要涉及两个指针 i 和 j，一开始 `i = l - 1，j = l`。我们需要实时维护两个指针使得任意时候，对于任意数组下标 kk，我们有如下条件成立：

1. $l \leq k \leq i$ 时， nums $[k] \leq$ pivot。
2. $i+1 \leq k \leq j-1$ 时， nums $[k]>$ pivot 。
3. $k=r$ 时， $n u m s[k]=$ pivot

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        import random                               # 导入随机数函数库
        def quicksort(nums,left,right):
            flag=nums[random.randint(left,right)]   # 随机初始化哨兵位置
            i,j=left,right                          # 设定从左到右的指针i，从右到左的指针j
            while i<=j:
                while nums[i]<flag: i+=1            # i从左往右扫，找到大于等于flag的数。
                while nums[j]>flag: j-=1            # j从右往左扫，找到小于等于flag的数。
                if i<=j:
                    nums[i],nums[j]=nums[j],nums[i] # 交换左右指针下标对应的数值
                    i+=1                            # 左指针继续往右走
                    j-=1                            # 右指针继续往左走
            if i<right: quicksort(nums,i,right)     # 递归解决flag左边的低位数组的排序
            if j>left:  quicksort(nums,left,j)      # 递归解决flag右边的低位数组的排序
        quicksort(nums,0,len(nums)-1)               # 函数入口，将整个数组的信息传入
        return nums  
```



### 2. [不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

> 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
>
> ```
> 输入: a = 1, b = 1
> 输出: 2
> ```

==使用位运算来实现==：

1. 进行异或运算，计算两个数各个位置上的相加，不考虑进位；
2. 进行位与运算，然后左移一位，计算进位值；
3. 把异或运算的结果赋给 num1，把进位值赋给 num2，依此循环，进位值为空的时候结束循环，num1 就是两数之和。

==基本思路==：回顾十进制加法原理

以 `5 + 7 = 12`为例，分步走：

1. 相加各位的值，不算进位，得到 2
2. 计算进位值，得到 10。如果这一步的进位值为 0，如果这一步的进位值为 0，那么第一步得到的值就是最终结果
3. 重复上述两步，只是相加的值变成上述两步得到的结果 2 和 10，得到 12

==相同思想运用于二进制加法运算==：

同样我们可以用三步走的方式计算二进制值相加，5 = （101），7 = （111）

1. 相加各位的值，不算进位，得到 010，二进制每位相加就相当于各位做==异或操作==，101 ^ 111。
2. 计算进位值，得到 1010，相当于各位做与操作得到 101，再向左移一位得到 1010，(101 & 111) << 1。
3. 重复上述两步， 各位相加 010 ^ 1010 = 1000，进位值为 100 = (010 & 1010) << 1 。
4. 继续重复上述两步：1000 ^ 100 = 1100，进位值为 0，跳出循环，1100 为最终结果。

==有点难，先放着==

```python
```



### 3. [数组中的第 K 个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

> 给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。
>
> 请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。
>
> ```
> 输入: [3,2,1,5,6,4] 和 k = 2
> 输出: 5
> ```

#### 3.1 大根堆：调库

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        maxHeap = []
        for x in nums:
            heapq.heappush(maxHeap, -x)
        for _ in range(k - 1):
            heapq.heappop(maxHeap)
        return -maxHeap[0]
```

### 4. 有多少小于当前数字的数字

> 给你一个数组 nums，对于其中每个元素 nums[i]，请你统计数组中比它小的所有数字的数目。
>
> 换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i] 。
>
> 以数组形式返回答案。
>
> 输入：nums = [8,1,2,2,3]
> 输出：[4,0,1,1,3]
> 解释： 
> 对于 nums[0]=8 存在四个比它小的数字：（1，2，2 和 3）。 
> 对于 nums[1]=1 不存在比它小的数字。
> 对于 nums[2]=2 存在一个比它小的数字：（1）。 
> 对于 nums[3]=2 存在一个比它小的数字：（1）。 
> 对于 nums[4]=3 存在三个比它小的数字：（1，2 和 2）。

首先要找小于当前数字的数字，那么从小到大排序之后，该数字之前的数字就是比它小的了。

所以可以定义一个新数组，将数组排个序。**排序之后，其实每一个数值的下标就代表这前面有几个比它小的了**。

用一个哈希表 hash（本题可以就用一个数组）来做数值和下标的映射。这样就可以通过数值快速知道下标（也就是前面有几个比它小的）。

此时有一个情况，就是数值相同怎么办？

例如，数组：1 2 3 4 4 4 ，第一个数值 4 的下标是 3，第二个数值 4 的下标是 4 了。

这里就需要一个技巧了，**在构造数组 hash 的时候，从后向前遍历，这样 hash 里存放的就是相同元素最左面的数值和下标了**。

最后在遍历原数组 nums，用 hash 快速找到每一个数值对应的 小于这个数值的个数。存放在将结果存放在另一个数组中。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4mz9vCUVLwn1552ibh2kA8GibLP6fhcekawCepf9wicEOzNPJftUG1N24sfmaYvTv4QfBBkuWFB35Ng/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        res = nums[:]
        hash = dict()
        res.sort() # 从小到大排序之后，元素下标就是小于当前数字的数字
        for i, num in enumerate(res):
            if num  not in hash.keys(): # 遇到了相同的数字，那么不需要更新该 number 的情况
                hash[num] = i       
        for i, num in enumerate(nums):
            res[i] = hash[num]
        return res
```

### 5. 比较含退格的字符串

> 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。
>
> 注意：如果对空文本输入退格字符，文本继续为空。
>
> ```
> 输入：s = "ab#c", t = "ad#c"
> 输出：true
> 解释：s 和 t 都会变成 "ac"。
> 
> 输入：s = "ab##", t = "c#d#"
> 输出：true
> 解释：s 和 t 都会变成 ""。
> ```

我的解法：从后向前遍历，双指针，好吧，这种想法无法解决 ‘ab##’ 的问题。

#### 5.1 栈

这道题目一看就是要使用栈的节奏，这种匹配消除问题也是栈的擅长所在。**那么本题，确实可以使用栈的思路，但是没有必要使用栈，因为最后比较的时候还要比较栈里的元素，有点麻烦**。这里直接使用字符串 string，来作为栈，末尾添加和弹出，string 都有相应的接口，最后比较的时候，只要比较两个字符串就可以了，比比较栈里的元素方便一些。

```python
class Solution:

    def get_string(self, s: str) -> str :
        bz = []
        for i in range(len(s)) :
            c = s[i]
            if c != '#' :
                bz.append(c) # 模拟入栈
            elif len(bz) > 0: # 栈非空才能弹栈
                bz.pop() # 模拟弹栈
        return str(bz)

    def backspaceCompare(self, s: str, t: str) -> bool:
        return self.get_string(s) == self.get_string(t)
        pass
```



#### 5.2 双指针：从后向前

当然还可以有使用 O (1) 的空间复杂度来解决该问题。

同时从后向前遍历 S 和 T（i 初始为 S 末尾，j 初始为 T 末尾），记录 # 的数量，模拟消除的操作，如果 # 用完了，就开始比较 S [i] 和 S [j]。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv7P7hElU0Ml0AWbPNfjHLkbXwXhKE6icW7WULW0U9QEX1WOmxcibO4saNzcLgMcItibePJ48OTQWQ3DA/640?wx_fmt=gif&wxfrom=5&wx_lazy=1)

如果 S [i] 和 S [j] 不相同返回 false，如果有一个指针（i 或者 j）先走到的字符串头部位置，也返回 false。

1. 准备两个指针 $i,j$ 分别指向 $S$ 和 $T$ 的末位字符，再准备两个变量 `skipS` 和 `skipT` 来分别存放 S 和 T 字符串中 # 的数量
2. 从后往前遍历 $S$，所遇情况有三:
   1. 若当前字符是 #，则 `skipS` 自增 1
   2. 若当前字符不是 # 且 `skipS` 不为 0，则`skipS` 自减 1
   3. 若当前字符不是 #，且 `skipS` 为 0，则代表当前字符不会被消除，我们可以用来和 $T$ 中的当前字符作比较。
3. 若对比过程出现  $S$ 和 $T$ 当前字符不匹配，则遍历结束，返回 false，若  $S$ 和 $T$ 都遍历结束，且能一一匹配，则返回 true

```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        i, j = len(S) - 1, len(T) - 1
        skipS = skipT = 0

        while i >= 0 or j >= 0:
            while i >= 0:
                if S[i] == "#":
                    skipS += 1
                    i -= 1
                elif skipS > 0:
                    skipS -= 1
                    i -= 1
                else:
                    break
            while j >= 0:
                if T[j] == "#":
                    skipT += 1
                    j -= 1
                elif skipT > 0:
                    skipT -= 1
                    j -= 1
                else:
                    break
            if i >= 0 and j >= 0:
                if S[i] != T[j]:
                    return False
            elif i >= 0 or j >= 0:
                return False
            i -= 1
            j -= 1
        
        return True

```

> 这对我来说是个难题，因为循环中套着循环，不太好想。

### 6. 旋转数组

> 给你一个数组，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。
>
> 输入: nums = [1,2,3,4,5,6,7], k = 3
> 输出: [5,6,7,1,2,3,4]
> 解释:
> 向右轮转 1 步: [7,1,2,3,4,5,6]
> 向右轮转 2 步: [6,7,1,2,3,4,5]
> 向右轮转 3 步: [5,6,7,1,2,3,4]

注意本题要求使用空间复杂度为 O(1) 的原地算法。

1. 反转整个字符串：[7,6,5,4,3,2,1]
2. 反转区间为前 k 的子串：[5,6,7,4,3,2,1]
3. 反转区间为 k 到末尾的子串：[5,6,7,1,2,3,4]

**需要注意的是，本题还有一个小陷阱，题目输入中，如果 k 大于 nums.size 了应该怎么办？**

例如，[1,2,3,4,5,6,7]  如果右移动 15 次的话，是 [7,1,2,3,4,5,6]  。

所以其实就是右移 k % nums.size () 次，即：15 % 7 = 1

```python
class Solution:
    def rotate(self, A: List[int], k: int) -> None:
        def reverse(i, j):
            while i < j:
                A[i], A[j] = A[j], A[i]
                i += 1
                j -= 1
        n = len(A)
        k = k % n
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```

### 7. [Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/)

> 输入："RD"
> 输出："Radiant"
> 解释：第一个参议员来自 Radiant 阵营并且他可以使用第一项权利让第二个参议员失去权力，因此第二个参议员将被跳过因为他没有任何权利。然后在第二轮的时候，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人

例如输入 "RRDDD"，执行过程应该是什么样呢？

- 第一轮：senate [0] 的 R 消灭 senate [2] 的 D，senate [1] 的 R 消灭 senate [3] 的 D，senate [4] 的 D 消灭 senate [0] 的 R，此时剩下 "RD"，第一轮结束！
- 第二轮：senate [0] 的 R 消灭 senate [1] 的 D，第二轮结束
- 第三轮：只有 R 了，R 胜利

估计不少同学都困惑，R 和 D 数量相同怎么办，究竟谁赢，**其实这是一个持续消灭的过程！** 即：如果同时存在 R 和 D 就继续进行下一轮消灭，轮数直到只剩下 R 或者 D 为止！

那么每一轮消灭的策略应该是什么呢？例如：RDDRD

第一轮：senate [0] 的 R 消灭 senate [1] 的 D，那么 senate [2] 的 D，是消灭 senate [0] 的 R 还是消灭 senate [3] 的 R 呢？

当然是消灭 senate [3] 的 R，因为当轮到这个 R 的时候，它可以消灭 senate [4] 的 D。

==所以消灭的策略是，尽量消灭自己后面的对手，因为前面的对手已经使用过权力了，而后序的对手依然可以使用权力消灭自己的同伴！==

==局部最优==：有一次权力机会，就消灭自己后面的对手

==全局最优==：为自己的阵营赢取最大利益

实现代码，在每一轮循环的过程中，去过模拟优先消灭身后的对手，其实是比较麻烦的。

这里有一个技巧，就是用一个变量记录当前参议员之前有几个敌对对手了，进而判断自己是否被消灭了。这个变量我用 flag 来表示。

```python
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        # R = true 表示本轮循环结束后，字符串里依然有 R。D 同理
        R , D = True, True

        # flag 表示当前参议院前面有几个敌对对手
        # 当 flag 大于 0 时，R 在 D 前出现，R 可以消灭 D。当 flag 小于 0 时，D 在 R 前出现，D可以消灭 R
        flag = 0

        senate = list(senate)
        while R and D: # 一旦 R 或者 D 为 false，就结束循环，说明本轮结束后只剩下 R 或者 D了
            R = False
            D = False
            for i in range(len(senate)):
                if senate[i] == 'R':
                    if flag < 0: 
                        senate[i] = '0' # 消灭 R，R 此时为 false
                    else: 
                        R = True # 如果没被消灭，本轮循环结束有 R
                    flag += 1
                if senate[i] == 'D':
                    if flag > 0: 
                        senate[i] = '0'
                    else: 
                        D = True
                    flag -= 1
        # 循环结束之后，R 和 D 只能有一个为 true
        return "Radiant" if R else "Dire"
```

### 8. 有效的山脉数组

> 给定一个整数数组 arr，如果它是有效的山脉数组就返回 true，否则返回 false。
>
> 让我们回顾一下，如果 arr 满足下述条件，那么它是一个山脉数组：
>
> arr.length >= 3
> 在 0 < i < arr.length - 1 条件下，存在 i 使得：
> arr[0] < arr[1] < ... arr[i-1] < arr[i]
> arr[i] > arr[i+1] > ... > arr[arr.length - 1]
>
> <img src="https://assets.leetcode.com/uploads/2019/10/20/hint_valid_mountain_array.png" alt="img" style="zoom:50%;" />
>
> ```
> 输入：arr = [0,3,2,1]
> 输出：true
> 
> 输入：arr = [3,5,5]
> 输出：false
> ```

判断是山峰，主要就是要严格的保存左边到中间，和右边到中间是递增的。

这样可以使用两个指针，left 和 right，让其按照如下规则移动，如图：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4tthKjaWUBf3KPFDOW1Dbxlhc2joYmGbXeiaNbmJvJwWNfl0wUZqCJ7TGvsLL6oYFp2cBtMdTrKaQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

**注意这里还是有一些细节，例如如下两点：**

- 因为 left 和 right 是数组下表，移动的过程中注意不要==数组越界==
- 如果 left 或者 right 没有移动，说明是一个单调递增或者递减的数组，依然不是山峰

```python
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        i = 0
        j = len(arr) - 1
        if len(arr) < 3: return False
        while i < len(arr)-1 and arr[i+1] > arr[i]:
            i += 1
        while j > 0 and arr[j-1] > arr[j]:
            j -= 1
        # i 和 j 相遇且都不在起始位置
        if i == j and i > 0 and j < len(arr) - 1:
            return True
        else:
            return False  
```

### 9. 独一无二的出现次数

> 给你一个整数数组 `arr`，请你帮忙统计数组中每个数的出现次数。
>
> 如果每个数的出现次数都是独一无二的，就返回 `true`；否则返回 `false`。
>
>  输入：arr = [1,2,2,1,1,3]
> 输出：true
> 解释：在该数组中，1 出现了 3 次，2 出现了 2 次，3 只出现了 1 次。没有两个数的出现次数相同。

这道题目数组在是哈希法中的经典应用，回归本题，**本题强调了 - 1000 <= arr [i] <= 1000**，那么就可以用数组来做哈希，arr [i] 作为哈希表（数组）的下标，那么 arr [i] 可以是负数，怎么办？负数不能做数组下标。

**此时可以定义一个 2000 大小的数组，例如 int count [2002];**，统计的时候，将 arr [i] 统一加 1000，这样就可以统计 arr [i] 的出现频率了。

题目中要求的是是否有相同的频率出现，那么需要再定义一个哈希表（数组）用来记录频率是否重复出现过，bool fre [1002]; 定义布尔类型的就可以了，**因为题目中强调 1 <= arr.length <= 1000，所以哈希表大小为 1000 就可以了**。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6KqK5icjMR6OupaM452DeDfuecicBib3U9epFmHPibpVU2ibtPRAebboBI8VXGQarZD2rYqwcfAWkSRZg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom: 67%;" />

```python
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        # 可以用字典
        res = {}
        for i in range(len(arr)):
            if arr[i] in res:
                res[arr[i]] += 1
            else:
                res[arr[i]] = 1
        # 标记相同频率是否重复出现
        temp = {}
        for _, fre in res.items():
            if fre not in temp:
                temp[fre] = 1
            else:
                return False
        return True
```

### 10. 寻找数组的中心下标

> 给你一个整数数组 nums ，请计算数组的 中心下标 。
>
> 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。
>
> 如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。
>
> 如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。
>
> 输入：nums = [1, 7, 3, 6, 5, 6]
> 输出：3
> 解释：
> 中心下标是 3 。
> 左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11 ，
> 右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11 ，二者相等。

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        numSum = sum(nums) #数组总和
        leftSum = 0
        for i in range(len(nums)):
            if numSum - leftSum -nums[i] == leftSum: #左右和相等
                return i
            leftSum += nums[i]
        return -1
```

### 11. [按奇偶排序数组 II](https://leetcode-cn.com/problems/sort-array-by-parity-ii/)

> 给定一个非负整数数组 nums，  nums 中一半整数是 奇数 ，一半整数是 偶数 。
>
> 对数组进行排序，以便当 nums[i] 为奇数时，i 也是 奇数 ；当 nums[i] 为偶数时， i 也是 偶数 。
>
> ```
> 输入：nums = [4,2,5,7]
> 输出：[4,5,2,7]
> 解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
> ```

```python
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        oddIndex = 1
        # 步长为2
        for i in range(0,len(nums),2): 
            # 偶数位遇到奇数
            if nums[i] % 2: 
                # 奇数位找偶数
                while  nums[oddIndex] % 2: 
                    oddIndex += 2
                nums[i], nums[oddIndex] = nums[oddIndex], nums[i]
        return nums
```

### 12. 搜索插入位置

> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
>
> 请必须使用时间复杂度为 O(log n) 的算法。
>
> ```
> 输入: nums = [1,3,5,6], target = 5
> 输出: 2
> ```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 二分法
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
        return right + 1
```

> 注意最后输出的是 right + 1。为什么呢？

### 13. 回文链表

> 给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。
>
> ```
> 输入：head = [1,2,2,1]
> 输出：true
> ```

#### 13.1 数组模拟

最直接的想法，就是把链表装成数组，然后再判断是否回文。

```python
# 数组模拟
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        nums = []
        cur = head
        while cur:
            nums.append(cur.val)
            cur = cur.next
        
        i = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] != nums[j]:
                return False
            else:
                i += 1
                j -= 1
        return True
```



#### 13.2 反转后半部分链表

分为如下几步：

- 用快慢指针，快指针有两步，慢指针走一步，快指针遇到终止位置时，慢指针就在链表中间位置
- 同时用 pre 记录慢指针指向节点的前一个节点，用来分割链表
- 将链表分为前后均等两部分，如果链表长度是奇数，那么后半部分多一个节点
- 将后半部分反转 ，得 cur2，前半部分为 cur1
- 按照 cur1 的长度，一次比较 cur1 和 cur2 的节点数值

如图所示：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv6pVsicR709O6tUmz8RDV8z5qDUsjvdFO4A2AemGz58TU05fB7dpiaANfGYTibW2mJAzWYdY9VQd7giaA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

```python
# 反转后半部分链表
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if head == None or head.next == None:
            return True
        slow, fast = head, head
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next

        pre.next = None # 分割链表
        cur1 = head # 前半部分
        cur2 = self.reverseList(slow) # 反转后半部分，总链表长度如果是奇数，cur2 比 cur1多一个节点
        while cur1:
            if cur1.val != cur2.val:
                return False
            cur1 = cur1.next
            cur2 = cur2.next
        return True

    def reverseList(self, head: ListNode) -> ListNode:
        # 反转链表
        cur = head
        pre = None
        while(cur != None):
            # 保存一下 cur 的下一个节点，因为接下来要改变 cur->next
            temp = cur.next
            # 反转
            cur.next = pre
            # 更新 pre、cur 指针
            pre = cur
            cur = temp
        return pre

```

终于明白指针的含义了，节点有两个属性：val 和 next，只要不改变 `node.val` 或者 `node.next`，链表就不会改变，pre、slow和 fast 就只是一个指针；如果想要改变链表，可以使用：`fast.next = fast.next.next` 类似这种代码。

### 14. 重排链表

> 给定一个单链表 L 的头节点 head ，单链表 L 表示为：
>
> L0 → L1 → … → Ln - 1 → Ln
> 请将其重新排列后变为：
>
> L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
> 不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
>
> ![img](https://pic.leetcode-cn.com/1626420311-PkUiGI-image.png)
>
> ```
> 输入：head = [1,2,3,4]
> 输出：[1,4,2,3]
> ```

#### 14.1 数组模拟

把链表放进数组中，然后通过双指针法，一前一后，来遍历数组，构造链表。

#### 14.2 双向队列模拟

把链表放进双向队列，然后通过双向队列一前一后弹出数据，来构造新的链表。这种方法比操作数组容易一些，不用双指针模拟一前一后了

```python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        d = collections.deque()
        tmp = head
        while tmp.next: # 链表除了首元素全部加入双向队列
            d.append(tmp.next)
            tmp = tmp.next
        tmp = head
        while len(d): # 一后一前加入链表
            tmp.next = d.pop()
            tmp = tmp.next
            if len(d):
                tmp.next = d.popleft()
                tmp = tmp.next
        tmp.next = None # 尾部置空
```



#### 14.3 直接分割链表

将链表分割成两个链表，然后把第二个链表反转，之后在通过两个链表拼接成新的链表。

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv74WqHkicBX3mIXXB3prdU8cGqgvKHb2YjeEk5YNVRKWPtjQmkX6HQ0bLticsibJMheb2ticwAEicag16g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片"  />

```python
# 方法三 反转链表
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if head == None or head.next == None:
            return True
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        right = slow.next # 分割右半边
        slow.next = None # 切断
        right = self.reverseList(right) #反转右半边
        left = head
        # 左半边一定比右半边长, 因此判断右半边即可
        while right:
            curLeft = left.next
            left.next = right
            left = curLeft

            curRight = right.next
            right.next = left
            right = curRight


    def reverseList(self, head: ListNode) -> ListNode:
        cur = head   
        pre = None
        while(cur!=None):
            temp = cur.next # 保存一下cur的下一个节点
            cur.next = pre # 反转
            pre = cur
            cur = temp
        return pre
```

### 15. 在排序数组中查找元素的第一个和最后一个位置

> 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
>
> 如果数组中不存在目标值 target，返回 [-1, -1]。
>
> ```
> 输入：nums = [5,7,7,8,8,10], target = 8
> 输出：[3,4]
> ```

下面我来把所有情况都讨论一下。

寻找 target 在数组里的左右边界，有如下三种情况：

- 情况一：target 在数组范围的右边或者左边，例如数组 {3, 4, 5}，target 为 2 或者数组 {3, 4, 5},target 为 6，此时应该返回 {-1, -1}
- 情况二：target 在数组范围中，且数组中不存在 target，例如数组 {3,6,7},target 为 5，此时应该返回 {-1, -1}
- 情况三：target 在数组范围中，且数组中存在 target，例如数组 {3,6,7},target 为 6，此时应该返回 {1, 1}

接下来，在去寻找左边界，和右边界了。

采用二分法来去寻找左右边界，为了让代码清晰，我分别写两个二分来寻找左边界和右边界。

**刚刚接触二分搜索的同学不建议上来就像如果用一个二分来查找左右边界，很容易把自己绕进去，建议扎扎实实的写两个二分分别找左边界和右边界**

这里我采用 `while (left <= right)` 的写法，区间定义为 `[left, right]`，即左闭右闭的区间。

```python
# 1、首先，在 nums 数组中二分查找 target；
# 2、如果二分查找失败，则 binarySearch 返回 -1，表明 nums 中没有 target。此时，searchRange 直接返回 {-1, -1}；
# 3、如果二分查找成功，则 binarySearch 返回 nums 中值为 target 的一个下标。然后，通过左右滑动指针，来找到符合题意的区间
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            while left<=right: # 不变量：左闭右闭区间
                middle = left + (right-left) // 2
                if nums[middle] > target:
                    right = middle - 1
                elif nums[middle] < target:
                    left = middle + 1
                else:
                    return middle
            # 如果没有找到，返回 -1
            return -1
        index = binarySearch(nums, target)
        # nums 中不存在 target，直接返回 {-1, -1}
        if index == -1:
            return [-1, -1]
        # nums 中存在 targe，则左右滑动指针，来找到符合题意的区间
        left, right = index, index
        # 向左滑动，找左边界，因为这是一个排序数组
        while left -1 >= 0 and nums[left - 1] == target:
            left -=1
        # 向右滑动，找右边界
        while right+1 < len(nums) and nums[right + 1] == target:
            right +=1
        return [left, right]
```



## 参考资料

1. 微信公众号：代码随想录
1. https://github.com/youngyangyang04/leetcode-master/tree/master/problems

