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

### 4. 长度最小的子数组

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

### 5. 螺旋矩阵

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
                # 指针向后移动两位
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



### 5. 链表相交

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

==算法==：我们求出两个链表的长度，并求出两个链表长度的差值，然后让 `curA` 移动到和 `curB` ==末尾对齐==的位置：

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

### 6. 环形链表

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

### 5. 快乐数

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

此时就要选择另一种数据结构：map ，map 是一种 key-value 的存储结构，可以用 key 保存数值，用 value 在保存数值所在的下表。

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

### 9. 三数之和

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

### 10. 四数之和

> 给你一个由 n 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且==不重复==的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：
>
> 1. `0 <= a, b, c, d < n`
> 2. a、b、c 和 d 互不相同
> 3. `nums[a] + nums[b] + nums[c] + nums[d] == target`
>
> 输入：`nums = [1,0,-1,0,-2,2], target = 0`
> 输出：`[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]`

四数之和的双指针解法是两层 for 循环 `nums [k] + nums [i]`为确定值，依然是循环内有 left 和 right 作为双指针，找出 `nums [k] + nums [i] + nums [left] + nums [right] == target`的情况，三数之和的时间复杂度是 $O(n^2)$，四数之和的时间复杂度是 $O(n^3)$。

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

对于字符串，我们定义两个指针（索引下标），一个从字符串前面，一个从字符串后米艾尼，两个指针同时向中间移动你，并交换元素。

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

### 6. 实现 strStr() 函数

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



### 7. 重复的子字符串

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

在 push 数据的时候，只要数据放进输入栈就好，但是在 pop 的时候，操作就复杂一些，输出栈如果为空，就把进栈数据==全部导入==，再从出栈弹出数据，如果输出栈不为空，则直接从出栈弹出数据就可以了。

最后如何判断队列为空呢？如果进栈和出栈都为空的话，说明模拟的队列为空了。

> 注意，在工业级代码开发中，最忌讳就是实现一个类似的函数，直接把代码粘贴过来改一改，这样代码会越来越乱，一定要懂得复用，功能相近的函数要抽象出来。

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

队列模拟栈，其实一个队列就够了。

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

### 5. 滑动窗口最大值

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

这是使用单调队列的经典题目。==难点==是如何求一个区间的最大值。

暴力方法，遍历一遍的过程中每次从窗口中在找到最大的数值，这样很明显是 $O (n * k)$ 的算法。

有的同学可能会想用一个**大顶堆（优先级队列）**来存放这个窗口里的 k 个数字，这样就可以知道最大的最大值是多少了， 但是问题是这个窗口是移动的，而大顶堆每次只能弹出最大值，我们无法移除其他数值，这就造成大顶堆维护的不是滑动窗口里面的数值了。

此时我们需要一个队列，随着窗口的移动，队列也一进一出，每次移动之后，队列告诉我们里面的最大值是什么。

每次窗口移动的时候，调用 `que.pop` (滑动窗口中移除元素的数值)，`que.push` (滑动窗口添加元素的数值)，然后 `que.front ()` 就返回我们要的最大值。

为实现这一点，队列里面的元素一定需要排序，而且最大值放在出口，但如果把窗口里的元素都放进队列里，窗口移动的时候，队列需要弹出元素。

那么问题来了，已经排序之后的队列 怎么能把窗口要移除的元素（这个元素可不一定是最大值）弹出呢。

其实队列没有必要维护窗口里的所有元素，只需要维护**有可能**成为窗口里最大值的元素就可以了，同时保证队列里面的元素数值是从大到小的

那么这个维护元素单调递减的队列就叫做**单调队列，即单调递减或单调递增的队列。**

![图片](https://mmbiz.qpic.cn/mmbiz_gif/ciaqDnJprwv4mCxur8W49qtZmumwtiax6R0axb2Svoib5fzy1ibMlLRFslLlq9TSG84soSCoicvH5jmlQUpKwHiaXZ6A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

对于窗口里的元素 {2, 3, 5, 1 ,4}，单调队列里只维护 {5, 4} 就够了，保持单调队列里单调递减，此时队列出口元素就是窗口里最大元素。

设计单调队列的时候，pop 和 push 操作要保持如下规则：

1. pop (value)：如果窗口移除的元素 value 等于单调队列的出口元素，那么队列弹出元素，否则不用任何操作
2. push (value)：如果 push 的元素 value 大于入口元素的数值，那么就将队列入口的元素弹出，直到 push 元素的数值小于等于队列入口元素的数值为止

保持如上规则，每次窗口移动的时候，只要问 `que.front ()` 就可以返回当前窗口的最大值。

为了更直观的感受到单调队列的工作过程，以题目示例为例，输入: `nums = [1,3,-1,-3,5,3,6,7]`, 和 `k = 3`，动画如下：

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
        每次弹出时比较当前弹出的数值是否等于队列出口元素的数值，如果相等则直接弹出
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
        # 先将前 k 的元素放进队列
        for i in range(k):
            que.push(nums[i])
        result.append(que.front())
        for i in range(k, len(nums)):
            # 滑动窗口移除最前面元素
            que.pop(nums[i - k])
            # 滑动窗口前加入最后面的元素
            que.push(nums[i])
            # 记录对应的最大值
            result.append(que.front())
        return result

sol = Solution()
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(sol.maxSlidingWindow(nums,k))
```

### 6. [前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)
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
            if len(pri_que) > k:  # 如果堆的大小大于了 K，则队列弹出，保证堆的大小一直为 k
                heapq.heappop(pri_que)

        # 找出前 K 个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result


sol = Solution()
nums = [1, 1, 1, 2, 2, 3, 1, 1]
k = 2
print(sol.topKFrequent(nums, k))
```

## 二叉树

### 1. 二叉树的基本知识

#### 1.1 二叉树的类型

在我们解题过程中二叉树有两种主要的形式：满二叉树和完全二叉树。

==满二叉树==：如果一棵二叉树只有度为 0 的节点和度为 2 的节点，并且度为 0 的节点在同一层上，则这棵二叉树为满二叉树。也可以说深度为 k，有 $2^k-1$ 个节点的二叉树。

==完全二叉树==：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到了最大值，并且最下面一层的节点都集中在该层最左边的若干位置上。若最底层为第 $h$ 层，则该层包含了 1 ~ $(2^h -1)$  个节点。

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



### 2. 二叉树的递归遍历

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

### 3. 二叉树的迭代遍历

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

在刚才的迭代过程中其实我们有连个操作：

1. 处理：将元素放进 result 数组中
2. 访问：遍历节点

前序遍历的代码不能和中序遍历通用，因为前序遍历的顺序是中左右，先访问的元素的==中间节点==，要处理的元素也是==中间节点==；但是中序遍历顺序是左中右，先访问的是二叉树顶部的节点，然后一层一层向下访问，知道达到树左面的==最底部==，再开始处理节点（也就是把节点的数值放进 result 数组中），**这就造成了处理顺序和访问顺序是不一致的。**

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



### 4. 二叉树的层序遍历

> 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。（即逐层地，从左到右访问所有节点）。
>
> ![img](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)
>
> 输入：`root = [3,9,20,null,null,15,7]`
> 输出：`[[3],[9,20],[15,7]]`

二叉树的层序遍历需要借助一个辅助数据结构即队列来实现，队列先进先出，符合一层一层遍历的逻辑，而栈先进后出适合模拟深度优先遍历也就是递归的逻辑。

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

首先要想清楚，判断对称二叉树要比较的是哪两个节点，可不是左右节点！而是要比较跟节点的 左右子树，所以在递归遍历的过程中，也是要==同时遍历两棵树==。   

==那应该如何选择遍历顺序？==

本题遍历顺序只能是”==后序遍历==“，因为我们要通过递归函数的返回值来判断两个子树的内侧节点是否相等。正因为要遍历两棵树而且要比较内侧和外侧节点，所以准确的来说是一个树的遍历顺序是左右中，一个树的遍历顺序是右左中。

#### 6.1递归法

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
        #首先排除空节点的情况
        if left == None and right != None:
            return False
        elif left != None and right == None:
            return False
        elif left == None and right == None:
            return True
        #排除了空节点，再排除数值不相同的情况
        elif left.val != right.val:
            return False
        
        #此时就是：左右节点都不为空，且数值相同的情况
        #此时才做递归，做下一层的判断
        outside = self.compare(left.left, right.right) #左子树：左、 右子树：右
        inside = self.compare(left.right, right.left) #左子树：右、 右子树：左
        isSame = outside and inside #左子树：中、 右子树：中 （逻辑处理）
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

本题可以使用前序（中左右），也可以使用后序遍历（左右中），使用前序求的就是深度，使用后序求的是高度。

**而根节点的高度就是二叉树的最大深度**，所以本题中我们通过后序求的根节点高度来求的二叉树最大深度。

==TO DO==：二叉树做的有点厌倦了，先做回溯算法调节下心情



## 回溯算法

回溯算法，也叫回溯搜索法，是一种搜索方式。

回溯是递归的副产品，只要有递归就会有回溯。回溯函数就是递归函数。

### 1. 回溯法的基本知识

回溯法并不是高效的算法，因为回溯法的本质是穷举，穷举所有可能，然后选出我们想要的答案，如果想让回溯法高效一些，可以加一些剪枝的操作，但也改变不了回溯法就是穷举的本质。

==回溯法解决的问题==：

- 组合问题：N 个数里面按照一定规则找出 k 个数的集合
- 切割问题：一个字符串按照 一定规则有几种切割方式
- 子集问题：一个 N 个数的集合里有多少符合条件的子集
- 排列问题：N 个数按一定规则全排列，有几种排列方式
- 棋盘问题：N 皇后，解数独等等

> 组合无序，排列有序。

==如何理解回溯法：==

回溯法解决的问题都可以抽象为树形结构，因为回溯法解决的都是在集合中递归查找子集

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

这里主要定义两个局部变量，一个用来存放符合条件的单一结果 result，一个用来存放符合条件结果的集合 path。

函数里一定有两个参数，既然是集合 n 里面取 k 的数，那么 n 和 k 是两个 int 型的参数。

然后还需要一个参数，为 int 型变量 `startIndex`，这个参数用来记录本层递归中，集合从哪里开始遍历（集合就是 [1,...,n] ），也就是下一层递归搜索的起始位置（如果可以重复的，应该就不需要这个参数）。

> **为什么要有这个 startIndex 呢？**
>
> 每次从集合中选取元素，可选择的范围随着选择的进行而收缩，调整可选择的范围，就要靠  `startIndex`

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
        # len(path)==k 时不管 sum 是否等于n都会返回
        if len(self.path) == k:  
            if self.sum_now == n:
                self.res.append(self.path[:])
            # 如果 len(path)==k 但是 和不等于 target，直接返回
            return
        # 集合固定为 9 个数
        for i in range(start_num, 10 - (k - len(self.path)) + 1):
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

> 给你一个无重复元素的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的所有不同组合 ，并以列表形式返回。你可以按任意顺序返回这些组合。
>
> candidates 中的 同一个数字可以无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
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
        # 确定终止条件
        if self.sum == target:
            self.res.append(self.result[:])
            return
        if self.sum > target:
            return

        # 进入单层循环逻辑：从 startindex 开始选取是为了保证在后面做选择时不会选到前面的数字避免重复
        for i in range(startindex, len(candidates)):
            self.result.append(candidates[i])
            self.sum += candidates[i]
            # 因为可以无限制选取同一个数字，所以是 i
            self.trackbacking(i, candidates, target)
            # 回溯
            self.result.pop()
            self.sum -= candidates[i]
        return self.res
```

==剪枝优化：==这个优化一般不容易想到，但是在求和问题中，排序后加剪枝是常见的套路！

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

这个题目的关键在==去重==！题目给的集合里有重复的元素，但是不能有重复的组合！！！

如果把所有的组合全部求出来，再用 set 或者 map 去重，这么做很容易**超时**！所以我们需要在搜索的过程中就去掉重复组合。

所谓去重，==其实就是使用过的元素不能重复选取==。

都知道组合问题可以抽象成树型结构，那么使用过在这个树形结构上是有两个维度的，一个维度是==同一树枝==上使用过，一个维度是==同一树层==上使用过，没有理解这两个层面的“使用过”是造成大家没有彻底理解去重的根本原因。

题目要求：元素在同一组合是就可以重复使用的，只要给的集合有重复的数字，但是这两个组合不能相同，所以我们==要去重的是同一树层上的“使用过”，因为同一树枝上的都是一个组合里的元素，不用去重==。以 `candidates = [1, 1, 2], target = 3`为例：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv48aCU4UTGAaibHh1UFayia1yBvRvuXqu2Z4jnaY2fEhUoL3Ggr0zxN7vgzKBRHO7QmeBy5BO1BqeFg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

==回溯三部曲：==

1. **递归函数参数**：与上一题相同，此时还需要加一个 ==bool 型数组 used==，用来记录同一树枝上的元素是否使用过，这个集合去重的重任就是 used 完成的。

2. **递归终止条件**：终止条件为 `sum > target` 和 `sum == target`。
3. **单层搜索的逻辑**：如何判断同一树层上元素是否使用过了呢？

**如果 `candidates[i] == candidates[i - 1]` 并且 `used[i - 1] == false`，就说明：前一个树枝，使用了 candidates [i - 1]，也就是说同一树层使用过 candidates [i - 1]**。此时 for 循环里应该做 continue 的操作。

![图片](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv48aCU4UTGAaibHh1UFayia1yFn6HgwBDohL8uc9icx9afAMLSQKaibWwItd8bZHaL9WYvmTTX7IwAg9A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我在图中将 used 的变化用橘黄色标注上，可以看出在 `candidates [i] == candidates [i - 1]` 相同的情况下：

- `used [i - 1] == true`，说明同一树支 `candidates [i - 1]` 使用过
- `used [i - 1] == false`，说明同一树层 `candidates [i - 1]` 使用过

和上一题相比，同样是求组合总和，但就是因为其数组 candidates 有重复元素，而要求不能有重复的组合，难度提升了不少。

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
            if i >= 1 and candidates[i] == candidates[i-1] and self.used[i-1] == 0:
                continue

            self.result.append(candidates[i])
            self.sum += candidates[i]
            self.used[i] = 1
            # 这是在同一树层上去重
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

- 递归调用时，下一层递归的 startindex 要从 i + 2 开始，因为需要在字符串中加入分隔符，同时记录分隔符的数量 pointNum 要加 1。
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

## 贪心算法

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
            if res <len(g) and s[i] >= g[res]:  #小饼干先喂饱小胃口
                res += 1
        return res
```

### 3. 摆动序列

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
        # 题目里nums长度大于等于1，当长度为1时，其实到不了for循环里去，所以不用考虑nums长度
        preC, curC, res = 0,0,1  
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            # 差值为0时，不算摆动
            if curC * preC <= 0 and curC !=0:  
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

### 4. 最大子数组和

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

从代码角度上来讲：遍历 `nums`，从头开始用 `count` 累积，如果 `count` 一旦加上 `nums [i]` 变为负数，那么就应该从 `nums [i+1]` 开始从 0 累积 count 了，因为已经变为负数的 `count`，只会==拖累总和==。（很有道理）

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

==局部最优==：收集每天的正利润；==全局最优==：求得最大利润

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

### 6. 跳跃游戏

> 给定一个非负整数数组 nums ，你最初位于数组的第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。
>
> ```
> 输入：nums = [3,2,1,0,4]
> 输出：false
> 解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
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

### 7. 跳跃游戏 II

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

> 不管你从哪里起跳，步数加一之后指针移到最大的范围处

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
2. 从前向后遍历，遇到负数将其变为正数，同时 `K = K-1`
3. 如果 K 还大于 0，那么反复转变数值最小的元素，将 K 用完
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
            A[-1] *= (-1)**K #取A最后一个数只需要写-1
        return sum(A)
```





## 动态规划

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

我们发现，在计算 `a[6]` 和 `a[7]` 的时候，我们都用了 `a[8]`，也就是重复利用了结果！这就是动态规划！

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

另外，动态规划算法通常以自顶向上的方式解各子问题，而贪心算法通常自顶向下的方式进行。

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

==主要思路就是==：

1. 如果障碍物出现在终点，直接返回 0，永远也走不到
2. 如果障碍物在最右边或者最下边，该位置以上（包括该位置）或该位置以左（包括该位置）全部设置为 0
3. 如果障碍物出现在中间，将该位置的 dp 值设置为 0

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4ACEvjQUThueyLtEmtKZh1Oiak3icibK9TgKzrkoMpQVKQn5GbVLNYpVbwYfoIsoiaSniaKBibibJHYBkkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

自己独立思考完成！很棒！虽然提交了 4 次才成功，只要是一些边界条件和初始化没有好好审查。

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

### 9. 不同的二叉搜索树

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
3. 当 2 为头节点时，其左右子树都只有一个节点，布局是不是和 n 为 1 的时候只有一棵树的布局也是一样的啊！、

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
2. 确定递推公式：p [i] += dp [以 j 为头结点左子树节点数量] * dp [以 j 为头结点右子树节点数量] ，j 相当于是头结点的元素，从 1 遍历到 i 为止。所以递推公式：dp [i] += dp [j - 1] * dp [i - j]
3. dp 数组如何初始化：dp[0] = 1，从定义上讲，空节点也是一棵二叉树
4. 确定遍历顺序：从递归公式可以看出，节点数为 i 的状态是依靠 i 之前节点数的状态。那么遍历 i 里面每一个数作为头结点的状态，用 j 来遍历。
5. 举例推导 dp 数组：

<img src="https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5Eev9at7TiapAd6lv3wXnuJvgCSmmkVuU7xbK82cHl1X26iaD6ULLWI3eJTiaIo0yTj58YnIsnXuPxA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:80%;" />

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[-1]
```



## 参考资料

1. 微信公众号：代码随想录
1. https://github.com/youngyangyang04/leetcode-master/tree/master/problems

