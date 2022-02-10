Leetcode 刷题笔记

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

> 原地移除元素，并返回移除后数组的新长度。要求不适用额外的数组空间，必须使用 $O(1)$ 额外空间并原地修改输入数组。你不需要考虑数组中超出新长度后面的元素。

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

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291723017.png" alt="image-20220129172318972" style="zoom: 80%;" />

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

> 给你一个按==非递减顺序==排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按非递减顺序排序。
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

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201291931420.png" alt="image-20220129193153380" style="zoom:80%;" />

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

该解法超出时间限制了

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

所谓滑动窗口，就是不断调节子序列的起始位置和终止位置，从而得出我们想要的结果。

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201292032148.png" alt="image-20220129203228105" style="zoom:67%;" />

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

==为什么时间复杂度是 $O(n^2)$==？

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

> 对难度中等的题目，完全没思路，我天。

这道题目可以说在面试中出现频率较高的题目，不涉及什么算法，就是模拟过程，但十分考察对代码的掌控能力。

==循环不变量原则==，模拟顺时针画矩阵的过程：

- 填充上行从左到右
- 填充右列从上到下
- 填充下行从右到左
- 填充左列从下到上

由外向内一圈一圈这么画下去。

可以发现这里的边界条件非常多，在一个循环中，如此多的边界条件，吐过不按照固定规则来遍历，那就是一进循环深似海，从此 offer 是路人。

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

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202201302119374.png" alt="image-20220130211934331" style="zoom: 80%;" />

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

==算法==：我们求出两个链表的长度，并求出两个链表长度的差值，然后让 `curA` 移动到和 `curB` 末尾对齐的位置：

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
> 那么 fast 指针走到环入口 3 的时候，已经走了 $k+n$ 个节点，slow 相应走了 $(k+n)/2$ 个节点，因为 $k$ 小于 $n$，所以 $(k+n)/2$ 一定小于 $n$，这说明 slow  一定没有走到环入口 3，而 fast 已经到环入口 3 了，也就是**在 slow 开始走的那一环已经和 fast 相遇了**。

因为 fast 指针是一步走两个节点，slow 指针一步走一个节点， 所以 fast 指针走过的节点数 = slow 指针走过的节点数 * 2：
$$
\begin{aligned}
(x+y) * 2&=x+y+n(y+z)\\
x+y&=n(y+z)
\end{aligned}
$$
因为要找环形的入口，那么要求的是 x，因为 x 表示 头结点到 环形入口节点的的距离。整理如下：
$$
x=(n-1)(y+z)+z
$$
这就意味着，**从头结点出发一个指针，从相遇节点也出发一个指针，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是 环形入口的节点**。

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

下面给出用字典做赎金信的代码：

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
3. 如果 `nums [i] + nums [left] + nums [right] > 0`  就说明此时三数之和大了，因为数组是排序后的，所以 right 就应该向左移动，这样才能让三数之和小一些
4. 如果 `nums [i] + nums [left] + nums [right] < 0` 说明此时三数之和小了，left 就向右移动，才能让三数之和大一些，直到 left 与 right 相遇为止

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

> 给你一个由 n 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且不重复的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：
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
    from functools import reduce
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

<img src="https://gitee.com/lockegogo/markdown_photo/raw/master/202202021219402.png" alt="image-20220202121911348" style="zoom: 80%;" />

> **从前向后填充可以吗？**
>
> 从前向后填充就是 $O(n^2)$ 的算法，因为每次添加元素都要将添加元素之后的左右元素向后移动。

其实很多数组填充类的问题，都可以先预先给数组扩容带填充后的大小，然后在从后向前进行操作。

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

1. 初始化
2. 处理前后缀不相同的情况
3. 处理前后缀相同的情况

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





## 参考资料

1. 微信公众号：代码随想录
1. https://github.com/youngyangyang04/leetcode-master/tree/master/problems

