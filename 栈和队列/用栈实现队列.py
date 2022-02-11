"""
请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：
1. void push(int x) 将元素 x 推到队列的末尾
2. int pop() 从队列的开头移除并返回元素
3. int peek() 返回队列开头的元素
4. boolean empty() 如果队列为空，返回 true ；否则，返回 false


说明：
你只能使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/implement-queue-using-stacks
"""


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



# Your MyQueue object will be instantiated and called as such:

obj = MyQueue()
obj.push(5)
obj.push(8)
param_2 = obj.pop()
param_3 = obj.peek()
param_4 = obj.empty()
