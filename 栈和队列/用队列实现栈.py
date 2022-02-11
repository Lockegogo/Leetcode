
from collections import deque


class MyStack:

    def __init__(self):
        """
        1. python 可以用 list 实现队列，但是在使用 pop(0) 时时间复杂度为 O(n)，因此这里可以使用双向队列
        2. 我们保证只执行 popleft() 和 append()，因为 deque 可以用索引访问
        in：存所有数据
        out：仅在 pop 的时候会用到
        """
        self.queue_in = deque()
        self.queue_out = deque()

    def push(self, x: int) -> None:
        self.queue_in.append(x)

    def pop(self) -> int:
        """
        函数功能：移除并返回栈顶元素

        1. 首先确认不空
        2. 因为队列的特殊性先进先出，所以我们只有在 pop() 的时候才会使用 queue_out
        3. 先把 queue_in 中的所有元素（除了最后一个）依次放入 queue_out
        4. 交换 in 和 out，此时 out 里只有一个元素
        5. 把 out 中的 pop 出来，即是原队列中的最后一个

        tip：这不能像栈实现队列一样，因为另一个queue也是FIFO，如果执行pop()它不能像
        stack一样从另一个pop()，所以干脆in只用来存数据，pop()的时候两个进行交换
        """
        if self.empty():
            return None
        for i in range(len(self.queue_in)-1):
            # 弹出最左边的元素，也就是先加进来的元素
            self.queue_out.append(self.queue_in.popleft())
        # 交换 in 和 out
        self.queue_in, self.queue_out = self.queue_out, self.queue_in
        # 交换之后，把现在的 out 之前的 in 留下的最后一个元素弹出即可
        return self.queue_out.popleft()

    def top(self) -> int:
        """
        功能：返回栈顶元素

        1. 首先确认不空
        2. 仅有in会存放数据，所以返回第一个即可
        """
        if self.empty():
            return None
        
        return self.queue_in[-1]

    def empty(self) -> bool:
        """
        因为只有 in 存了数据，只要判断 in 是不是有数据即可
        """
        return len(self.queue_in) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
