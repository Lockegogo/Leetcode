# Definition for a binary tree node.
class Node:
    def __init__(self, val):
        # val 是传入的值
        # 下面三个都是 Node 类的属性，初始化这些属性
        # 函数内新引入的变量均为局部变量，故无论是设置还是使用它的属性都得利用 self. 的方式
        # 如果不加 self. 这个变量就无法在 init 函数之外被使用
        self.left = None
        self.right = None
        self.val = val

    def insert(self, val):
    # 将新值与父节点进行比较
        if self.val:  # 非空
            if val < self.val:            #新值较小，放左边
                if self.left is None:       #若空，则新建插入节点
                    self.left = Node(val)
                else:                       #否则，递归往下查找
                    self.left.insert(val)
            elif val > self.val:          #新值较大，放右边
                if self.right is None:      #若空，则新建插入节点
                    self.right = Node(val)
                else:                       #否则，递归往下查找
                    self.right.insert(val)
        else:
            self.val = val                

    # 打印这棵树，中序遍历
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.val),
        if self.right:
            self.right.PrintTree()

# 使用insert方法添加节点
root = Node(12)
root.insert(6)
root.insert(14)
root.insert(3)


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
    def invertTree(self, root):
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