# 二叉树的构建
from unittest import result


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
            if data < self.data:  # 新值较小，放左边
                if self.left is None:  # 若空，则新建插入节点
                    self.left = Node(data)
                else:  # 否则，递归往下查找
                    self.left.insert(data)
            elif data > self.data:  # 新值较大，放右边
                if self.right is None:  # 若空，则新建插入节点
                    self.right = Node(data)
                else:  # 否则，递归往下查找
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
