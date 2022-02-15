
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
        

# 递归法
class Solution:
    def levelOrder(self, root):
        # 确定递归函数的参数和返回值
        res = []
        def helper(root, depth):
            # 递归的结束条件
            if not root: 
                return []
            # start the current depth
            if len(res) == depth: 
                # 添加一个新列表
                res.append([]) 
            # 确定单层递归的逻辑
            res[depth].append(root.val) 
            # process child nodes for the next depth
            if  root.left: 
                helper(root.left, depth + 1) 
            if  root.right: 
                helper(root.right, depth + 1)
        helper(root, 0)
        return res
