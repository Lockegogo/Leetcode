"""
给你两个单链表的头节点 headA 和 headB，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null。

示例：
输入：listA = [4,1,8,4,5], listB = [5,0,1,8,4,5]
输出：Reference of the node with value = 8
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        countA = 0
        countB = 0
        curA = headA
        curB = headB

        while curA != None:
            curA = curA.next
            countA += 1

        while curB != None:
            curB = curB.next
            countB += 1

        # 让 curA 指向长链表
        if countA < countB:
            curA = headB
            curB = headA
        else:
            curA = headA
            curB = headB

        gap = abs(countA-countB)
        for i in range(gap):
            # 末尾对齐
            curA = curA.next

        while curA != None:
            if curA == curB:
                return curA
            curA = curA.next
            curB = curB.next

# 快慢法则
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
