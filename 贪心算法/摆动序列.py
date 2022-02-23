class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        # 题目里 nums 长度大于等于 1，当长度为 1 时，其实到不了 for 循环里去，所以不用考虑 nums长度
        preC, curC, res = 0, 0, 1
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            # 差值为 0 时，不算摆动
            if curC * preC <= 0 and curC != 0:
                res += 1
                # 如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
                preC = curC
        return res
