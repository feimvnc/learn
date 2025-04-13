
'''
Given an array, find the max sum of subarray.
Input: [2,-3,4,-1,2,3,-7,-2]
Output: 8
Explanation: [4,-1,2,3] = 8

array - subarray - sum -> prefix sum 
'''

from typing import List 

class Solution:

    def max_subarray(self, nums: List[int]) -> int:
        if not nums or len(nums) == 0:
            return -1
        
        maxx = 0
        prefix_sum = 0
        min_prefix_sum = 0

        i = 0
        while i < len(nums):
            prefix_sum += arr[i]
            maxx = max(maxx, prefix_sum - min_prefix_sum)
            min_prefix_sum = min(min_prefix_sum, prefix_sum)
            print(prefix_sum, min_prefix_sum, maxx)
            i += 1

        return maxx

arr = [2,-3,4,-1,2,3,-7,-2]
sol = Solution()
res = sol.max_subarray(arr)
print(res)
