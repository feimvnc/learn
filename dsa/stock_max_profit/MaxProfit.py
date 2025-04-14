"""
Given an array, find the max difference.
Input: [102,99,103,102,104,107,100,98]
Output: 8
Explanation: [99,107] = 8

minimun_point - max_point -> max profit

price                       102,99,103,102,104,107,100,98
before minimun  infiniti    102,99, 99, 99, 99, 99, 99, 98
max profit      0             0  0   4   4   5   8   8   8

O(N) - single for loop
O(1) - space 
"""

import sys
from typing import List 

class Solution:

    def max_profit(self, prices: List[int]) -> int:
        max_profit = 0
        prev_min = sys.maxsize  # set max value

        i = 0
        while i < len(prices):
            # find running minimun so far
            prev_min = prices[i] if prices[i] < prev_min else prev_min
            
            # find running max so far
            max_profit = prices[i] - prev_min if prices[i] - prev_min > max_profit else max_profit
            i += 1
            print(prev_min, max_profit)

        return max_profit 
    

prices = [102,99,103,102,104,107,100,98]
sol = Solution()
res = sol.max_profit(prices)
print(res)
