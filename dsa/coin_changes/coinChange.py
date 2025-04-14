"""
Input:
coins = [1,3,5]
amount: 11
Find the mininum number of coins required to sum up to amount

Output: 
3 (1, 5, 5, )

"""

class Solution:
    def coin_changes(self, coins, amount) -> int:
        dp = [amount+1] * (amount+1)
        dp[0] = 0  # must set first value

        # while loop
        a = 0
        while a < amount+1:
            for c in coins:
                if a-c >= 0:
                    print(a, c, dp[a])
                    dp[a] = min(dp[a], dp[a-c]+1)
            a += 1

        # for loop
        # for a in range(amount+1):
        #     for c in coins:
        #         if a - c >= 0:
        #             dp[a] = min(dp[a], dp[a-c]+1)  # prev used count
        return dp[amount]

coins = [1,3,5]
amount = 11
sol = Solution()
res = sol.coin_changes(coins, amount)
print(res)
