# Formular
# Height = (1 << ((n >> 1) + 1)) - 1 << n % 2
"""
python3 UtopianTree.py 
5 > 14
0 1
1 2
2 3
3 6
4 7
5 14
5 > 14
"""

class Solution:

    def utopian_tree(self, n):
        return (1 << ((n >> 1) + 1)) - 1 << n % 2

    def utopian_tree_brute_force(self, n):
        height = 1
        start = 0
        print(start, height)

        for i in range(1, n+1):
            if i % 2 == 1:  # odd
                height *= 2
            else:
                height += 1
            print(i, height)
        return height

    def utopian_tree_spring(self, n):
        spring = (2 ** (n//2+1)) - 1
        if n % 2 == 0:
            return spring
        else:
            return (2 * spring)

sol = Solution()
n = 5
res = sol.utopian_tree(n)
print("{} > {}".format(n, res))

res = sol.utopian_tree_brute_force(n)
print("{} > {}".format(n, res))
n=5
res = sol.utopian_tree_spring(n)
print("{} > {}".format(n, res))
