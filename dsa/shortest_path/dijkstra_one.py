import heapq

class Solution:
    def calculate_distances(self, g, start):
        res = {v: float('infinity') for v in g}
        res[start] = 0
        pq = [(0, start)]

        while len(pq) > 0:
            print(pq)
            cur_res, cur_v = heapq.heappop(pq)
            if cur_res > res[cur_v]:
                continue
            for neig, weight in g[cur_v].items():
                temp = cur_res + weight 
                if temp < res[neig]:
                    res[neig] = temp
                    heapq.heappush(pq, (temp, neig))
        return res


g = {
    'u': {'v':2, 'w':5, 'x':1},
    'v': {'u':2, 'x':2, 'w':3},
    'w': {'v':3, 'u':5, 'x':3, 'y':1, 'z':5},
    'x': {'u':1, 'v':2, 'w':3, 'y':1},
    'y': {'x':1, 'w':1, 'z':1},
    'z': {'w':5, 'y':1},
}    

s = Solution()
res = s.calculate_distances(g, 'x')
print(res)


"""
python3 dijkstra_one.py 
[(0, 'x')]
[(1, 'u'), (1, 'y'), (3, 'w'), (2, 'v')]
[(1, 'y'), (2, 'v'), (3, 'w')]
[(2, 'v'), (2, 'z'), (2, 'w'), (3, 'w')]
[(2, 'w'), (2, 'z'), (3, 'w')]
[(2, 'z'), (3, 'w')]
[(3, 'w')]
{'u': 1, 'v': 2, 'w': 2, 'x': 0, 'y': 1, 'z': 2}

"""