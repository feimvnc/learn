from collections import Counter

class Solution: 

    def repeated_string(self, s, n, ch) -> int:
        count = 0
        new_s = ''
        new_s_len = 0

        while new_s_len <= n:
            new_s = new_s + s
            new_s_len = len(new_s)
        new_s = new_s[:n]
        print(new_s, new_s_len)
        
        for c in new_s:
            if c == ch:
                count += 1

        return count

    def repeated_string_counter(self, s, n, ch) -> int:
        new_s = ''
        new_s_len = 0

        while new_s_len <= n:
            new_s = new_s + s
            new_s_len = len(new_s)
        new_s = new_s[:n]
        print(new_s, new_s_len)
        counter = Counter(new_s)
        return counter[ch]

    def repeated_string_single_char(self, s, n, ch) -> int:
        count = 0
        new_s = ''
        while len(new_s) < n:
            for e in s:
                if e == ch:
                    count += 1
                new_s += e
                if len(new_s) == n:
                    break
        print(new_s, len(new_s))
        return count

    def repeated_string_reduce(self, s, n, ch) -> int:
        count = 0
        pos = 0
        new_s = ''
        while len(new_s) < n:
            new_s += s[pos]
            if s[pos] == ch:
                count += 1
            pos = len(new_s) % len(s)
        print(new_s, n)
        return count
            
            


s = "abc"
n = 10
c = "a"

sol = Solution()
res = sol.repeated_string(s, n, c)
print(res)

res = sol.repeated_string_counter(s, n, c)
print(res)

res = sol.repeated_string_single_char(s, n, c)
print(res)

res = sol.repeated_string_reduce(s, n, c)
print(res)