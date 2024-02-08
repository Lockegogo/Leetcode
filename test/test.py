def func(s, k):
    calc = lambda x: 1 if x == 1 else (2 if x < 10 else (3 if x < 100 else 4))

    n = len(s)
    f = [[10**9] * (k + 1) for _ in range(n + 1)]
    f[0][0] = 0

    for i in range(1, n + 1):
        for j in range(min(k, i) + 1):
            if j > 0:
                f[i][j] = f[i - 1][j - 1]
            same = diff = 0
            for i0 in range(i, 0, -1):
                if s[i0 - 1] == s[i - 1]:
                    same += 1
                    f[i][j] = min(f[i][j], f[i0 - 1][j - diff] + calc(same))
                else:
                    diff += 1
                    if diff > j:
                        break
    return f[n][k]


# arr = list(s)
# # ord()  chr()
# record = [0] * 26
# for i in range(len(s)):
#     record[ord(s[i]) - ord("a")] += 1
# # print(record)

# record.sort()

# length = 0
# for i in range(26):
#     if record[i] > 0:
#         length += 1

# for i in range(26):
#     if record[i] > 0 and n - record[i] >= 0:
#         n = n - record[i]
#         length -= 1

# print(length * 2)
