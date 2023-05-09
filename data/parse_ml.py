import sys

data = dict()
with open(sys.argv[1], 'r') as f:
    for l in f:
        uid, iid, score, time = l.split()
        if uid not in data:
            data[uid] = []
        data[uid].append(iid)

for u, lst in data.items():
    print(f'{u} {" ".join(lst)}')
'''
data format of ml100K
1	17	3	875073198
1	47	4	875072125
1	64	5	875072404
1	90	4	878542300
1	92	3	876892425
user_id, movie_id, rating, timestamp

data format of pytorch github
0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126
user_id, item_id
'''
