from tqdm import tqdm
import random

N = 10
tries = int(1e6)

pages = [i+1 for i in range(N)]
succ = 0
for i in tqdm(range(tries)):
    random.shuffle(pages)
    a,b,x = tuple(pages[:3])

    res = None
    if random.randint(0,1)==0:
        if a<x:
            res = a<b
        else:
            res = b<a
    else:
        if b<x:
            res = b<a
        else:
            res = a<b
    succ += int(res)

print(succ/tries)