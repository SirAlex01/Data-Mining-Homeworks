from tqdm import tqdm
import random

N = 7
tries = int(1e6)

pages = [i for i in range(N)]
succ = 0
for i in tqdm(range(tries)):
    random.shuffle(pages)
    a,b = tuple(pages[:2])
    x = pages[random.randint(0, len(pages)-1)]

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

    #print(a,b,x,res)

print(succ/tries)