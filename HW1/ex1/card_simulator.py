import random
from tqdm import tqdm

# Build the probability space
values = [str(i) for i in range(1,11)]
values += ["J", "Q", "K"]
straight = values + ["1"]
suits = ["H", "D", "C", "S"]

omega = [i+j for i in suits for j in values]
print('event space: ', omega, sep = '\n')

#define the experiment's parameters
EXPERIMENT = 4
N = 0

if EXPERIMENT == 0:
    N = 4
elif EXPERIMENT == 1:
    N = 7
elif EXPERIMENT == 2 or EXPERIMENT == 3:
    N = 3
elif EXPERIMENT == 4:
    N = 5

num_tries = int(1e9)
successes = 0

for _ in tqdm(range(num_tries)):
        
    #extract N cards without repetition
    random.shuffle(omega)
    extracted = omega[:N]

    #perform the desired experiment
    res = None
    if EXPERIMENT == 0:
        res = False
        for e in extracted:
            if e[0] == 'C':
                res = True
                break

    elif EXPERIMENT == 1:
        res = False
        for e in extracted:
            if e[0] == 'C' and not res:
                res = True
            elif e[0] == 'C' and res:
                res = False
                break

    elif EXPERIMENT == 2:
        s = extracted[0][0]
        res = True
        for e in extracted[1:]:
            if e[0]!=s:
                res = False
                break

    elif EXPERIMENT == 3:
        res = True
        for e in extracted:
            if e[1:] != str(7):
                res = False
                break

    elif EXPERIMENT == 4:
        res = True
        s = extracted[0][0]
        ss = True

        #sort extracted to recover the sequence
        extracted = sorted(extracted, key=lambda x: values.index(x[1:]))

        #swap ace and 10 to have 10->A straight
        if extracted[0][1:] == '1' and extracted[1][1:] == '10':
            e = extracted.pop(0)
            extracted.append(e)


        ind = values.index(extracted[0][1:])
        for e in extracted[1:]:
            ind += 1
            ss = ss and s == e[0]
            if ind >= len(straight) or e[1:] != straight[ind]:
                res = False
                break
        if res and ss:
            res = False
    
    #if res: print(extracted)

    successes += int(res)

print(successes/num_tries)