from math import factorial

def multinomial_coeff(n, ks):
    numerator = factorial(n)
    denominator = 1
    for k in ks:
        denominator *= factorial(k)
    return numerator // denominator


EXERCISE = 4
n = 11
p = 0.3
res = None
if EXERCISE == 3:
    assert(n % 2 == 0)
    res = multinomial_coeff(n,[n // 2]*2) * (p**n) * ((1-p)**(multinomial_coeff(n, [2,n-2])-n)) * (factorial(n//2-1)**2) / 8
elif EXERCISE == 4:
    res = 0.0
    for i in range(3, n-2):
        for j in range(3, n-i+1):
            res += multinomial_coeff(n,[i,j,n-i-j]) * factorial(i-1) * factorial(j-1) * p**(i+j) * (1-p)**(multinomial_coeff(n, [2,n-2])-i-j) / 8 
print(res)


