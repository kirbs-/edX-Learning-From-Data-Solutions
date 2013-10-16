from random import choice

def flip(): return choice([0,1])
def min_heads(coins):
    heads = 10
    min_flip = None
    for i, flips in enumerate(coins):
        if sum(flips) < heads:
            heads = sum(flips)
            min_flip = flips
    return min_flip
def frac(flips): return sum(flips) / 10.0

def run():
    coins = [[flip() for i in range(10)] for j in range(1000)]
    c1 = coins[0]
    crand = choice(coins)
    cmin = min_heads(coins)
    v1 = frac(c1)
    vrand = frac(crand)
    vmin = frac(cmin)
    return vmin

v = 0
n = 1000
for i in range(n):
    v += run()
print v / n