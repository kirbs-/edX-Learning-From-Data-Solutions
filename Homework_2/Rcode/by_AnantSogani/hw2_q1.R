#!/usr/bin/env Rscript

#
# Number of Runs of the Experiment.
# Set to 100000 for the formal answer.
#
R = 100

# Sample Size.
N = 10

# Number of Samples.
M = 1000

#
# Vector Initialization to Null Vector.
# V: Each entry V[i] is fraction of Heads from N flips of Coin i.
#
V = 0

#
# Vector Initialization to Null Vector.
# v: Index 0 = V[1], Index 1 = V[rand], Index 2 = Vmin.
#
v = 0

for (i in 1:R) {

    # 0 for Tails, 1 for Heads.
    for (i in 1:M) {
        V[i] = mean(sample(0:1, N, replace=TRUE))
    }

    v = v + c(V[1], V[sample(1:M, 1)], min(V))
}

v = v / R

print("v1  vrand vmin");
print(v)
