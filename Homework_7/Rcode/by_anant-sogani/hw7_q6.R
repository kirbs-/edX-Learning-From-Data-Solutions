#!/usr/bin/env Rscript

# Number of Monte Carlo trials.
n = 1000000

# Expected Value of e1.
print(mean(runif(n)))

# Expected Value of e2.
print(mean(runif(n)))

# Expected Value of e = min(e1, e2).
print(mean(pmin(runif(n), runif(n))))
