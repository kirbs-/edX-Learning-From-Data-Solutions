#!/usr/bin/env Rscript

# Base Expression in Error Function.
base = function (x) {
    u = x[1]
    v = x[2]
    (u * exp(v) - 2 * v * exp(-u))
}

# Error Function that needs to be minimized.
E = function (x) base(x)^2

# Gradient of the Error Function.
gradE = function (x) {
    cmn = 2 * base(x) 
    u = x[1]
    v = x[2]
    dE_du = cmn * (exp(v) + 2 * v * exp(-u))
    dE_dv = cmn * (u * exp(v) - 2 * exp(-u))
    c(dE_du, dE_dv)
}

# Initialization: Input Vector to Error Function.
x = c(1, 1)

# Learning Rate.
eta = 0.1

# Iterations Counter.
iter = 0

print("Initialization:")
print(paste("E      =", E(x)))
print(paste("(u, v) =", x[1], x[2]))
print(paste("Eta    =", eta))

# Run GD.
while (1) {

    # Terminating Condition.
    if (E(x) < 10^(-14)) break

    # Descend!
    p = gradE(x); q = sqrt(sum(p^2))
    print(p); print(p/q);
    x = x - eta * gradE(x)

    # Counter.
    iter = iter + 1

    print("------------------------------")
    print(paste("After", iter, "iterations"))
    print(paste("E      =", E(x)))
    print(paste("(u, v) =", x[1], x[2]))
}
