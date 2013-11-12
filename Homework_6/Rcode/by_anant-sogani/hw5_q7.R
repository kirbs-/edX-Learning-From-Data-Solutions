#!/usr/bin/env Rscript

# Base Expression in Error Function.
base = function (u, v) (u * exp(v) - 2 * v * exp(-u))

# Error Function that needs to be minimized.
E = function (u, v) base(u, v)^2

# Gradient of the Error Function w.r.t u.
dE_du = function (u, v) {
    2 * base(u, v) * (exp(v) + 2 * v * exp(-u))
}

# Gradient of the Error Function w.r.t v.
dE_dv = function (u, v) {
    2 * base(u, v) * (u * exp(v) - 2 * exp(-u))
}

# Initialization: Input Vector to Error Function.
u = 1; v = 1

# Learning Rate.
eta = 0.1

# Iterations Counter.
iter = 1

print("Initialization:")
print(paste("Eta    =", eta))
print(paste("(u, v) =", u, v))
print(paste("E      =", E(u, v)))

# Run GD.
while (iter <= 15) {

    print("-----------------------------")
    print(paste("In iteration", iter, ":"))

    # Co-ordinate Descent!
    u = u - eta * dE_du(u, v)
    print("After u-Descent")
    print(paste("(u, v) =", u, v))
    print(paste("E      =", E(u, v)))

    v = v - eta * dE_dv(u, v)
    print("After v-Descent")
    print(paste("(u, v) =", u, v))
    print(paste("E      =", E(u, v)))

    # Counter.
    iter = iter + 1
}
