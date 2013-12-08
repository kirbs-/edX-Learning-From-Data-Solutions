#!/usr/bin/env Rscript

# Input.
X = matrix(c(1, 0, 0, 1, 0, -1, -1, 0, 0, 2, 0, -2, -2, 0),
           ncol = 2, byrow = TRUE)

# Labels.
Y = c(-1, -1, -1, 1, 1, 1, 1)

# Sample Size.
N = length(Y)

# Non-Linear Transform.
Z = cbind(X[, 2]^2 - 2 * X[, 1] - 1,
          X[, 1]^2 - 2 * X[, 2] + 1)

#
# METHOD 1: Geometry.
#
name = "margin.png"
png(name)

# plot(X[, 1], X[, 2], xlab = "x1", ylab = "x2",
     # main = "X",
     # col = 3 + Y, pch = 20)

plot(Z[, 1], Z[, 2], xlab = "z1", ylab = "z2",
     main = "METHOD 1: Maximize Margin using Geometry",
     col = 3 + Y, pch = 15)

dev.off()
browseURL(name)

#
# METHOD 2: QP Package.
#
library(LowRankQP)

# Create the Quadratic Programing Parameters for LowRankQP().
Vmat = (Y * Z) %*% t(Y * Z)
dvec = rep(-1, N)
Amat = t(Y)
bvec = 0
uvec = rep(10000, N)

# Minimize the Quadratic Function!
solution = LowRankQP(Vmat, dvec, Amat, bvec, uvec, method = "LU")

# Obtain Alphas.
a = c(zapsmall(solution$alpha))

# Weights.
w = colSums(a * Y * Z)

# Workaround for "special" behavior of sample().
resample = function (x, ...) x[sample.int(length(x), ...)]

# Bias/Threshold Term. Take any one, all are equal.
b = resample(((1 / Y) - (Z %*% w))[a > 0], 1)

print("METHOD 2: Quadratic Programming")
print(paste("#SV's =", sum(a > 0)))
print("Weights (w1, w2, b):")
print(c(w, b))
