#!/usr/bin/env Rscript

# Input.
X = matrix(c(1, 0, 0, 1, 0, -1, -1, 0, 0, 2, 0, -2, -2, 0),
           ncol = 2, byrow = TRUE)

# Labels.
Y = c(-1, -1, -1, 1, 1, 1, 1)

# Sample Size.
N = length(Y)

#
# METHOD 1: QP Package.
#
library(LowRankQP)

# Kernel.
K = function (x, x_) (1 + t(x) %*% x_)^2

# Create the Quadratic Programing Parameters for LowRankQP().
Vmat = matrix(nrow = N, ncol = N)
for (i in 1:N)
for (j in 1:N)
    Vmat[i, j] = Y[i] * Y[j] * K(X[i, ], X[j, ])

dvec = rep(-1, N)
Amat = t(Y)
bvec = 0
uvec = rep(10000, N)

# Minimize the Quadratic Function!
solution = LowRankQP(Vmat, dvec, Amat, bvec, uvec, method = "LU")

# Obtain Alphas.
a = c(zapsmall(solution$alpha))

# Number of Support Vectors.
print("METHOD 1: Quadratic Programming")
print(paste("#SV's =", sum(a > 0)))

#
# METHOD 2: SVM Package.
#
library(e1071)

model = svm(X, Y, scale = FALSE,
            type = "C-classification", cost = 10^6,
            kernel = "polynomial",
            gamma = 1, coef0 = 1, degree = 2)

# Number of Support Vectors. 
print("METHOD 2: SVM Package")
print(paste("#SV's =", nrow(model$SV)))
