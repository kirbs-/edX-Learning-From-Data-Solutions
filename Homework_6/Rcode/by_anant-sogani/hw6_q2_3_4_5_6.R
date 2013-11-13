#!/usr/bin/env Rscript

# Training: In-Sample Data.
D = as.matrix(read.table("http://work.caltech.edu/data/in.dta"))

# X Part of Input.
X = cbind(D[, 'V1'], D[, 'V2'])

# Y Part of Input.
Y = D[, 'V3']

# Testing: Out-of-Sample Data.
Dout = as.matrix(read.table("http://work.caltech.edu/data/out.dta")) 

# Xout Part. 
Xout = cbind(Dout[, 'V1'], Dout[, 'V2'])

# Yout Part.
Yout = Dout[, 'V3']

# Non-Linear Transformation.
phi = function (x) {
    c(1, x[1], x[2], x[1]^2, x[2]^2, x[1] * x[2],
      abs(x[1] - x[2]), abs(x[1] + x[2]))
}

# X in Z Space.
Z = t(apply(X, 1, phi))

# See the effects of varying Lambda.
k = c(-Inf, -3, -2, -1, 0, 1, 2, 3)

for (lambda in 10^k) {

    # Final Weights as per Linear Regression with Weight Decay.
    w_reg = solve(t(Z) %*% Z + lambda * diag(8)) %*% t(Z) %*% Y

    # Final Hypothesis used for Classification.
    g = function (x) { sign(t(w_reg) %*% phi(x)) }

    # In/Out of Sample Errors: Fraction of misclassified points.
    Ein  = mean(apply(X,    1, g) != Y)
    Eout = mean(apply(Xout, 1, g) != Yout)

    # Results.
    print("---------------------------------------------")
    print(paste("LAMBDA                    =", lambda))
    print(paste("In-Sample Error           =", Ein))
    print(paste("Out-of-Sample Error       =", Eout))
}
