#!/usr/bin/env Rscript

# Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

# Test Set.
Dout = as.matrix(read.table("http://www.amlbook.com/data/zip/features.test"))
Xout = cbind(Dout[, 'V2'], Dout[, 'V3'])
Yout = Dout[, 'V1']

# Normal Transformation.
normal = function (x) c(1, x)

# Non-Linear Transformation.
phi = function (x) {
    c(1, x[1], x[2], x[1] * x[2], x[1]^2, x[2]^2)
}

# Linear Regression with Regularization.
one_vs_rest = function (d, transform) {

    # Training Set.
    Xtrain = X
    Ytrain = ifelse((Y == d), +1, -1)

    if (transform) { trans = phi;    dim = 6 }
    else           { trans = normal; dim = 3 }

    # Z Space.
    Ztrain = t(apply(Xtrain, 1, trans))

    # Final Weights as per Linear Regression with Regularization.
    lambda  = 1
    inverse = solve(t(Ztrain) %*% Ztrain + lambda * diag(dim))
    w_reg   = inverse %*% t(Ztrain) %*% Ytrain

    # Final Hypothesis used for Classification.
    g = function (x) { sign(t(w_reg) %*% trans(x)) }

    # Testing Set.
    Xtest = Xout
    Ytest = ifelse((Yout == d), +1, -1)

    # In/Out of Sample Errors: Fraction of misclassified points.
    ein  = mean(apply(Xtrain, 1, g) != Ytrain)
    eout = mean(apply(Xtest,  1, g) != Ytest)

    print(paste(d, "vs all, transform =", transform,
                "ein:", round(ein, 6), "eout:", round(eout, 6)))
}

print("EXPERIMENT 1: LOWEST >>Ein<< without transformation")
for (d in 5:9) {
    one_vs_rest(d, transform = 0)
}

print("...")
print("EXPERIMENT 2: LOWEST >>Eout<< with transformation PHI")
for (d in 0:4) {
    one_vs_rest(d, transform = 1)
}

print("...")
print("EXPERIMENT 3: Performance with/without transformation")
for (d in 0:9) {
    one_vs_rest(d, transform = 1)
    one_vs_rest(d, transform = 0)
}
