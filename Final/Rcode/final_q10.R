#!/usr/bin/env Rscript

# The digits in the One-vs-One Classifier.
d = 1; d_ = 5

# Complete Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

# Data Set revelant to the 1-vs-5 classifier.
select = (Y == d) | (Y == d_)
Xtrain = X[select, ]
Ytrain = ifelse((Y[select] == d), +1, -1)

# Complete Out-of-Sample Data.
Dout = as.matrix(read.table("http://www.amlbook.com/data/zip/features.test"))
Xout = cbind(Dout[, 'V2'], Dout[, 'V3'])
Yout = Dout[, 'V1']

# Out-of-Sample Data revelant to the 1-vs-5 classifier.
select = (Yout == d) | (Yout == d_)
Xtest  = Xout[select, ]
Ytest  = ifelse((Yout[select] == d), +1, -1)

# Non-Linear Transformation.
phi = function (x) {
    c(1, x[1], x[2], x[1] * x[2], x[1]^2, x[2]^2)
}

for (lambda in 10^c(-2, 0)) {

    # Non-Linear Transform.
    Ztrain = t(apply(Xtrain, 1, phi))

    # Final Weights as per Linear Regression with Regularization.
    inverse = solve(t(Ztrain) %*% Ztrain + lambda * diag(6))
    w_reg   = inverse %*% t(Ztrain) %*% Ytrain

    # Final Hypothesis used for Classification.
    g = function (x) { sign(t(w_reg) %*% phi(x)) }

    # In/Out of Sample Errors: Fraction of misclassified points.
    ein  = mean(apply(Xtrain, 1, g) != Ytrain)
    eout = mean(apply(Xtest,  1, g) != Ytest)

    print(paste("LAMBDA:", lambda,
                "ein:", round(ein, 6), "eout:", round(eout, 6)))
}
