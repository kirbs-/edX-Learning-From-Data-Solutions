#!/usr/bin/env Rscript

# Input Data Set.
D = as.matrix(read.table("http://work.caltech.edu/data/in.dta"))

# X Part of Input.
X = cbind(D[, 'V1'], D[, 'V2'])

# Y Part of Input.
Y = D[, 'V3']

# Out-of-Sample Data.
Dout = as.matrix(read.table("http://work.caltech.edu/data/out.dta"))

# Xout Part. 
Xout = cbind(Dout[, 'V1'], Dout[, 'V2'])

# Yout Part.
Yout = Dout[, 'V3']

# Family of Non-Linear Transformations.
phi = function (x, k) {
    phi8 = c(1, x[1], x[2], x[1]^2, x[2]^2, x[1] * x[2],
             abs(x[1] - x[2]), abs(x[1] + x[2]))
    k = k + 1
    phi8[1:k]
}

experiment = function (Xtrain, Ytrain, Xval, Yval) {

    # Obtain g- and Eval for 5 models.
    for (k in c(3, 4, 5, 6, 7)) {

        # Non-Linear Transform.
        Z = t(apply(Xtrain, 1, phi, k))

        # Final Weights.
        wlin = solve(t(Z) %*% Z) %*% t(Z) %*% Ytrain

        # Final Hypothesis used for Classification.
        g_ = function (x) { sign(t(wlin) %*% phi(x, k)) }

        # Validation Error.
        Eval = mean(apply(Xval, 1, g_) != Yval)

        # Out-of-Sample Error.
        Eout = mean(apply(Xout, 1, g_) != Yout)

        print("---------------------------------------------")
        print(paste("Model phi0 to phi", k, sep = ""))
        print(paste("Validation Error    =", Eval))
        print(paste("Out-of-Sample Error =", Eout))
    }
}

print("EXPERIMENT #1:")
print("Training Set (N - K) = 25, Validation Set (K) = 10")
experiment(X[1:25, ], Y[1:25], X[26:35, ], Y[26:35])

print("**************************************************")

print("EXPERIMENT #2:")
print("Training Set (N - K) = 10, Validation Set (K) = 25")
experiment(X[26:35, ], Y[26:35], X[1:25, ], Y[1:25])
