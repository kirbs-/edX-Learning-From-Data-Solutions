#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 100

# Number of Out-of-Sample Test Points.
T = 1000

#
# Per Experiment: RBFk9
# Fraction of OUT-OF-SAMPLE Mismatches gRBFk9(x) != f(x).
#
EoutRBFk9 = vector(length = R)

#
# Per Experiment: RBFk12
# Fraction of OUT-OF-SAMPLE Mismatches gRBFk12(x) != f(x).
#
EoutRBFk12 = vector(length = R)

#
# Per Experiment: RBFk9
# Fraction of IN-SAMPLE Mismatches gRBFk9(x) != y.
#
EinRBFk9 = vector(length = R)

#
# Per Experiment: RBFk12
# Fraction of IN-SAMPLE Mismatches gRBFk12(x) != y.
#
EinRBFk12 = vector(length = R)

#
# Per Experiment: RBFk9
# Number of Iterations in Lloyd's Algorithm.
#
iterNk9 = vector(length = R)

#
# Per Experiment: RBFk12
# Number of Iterations in Lloyd's Algorithm.
#
iterNk12 = vector(length = R)

# Discard Counts.
discardRBFk9 = 0; discardRBFk12 = 0

# Observe X.
random_x = function (void) runif(2, -1, 1)

# Target Function.
f = function (x) sign(x[2] - x[1] + 0.25 * sin(pi * x[1]))

# Square of the Euclidean Distance between two points.
norms = function (u, x) { t(u - x) %*% (u - x) }

# Workaround for the "special" behavior of colMeans.
average = function (X) {
        if (is.null(nrow(X))) X else colMeans(X)
}

# Model: Radial Basis Functions.
rbf = function (X, Y, K, gamma = 1.5) {

    #
    # STEP 1: Calculate K centers using Lloyd's Algorithm.
    #
     
    # Set/Cluster Membership.
    Sx = vector(length = N)

    # Initialize Centers: Random points in X-Space.
    U = t(sapply(1:K, random_x))
 
    # Number of Iterations.
    iterN = 1

    while (1) {

        # "Fix Centers" : Calculate Set Membership for each x.
        Sx = apply(X, 1, function (x) {
                              which.min(apply(U, 1, norms, x))
                         })

        # Discard Run if any Set is Empty.
        if (any(!(1:K %in% Sx))) return(list('discard' = TRUE))
        
        # "Fix Set Membership" : Calculate Centers.
        Unew = t(sapply(1:K, function (k) { average(X[Sx == k, ]) }))

        # Termination Condition.
        if (all(U == Unew)) break

        # Prepare for next Iteration.
        U     = Unew
        iterN = iterN + 1
    }

    #
    # STEP 2: Calculate Weights that minimize Mean Squared Error.
    #
     
    # Prepare Matrix for Pseudo-Inverse.
    phi = matrix(nrow = N, ncol = K + 1)
    phi[, 1] = 1
    for (i in 1:N)
    for (j in 1:K)
        phi[i, j + 1] = exp(-gamma * norms(U[j, ], X[i, ]))

    # Final Weights.
    weights = solve(t(phi) %*% phi) %*% t(phi) %*% Y
    b       = weights[+1]
    w       = weights[-1]

    #
    # STEP 3: Create Final Hypothesis.
    #
    g = function (x) {
        s = b
        for (k in 1:K) s = s + w[k] * exp(-gamma * norms(U[k, ], x))
        sign(s)
    }

    return(list('discard' = FALSE, 'g' = g, 'iterN' = iterN))
}

# In-Sample Error Calculations.
ein = function (gRBFk9, gRBFk12, Xin, Yin) {

    einRBFk9  = mean(apply(Xin, 1, gRBFk9)  != Yin)
    einRBFk12 = mean(apply(Xin, 1, gRBFk12) != Yin)

    return(list('rbfk9' = einRBFk9, 'rbfk12' = einRBFk12))
}

# Out-of-Sample Error Calculations.
eout = function (f, gRBFk9, gRBFk12) {

    # Generate Out-of-Sample Data.
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)

    eoutRBFk9  = mean(apply(Xout, 1, gRBFk9)  != Yout)
    eoutRBFk12 = mean(apply(Xout, 1, gRBFk12) != Yout)

    return(list('rbfk9' = eoutRBFk9, 'rbfk12' = eoutRBFk12))
}

# Number of Effective Runs.
r = 0

while (1) {

    # Create Data Set.
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)

    # RBF with K = 9: Final Hypothesis.
    ret     = rbf(X, Y, 9)
    if (ret$discard) { discardRBFk9 = discardRBFk9 + 1; next }
    gRBFk9  = ret$g
    iterNk9 = ret$iterN

    # RBF with K = 12: Final Hypothesis.
    ret      = rbf(X, Y, 12)
    if (ret$discard) { discardRBFk12 = discardRBFk12 + 1; next }
    gRBFk12  = ret$g
    iterNk12 = ret$iterN

    r = r + 1

    # Out-of-Sample Errors.
    ret           = eout(f, gRBFk9, gRBFk12)
    EoutRBFk9[r]  = ret$rbfk9
    EoutRBFk12[r] = ret$rbfk12

    # In-Sample Errors.
    ret          = ein(gRBFk9, gRBFk12, X, Y)
    EinRBFk9[r]  = ret$rbfk9
    EinRBFk12[r] = ret$rbfk12

    # All Runs done.
    if (r == R) break
}

print(paste("Sample Size               =", N))
print(paste("Effective #Runs           =", R))
print(paste("RBF K9:  Discarded #Runs  =", discardRBFk9))
print(paste("RBF K12: Discarded #Runs  =", discardRBFk12))
print(paste("RBF K9:  Avg #Iterations  =", mean(iterNk9)))
print(paste("RBF K12: Avg #Iterations  =", mean(iterNk12)))
print(paste("RBF K9:  Avg Ein          =", mean(EinRBFk9)))
print(paste("RBF K12: Avg Ein          =", mean(EinRBFk12)))
print(paste("Test Data Size            =", T))
print(paste("RBF K9:  Avg Eout         =", mean(EoutRBFk9)))
print(paste("RBF K12: Avg Eout         =", mean(EoutRBFk12)))
print(paste("Frequency of Option [a]   =",
            sum((EinRBFk9 > EinRBFk12) & (EoutRBFk9 < EoutRBFk12))))
print(paste("Frequency of Option [b]   =",
            sum((EinRBFk9 < EinRBFk12) & (EoutRBFk9 > EoutRBFk12))))
print(paste("Frequency of Option [c]   =",
            sum((EinRBFk9 < EinRBFk12) & (EoutRBFk9 < EoutRBFk12))))
print(paste("Frequency of Option [d]   =",
            sum((EinRBFk9 > EinRBFk12) & (EoutRBFk9 > EoutRBFk12))))
print(paste("Frequency of Option [e]   =",
            sum((EinRBFk9  == EinRBFk12) &
                (EoutRBFk9 == EoutRBFk12))))
