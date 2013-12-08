#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 100

# Number of Out-of-Sample Test Points.
T = 1000

#
# Per Experiment: RBFg1_5
# Fraction of OUT-OF-SAMPLE Mismatches gRBFg1_5(x) != f(x).
#
EoutRBFg1_5 = vector(length = R)

#
# Per Experiment: RBFg2
# Fraction of OUT-OF-SAMPLE Mismatches gRBFg2(x) != f(x).
#
EoutRBFg2 = vector(length = R)

#
# Per Experiment: RBFg1_5
# Fraction of IN-SAMPLE Mismatches gRBFg1_5(x) != y.
#
EinRBFg1_5 = vector(length = R)

#
# Per Experiment: RBFg2
# Fraction of IN-SAMPLE Mismatches gRBFg2(x) != y.
#
EinRBFg2 = vector(length = R)

#
# Per Experiment: RBFg1_5
# Number of Iterations in Lloyd's Algorithm.
#
iterNg1_5 = vector(length = R)

#
# Per Experiment: RBFg2
# Number of Iterations in Lloyd's Algorithm.
#
iterNg2 = vector(length = R)

# Discard Counts.
discardRBFg1_5 = 0; discardRBFg2 = 0

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
rbf = function (X, Y, gamma, K = 9) {

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
ein = function (gRBFg1_5, gRBFg2, Xin, Yin) {

    einRBFg1_5 = mean(apply(Xin, 1, gRBFg1_5) != Yin)
    einRBFg2   = mean(apply(Xin, 1, gRBFg2)   != Yin)

    return(list('rbfg1_5' = einRBFg1_5, 'rbfg2' = einRBFg2))
}

# Out-of-Sample Error Calculations.
eout = function (f, gRBFg1_5, gRBFg2) {

    # Generate Out-of-Sample Data.
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)

    eoutRBFg1_5 = mean(apply(Xout, 1, gRBFg1_5) != Yout)
    eoutRBFg2   = mean(apply(Xout, 1, gRBFg2)   != Yout)

    return(list('rbfg1_5' = eoutRBFg1_5, 'rbfg2' = eoutRBFg2))
}

# Number of Effective Runs.
r = 0

while (1) {

    # Create Data Set.
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)

    # RBF with Gamma = 1.5: Final Hypothesis.
    ret       = rbf(X, Y, gamma = 1.5)
    if (ret$discard) { discardRBFg1_5 = discardRBFg1_5 + 1; next }
    gRBFg1_5  = ret$g
    iterNg1_5 = ret$iterN

    # RBF with Gamma = 2: Final Hypothesis.
    ret     = rbf(X, Y, gamma = 2)
    if (ret$discard) { discardRBFg2 = discardRBFg2 + 1; next }
    gRBFg2  = ret$g
    iterNg2 = ret$iterN

    r = r + 1

    # Out-of-Sample Errors.
    ret            = eout(f, gRBFg1_5, gRBFg2)
    EoutRBFg1_5[r] = ret$rbfg1_5
    EoutRBFg2[r]   = ret$rbfg2

    # In-Sample Errors.
    ret           = ein(gRBFg1_5, gRBFg2, X, Y)
    EinRBFg1_5[r] = ret$rbfg1_5
    EinRBFg2[r]   = ret$rbfg2

    # All Runs done.
    if (r == R) break
}

print(paste("Sample Size               =", N))
print(paste("Effective #Runs           =", R))
print(paste("RBF G1_5: Discarded #Runs =", discardRBFg1_5))
print(paste("RBF G2:   Discarded #Runs =", discardRBFg2))
print(paste("RBF G1_5: Avg #Iterations =", mean(iterNg1_5)))
print(paste("RBF G2:   Avg #Iterations =", mean(iterNg2)))
print(paste("RBF G1_5: Avg Ein         =", mean(EinRBFg1_5)))
print(paste("RBF G2:   Avg Ein         =", mean(EinRBFg2)))
print(paste("Test Data Size            =", T))
print(paste("RBF G1_5: Avg Eout        =", mean(EoutRBFg1_5)))
print(paste("RBF G2:   Avg Eout        =", mean(EoutRBFg2)))
print(paste("Frequency of Option [a]   =",
            sum((EinRBFg1_5 > EinRBFg2) & (EoutRBFg1_5 < EoutRBFg2))))
print(paste("Frequency of Option [b]   =",
            sum((EinRBFg1_5 < EinRBFg2) & (EoutRBFg1_5 > EoutRBFg2))))
print(paste("Frequency of Option [c]   =",
            sum((EinRBFg1_5 < EinRBFg2) & (EoutRBFg1_5 < EoutRBFg2))))
print(paste("Frequency of Option [d]   =",
            sum((EinRBFg1_5 > EinRBFg2) & (EoutRBFg1_5 > EoutRBFg2))))
print(paste("Frequency of Option [e]   =",
            sum((EinRBFg1_5  == EinRBFg2) &
                (EoutRBFg1_5 == EoutRBFg2))))
