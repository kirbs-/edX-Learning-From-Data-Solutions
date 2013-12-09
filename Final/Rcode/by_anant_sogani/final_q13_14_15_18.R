#!/usr/bin/env Rscript

library(e1071)

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 100

# Number of Out-of-Sample Test Points.
T = 1000

# Number of Clusters in RBF. Change to 12 for q15.
K = 9

#
# Per Experiment: SVM
# Fraction of OUT-OF-SAMPLE Mismatches gSVM(x) != f(x).
#
EoutSVM = vector(length = R)

#
# Per Experiment: RBF
# Fraction of OUT-OF-SAMPLE Mismatches gRBF(x) != f(x).
#
EoutRBF = vector(length = R)

#
# Per Experiment: SVM
# Fraction of IN-SAMPLE Mismatches gSVM(x) != y.
#
EinSVM = vector(length = R)

#
# Per Experiment: RBF
# Fraction of IN-SAMPLE Mismatches gRBF(x) != y.
#
EinRBF = vector(length = R)

#
# Per Experiment: RBF
# Number of Iterations in Lloyd's Algorithm.
#
iterN = vector(length = R)

# Discard Counts.
discardSVM = 0; discardRBF = 0;

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

# Paintings are necessary to visualize things :)
paint_rbf_clustering_init = function () {
    unlink("rbf_clustering*.png")
}

paint_rbf_clustering = function (X, Y, U, Sx, iterN) {

    name = paste("rbf_clustering_iter_", iterN, ".png", sep = "")
    png(name)

    colors = c("brown", "darkgoldenrod", "blue4", "cornflowerblue",
               "aquamarine4", "chartreuse", "chartreuse3",
               "coral", "burlywood", "blueviolet")

    title = paste("Lloyd's Algorithm Iteration #", iterN)

    plot(X[, 1], X[, 2], xlab = "x1", ylab = "x2", pch = 20,
         main = title, col = colors[Sx])

    for (k in 1:nrow(U))
        points(U[k, 1], U[k, 2], col = colors[k], bg = colors[k],
               pch = 25)

    dev.off()
}

# Model: Radial Basis Functions.
rbf = function (X, Y, gamma) {

    #
    # STEP 1: Calculate K centers using Lloyd's Algorithm.
    #
     
    # Set/Cluster Membership.
    Sx = vector(length = N)

    # Initialize Centers: Random points in X-Space.
    U = t(sapply(1:K, random_x))
 
    # Number of Iterations.
    iterN = 1

    # paint_rbf_clustering_init()

    while (1) {

        # "Fix Centers" : Calculate Set Membership for each x.
        Sx = apply(X, 1, function (x) {
                              which.min(apply(U, 1, norms, x))
                         })

        # paint_rbf_clustering(X, Y, U, Sx, iterN)

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

    b = weights[+1]
    w = weights[-1]

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

# Model: Hard-Margin SVM with RBF Kernel.
hsvm = function (X, Y, gamma) {

    model = svm(X, Y, scale = FALSE,
                type = "C-classification", cost = 1e10,
                kernel = "radial", gamma = gamma)

    # Discard Run if Data Set is not linearly separable.
    ein     = mean(predict(model, X) != Y)
    discard = ifelse(ein == 0, FALSE, TRUE)

    return(list('discard' = discard, 'g' = model))
}

# In-Sample Error Calculations.
ein = function (gSVM, gRBF, Xin, Yin) {

    einSVM = mean(predict(gSVM, Xin)  != Yin)
    einRBF = mean(apply(Xin, 1, gRBF) != Yin)

    return(list('svm' = einSVM, 'rbf' = einRBF))
}

# Out-of-Sample Error Calculations.
eout = function (f, gSVM, gRBF) {

    # Generate Out-of-Sample Data.
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)

    eoutSVM = mean(predict(gSVM, Xout)  != Yout)
    eoutRBF = mean(apply(Xout, 1, gRBF) != Yout)

    return(list('svm' = eoutSVM, 'rbf' = eoutRBF))
}

# Number of Effective Runs.
r = 0

while (1) {

    # Create Data Set.
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)

    # Hard-Margin SVM: Final Hypothesis.
    ret  = hsvm(X, Y, gamma = 1.5)
    if (ret$discard) { discardSVM = discardSVM + 1; next }
    gSVM = ret$g

    # RBF: Final Hypothesis.
    ret  = rbf(X, Y, gamma = 1.5)
    if (ret$discard) { discardRBF = discardRBF + 1; next }
    gRBF = ret$g

    r = r + 1

    # Iterations needed by Lloyd's Algorithm to converge.
    iterN[r] = ret$iterN

    # Out-of-Sample Errors for SVM and RBF.
    ret        = eout(f, gSVM, gRBF)
    EoutSVM[r] = ret$svm
    EoutRBF[r] = ret$rbf

    # In-Sample Errors for SVM and RBF.
    ret       = ein(gSVM, gRBF, X, Y)
    EinSVM[r] = ret$svm
    EinRBF[r] = ret$rbf

    # All Runs done.
    if (r == R) break
}

print(paste("Sample Size               =", N))
print(paste("Effective      #Runs      =", R))
print(paste("SVM: Discarded #Runs      =", discardSVM))
print(paste("RBF: Discarded #Runs      =", discardRBF))
print(paste("RBF: Number of Clusters K =", K))
print(paste("RBF: Average #Iterations  =", mean(iterN)))
print(paste("SVM: Average Ein          =", mean(EinSVM)))
print(paste("RBF: Average Ein          =", mean(EinRBF)))
print(paste("Test Data Size            =", T))
print(paste("SVM: Average Eout         =", mean(EoutSVM)))
print(paste("RBF: Average Eout         =", mean(EoutRBF)))
print(paste("SVM beats RBF wrt Eout    =", 
            100 * mean(EoutSVM < EoutRBF), "%",
            " of the times"))
print(paste("RBF Zero Ein              =", 
            100 * mean(EinRBF == 0), "%",
            " of the times"))
