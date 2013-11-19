#!/usr/bin/env Rscript

library(LowRankQP)

# Number of Runs.
R = 1000

# Sample Size. Change to N = 100 for q9-10.
N = 10

# Target Function: Weight Vector and defining Points.
wf = NA; a = NA; b = NA

# Perceptron Learning Algorithm: Final Hypothesis.
wgPLA = NA; gPLA = NA

# Support Vector Machine: Final Hypothesis.
wgSVM = NA; gSVM = NA

# Number of Out-of-Sample Test Points.
T = 10000

#
# Per Experiment: PLA
# Fraction of OUT-OF-SAMPLE Mismatches gPLA(x) != f(x).
#
EoutPLA = vector(length = R)

#
# Per Experiment: SVM
# Fraction of OUT-OF-SAMPLE Mismatches gSVM(x) != f(x).
#
EoutSVM = vector(length = R)

#
# Per Experiment: PLA
# Fraction of IN-SAMPLE Mismatches gPLA(x) != y.
#
EinPLA = vector(length = R)

#
# Per Experiment: SVM
# Fraction of IN-SAMPLE Mismatches gSVM(x) != y.
#
EinSVM = vector(length = R)

#
# Per Experiment: PLA
# Number of Iterations.
#
iterN = vector(length = R)

#
# Per Experiment: SVM
# Number of Support Vectors
#
svN = vector(length = R)

# Generators.
random_x = function (void) { c(1, runif(2, -1, 1)) }
random_f = function () {
    a <<- random_x()
    b <<- random_x()

    wf[1] <<- (a[2] * b[3]) - (b[2] * a[3])
    wf[2] <<- a[3] - b[3]
    wf[3] <<- b[2] - a[2]

    function (x) { sign(t(wf) %*% x) }
}

# Workaround for "special" behavior of sample().
resample = function (x, ...) x[sample.int(length(x), ...)]

# Perceptron Learning Algorithm.
pla = function (X, Y) {

    # Initial Weights.
    w = rep(0, 3)

    # Number of Iterations.
    iterN = 1

    while (1) {

        # Current Hypothesis.
        h = function (x) { sign(t(w) %*% x) }

        # Pick a misclassified point at random, otherwise terminate.
        miscl = (apply(X, 1, h) != Y)
        if (sum(miscl) == 0) break
        n = resample(c(1:N)[miscl], 1)

        # Update the Weight Vector.
        w = w + Y[n] * X[n, ]

        iterN = iterN + 1
    }

    # (i)  Final Weights 
    # (ii) Number of Iterations.
    return(list('w' = w, 'iterN' = iterN))
}

#
# Support Vector Machine One-Step Solution.
#
svm = function (X, Y) {

    # Create the Quadratic Programing Parameters for LowRankQP().

    # Matrix.
    Vmat = (Y * X) %*% t(Y * X)

    # Vector.
    dvec = rep(-1, N)

    # Constraints Matrix.
    Amat = t(Y)

    # Constraints.
    bvec = 0

    # (Practical) Upper Bound on Alpha. Theoretically infinite.
    uvec = rep(10000, N)

    # Minimize the Quadratic Function!
    solution = LowRankQP(Vmat, dvec, Amat, bvec, uvec,
                         method = "LU")

    # Obtain Alphas.
    a = c(zapsmall(solution$alpha))

    # Weights.
    w = colSums(a * Y * X)

    # Bias/Threshold Term. Take any one, all are equal.
    b = resample(((1 / Y) - (X %*% w))[a > 0], 1)

    # Number of Support Vectors.
    svN = sum(a > 0)

    # (i)  Final Weights 
    # (ii) Number of Support Vectors.
    return(list('w' = c(b, w), 'svN' = svN))
}

# In-Sample Error Calculations.
ein  = function (gPLA, gSVM, Xin, Yin) {

    einPLA = mean(apply(Xin, 1, gPLA) != Yin)
    einSVM = mean(apply(Xin, 1, gSVM) != Yin)

    return(list('pla' = einPLA, 'svm' = einSVM))
}

# Out-of-Sample Error Calculations.
eout = function (f, gPLA, gSVM) {

    # Generate Out-of-Sample Data.
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)

    eoutPLA = mean(apply(Xout, 1, gPLA) != Yout)
    eoutSVM = mean(apply(Xout, 1, gSVM) != Yout)

    return(list('pla' = eoutPLA, 'svm' = eoutSVM))
}

# Number of Effective Runs.
r = 0

while (1) {

    # Generate Target Function.
    f = random_f()

    # Generate Data Set.
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)

    # Discard Run if all points are on the same side.
    if (abs(sum(Y)) == N) next

    r = r + 1

    # PLA: Final Weights and Hypothesis.
    ret      = pla(X, Y)
    iterN[r] = ret$iterN
    wgPLA    = ret$w
    gPLA     = function (x) { sign(t(wgPLA) %*% x) }

    # SVM: Final Weights and Hypothesis.
    ret    = svm(X[, -1], Y)
    svN[r] = ret$svN
    wgSVM  = ret$w
    gSVM   = function (x) { sign(t(wgSVM) %*% x) }

    # Out-of-Sample Errors for PLA and SVM.
    ret = eout(f, gPLA, gSVM)
    EoutPLA[r] = ret$pla
    EoutSVM[r] = ret$svm

    # In-Sample Errors for PLA and SVM.
    ret = ein(gPLA, gSVM, X, Y)
    EinPLA[r] = ret$pla
    EinSVM[r] = ret$svm

    # All Runs done.
    if (r == R) break
}

print(paste("Number of Runs            =", R))
print(paste("Sample Size               =", N))
print(paste("PLA: Average #Iterations  =", mean(iterN)))
print(paste("SVM: Average #SV's        =", mean(svN)))
print(paste("PLA: Average Ein          =", mean(EinPLA)))
print(paste("SVM: Average Ein          =", mean(EinSVM)))
print(paste("Test Data Size            =", T))
print(paste("PLA: Average Eout         =", mean(EoutPLA)))
print(paste("SVM: Average Eout         =", mean(EoutSVM)))
print(paste("SVM beats PLA             =", 
            100 * mean(EoutSVM < EoutPLA), "%",
            " of the times"))

paint = function () {

    name = "pla_vs_svm.png"
    png(name)

    plot(0, 0, type = "n", xlab = "x1", ylab = "x2",
         main = "PLA vs SVM")

    ch = 16; color = 3

    for (i in 1:N) {
        x = X[i, ]
        y = Y[i]
        points(x[2], x[3], pch=(ch + y), col=(color + gPLA(x)))
    }

    wfcolor    = "dark green"; wflty = 3
    wgPLAcolor = "dark green"; wgPLAlty = 2
    wgSVMcolor = "dark green"; wgSVMlty = 1

    points(a[2], a[3], pch=20, col="yellow")
    points(b[2], b[3], pch=20, col="yellow")

    abline(a=(-wf[1]/wf[3]), b = (-wf[2]/wf[3]),
           col=wfcolor, lty=wflty)
    abline(a=(-wgPLA[1]/wgPLA[3]), b = (-wgPLA[2]/wgPLA[3]),
           col=wgPLAcolor, lty=wgPLAlty)
    abline(a=(-wgSVM[1]/wgSVM[3]), b = (-wgSVM[2]/wgSVM[3]),
           col=wgSVMcolor, lty=wgSVMlty)

    legend("topright", c("f(x)", "gPLA(x)", "gSVM(x)"),
           col=c(wfcolor, wgPLAcolor, wgSVMcolor),
           lty=c(wflty, wgPLAlty, wgSVMlty))

    dev.off()
    browseURL(name)
}

# Paint result of the Last Experiment.
paint()
