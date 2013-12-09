#!/usr/bin/env Rscript

# Number of Runs of each Experiment.
R = 1000

# Sample Sizes to Experiment on.
Ns = c(10, 100)

#
# Per Experiment:
# Number of Iterations taken by PLA to converge.
#
IterN = vector(length = R)

#
# Per Experiment:
# Fraction of IN-SAMPLE Mismatches g(x) != y.
#
Ein = vector(length = R) 

#
# Per Experiment:
# Fraction of OUT-OF-SAMPLE Mismatches g(x) != f(x).
#
Eout = vector(length = R)
 
#
# Target Function and its Weight Vector.
# a, b are the Random Points which define the Function.
#
f = NA; wf = NA; a = NA; b = NA

# Size of Test Set.
T = 1000

# Generators.
random_x = function (void) c(1, runif(2, -1, 1))
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

# The Perceptron Learning Algorithm.
pla = function (X, Y) {

    # Initialize Weight Vector.
    w = rep(0, 3)

    # Iteration Count.
    iterN = 1

    while (1) {

        # Current Hypothesis.
        h = function (x) sign(t(w) %*% x)

        # Pick a misclassified point at random, else done.
        misclassify = (apply(X, 1, h) != Y)
        if (!any(misclassify)) return(list('iterN' = iterN, 'w' = w))
        n  = resample(which(misclassify), 1)

        # Update the Weight Vector.
        w  = w + Y[n] * X[n, ]

        iterN = iterN + 1
    }
}

ein = function (X, Y, g) mean(apply(X, 1, g) != Y)

eout = function (g) {

    # Generate Test Set.
    Xout = t(sapply(1:T, random_x))

    mean(apply(Xout, 1, g) != apply(Xout, 1, f))
}

# Run Experiments.
for (N in Ns) {

    for (r in 1:R) {

        # Generate Target Function.
        f = random_f()

        # Generate Data Set.
        X = t(sapply(1:N, random_x))
        Y = apply(X, 1, f)

        # Run PLA and obtain the Final Hypothesis.
        ret      = pla(X, Y)
        IterN[r] = ret$iterN
        wg       = ret$w
        g        = function (x) sign(t(wg) %*% x)

        # Store In/Out-of-Sample Errors.
        Ein[r]  = ein(X, Y, g)
        Eout[r] = eout(g)
    }

    # Stats. 
    print("-------------------------------")
    print(paste("SAMPLE SIZE     = ", N))
    print("-------------------------------")
    print(paste("Number of Runs  = ", R))
    print(paste("Avg #Iterations = ", mean(IterN)))
    print(paste("Avg Ein         = ", mean(Ein)))
    print(paste("Avg Eout        = ", mean(Eout)))

}

paint = function () {
    name = "perceptron.png"
    png(name)

    plot(0, 0, type="n", xlab="x1", ylab="x2",
         main="Perceptron Learning Algorithm")

    ch = 16; color = 3

    points(X[, 2], X[, 3], pch = ch + Y,
           col = color + apply(X, 1, g))

    wgcolor = "black"; wfcolor = "dark green"
    wglty   = 1      ; wflty   = 2

    abline(a=(-wf[1]/wf[3]), b = (-wf[2]/wf[3]),
           col=wfcolor, lty=wflty, lwd=1)

    abline(a=(-wg[1]/wg[3]), b = (-wg[2]/wg[3]),
           col=wgcolor, lty=wglty)

    legend("topright", c("f(x)", "g(x)"),
           col=c(wfcolor, wgcolor),
           lty=c(wflty, wglty))

    dev.off()
    browseURL(name)
}

# Paint the results of the last Experiment.
paint()
