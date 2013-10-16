#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 1000

# X part of set of samples.
X = NA

# Y part of set of samples.
Y = NA

# Hypothesis from Linear Regression.
gmin = NA; wmin = NA

# Final Hypothesis.
g = NA

#
# Fraction of IN-SAMPLE Mismatches
# g(x) != y per Experiment.
#
Ein = vector(length=R)

#
# Fraction of OUT-OF-SAMPLE Mismatches
# g(x) != f(x) per Experiment.
#
Eout = vector(length=R)

random_x = function (void) {
    c(1, runif(2, -1, 1))
}

f = function (x) sign(sum(x^2) - 0.6 - 1)

flip = function (y) -y

noise = function () {
    s = sample(1:N, N / 10)
    Y[s] = flip(Y[s])
    Y
}

minw = function () {
    Xdagger = solve(t(X) %*% X) %*% t(X)
    w = Xdagger %*% Y
}

create_h = function (w) {
    function (x) sign(x %*% w)
}

ein = function () mean(apply(X, 1, g) != Y)

eout = function () {
    T = 1000
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)
    mean(apply(Xout, 1, g) != Yout)
}

for (i in 1:R) {
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)
    Y = noise()

    wmin = minw()
    g = gmin = create_h(wmin)

    Ein[i] = ein()
    Eout[i] = eout()
}

print("Linear Regression")
print(paste("Sample Size     = ", N))
print(paste("Number of Runs  = ", R))
print(paste("Mean Ein        = ", mean(Ein)))
print(paste("Mean Eout       = ", mean(Eout)))

paint = function () {
    name = "linear_regression.png"
    png(name)

    plot(0, 0, type="n", xlab="x1", ylab="x2",
         main="Linear Regression on Non-LS Input")

    ch = 16; color = 3

    for (i in 1:N) {
        x = X[i, ]
        y = Y[i]
        points(x[2], x[3], pch=(ch + y), col=(color + g(x)))
    }

    wycolor   = "dark green"; wylty = 2
    wmincolor = "black";      wminlty = 1

    abline(a=(-wmin[1]/wmin[3]), b = (-wmin[2]/wmin[3]),
           col=wmincolor, lty=wminlty)

    dev.off()
    browseURL(name)
}

# Paint the results of the last Experiment.
paint()
