#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 100

# X part of set of samples.
Xs = NA

# Y part of set of samples.
Ys = NA

# Weight Vector of Random Target Function.
wf = NA

# Final Hypothesis.
g = NA; wg = NA

#
# Fraction of IN-SAMPLE Mismatches
# g(x) != y per Experiment.
#
Ein = NA

#
# Fraction of OUT-OF-SAMPLE Mismatches
# g(x) != f(x) per Experiment.
#
Eout = NA

random_x = function (void) {
    x = runif(2, -1, 1)
}

proper = function (x) {
    append(1, x)
}

random_f = function () {
    a = random_x()
    b = random_x()
    
    wf[1] <<- (a[1] * b[2]) - (b[1] * a[2])
    wf[2] <<- a[2] - b[2]
    wf[3] <<- b[1] - a[1]

    function (x) { sign(t(wf) %*% proper(x)) }
}

getX = function () {
    X = vector()
    for (i in 1:N) {
        X = c(X, proper(unlist(Xs[i])))
    }
    X = matrix(X, nrow=N, ncol=3, byrow=TRUE)
}

getY = function () {
    Y = vector()
    for (i in 1:N) {
        Y = c(Y, unlist(Ys[i]))
    }
    Y = matrix(Y, nrow=N, ncol=1, byrow=TRUE)
}

minw = function (X, Y) {
    Xdagger = solve(t(X) %*% X) %*% t(X)
    w = Xdagger %*% Y
    w = c(w)
}

create_h = function (w) {
    function (x) { sign(t(w) %*% proper(x)) }
}

ein = function () {
    mismatchN = 0

    for (i in 1:N) {
        x = unlist(Xs[i])
        y = unlist(Ys[i])    
        if (g(x) != y) {
            mismatchN = mismatchN + 1
        } 
    }

    avg = mismatchN / N
}

eout = function () {
    T = 1000; mismatchN = 0

    for (i in 1:T) {
        x = random_x()
        if (g(x) != f(x)) {
            mismatchN = mismatchN + 1
        }
    }

    avg = mismatchN / T
}

for (i in 1:R) {
    f   = random_f()
    Xs  = lapply(1:N, random_x)
    Ys  = lapply(Xs, f)

    X   = getX()
    Y   = getY()
    wg  = minw(X, Y)
    g   = create_h(wg)

    Ein[i]  = ein()
    Eout[i] = eout()
}

print("Linear Regression Classification")
print(paste("Sample Size     = ", N))
print(paste("Number of Runs  = ", R))
print(paste("Mean Ein        = ", mean(Ein)))
print(paste("Mean Eout       = ", mean(Eout)))

paint = function () {
    name = "linear_regression.png"
    png(name)

    plot(0, 0, type="n", xlab="x1", ylab="x2",
         main="Linear Regression Classification")

    ch = 16; color = 3

    for (i in 1:N) {
        x = unlist(Xs[i])
        y = unlist(Ys[i])
        points(x[1], x[2], pch=(ch + y), col=(color + g(x)))
    }

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
