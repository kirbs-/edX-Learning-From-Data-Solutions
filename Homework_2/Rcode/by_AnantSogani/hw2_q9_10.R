#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 10

# Sample Size.
N = 1000

# X part of set of samples.
X = NA

# Y part of set of samples.
Y = NA

# Hypothesis from Linear Regression.
gmin = NA; wmin_ = NA

# Final Hypothesis.
g = NA

#
# Per Experiment:
# W Vector returned by Linear Regression.
#
W_ = matrix(nrow=R, ncol=6)

#
# Per Experiment:
# Fraction of IN-SAMPLE Mismatches g(x) != y.
#
Ein = vector(length=R)

#
# Per Experiment:
# Fraction of OUT-OF-SAMPLE Mismatches g(x) != f(x).
#
Eout = vector(length=R)

# Number of Candidate Hypotheses.
Nhc = 5

# Set of Candidate Hypotheses.
Hc = NA

#
# Per Experiment:
# Fraction of Agreements between Candidate Hypotheses
# and the Final Hypothesis
# g(x) == hc(x)
#
A = matrix(nrow=R, ncol=Nhc)

random_x = function (void) c(1, runif(2, -1, 1))

f = function (x) sign(sum(x^2) - 0.6 - 1)

flip = function (y) -y

noiseY = function (Y, n) {
    return(Y)
    s = sample(1:n, n / 10)
    Y[s] = flip(Y[s])
    Y
}

x2z = function (x) c(x, x[2]*x[3], x[2]^2, x[3]^2)

minw_ = function () {
    Zdagger = solve(t(Z) %*% Z) %*% t(Z)
    w = Zdagger %*% Y
}

create_h = function (w_) {
    force(w_)
    function (x) sign(x2z(x) %*% w_)
}

ein = function () mean(apply(X, 1, g) != Y)

eout = function () {
    T = 1000
    Xout = t(sapply(1:T, random_x))
    Yout = apply(Xout, 1, f)
    Yout = noiseY(Yout, T)
    mean(apply(Xout, 1, g) != Yout)
}

candidates = function () {

    Whc_ = matrix(c(-1, -0.05, 0.08, 0.13, 1.5 , 1.5,
                  -1, -0.05, 0.08, 0.13, 1.5 , 15,
                  -1, -0.05, 0.08, 0.13, 15  , 1.5,
                  -1, -1.5 , 0.08, 0.13, 0.05, 0.05,
                  -1, -0.05, 0.08, 1.5 , 0.15, 0.15),
                nrow=5, ncol=6, byrow=TRUE);

    Hc = apply(Whc_, 1, create_h)
}

agreeC = function (hc, X) mean(apply(X, 1, hc) == apply(X, 1, g))

agreement = function () {
    T = 1000
    Xout = t(sapply(1:T, random_x))
    sapply(Hc, agreeC, Xout)
}


Hc = candidates()

for (i in 1:R) {
    X = t(sapply(1:N, random_x))
    Y = apply(X, 1, f)
    Y = noiseY(Y, N)

    Z = t(apply(X, 1, x2z))
    wmin_ = minw_()
    g = gmin = create_h(c(wmin_))

    Ein[i]  = ein()
    Eout[i] = eout()

    A[i, ]  = agreement()
    W_[i, ] = wmin_
}

print("Linear Regression")
print(paste("Sample Size     = ", N))
print(paste("Number of Runs  = ", R))
print(paste("Mean Ein        = ", mean(Ein)))
print(paste("Mean Eout       = ", mean(Eout)))
print(colMeans(A))
print("Average Weight Vector:")
print(colMeans(W_))

paint = function () {
    name = "linear_regression.png"
    png(name)

    plot(0, 0, type="n", xlab="x1", ylab="x2",
         main="Linear Regression on Non-LS Input with Z feature")

    ch = 16; color = 3

    for (i in 1:N) {
        x = X[i, ]
        y = Y[i]
        points(x[2], x[3], pch=(ch + y), col=(color + g(x)))
    }

    dev.off()
    browseURL(name)
}

# Paint the results of the last Experiment.
paint()
