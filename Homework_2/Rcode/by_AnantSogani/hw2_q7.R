#!/usr/bin/env Rscript

# Number of Runs of the Experiment.
R = 1000

# Sample Size.
N = 10

# X part of set of samples.
Xs = NA

# Y part of set of samples.
Ys = NA

# Weight Vector of Random Target Function.
wf = NA

# Intermediate Hypothesis from Linear Regression.
gmin = NA; wmin = NA

# Final Hypothesis.
g = NA; wg = NA

# Number of Iterations per Experiment.
Iter = NA

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

# Profile or not to Profile the program.
profile = FALSE

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

pick_xy = function (h) {
    i = sample(1:N, 1)

    count = 1
    while (count <= N ) {
        
        x = unlist(Xs[i])
        y = unlist(Ys[i])

        if (h(x) != y) {
            return(list("x"=x, "y"=y, "unclassified"=TRUE))
        } 

        count = count + 1
        i = i + 1
        if (i > N) i = 1
    }

    return(list("unclassified"=FALSE))
}

update_w = function (w, x, y) {
    w = w + y * proper(x)
}

perceptron = function () {
    w = wmin
    h = create_h(w)

    iterN = 0

    while (1) {
        point = pick_xy(h)

        if (point$unclassified == FALSE) {
            return(list("N"=iterN, "g"=h, "wg"=w))
        }

        x = point$x
        y = point$y

        w = update_w(w, x, y)
        h = create_h(w)

        iterN = iterN + 1
    }
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

profile_start = function () {
    if (profile == TRUE) {
        Rprof("profile.out")
    }
}

profile_stop = function () {

    if (profile == TRUE) {
        Rprof(NULL)
        prof = summaryRprof("profile.out")
        unlink("profile.out")
        print(paste("Seconds taken   = ", prof$sampling.time))
    }
}

profile_start()

for (i in 1:R) {
    f   = random_f()
    Xs  = lapply(1:N, random_x)
    Ys  = lapply(Xs, f)

    X   = getX()
    Y   = getY()
    wmin = minw(X, Y)

    out = perceptron()
    g   = out$g
    wg  = out$wg
    Iter[i] = out$N

    Ein[i]  = ein()
    Eout[i] = eout()
}

print("Linear Regression aided Perceptron")
print(paste("Sample Size     = ", N))
print(paste("Number of Runs  = ", R))
print(paste("Mean Iterations = ", mean(Iter)))
print(paste("Mean Ein        = ", mean(Ein)))
print(paste("Mean Eout       = ", mean(Eout)))

paint = function () {
    name = "linear_regression_aided_perceptron.png"
    png(name)

    plot(0, 0, type="n", xlab="x1", ylab="x2",
         main="Linear Regression aided Perceptron")

    ch = 16; color = 3

    for (i in 1:N) {
        x = unlist(Xs[i])
        y = unlist(Ys[i])
        points(x[1], x[2], pch=(ch + y), col=(color + g(x)))
    }

    wfcolor   = "dark green"; wflty = 2
    wmincolor = "dark gray";  wminlty = 3
    wgcolor   = "black";      wglty = 1

    abline(a=(-wf[1]/wf[3]), b = (-wf[2]/wf[3]),
           col=wfcolor, lty=wflty)

    abline(a=(-wmin[1]/wmin[3]), b = (-wmin[2]/wmin[3]),
           col=wmincolor, lty=wminlty)

    abline(a=(-wg[1]/wg[3]), b = (-wg[2]/wg[3]),
           col=wgcolor, lty=wglty)

    legend("topright", c("f(x)", "gmin(x)", "g(x)"),
           col=c(wfcolor, wmincolor, wgcolor),
           lty=c(wflty, wminlty, wglty))

    dev.off()
    browseURL(name)
}

# Paint the results of the last Experiment.
paint()

profile_stop()
