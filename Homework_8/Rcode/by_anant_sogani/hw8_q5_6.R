#!/usr/bin/env Rscript

library(e1071)

# Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

# Test Set.
Dout = as.matrix(read.table("http://www.amlbook.com/data/zip/features.test"))
Xout = cbind(Dout[, 'V2'], Dout[, 'V3'])
Yout = Dout[, 'V1']

one_vs_one = function (d, d_, Q, C) {

    # Training Data.
    select = (Y == d) | (Y == d_)
    Xtrain = X[select, ]
    Ytrain = ifelse((Y[select] == d), +1, -1)

    # Train SVM.
    model = svm(Xtrain, Ytrain, scale = FALSE,
                type = "C-classification", cost = C,
                kernel = "polynomial",
                gamma = 1, coef0 = 1, degree = Q)

    # Number of Support Vectors. 
    svN = nrow(model$SV)

    # In-Sample Error.
    ein = mean(predict(model, Xtrain) != Ytrain)

    # Testing Data.
    select = (Yout == d) | (Yout == d_)
    Xtest  = Xout[select, ]
    Ytest  = ifelse((Yout[select] == d), +1, -1)

    # Out-of-Sample Error.
    eout = mean(predict(model, Xtest) != Ytest)

    return(list('svN' = svN, 'ein' = ein, 'eout' = eout))
}

print("Experiment 1: How #SV, Ein, Eout vary with C")
for (C in c(0.001, 0.01, 0.1, 1)) {
    Q = 2
    ret = one_vs_one(1, 5, Q, C)
    print(paste("C:", C, "#SV:", ret$svN,
                "Ein:",  round(ret$ein, 5),
                "Eout:", round(ret$eout, 5)))
}

print("")
print("Experiment 2: How #SV, Ein, Eout vary with Q")
for (C in c(0.0001, 0.001, 0.01, 1)) {
    Q = 2
    ret = one_vs_one(1, 5, Q, C)
    print(paste("Q:", Q, "C:", C, "#SV:", ret$svN,
                "Ein:",  round(ret$ein, 5),
                "Eout:", round(ret$eout, 5)))

    Q = 5
    ret = one_vs_one(1, 5, Q, C)
    print(paste("Q:", Q, "C:", C, "#SV:", ret$svN,
                "Ein:",  round(ret$ein, 5),
                "Eout:", round(ret$eout, 5)))
}
