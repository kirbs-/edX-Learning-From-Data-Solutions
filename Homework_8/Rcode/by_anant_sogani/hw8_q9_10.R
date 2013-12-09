#!/usr/bin/env Rscript

library(e1071)

# The digits in the One-vs-One Classifier.
d = 1; d_ = 5

# Complete Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

# Data Set revelant to the 1-vs-5 classifier.
select = (Y == d) | (Y == d_)
Xtrain = X[select, ]
Ytrain = ifelse((Y[select] == d), +1, -1)

# Complete Out-of-Sample Data.
Dout = as.matrix(read.table("http://www.amlbook.com/data/zip/features.test"))
Xout = cbind(Dout[, 'V2'], Dout[, 'V3'])
Yout = Dout[, 'V1']

# Out-of-Sample Data revelant to the 1-vs-5 classifier.
select = (Yout == d) | (Yout == d_)
Xtest  = Xout[select, ]
Ytest  = ifelse((Yout[select] == d), +1, -1)

# Train SVM with RBF Kernel for different Cost values C.
for (C in 10^c(-2, 0, 2, 4, 6)) {

    model = svm(Xtrain, Ytrain, scale = FALSE,
                type = "C-classification", cost = C,
                kernel = "radial", gamma = 1)

    # In-Sample Error.
    ein = mean(predict(model, Xtrain) != Ytrain)

    # Out-of-Sample Error.
    eout = mean(predict(model, Xtest) != Ytest)

    print(paste("C:", C, "ein:", round(ein, 6),
                "eout:", round(eout, 5)))
}
