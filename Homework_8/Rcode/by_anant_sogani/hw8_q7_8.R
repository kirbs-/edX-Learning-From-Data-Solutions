#!/usr/bin/env Rscript

library(e1071)

# Number of Runs of the Experiment.
R = 100

#
# Per Experiment:
# Cross-Validation Error of the 5 Models.
#
Ecv = matrix(nrow = R, ncol = 5)

#
# Per Experiment:
# Best Model Hm*.
#
Mstar = vector(length = R)

# Complete Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

# Data Set revelant to 1-vs-5 classifier.
d = 1; d_ = 5
select = (Y == d) | (Y == d_)
X = X[select, ]
Y = ifelse((Y[select] == d), +1, -1)

# Train SVM and Return 10-fold Cross-Validation Error.
one_vs_one = function (Xr, Yr, C) {

    model = svm(Xr, Yr, scale = FALSE,
                type = "C-classification", cost = C,
                kernel = "polynomial",
                gamma = 1, coef0 = 1, degree = 2,
                cross = 10)

    return(1 - (model$tot.accuracy / 100));
}

# Experiment.
for (r in 1:R) {

    # Create a Random Permutation of the Data Set.
    p = sample(length(Y))
    Xr = X[p, ]
    Yr = Y[p]

    # Obtain Cross-Validation Error for 5 SVM classifiers. 
    for (i in 1:5) {
        C = 10^(i - 5)
        Ecv[r, i] = one_vs_one(Xr, Yr, C)
    }

    # Best Hypothesis.
    Mstar[r] = which.min(Ecv[r, ])
}

print("Hypothesis Number, Frequency")
print(table(Mstar))
print("Average Ecv for each Hypothesis")
print(colMeans(Ecv))
