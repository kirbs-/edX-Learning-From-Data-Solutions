#!/usr/bin/env Rscript

library(e1071)

# Data Set.
D = as.matrix(read.table("http://www.amlbook.com/data/zip/features.train"))
X = cbind(D[, 'V2'], D[, 'V3'])
Y = D[, 'V1']

one_vs_rest = function (d) {

    Xtrain = X
    Ytrain = ifelse((Y == d), +1, -1)

    # Train SVM.
    model = svm(Xtrain, Ytrain, scale = FALSE,
                type = "C-classification", cost = 0.01,
                kernel = "polynomial",
                gamma = 1, coef0 = 1, degree = 2)

    # Number of Support Vectors. 
    svN = nrow(model$SV)

    # In-Sample Error.
    ein = mean(predict(model, Xtrain) != Ytrain)

    return(list('svN' = svN, 'ein' = ein))
}

print("Search for the HIGHEST In-Sample Error:")
print("Classifier, # of SV's, Ein")
for (d in c(0, 2, 4, 6, 8)) {
    ret = one_vs_rest(d)
    print(paste(d, "vs rest,", ret$svN, ret$ein))
}

print("")
print("Search for the LOWEST In-Sample Error:")
print("Classifier, # of SV's, Ein")
for (d in c(1, 3, 5, 7, 9)) {
    ret = one_vs_rest(d)
    print(paste(d, "vs rest", ret$svN, ret$ein))
}
