# Get data
digits.train <- read.table(url("http://www.amlbook.com/data/zip/features.train"))
digits.test <- read.table(url("http://www.amlbook.com/data/zip/features.test"))

colnames(digits.train) <- c("digit", "symmetry", "intensity")
colnames(digits.test) <- c("digit", "symmetry", "intensity")

digits.train$digit <- factor(digits.train$digit)
digits.test$digit <- factor(digits.test$digit)


# Visualize
ggplot(digits.train, aes(x = symmetry, y = intensity, color = digit)) + 
  geom_point()

summary(digits.train)
summary(digits.train$digit)

q2_4 <- t(sapply(0:9, function(myDigit) {
  fit <- svm(digit == myDigit ~ ., data = digits.train, 
             scale = FALSE, 
             kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 0.01,
             type = "C-classification")
  
  # fit
  pred <- predict(fit, digits.train[-1])
  
  Ein = sum(pred != (digits.train$digit == myDigit))/nrow(digits.train)
  table(pred, (digits.train$digit == myDigit))
  cbind(myDigit, Ein, fit$tot.nSV)
}))


digits.train.sub <- subset(digits.train, digit == 1 | digit == 5)
digits.test.sub <- subset(digits.test, digit == 1 | digit == 5)

q5 <- t(sapply(c(0.0001, 0.001, 0.01, 0.1, 1), function(cst) {
  myDigit = 5
  fit <- svm(digit == myDigit ~ ., data = digits.train.sub, 
             scale = FALSE, 
             kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = cst,
             type = "C-classification")
  pred <- predict(fit, digits.train.sub[-1])
  
  Ein = sum(pred != (digits.train.sub$digit == myDigit))/nrow(digits.train.sub)
  
  pred <- predict(fit, digits.test.sub[-1])
  Eout = sum(pred != (digits.test.sub$digit == myDigit))/nrow(digits.test.sub)
  
  cbind(cst, Ein, Eout, fit$tot.nSV)
}))

# q6, run q5 with degree = 5

myDigit = 5
q7 <- tune(svm,digit == myDigit ~ ., data = digits.train.sub, 
           scale = FALSE,      type = "C-classification",
           kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, 
           ranges = list(cost = c(.0001, .001, .01, .1, 1)),
           tunecontrol = tune.control(sampling = "cross", cross = 10, nrepeat = 100))

q7

# Another way
rslts <- lapply(1:100, function(x) {
  ordIdx <- sample(nrow(digits.train.sub))
  fit.0001 <- svm(digit == myDigit ~ ., data = digits.train.sub[ordIdx, ], 
                  scale = FALSE, seed = x,
                  kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 0.0001,
                  type = "C-classification", cross = 10)
  
  fit.001 <- svm(digit == myDigit ~ ., data = digits.train.sub[ordIdx, ], 
                 scale = FALSE, seed = x,
                 kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 0.001,
                 type = "C-classification", cross = 10)
  
  fit.01 <- svm(digit == myDigit ~ ., data = digits.train.sub[ordIdx, ], 
                scale = FALSE, seed = x,
                kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 0.01,
                type = "C-classification", cross = 10)
  
  fit.1 <- svm(digit == myDigit ~ ., data = digits.train.sub[ordIdx, ], 
               scale = FALSE, seed = x,
               kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 0.1,
               type = "C-classification", cross = 10)
  
  fit1 <- svm(digit == myDigit ~ ., data = digits.train.sub[ordIdx, ], 
              scale = FALSE, seed = x,
              kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 1,
              type = "C-classification", cross = 10)

  1 - cbind(fit.0001$tot.accuracy, fit.001$tot.accuracy, fit.01$tot.accuracy, fit.1$tot.accuracy, fit1$tot.accuracy)/100
})

summary(sapply(rslts, function(x) which(x == min(x))[1]))
summary(sapply(rslts, function(x) x[2]))

myDigit = 5
q9_10 <- t(sapply(c(1e-2, 1, 1e2, 1e4, 1e6), function(cst) {
  fit <- svm(digit == myDigit ~ ., data = digits.train.sub, 
             scale = FALSE, 
             kernel = "radial", gamma = 1, cost = cst,
             type = "C-classification")
  pred <- predict(fit, digits.train.sub[-1])
  
  Ein = sum(pred != (digits.train.sub$digit == myDigit))/nrow(digits.train.sub)
  
  pred <- predict(fit, digits.test.sub[-1])
  Eout = sum(pred != (digits.test.sub$digit == myDigit))/nrow(digits.test.sub)
  
  cbind(cst, Ein, Eout, fit$tot.nSV)
}))


q9_10
