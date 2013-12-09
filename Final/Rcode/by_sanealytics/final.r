
# Get data
digits.train <- read.table(url("http://www.amlbook.com/data/zip/features.train"))
digits.test <- read.table(url("http://www.amlbook.com/data/zip/features.test"))

colnames(digits.train) <- c("digit", "symmetry", "intensity")
colnames(digits.test) <- c("digit", "symmetry", "intensity")

digits.train$digit <- factor(digits.train$digit)
digits.test$digit <- factor(digits.test$digit)

library(ggplot2)

ggplot(digits.train, aes(x = symmetry, y = intensity, color = digit)) + 
  geom_point()

linearRegression <- function(x, y, lambda, invFn = function(x) solve(x)) {
  # Calculate pseudo inverse
  pseudo = invFn(t(x) %*% x + lambda * diag(ncol(x))) %*% t(x)
  w = pseudo %*% y
  Ein = sum(y != sign(t(w) %*% t(x)))/length(y)
  # error = sum(y != sign(t(t(w) %*% t(x)))) # In case of only one record
  list(w = w, Ein = Ein)
}

phi.1 <- function(x) as.matrix(cbind(bias = 1, x[,1], x[,2]))

phi.2 <- function(x) as.matrix(cbind(bias = 1, x[,1], x[, 2], x[, 1] * x[,2], x[,1]^2, x[,2]^2))

q7_q8_q9 <- t(sapply(0:9, function(myDigit) {
  lambda = 1
  y.train = ifelse(digits.train$digit == myDigit, 1, -1)
  x.train.1 = phi.1(digits.train[, -1]) # Could be outside the loop
  x.train.2 = phi.2(digits.train[, -1]) # Could be outside the loop
  
  fit.1 <- linearRegression(x.train.1, y.train, lambda)
  fit.2 <- linearRegression(x.train.2, y.train, lambda)
  
  y.test = ifelse(digits.test$digit == myDigit, 1, -1)
  x.test.1 = phi.1(digits.test[, -1])
  x.test.2 = phi.2(digits.test[, -1])
  
  Eout.1 = sum(y.test != sign(t(fit.1$w) %*% t(x.test.1)))/length(y.test)
  Eout.2 = sum(y.test != sign(t(fit.2$w) %*% t(x.test.2)))/length(y.test)
  
  Err.1 = Eout.1 - fit.1$Ein
  Err.2 = Eout.2 - fit.2$Ein
  
  cbind(myDigit, fit.1$Ein, Eout.1, Err.1, 
        fit.2$Ein, Eout.2, Err.2, 100*(Eout.2 - Eout.1), Eout.2 <= 0.95 * Eout.1)
}))

colnames(q7_q8_q9) <- c("Digit", "Ein.1", "Eout.1", "Err.1",
                        "Ein.2", "Eout.2", "Err.2", "diff", "check")

q7_q8_q9

ggplot(as.data.frame(q7_q8_q9), aes(x=Digit)) + 
  geom_line(aes(y = Eout.1, color="psi1")) + 
  geom_line(aes(y = Eout.2, color = "psi2")) 


# Q10
digits.train.sub <- subset(digits.train, digit == 1 | digit == 5)
digits.test.sub <- subset(digits.test, digit == 1 | digit == 5)
myDigit = 1

require(foreach)
foreach(lambda=c(0.01, 1), .combine = rbind) %do% {
  y.train = ifelse(digits.train.sub$digit == myDigit, 1, -1)
  x.train.2 = phi.2(digits.train.sub[, -1])
  fit.2 <- linearRegression(x.train.2, y.train, lambda)
  
  y.test = ifelse(digits.test.sub$digit == myDigit, 1, -1)
  x.test.2 = phi.2(digits.test.sub[, -1])
  
  Eout.2 = sum(y.test != sign(t(fit.2$w) %*% t(x.test.2)))/length(y.test)
  
  cbind(Ein.2 = fit.2$Ein, Eout.2)
}

# Q11
x = rbind(c(1,0), c(0,1), c(0,-1),
          c(-1,0), c(0,2), c(0,-2),
          c(-2,0))

y = c(-1, -1, -1, +1, +1, +1, +1)

plot(x, pch = y+2)

phi <- function(x) as.matrix(cbind(x[,2]^2 - 2 * x[,1] - 1, x[,1]^2 - 2 * x[,2] + 1))

plot(phi(x), pch = y + 2)

abline(-3, 7)

fit <- linearRegression(cbind(1, phi(x)), y, 0)
abline(-fit$w[1]/fit$w[3], -fit$w[2]/fit$w[3])

w1=-1; w2=1; b = -0.5
abline(-w1/w2, -b/w2, col = "red")
abline(-b/w2, -w1/w2, col = "red")
abline(w1/w2, b/w2, col = "red")
w1=1; w2=-1; b = -0.5
abline(-w1/w2, -b/w2, col = "blue")
abline(-b/w2, -w1/w2, col = "blue")
abline(w1/w2, b/w2, col = "blue")
w1=1; w2=1e6; b = -0.5
abline(-w1/w2, -b/w2, col = "green")
abline(-b/w2, -w1/w2, col = "green")
abline(w1/w2, b/w2, col = "green")
w1=0; w2=1; b = -0.5
abline(-w1/w2, -b/w2, col = "purple")
abline(-b/w2, -w1/w2, col = "purple")
abline(w1/w2, b/w2, col = "purple")

# Got Q11 wrong

# Q12
require(e1071)
fit <- svm(x, y, 
           scale = TRUE, 
           kernel = "polynomial", degree = 2, gamma = 1, coef0 = 1, cost = 1e16,
           type = "C-classification")

fit$tot.nSV

# Q13
f <- function(x) sign(x[,2] - x[,1] + 0.25 * sin(pi * x[,1]))
Eins <- sapply(1:10000, function(run) {
  x <- cbind(runif(100, -1, 1), runif(100, -1, 1))
  y <- f(x)
  # plot(x, pch=y+2)
  
  fit <- svm(x, y, 
             scale = TRUE, 
             kernel = "radial", gamma = 1.5, cost = 1e16,
             type = "C-classification")
  
  #fit$tot.nSV
  #points(x[fit$index,], col='blue', pch=14, lwd=4)
  sum(fit$fitted != y)
})

summary(Eins)

# Q14
# RBF
phi <- function(gamma, x, k) cbind(bias = 1, 
                                   t(apply(x,1, function(x1) 
                                     apply(k, 1, function(s) 
                                       exp(- gamma * Norm(x1, s)^2)))))

# f <- function(x) sign(tanh(x[,1] - x[,2]))
k = 12 # 9
gamma = 1.5
N <- 100
runs <- 1000

q14_15 <- sapply(seq(runs), function(run) {
  x.train <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.train <- f(x.train)
  fit.svm <- svm(x.train, y.train, 
                 scale = TRUE, 
                 kernel = "radial", gamma = gamma, cost = 1e16,
                 type = "C-classification")
  
  # Get centers
  fit.kmeans = kmeans(x.train, k)
  # Get RBF
  x.train.phi = phi(gamma = gamma, x.train, fit.kmeans$centers)
  fit.rbf <- linearRegression(x.train.phi, y.train, 0)
  
  x.test <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.test <- f(x.test)
  
  Eout.svm <- sum(y.test != predict(fit.svm, x.test))/length(y.test)
  pred.rbf <- sign(phi(gamma, x.test, fit.kmeans$centers) %*% fit.rbf$w)
  Eout.rbf <- sum(y.test != pred.rbf)/length(y.test)
  
  cbind(Eout.svm, Eout.rbf, Eout.rbf > Eout.svm)
})


summary(t(q14_15)*100)
sum(q14_15[3,])*100/runs

# Q16
q16 <- sapply(seq(runs), function(run) {
  x.train <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.train <- f(x.train)
  
  x.test <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.test <- f(x.test)
  
  k = 9
  # Get centers
  fit.kmeans = kmeans(x.train, k)
  # Get RBF
  x.train.phi = phi(gamma = gamma, x.train, fit.kmeans$centers)
  fit.rbf.9 <- linearRegression(x.train.phi, y.train, 0)
  
  pred.rbf <- sign(phi(gamma, x.test, fit.kmeans$centers) %*% fit.rbf.9$w)
  Eout.rbf.9 <- sum(y.test != pred.rbf)/length(y.test)
  
  k = 12
  # Get centers
  fit.kmeans = kmeans(x.train, k)
  # Get RBF
  x.train.phi = phi(gamma = gamma, x.train, fit.kmeans$centers)
  fit.rbf.12 <- linearRegression(x.train.phi, y.train, 0)
    
  pred.rbf <- sign(phi(gamma, x.test, fit.kmeans$centers) %*% fit.rbf.12$w)
  Eout.rbf.12 <- sum(y.test != pred.rbf)/length(y.test)
  
  
  cbind(fit.rbf.9$Ein, fit.rbf.12$Ein, Eout.rbf.9, Eout.rbf.12)
})


summary(t(q16)*100)


# Q17
k = 9
q17 <- sapply(seq(runs), function(run) {
  x.train <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.train <- f(x.train)
  
  x.test <- cbind(runif(N, -1, 1), runif(N, -1, 1))
  y.test <- f(x.test)
  
  gamma = 1.5
  # Get centers
  fit.kmeans = kmeans(x.train, k)
  # Get RBF
  x.train.phi = phi(gamma = gamma, x.train, fit.kmeans$centers)
  fit.rbf.9 <- linearRegression(x.train.phi, y.train, 0)
  
  pred.rbf <- sign(phi(gamma, x.test, fit.kmeans$centers) %*% fit.rbf.9$w)
  Eout.rbf.9 <- sum(y.test != pred.rbf)/length(y.test)
  
  gamma = 2
  # Get centers
  fit.kmeans = kmeans(x.train, k)
  # Get RBF
  x.train.phi = phi(gamma = gamma, x.train, fit.kmeans$centers)
  fit.rbf.12 <- linearRegression(x.train.phi, y.train, 0)
  
  pred.rbf <- sign(phi(gamma, x.test, fit.kmeans$centers) %*% fit.rbf.12$w)
  Eout.rbf.12 <- sum(y.test != pred.rbf)/length(y.test)
  
  
  cbind(fit.rbf.9$Ein, fit.rbf.12$Ein, Eout.rbf.9, Eout.rbf.12)
})

summary(t(q17)*100)
sum(q17[1,]==0)/runs * 100

