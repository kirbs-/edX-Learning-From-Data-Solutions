# hw7

phi <- function(X){
  x1 <- X$x1; x2 <- X$x2
  Xtransf <- cbind(1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2), X$y)
  colnames(Xtransf) <- c(paste0("b", 0:(ncol(Xtransf)-2)), "y")
  return(as.data.frame(Xtransf))
}
Solve <- function(X, y, lambda=0){
  X <- as.matrix(X)
  L <- diag(lambda, ncol(X))
  y <- as.matrix(y)
  W <- solve(t(X)%*%X + L)%*%t(X)%*%y
  return(as.vector(W))
}
distance <- function(x, w) {
  return(sum(t(w)%*%x))
}
classify.linear <- function(X, w) {
  distances = apply(as.data.frame(X), 1, distance, w = w)
  return(ifelse(distances < 0, -1, +1))
}
setwd("~/Documents/Courses/CalTech_LearningFromData_2/hw6")
indat <- read.csv("in.txt", header = F, sep = "\t")
outdat <- read.csv("out.txt", header = F, sep = "\t")
colnames(indat) <- colnames(outdat) <- c("x1", "x2", "y")

train <- phi(indat[1:25,])
valid <- phi(indat[26:35,])
test <- phi(outdat)

# Check coefs using test[1:100,]
# K=1:7
# check <- sapply(K, function(k){
#   W <- Solve(test[1:100, 1:(k+1)], test$y[1:100])
#   cat("k:", k, "\tW:", W, "\n")
#   fit1 <- classify.linear(valid[,1:(k+1)], W)
#   fit2 <- classify.linear(test[,1:(k+1)], W)
#   Ev <- sum(fit1 != valid$y)/length(fit1)
#   Eo <- sum(fit2 != test$y)/length(fit2)
#   c(Ev, Eo)
# })
# check

# Q1
K=3:7
Evalid <- sapply(K, function(k){
  W <- Solve(train[,1:(k+1)], train$y)
  cat("k:", k, "\tW:", W, "\n")
  fit <- classify.linear(valid[,1:(k+1)], W)
  sum(valid$y != fit)/length(valid$y)
})
Evalid
K[which.min(Evalid)]

# Q2
Eout <- sapply(K, function(k){
  W <- Solve(train[,1:(k+1)], train$y)
  cat("k:", k, "W:", W, "\n")
  fit <- classify.linear(test[,1:(k+1)], W)
  sum(test$y != fit)/length(test$y)
})
Eout
min(Eout)
K[which.min(Eout)]

# Q3
Evalid <- sapply(K, function(k){
  W <- Solve(valid[,1:(k+1)], valid$y)
  cat("k:", k, "W:", W, "\n")
  fit <- classify.linear(train[,1:(k+1)], W)
  sum(train$y != fit)/length(train$y)
})
Evalid
K[which.min(Evalid)]

# Q4
Eout <- sapply(K, function(k){
  W <- Solve(valid[,1:(k+1)], valid$y)
  fit <- classify.linear(test[,1:(k+1)], W)
  sum(test$y != fit)/length(test$y)
})
Eout
min(Eout)
K[which.min(Eout)]

# Q5
K=3:7
Evalid <- sapply(K, function(k){
  W <- Solve(train[,1:(k+1)], train$y)
  fit <- classify.linear(valid[,1:(k+1)], W)
  sum(valid$y != fit)/length(valid$y)
})
k <- K[which.min(Evalid)]
W <- Solve(train[,1:(k+1)], train$y)
fit <- classify.linear(test[,1:(k+1)], W)
Eout1 <- sum(test$y != fit)/length(test$y)

Evalid <- sapply(K, function(k){
  W <- Solve(valid[,1:(k+1)], valid$y)
  fit <- classify.linear(train[,1:(k+1)], W)
  sum(train$y != fit)/length(train$y)
})
k <- K[which.min(Evalid)]
W <- Solve(valid[,1:(k+1)], valid$y)
fit <- classify.linear(test[,1:(k+1)], W)
Eout2 <- sum(test$y != fit)/length(test$y)

cat("Using train:", Eout1, "\tusing valid", Eout2)

# Q6
B = 1:1e4
estim <- lapply(B, function(i){
  e <- runif(2)
  c(e, min(e))
})
estim <- colMeans(do.call(rbind, estim))
estim

# Q7: rho = 3, E1 = E2 = 1/2: Got wrong here!
Rho <- seq(0, 3, len = 100)
E1 <- c()
for(r in Rho){
  X <- data.frame(x1 = c(-1, r, 1), x2 = c(0, 1, 0))
  e <- c()
  for(i in 1:nrow(X)){
    m <- lm(x2 ~ 1, data = X[-i,])
    fit <- predict(m, data.frame(x1 = X[i,1], x2 = X[i, 2]))
    e <- c(e, sum((fit - X[i,2])^2))
    }
  E1 <- c(E1, 1/nrow(X)*sum(e))
}

E2 <- c()
for(r in Rho){
  X <- data.frame(x1 = c(-1, r, 1), x2 = c(0, 1, 0))
  e <- c()
  for(i in 1:nrow(X)){
    m <- lm(x2 ~ x1, data = X[-i,])
    fit <- predict(m, data.frame(x1 = X[i,1], x2 = X[i, 2]))
    e <- c(e, sum((fit - X[i,2])^2))
  }
  E2 <- c(E2, 1/nrow(X)*sum(e))
}
plot(Rho, E1, ylim = range(0, 4)); lines(Rho, E2)
abline(h = sqrt(sqrt(3)+4), col = "blue")
abline(h = sqrt(sqrt(3)-1), col = "red")
abline(h = sqrt(9+4*sqrt(6)), col = "green")
abline(h = sqrt(9-sqrt(6)), col = "cyan")

# Helpers for Q8 to Q10
require(e1071)
initLine <- function(lo=-.95, hi=.95){
  x1 <- runif(2, lo, hi)
  x2 <- runif(2, lo, hi)
  b = -(x2[1] - x1[1])
  a = x2[2] - x1[2]
  c = -a*x1[1] - b*x1[2]
  w = c(a, b)
  w0 = c
  return(c(w0, w))
}
distance <- function(x, w) {
  return(sum(t(w)%*%x))
}
classify.linear <- function(X, w) {
  distances = apply(as.data.frame(X), 1, distance, w = w)
  return(ifelse(distances < 0, -1, +1))
}
perceptron <- function(X, Class, rate = 0.25, W = rep(0, ncol(X)), Plot = FALSE){
  iter = 0
  missClass = TRUE
  while(missClass & iter<1000){
    iter = iter  + 1
    missClass = FALSE
    tmpClass <- classify.linear(X, W)
    if(any(tmpClass != Class)){
      i = sample(which(tmpClass != Class), 1)
      W[1] = W[1] + Class[i]*rate
      W[-1] = W[-1] + Class[i]*X[i,-1]*rate
      missClass = TRUE
      }
    }
  if(iter == 1000)
    iter <- NA
  err <- sum(classify.linear(X, W) != Class)/length(Class)
  return(list(W = W, class = classify.linear(X, W), Err = err))
}

# Q8
B = 1000; N = 10
Test <- lapply(1:B, function(i){
  y <- rep(1, N)
  while(length(unique(y))<2){
    X <- matrix(runif(N*2, -1, 1), N, 2)
    X <- cbind(1, X)
    colnames(X) <- paste0("x", 0:2)
    winit <- initLine()
    y <- classify.linear(X, winit)
    }
  # Learn
  P <- perceptron(X, y)
  S <- svm(y ~ ., data=as.data.frame(X), scale = FALSE, cost = 1000,
           kernel = "linear", type = "C")
  
  # Eout on fresh points
  Xout <- matrix(runif(1000*2, -1, 1), 1000, 2)
  Xout <- cbind(1, Xout)
  colnames(Xout) <- paste0("x", 0:2)
  yout <- classify.linear(Xout, winit)
  
  fitP <- classify.linear(Xout, P$W)
  Eper <- sum(fitP != yout)/length(yout)

  fitS <- predict(S, as.data.frame(Xout))
  Esvm <- sum(fitS != yout)/length(yout)

  return(c(Eper = Eper, Esvm = Esvm, nSV = S$tot.nSV))
  })
Test <- as.data.frame(do.call(rbind, Test))
head(Test)
Best <- Test$Esvm < Test$Eper
perf <- sum(Best, na.rm = TRUE)/sum(!is.na(Best))*100
cat("Gsvm best in:", perf)

# Q9 & 10
B = 1000; N = 100
Test <- lapply(1:B, function(i){
  y <- rep(1, N)
  while(length(unique(y))<2){
    X <- matrix(runif(N*2, -1, 1), N, 2)
    X <- cbind(1, X)
    colnames(X) <- paste0("x", 0:2)
    winit <- initLine()
    y <- classify.linear(X, winit)
  }
  # Learn
  P <- perceptron(X, y)
  S <- svm(y ~ ., data=as.data.frame(X[,-1]), scale = FALSE, cost = 1e12,
           kernel = "linear", type = "C")
  
  # Eout on fresh points
  Xout <- matrix(runif(1000*2, -1, 1), 1000, 2)
  Xout <- cbind(1, Xout)
  colnames(Xout) <- paste0("x", 0:2)
  yout <- classify.linear(Xout, winit)
  
  fitP <- classify.linear(Xout, P$W)
  Eper <- sum(fitP != yout)/length(yout)
  
  fitS <- predict(S, as.data.frame(Xout))
  Esvm <- sum(fitS != yout)/length(yout)
  
  return(c(Eper = Eper, Esvm = Esvm, nSV = S$tot.nSV))
})
Test <- as.data.frame(do.call(rbind, Test))
head(Test)
Best <- Test$Esvm < Test$Eper
perf <- sum(Best, na.rm = TRUE)/B*100
nSV <- mean(Test$nSV)
cat("Gsvm best in:", perf, "mean nSV:", round(nSV))


################################
op <- par(no.readonly=T)
require(e1071)
N = 100
y <- rep(1, N)
while(length(unique(y))<2){
  X <- matrix(runif(N*2, -1, 1), N, 2)
  X <- cbind(1, X)
  colnames(X) <- paste0("x", 0:2)
  winit <- initLine()
  y <- classify.linear(X, winit)
}
Xout <- matrix(runif(1000*2, -1, 1), 1000, 2)
Xout <- cbind(1, Xout)
colnames(Xout) <- paste0("x", 0:2)
yout <- classify.linear(Xout, winit)

# Learn
# P <- perceptron(X, y)

par(mfrow = c(2, 2))
for(C in c(1e1, 1e2, 1e3, 1e4)){
  S <- svm(y ~ ., data=as.data.frame(X), scale = FALSE, cost = C, kernel = "linear", type = "C")
  cat("Cost:", C, "\tnSV:", S$tot.nSV)
  
  fitS <- predict(S, as.data.frame(Xout))
  Err <- sum(fitS!=yout)/length(fitS)
  #Plot
  plot(Xout[,-1], pch = 19, col = c("red", "blue")[fitS], cex = .1,
       xlim = range(-1,1), ylim=range(-1,1),
       main = sprintf("Cost: %s, nSV: %s, Err: %s", C, S$tot.nSV, Err))
  points(X[S$index, -1], cex = 1.5, lwd = 4)
# abline(-winit[1]/winit[3], -winit[2]/winit[3])
# abline(-P$W[1]/P$W[3], -P$W[2]/P$W[3], col = "blue")

  ws <- t(S$coefs) %*% X[S$index,]
  b <- -S$rho
  abline(-(b+ws[1])/ws[3], -ws[2]/ws[3], lwd = 2, col = "red")
  abline(-(b+ws[1]+1)/ws[3], -ws[2]/ws[3], lty=2, col = "red")
  abline(-(b+ws[1]-1)/ws[3], -ws[2]/ws[3], lty=2, col = "red")
  }
par(op)

#######################
################################
op <- par(no.readonly=T)

par(mfrow = c(2, 2))
for(i in 1:4){
  N = 10

  # In-sample
  y <- rep(1, N)
  while(length(unique(y))<2){
    X <- matrix(runif(N*2, -1, 1), N, 2)
    X <- cbind(1, X)
    colnames(X) <- paste0("x", 0:2)
    winit <- initLine()
    y <- classify.linear(X, winit)
  }

  # Out-of-sample
  Xout <- matrix(runif(1000*2, -1, 1), 1000, 2)
  Xout <- cbind(1, Xout)
  colnames(Xout) <- paste0("x", 0:2)
  yout <- classify.linear(Xout, winit)

  # Learning
  S <- svm(y ~ ., data=as.data.frame(X), scale = FALSE, cost = 1000, kernel = "linear", type = "C")
  cat("Cost:", C, "\tnSV:", S$tot.nSV)
  
  # Using predict(svm-object) to predict new data
  fitS <- predict(S, as.data.frame(Xout))
  
  # Or manually
  w <- t(S$coefs)%*%X[S$index,]
  b <- -S$rho
  fit <- -sign(Xout%*%t(w) + b)
  all(fit == fitS)
  
  Err <- sum(fitS!=yout)/length(fitS)

  # Plot out-of-samples
  plot(Xout[,-1], pch = 19, col = c("red", "blue")[fitS],
       xlim = range(-1,1), ylim=range(-1,1),
       main = sprintf("Cost: %s, nSV: %s, Err: %s", C, S$tot.nSV, Err))
  points(X[S$index, -1], cex = 1.5, lwd = 4)
  
  # Add the decision boundary and margins
  ws <- t(S$coefs) %*% X[S$index,]
  b <- -S$rho
  abline(-(b+ws[1])/ws[3], -ws[2]/ws[3], lwd = 2, col = "red")
  abline(-(b+ws[1]+1)/ws[3], -ws[2]/ws[3],lty=2, col = "red")
  abline(-(b+ws[1]-1)/ws[3], -ws[2]/ws[3],lty=2, col = "red")
}
par(op)
