# hw8 using svm from e1071 R package
######################################
# getting data from url
######################################

getDat <- function(Url){
  dat <- readLines(url(Url))
  dat <- lapply(dat, function(l){
    l <- gsub("( )+", "_", l)
    unlist(strsplit(l, "_"))[-1]
  })
  dat <- as.data.frame(do.call(rbind, dat))
  for(i in 1:ncol(dat)) dat[,i] <- as.numeric(as.character(dat[,i]))
  colnames(dat) <- c("digits", "sym", "int")
  return(dat)
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
Err <- function(obs, fit){
  return(sum(obs != fit)/length(obs))
}
######################################

# load the kernlab package and the data
require(e1071)
options(digits = 15)

cat("Loading data...")
train <- getDat("http://www.amlbook.com/data/zip/features.train")
test <- getDat("http://www.amlbook.com/data/zip/features.test")
op <- par(no.readonly = TRUE)
cat("Done.\n\n")

phi <- function(X){
  x1 <- X[,1]; x2 <- X[,2]
  Xtransf <- cbind(1, x1, x2, x1*x2, x1^2, x2^2)
  colnames(Xtransf) <- paste0("b", 0:(ncol(Xtransf)-1))
  return(as.data.frame(Xtransf))
}
linModel <- function(train, test, label, lambda=1){
  Xtrain <- cbind(1, train[,-1])
  Xtest <- cbind(1, test[,-1])
  ytrain <- ifelse(train$digits==label, 1, -1)
  ytest <- ifelse(test$digits==label, 1, -1)
  W <- Solve(Xtrain, ytrain, lambda)
  fitTrain <- classify.linear(Xtrain, W)
  fitTest <- classify.linear(Xtest, W)
  Ein <- Err(ytrain, fitTrain)
  Eout <- Err(ytest, fitTest)
  return(cbind(Ein=Ein, Eout=Eout))
}
transformModel <- function(train, test, label, lambda=1){
  phiTrain <- phi(train[,-1])
  phiTest <- phi(test[,-1])
  ytrain <- ifelse(train$digits==label, 1, -1)
  ytest <- ifelse(test$digits==label, 1, -1)
  W <- Solve(phiTrain, ytrain, lambda)
  fitTrain <- classify.linear(phiTrain, W)
  fitTest <- classify.linear(phiTest, W)
  Ein <- Err(ytrain, fitTrain)
  Eout <- Err(ytest, fitTest)
  return(cbind(Ein=Ein, Eout=Eout))
}

# Q7
E <- lapply(5:9, function(label){
  E <- linModel(train, test, label)
  cbind(label = label, E)
  })
E <- as.data.frame(do.call(rbind, E))
Min <- E$label[which.min(E$Ein)]
cat("Q7:", "\tMin Ein for", Min, "vs. all: Ein =", min(E$Ein), "\n")
cat("\n")

# Q8
E <- lapply(0:4, function(label){
  E <- transformModel(train, test, label)
  cbind(label = label, E)
})
E <- as.data.frame(do.call(rbind, E))
Min <- E$label[which.min(E$Eout)]
cat("Q8:", "\tMin Eout for", Min, "vs. all: Eout =", min(E$Eout), "\n")
cat("\n")

# Q9
perf <- lapply(0:9, function(label){
  # non-tranform
  ELin <- linModel(train, test, label)
  # transform
  Etransf <- transformModel(train, test, label)
  cbind(label = label, ELin, Etransf)
})
perf <- as.data.frame(do.call(rbind, perf))
colnames(perf) <- c("label", "EinLin", "EoutLin", "EinTrans", "EoutTrans")
A <- all(perf$EoutTrans - perf$EinTrans > perf$EoutLin - perf$EinLin)
B <- all(perf$EoutTrans < perf$EoutLin*.95)
C <- all(perf$EoutTrans == perf$EoutLin)
D <- all(perf$EoutTrans > perf$EoutLin*1.05)
E <- perf$EoutTrans[perf$label==5] < perf$EoutLin[perf$label==5]
cat("Q9:\t")
cat("a:", A, "\tb:", B, "\tc:", C, "\td:", D, "\te:", E, "\n")
cat("\n")

# Q10
train2 <- train[which(train$digits %in% c(1, 5)),]
test2 <- test[which(test$digits %in% c(1, 5)),]
perf <- lapply(c(1e-2, 1), function(lambda){
  E <- transformModel(train2, test2, label=5, lambda)
  cbind(lambda = lambda, E)
})
perf <- as.data.frame(do.call(rbind, perf))
overFit <- perf$Eout - perf$Ein
A <- overFit[1]>overFit[2]
B <- perf$Ein[perf$lambda==1] == perf$Ein[perf$lambda==.01]
C <- perf$Eout[perf$lambda==1] == perf$Eout[perf$lambda==.01]
D1 <- perf$Ein[perf$lambda==1] > perf$Ein[perf$lambda==.01]
D2 <- perf$Eout[perf$lambda==1] > perf$Eout[perf$lambda==.01]
E1 <- perf$Ein[perf$lambda==1] < perf$Ein[perf$lambda==.01]
E2 <- perf$Eout[perf$lambda==1] < perf$Eout[perf$lambda==.01]
cat("Q10:\t")
cat("a:", A, "\tb:", B, "\tc:", C, "\td:", D1 & D2, "\te:", E1 & E2, "\n")
cat("\n")

# Q11
X <- matrix(c(1,0, 0,1, 0,-1, -1,0, 0,2, 0,-2, -2,0), 7, 2, byrow = TRUE)
y <- c(-1, -1, -1, 1, 1, 1, 1)
colnames(X) <- c("x1", "x2")
z1 <- X[,2]^2 - 2*X[,1] - 1
z2 <- X[,1]^2 - 2*X[,2] + 1
Z <- cbind(z1=z1, z2=z2)

# Q12
model <- svm(y~., data=Z, scale = FALSE, cost = 1e6, coef0 = 1,
             type = "C-classification", kernel="polynomial", degree = 2)
cat("Q12: number of SVs:", model$tot.nSV, "\n")
cat("\n")

a <- model$coefs
b <- -model$rho
idx <- model$index
w <- t(a)%*%Z[idx,]
plot(Z, col = y+2)
abline(-(b)/w[2], -w[1]/w[2], lwd = 2, col = "red")
abline(-(b+1)/w[2], -w[1]/w[2], lty=2, col = "red")
abline(-(b-1)/w[2], -w[1]/w[2], lty=2, col = "red")

# RBF helper functions
generateData <- function(n){
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  return(cbind.data.frame(x1 = x1, x2 = x2))
}
target <- function(X){
  x1 <- X$x1; x2 <- X$x2
  return(sign(x2 - x1 + .25*sin(pi*x1)))
}
kerMat <- function(X, M, g){
  # Regular RBF: compute the phi matrix
  ker <- function(x, g){exp(-g*t(x)%*%x)}
  Phi <- lapply(1:nrow(M), function(i){
    sapply(1:nrow(X), function(j) ker(as.numeric(X[j,]-M[i,]), g))
    })
  return(do.call(cbind, Phi))
}
kerMat2 <- function(X, M, g){
  # Nice but not faster!
  ker <- function(x, g){exp(-g*t(x)%*%x)}
  Phi <- outer(1:nrow(X), 1:nrow(M),
               Vectorize(function(i, j, g) ker(as.numeric(X[i,]-M[j,]), g)), g)
  return(Phi)
}
# Q13
B = 10000; N = 100
out <- sapply(1:B, function(b){
  if(b%%(B/10)==0) cat("B:", b, "\t")
  X <- generateData(N)
  y <- target(X)
  model <- svm(y~., data=X, cost = 1e6, gamma = 1, type = "C-classification", kernel = "radial")
  fit <- predict(model)
  return(!all(fit == y))
#   return(sum(fit!=y)/length(fit))  
  })
cat("\nQ13:", sum(out)/length(out)*100)
cat("\n\n")

# Q14
B = 300; N = 100; K = 9; g = 1.5
out <- sapply(1:B, function(b){
  if(b%%(B/10)==0) cat("B:", b, "\t")
  emptyK <- TRUE
  while(emptyK){
    X <- generateData(N)
    y <- target(X)
    km <- kmeans(X, centers = K, iter.max = 15, nstart = 10)
    clusters <- km$cluster
    emptyK <- length(unique(clusters)) != K
    }
  M <- km$centers
  kMatrix <- kerMat(X, M, g)
  wL <- Solve(cbind(1, kMatrix), y)
  
  modelS <- svm(y~., data=X, gamma = g, cost = 1e6, type = "C-classification", kernel = "radial")

  # Compute Eout only (on a fresh data set)
  newX <- generateData(N*10)
  newY <- target(newX)
  fitreg <- classify.linear(cbind(1, kerMat(newX, M, g)), wL)
  Ereg <- Err(newY, fitreg)
  fitker <- predict(modelS, as.data.frame(newX))
  Eker <- Err(newY, fitker)
  
#   par(mfrow = c(1, 2))
#     xtest <- seq(-1, 1, len = 1e3)
#     plot(X, col = clusters, main = "training set - k = 9"); points(M, pch = 19, col = 1:K, cex = 1.5)
#     lines(xtest, xtest - .25*sin(pi*xtest), lwd = 3)
#     lines(xFit, yhat)
#     plot(newX, col = fitreg+2, main = "test set"); points(newX, pch = 19, cex = .5, col = newY+2)
#     lines(xtest, xtest - .25*sin(pi*xtest), lwd = 3)
#   #plot(newX, col = c(1,3)[fitker], main = Eker); points(newX, pch = 19, cex = .5, col = newY+2)
#   par(op)

  return(Eker < Ereg)  
})
cat("\nQ14:", sum(out)/length(out)*100)
cat("\n\n")

# Q15: same as Q14 but use K = 12, gamma = 1.5
B = 300; N = 100; K = 12; g = 1.5
out <- sapply(1:B, function(b){
  if(b%%(B/10)==0) cat("B:", b, "\t")
  emptyK <- TRUE
  while(emptyK){
    X <- generateData(N)
    y <- target(X)
    km <- kmeans(X, centers = K, iter.max = 15, nstart = 10)
    emptyK <- length(unique(km$cluster)) != K
  }
  M <- km$centers
  wL <- Solve(cbind(1,kerMat(X, M, g)), y)
  modelS <- svm(y~., data=X, gamma = g, cost = 1e6,
                type = "C-classification", kernel = "radial")
  
  # Compute Eout only (on a fresh data set)
  newX <- generateData(N*10)
  newY <- target(newX)
  fitreg <- classify.linear(cbind(1, kerMat(newX, M, g)), wL)
  Ereg <- Err(newY, fitreg)
  fitker <- predict(modelS, data.frame(newX))
  Eker <- Err(newY, fitker)
#   par(mfrow = c(1, 2))
#     plot(newX, col = fitreg+2, main = Ereg)
#     plot(newX, col = as.numeric(fitker)+2, main = Eker)
#   par(op)
  return(Eker < Ereg)
})
cat("\nQ15:", sum(out)/length(out)*100)
cat("\n\n")

# Q16: use regular RBF only, with gamma = 1.5 and K from 9 to 12
testA <- function(Ein, Eout){
  n <- length(Ein)
  a1 <- all(sapply(2:n, function(i) Ein[i]<Ein[i-1]))
  a2 <- all(sapply(2:n, function(i) Eout[i]>Eout[i-1]))
  return(a1 & a2)
}
testB <- function(Ein, Eout){
  n <- length(Ein)
  b1 <- all(sapply(2:n, function(i) Ein[i]>Ein[i-1]))
  b2 <- all(sapply(2:n, function(i) Eout[i]<Eout[i-1]))
  return(b1 & b2)
}
testC <- function(Ein, Eout){
  n <- length(Ein)
  c1 <- all(sapply(2:n, function(i) Ein[i]>Ein[i-1]))
  c2 <- all(sapply(2:n, function(i) Eout[i]>Eout[i-1]))
  return(c1 & c2)
}
testD <- function(Ein, Eout){
  n <- length(Ein)
  d1 <- all(sapply(2:n, function(i) Ein[i]<Ein[i-1]))
  d2 <- all(sapply(2:n, function(i) Eout[i]<Eout[i-1]))
  return(d1 & d2)
}
testE <- function(Ein, Eout){
  e1 <- length(unique(Ein)) == 1
  e2 <- length(unique(Eout)) == 1
  return(e1 & e2)
}

B = 300; N = 100; K = c(9,12); g = 1.5
out <- mclapply(1:B, function(b){
  if(b%%(B/10)==0) cat("B:", b, "\t")
  emptyK <- TRUE
  while(emptyK){
    X <- generateData(N)
    y <- target(X)
    kms <- lapply(K, function(k) kmeans(X, centers = k, iter.max = 15, nstart = 10))
    emptyK <- sapply(1:length(kms), function(i){
      km <- kms[[i]]
      length(unique(km$cluster)) != K[i]
      })
    emptyK <- any(emptyK)
    }
  newX <- generateData(N*10)
  newY <- target(newX)
  tmp <- mclapply(kms, function(km){
    M <- km$centers
    kmat <- cbind(1, kerMat(X, M, g))
    wL <- Solve(kmat, y)
    fitin <- classify.linear(kmat, wL)
    Ein <- Err(y, fitin)
    fitout <- classify.linear(cbind(1, kerMat(newX, M, g)), wL)
    Eout <- Err(newY, fitout)
#     par(mfrow = c(1, 2))
#     plot(X, col = fitin+2, main = Ein)
#     plot(newX, col = fitout+2, main = Eout)
#     par(op)
    return(c(Ein = Ein, Eout = Eout))
    }, mc.cores = 2)
  E <- as.data.frame(do.call(rbind, tmp))
  A=testA(E$Ein, E$Eout);B=testB(E$Ein, E$Eout);C=testC(E$Ein, E$Eout)
  D=testD(E$Ein, E$Eout);E=testE(E$Ein, E$Eout)
  return(cbind(A=A, B=B, C=C, D=D, E=E))
}, mc.cores = 2)
out <- as.data.frame(do.call(rbind, out))
scores <- apply(out, 2, function(x) sum(x)/length(x))
cat("\nQ16:", names(scores)[which.max(scores)])
cat("\nScores:", scores)
cat("\n\n")

# Q17: use regular RBF only, with K = 9 and gamma from 1.5 to 2 (step = .1)
require(parallel)
B = 300; N = 100; K = 9; G = c(1.5, 2)
out <- mclapply(1:B, function(b){
  if(b%%(B/10)==0) cat("\nB:", b, "\t")
  emptyK <- TRUE
  while(emptyK){
    X <- generateData(N)
    y <- target(X)
    km <- kmeans(X, centers = K, iter.max = 10, nstart = 10)
    clusters <- km$cluster
    emptyK <- length(unique(clusters)) != K
  }
  M <- km$centers
  newX <- generateData(N*10)
  newY <- target(newX)
  tmp <- mclapply(G, function(g){
#    cat("\ng", g, "\t")
    kmat <- cbind(1, kerMat(X, M, g))
    wL <- Solve(kmat, y)
    fitin <- classify.linear(kmat, wL)
    fitout <- classify.linear(cbind(1, kerMat(newX, M, g)), wL)
    Ein <- Err(y, fitin)
    Eout <- Err(newY, fitout)
#    cat("Ein:", Ein, "Eout:", Eout)
#     par(mfrow = c(1, 2))
#       plot(X, col = fitin+2, main = paste(g, Ein))
#       plot(newX, col = fitout+2, main = paste(g, Eout))
#     par(op)
    return(c(Ein = Ein, Eout = Eout))
    }, mc.cores = 2)
  E <- as.data.frame(do.call(rbind, tmp))
  A=testA(E$Ein, E$Eout);B=testB(E$Ein, E$Eout);C=testC(E$Ein, E$Eout)
  D=testD(E$Ein, E$Eout);E=testE(E$Ein, E$Eout)
  return(cbind(A=A, B=B, C=C, D=D, E=E))
}, mc.cores = 2)
out <- as.data.frame(do.call(rbind, out))
scores <- apply(out, 2, function(x) sum(x)/length(x))
cat("\nQ17:", names(scores)[which.max(scores)])
cat("\nScores:", scores)
cat("\n\n")

# Q18
B = 1000; N = 100; K = 9; g = 1.5
out <- sapply(1:B, function(b){
  if(b%%(B/10)==0) cat("B:", b, "\t")
  emptyK <- TRUE
  while(emptyK){
    X <- generateData(N)
    y <- target(X)
    km <- kmeans(X, centers = K, iter.max = 15, nstart = 10)
    emptyK <- length(unique(km$cluster)) != K
  }
  M <- km$centers
  kmat <- cbind(1, kerMat(X, M, g))
  wL <- Solve(kmat, y)
  fitin <- classify.linear(kmat, wL)
  return(all(fitin==y))
})
#out <- do.call(c, out)
cat("Q18:", sum(out)/length(out)*100)
cat("\n\n")


# # Q20 simulate data then: g1 = ax+b, g2 is quadratic
# x <- rnorm(1000)
# y <- 2*x + rnorm(length(newX), 0, 1.5)
# g1 <- lm(y ~ x)
# g2 <- lm(y ~. , data=data.frame(x1=x^4, x2=x^3, x3=x^2))
# plot(x, y)
# lines(sort(x), predict(g1)[order(x)], col = "blue", lwd = 3)
# lines(sort(x), predict(g2)[order(x)], col = "red", lwd = 3)
# 
# newX <- rnorm(1000)
# newY <- 2*newX + rnorm(length(newX), 0, 1.5)
# fitg1 <- predict(g1, data.frame(x=newX))
# fitg2 <- predict(g2, data.frame(x1=newX^4, x2=newX^3, x3=newX^2))
# Eg1 <- 1/length(newY)*sum((fitg1 - newY)^2)
# Eg2 <- 1/length(newY)*sum((fitg2 - newY)^2)
# gFit <- 1/2*(fitg1 + fitg2)
# Eg <- 1/length(newY)*sum((gFit - newY)^2)
# Eg1; Eg2; 1/2*(Eg1+Eg2); Eg
# cat("Eg < Eg1:", Eg < Eg1)
# cat("Eg < mean(Eg1, Eg2):", Eg <= 1/2*(Eg1+Eg2))
# cat("Eg < min(Eg1, Eg2):", Eg <= min(Eg1,Eg2))
# cat("Eg is between Eg1, Eg2:", !Eg<=min(Eg1, Eg2) & !Eg>=max(Eg1, Eg2))
