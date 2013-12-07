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
######################################

# load the kernlab package and the data
require(e1071)
options(digits = 15)

cat("Loading data...")
train <- getDat("http://www.amlbook.com/data/zip/features.train")
test <- getDat("http://www.amlbook.com/data/zip/features.test")
op <- par(no.readonly = TRUE)
cat("Done.\n\n")

# Q2
C = 0.01; Q = 2
Ein <- lapply(c(0, 2, 4, 8), function(label){
  y <- ifelse(train$digits==label, 1, -1)
  svp <- svm(y ~ ., data=as.data.frame(train[,-1]), scale = FALSE, coef0=1,
              type="C-classification", kernel='polynomial',
             gamma = 1, cost = C, degree = Q)
  fitSvm <- predict(svp)
  E <- sum(fitSvm != y)/length(y)
  cbind(label = label, Ein = E)
})
Ein <- as.data.frame(do.call(rbind, Ein))
#Ein
Max <- Ein$label[which.max(Ein$Ein)]
cat("Q2:", "\tMax Ein for", Max, "vs. all: Ein =", max(Ein$Ein), "\n")
cat("\n")

# Q3
C = 0.01; Q = 2
Ein <- lapply(c(1, 3, 5, 7, 9), function(label){
  y <- ifelse(train$digits==label, 1, -1)
  svp <- svm(y ~ ., data=as.data.frame(train[,-1]), scale = FALSE, coef0=1,
             type="C-classification", kernel='polynomial',
             gamma = 1, cost = C, degree = Q)
  fitSvm <- predict(svp)
  E <- sum(fitSvm != y)/length(y)
  cbind(label = label, Ein = E)
})
Ein <- as.data.frame(do.call(rbind, Ein))
#Ein
Min <- Ein$label[which.min(Ein$Ein)]
cat("Q3:", "\tMin Ein for", Min, "vs. all: Ein =", min(Ein$Ein), "\n")
cat("\n")

# par(mfrow = c(1, 2))
# for(label in c(0, 1)){
#   y <- ifelse(train$digits==label, 1, -1)
#   plot(train[,-1], pch = y+2, col = y+2, main = sprintf("%s Vs. all", label))
# }
# par(op)

# Q4
C = 0.01; Q = 2
y <- as.factor(ifelse(train$digits==0, 1, -1))
svp0 <- svm(y ~ ., data=as.data.frame(train[,-1]), scale = FALSE, coef0=1,
            type="C-classification", kernel='polynomial',
            gamma = 1, cost = C, degree = Q)
nSv0 <- svp0$tot.nSV

y <- as.factor(ifelse(train$digits==1, 1, -1))
svp1 <- svm(y ~ ., data=as.data.frame(train[,-1]), scale = FALSE, coef0=1,
            type="C-classification", kernel='polynomial',
            gamma = 1, cost = C, degree = Q)
nSv1 <- svp1$tot.nSV

cat("Q4\n")
cat("0 vs. all:\t", nSv0, "\n1 vs all:\t", nSv1, "\n")
cat("Absolute diff. of SVs:", abs(nSv0 - nSv1), "\n")
cat("\n")

# Q5
train2 <- train[which(train$digits %in% c(1, 5)),]
test2 <- test[which(test$digits %in% c(1, 5)),]
yin <- ifelse(train2$digits == 1, 1, -1)
yout <- ifelse(test2$digits == 1, 1, -1)

# par(mfrow = c(1, 2))
# plot(train2[,-1], pch = as.character(train2$digits), col = train2$digits, main = "Training set")
# plot(test2[,-1], pch = as.character(test2$digits), col = test2$digits, main = "Test set")
# par(op)

Q = 2
Cost <- c(1e-3, 1e-2, 1e-1, 1)
out <- lapply(Cost, function(C){
  svp <- svm(yin ~ ., data=as.data.frame(train2[,-1]), scale = FALSE, coef0=1,
             type="C-classification", kernel='polynomial',
             gamma = 1, cost = C, degree = Q)
  nsv <- svp$tot.nSV
  fitin <- predict(svp)
  Ein <- sum(fitin != yin)/length(yin)
  
  # out_of_samples
  fitout <- predict(svp, test2[,-1])
  Eout <- sum(fitout != yout)/length(yout)
  
  return(c(C = C, nSv = nsv, Ein = Ein, Eout = Eout))
})
out <- as.data.frame(do.call(rbind, out))
#out
dsv <- diff(out$nSv, lag=1)
dEout <- diff(out$Eout, lag=1)
A <- all(dsv<0); B <- all(dsv>0); C <- all(dEout<0); D <- out$Ein[which.max(out$C)]==min(out$Ein)
cat("Q5\n")
cat("a:", A, "\n")
cat("b:", B, "\n")
cat("c:", C, "\n")
cat("d:", D, "\n")
cat("e:", !any(A, B, C, D), "\n")
cat("\n")

# Q6
train2 <- train[which(train$digits %in% c(1, 5)),]
test2 <- test[which(test$digits %in% c(1, 5)),]
yin <- ifelse(train2$digits == 1, 1, -1)
yout <- ifelse(test2$digits == 1, 1, -1)

out <- lapply(c(1e-4, 1e-3, 1e-2, 1), function(C){
  tmp <- lapply(c(2, 5), function(Q){
    svp <- svm(yin ~ ., data=as.data.frame(train2[,-1]), scale = FALSE, coef0=1,
               type="C-classification", kernel='polynomial',
               gamma = 1, cost = C, degree = Q)
    nsv <- svp$tot.nSV
    fitin <- predict(svp)
    Ein <- sum(fitin != yin)/length(yin)
    
    # out_of_samples
    fitout <- predict(svp, test2[,-1])
    Eout <- sum(fitout != yout)/length(yout)
    
    return(c(C = C, Q = Q, nSv = nsv, Ein = Ein, Eout = Eout))
  })
  do.call(rbind, tmp)
})
out <- as.data.frame(do.call(rbind, out))
#out
A <- out[out$C==1e-4,]; A <- A$Ein[A$Q==5]>A$Ein[A$Q==2]
B <- out[out$C==1e-3,]; B <- B$nSv[B$Q==5]<B$nSv[B$Q==2]
C <- out[out$C==1e-2,]; C <- C$Ein[C$Q==5]>C$Ein[C$Q==2]
D <- out[out$C==1,]; D <- D$Eout[D$Q==5]<D$Eout[D$Q==2]
cat("Q6\n")
cat("a:", A, "\n")
cat("b:", B, "\n")
cat("c:", C, "\n")
cat("d:", D, "\n")
cat("e:", !any(A, B, C, D), "\n")
cat("\n")


# Q7
train2 <- train[which(train$digits %in% c(1, 5)),]
yin <- ifelse(train2$digits == 1, 1, -1)

Q = 2
B <- 100
Cost <- c(1e-4, 1e-3, 1e-2, 1e-1, 1)
nFold <- 10

bestCost <- lapply(1:B, function(b){
  ord <- sample(1:length(yin))
  tmp <- lapply(Cost, function(C){
      svp <- svm(yin[ord] ~ ., data=as.data.frame(train2[ord,-1]), scale = FALSE, coef0=1,
                 type="C-classification", kernel='polynomial',
                 cost = C, degree = Q, gamma = 1, cross = nFold)
      return(1 - svp$tot.accuracy/100)
      })
      E <- do.call(c, tmp)
      return(E)
      })
bestCost <- as.data.frame(do.call(rbind, bestCost))
Freqs <- apply(bestCost, 1, function(x) Cost[which.min(x)])
#table(Freqs)
best <- which.max(table(Freqs))
best <- as.numeric(names(best))
#apply(bestCost, 2, function(x) mean(x, na.rm = TRUE))
cat("Q7\t")
cat("Best cost:", best, "\n")
cat("\n")


# Q8
train2 <- train[which(train$digits %in% c(1, 5)),]
yin <- ifelse(train2$digits == 1, 1, -1)
C = best # from Q7
Q = 2
B <- 100
nFold <- 10

Ecv <- sapply(1:B, function(b){
  Ord <- sample(1:length(yin))
  train2 <- train2[Ord,]
  yin <- yin[Ord]
  svp <- svm(yin ~ ., data=as.data.frame(train2[,-1]), scale = FALSE, coef0=1,
             type="C-classification", kernel='polynomial',
             gamma = 1, cost = C, degree = Q, cross = nFold)
  return(1 - svp$tot.accuracy/100)
})
cat("Q8\t")
cat("Ave. Ecv:", mean(Ecv), "\n")
cat("\n")

# Q9/10
train2 <- train[which(train$digits %in% c(1, 5)),]
test2 <- test[which(test$digits %in% c(1, 5)),]
yin <- ifelse(train2$digits == 1, 1, -1)
yout <- ifelse(test2$digits == 1, 1, -1)

output <- lapply(c(1e-2, 1, 1e2, 1e4, 1e6), function(C){
  svp <- svm(yin ~ ., data=as.data.frame(train2[,-1]), scale = FALSE,
              type="C-classification", kernel='radial', cost=C, gamma=1)
  fitin <- predict(svp)
  Ein <- sum(fitin != yin)/length(yin)
  
  # out_of_samples
  fitout <- predict(svp, test2[,-1])
  Eout <- sum(fitout != yout)/length(yout)
  
  return(c(C = C, Ein = Ein, Eout = Eout))
})
output <- as.data.frame(do.call(rbind, output))
#output
cat("Q9: lowest Ein for C =", output$C[which.min(output$Ein)], "\n\n")
cat("Q10: lowest Eout for C =", output$C[which.min(output$Eout)], "\n\n")
cat("\n")
