# Perceptron
PLA <- function(x, y, maxit = 1000, plotLn = FALSE, traceFlg = FALSE) {
  # Add in bias?
  x <- cbind(x, bias = rep(1, nrow(x)))
  # Initialize
  iter <- 0
  Cost <- 100
  w <- matrix(rep(0, ncol(x)), ncol=ncol(x))
  
  while(Cost > 0 & iter < maxit) {
    iter <- iter + 1
    # Predict
    h <- sign(x %*% t(w))
    
    misclassified <- which(h != y)
    Cost <- length(misclassified)
    
    if (Cost > 0) {
      # Update
      # hkReeves pointed out that sample(c(10),1) will give a random number between 1 and 10
      pickOne <- ifelse(Cost == 1, misclassified, sample(misclassified, 1))
      # Update weight vector
      # w <- w + t(y[pickOne]) %*% x[pickOne,]
      w <- w + y[pickOne] * x[pickOne,]
    }
    if(plotLn) abline(w[2], w[1], col = iter)
    if(traceFlg) print(paste("At iter", iter, "Cost ", Cost))
  }
  return(list(iter = iter, w = w))
}

# Pick which side of the line this should be on
chooseSide <- function(f, x) {
  m <- matrix(c(f[2,1] - f[1,1], x[1] - f[1,1], 
                f[2,2] - f[1,2], x[2] - f[1,2]), 
              byrow=TRUE, nrow=2)
  sign(det(m))
}


# Run many simulations
N <- 10
iterations = list()
for (runs in seq(1000)) {
  # Create linearly separable data
  # Choose a line in X^2 {-1, 1}
  f <- matrix(c(runif(2, min = -1, max = 1), -1, 1),
              nrow = 2)
  # Get uniformly distributed points in {-1, 1}
  x <- matrix(runif(2 * N, min=-1, max=1), ncol=2)
  # Figure out which side of our line they are on
  y <- apply(x, 1, function(z) chooseSide(f, z))
  
  # Fit perceptron model
  fit <- PLA(x, y, maxit=10000, plotLn=FALSE)
  iter <- fit$iter
  w <- fit$w
  iterations <- c(iterations, iter)
}

# Distribution of trials
summary(unlist(iterations))


# Test one
plot(x[,1], x[,2], pch = y+2, col = y+2)
lines(f, lwd = 3)

fit <- PLA(x, y, maxit=100, plotLn=FALSE, traceFlg=TRUE)

# Plot perceptron
abline(fit$w[2], fit$w[1], lty= 3)
title(paste("Perceptron at step", fit$iter))


# mydf <- as.data.frame(cbind(y, x))
# colnames(mydf) <- c("y", "x1", "x2")
# 
# library(ggplot2)
# qplot(x1, x2, data = mydf, color=factor(y)) + 
#   geom_abline(aes(intercept = w[2], slope = w[1]))

