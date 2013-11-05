# hw5

# Q1
s = .1; d = 8; eps = .008
N <- c(10, 25, 100, 500, 1000)
s^2*(1 - (d + 1)/N)
min(N[which(s^2*(1 - (d + 1)/N) > eps)])

# Q2
require(gplots)
f <- function(x1, x2, b0, b1, b2){return(b0 + b1*x1^2 + b2*x2^2)}
x1 <- x2 <- seq(-5, 5, len = 100)
out <- outer(x1, x2, f, 5, -10, 10)
image(x1, x2, out, col = colorpanel(100, "darkblue", "red"))
contour(x1, x2, out, add = T, col = "cyan")

# Q4
dEu <- function(u, v){
  return(2*(exp(v) + 2*v*exp(-u))*(u*exp(v) - 2*v*exp(-u)))
}
dEv <- function(u, v){
  return(2*(u*exp(v) - 2*exp(-u))*(u*exp(v) - 2*v*exp(-u)))
}

# Q5 & 6
E <- function(u, v){
  return((u*exp(v) - 2*v*exp(-u))^2)
}
dEu <- function(u, v){
  return(2*(exp(v) + 2*v*exp(-u))*(u*exp(v) - 2*v*exp(-u)))
}
dEv <- function(u, v){
  return(2*(u*exp(v) - 2*exp(-u))*(u*exp(v) - 2*v*exp(-u)))
}
eta <- 0.1
w <- c(1, 1)
eps <- 1e-14
counts <- 0
while(E(w[1], w[2]) > eps){
  w <- c(w[1] - eta*dEu(w[1],w[2]), w[2] - eta*dEv(w[1],w[2]))
  cat("E(u,v)", E(w[1], w[2]), "\t")
  cat("u", w[1], "v", w[2], "\n")
  counts = counts + 1
}
cat("Iterations:", counts, ",\tfinal values:", w)

# Q7
eta <- 0.1
w <- c(1, 1)
eps <- 1e-14
counts <- 0
while(counts <= 30){
  w[1] <- w[1] - eta*dEu(w[1],w[2])
  w[2] <- w[2] - eta*dEv(w[1],w[2])
  cat("E(u,v)", E(w[1], w[2]), "\t")
  cat("u", w[1], "v", w[2], "\n")
  counts = counts + 1
}
cat("Error:", E(w[1], w[2]), ",\tfinal values:", w)

# Q8 & 9
initLine <- function(){
  x1 <- runif(2, -.75, .75)
  x2 <- runif(2, -.75, .75)
  if(x2[1]>x1[1]){
    tmp <- x1
    x1 <- x2
    x2 <- tmp
  }
  b = -(x2[1] - x1[1])
  a = x2[2] - x1[2]
  c = -a*x1[1] - b*x1[2]
  w = c(a, b)
  w0 = c
  return(c(w0, w))
}
distance.from.plane <- function(x, w) {
  return(sum(t(w)%*%x))
}
classify.linear <- function(X, w) {
  distances = apply(as.data.frame(X), 1, distance.from.plane, w = w)
  return(ifelse(distances < 0, -1, +1))
}
crossEnt <- function(x, y, w){
  return(log(1 + exp(-y*t(w)%*%x)))
}
E <- function(X, y, w) {
  err <- sapply(1:nrow(X), function(i) crossEnt(X[i,], y[i], w))
  n <- length(err)
  return(1/n*sum(err))
}
sgd <- function(x, y, w){
  n <- length(y)
  etam <- y*x
  denom <- 1 + exp(y*t(w)%*%x)
  return(-etam/denom)
}
Norm <- function(w1, w2){
  sqrt(sum((w2 - w1)^2))
}
Eout <- function(n, w){
  test <- matrix(runif(n*2, -1, 1), n, 2)
  test <- cbind(1, test)
  realClass <- classify.linear(test, winit)
  return(E(test, realClass, w))  
}

Res <- c()
for(i in 1:100){
  cat("Run", i, "\n")
  n = 100
  X <- matrix(runif(n*2, -1, 1), n, 2)
  X <- cbind(1, X)
  winit <- initLine()
  y <- classify.linear(X, winit)

  eta = .01; w = c(0, 0, 0); tmp <- c(1, 1, 1); counts <- 0
  while(Norm(tmp, w)>0.01){
    w <- tmp
    for(i in sample(1:length(y))){
      tmp <- tmp - eta*sgd(X[i,], y[i], tmp)
      }
    counts = counts + 1
    }
    w <- tmp
    Res <- rbind(Res, c(counts, Eout(1000, w)))
  }
colMeans(Res)

# Graphics: linear and logistic regressions (using the last simulation within the loop)
op <- par(no.readonly = TRUE)

n = 200
X <- matrix(runif(n*2, -1, 1), n, 2)
X <- cbind(1, X)
y <- as.factor(classify.linear(X, w))

par(mfrow = c(1, 2), mar = c(5, 4.5, 4, 2))
plot(X[,-1], col = c("blue", "red")[y],
     xlab = expression(x[1]),
     ylab = expression(x[2]),
     main = expression(Y==sign(W^T*X)))
title(sub = sprintf("W = (%s, %s, %s)t", round(w[1], 3), round(w[2], 3), round(w[3], 3)))
abline(-w[1]/w[3], -w[2]/w[3], col = "goldenrod3", lty = 2, lwd = 4)

fit <- as.vector(t(w)%*%t(X))
pfit <- 1/(1+exp(-fit))
plot(c(0, 1)[y]~fit, col = c("blue", "red")[y],
     xlab = expression(W^T*X), ylab = "P(Y|X)", 
     main = expression(P(Y/X)==1/(1+e^-(W^T*X))))
lines(sort(fit), pfit[order(fit)], lwd = 4, col = "goldenrod3")
par(op)
