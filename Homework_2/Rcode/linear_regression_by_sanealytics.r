# Q1
library(plyr)

q1_Hoeffding <- function(...) {
  tot_coins = 1000
  tosses = 10
  # Toss 1000 coins 10 times
  # Could have used a shortcut here but being literal
#   coins <- matrix(rbinom(1000*tosses, 1, .5), nrow=1000,
#                   dimnames=list(paste0("coin", 1:1000),
#                                 paste0("trail", 1:tosses)))

  # Note: laply took a LOT longer than above
  coins <- laply(1:tot_coins, function(x) rbinom(tosses,1,.5))
  rownames(coins) <- paste0("coin", 1:tot_coins)
  colnames(coins) <- paste0("trial", 1:tosses)
  
  # Cound number of heads
  heads <- apply(coins, 1, sum)
  
  # pick first coin
#  c1 = coins[1,]
  
  # pick random coin
  randIdx <- sample(nrow(coins), 1)
#  c_rand = coins[randIdx, ]
  
  # Pick minimum 
  minIdx <- which(heads == min(heads))[1] # First one in case of tie
#  c_min <- coins[minIdx, ]
  
  cbind(v1 = heads[1]/tosses, vrand = heads[randIdx]/tosses, vmin = heads[minIdx]/tosses)
}


# Run experiment a bunch of times
exp_q1 <- laply(1:100000, q1_Hoeffding)

apply(exp_q1, 2, mean)
# v1    vrand     vmin 
# 0.499724 0.499814 0.037491 !!! 
# 0.499579 0.499819 0.037702 
# 0.499997 0.500360 0.037609 

linearRegression <- function(x, y) {
  # Add in bias
  x <- cbind(bias = rep(1, nrow(x)), x)
  
  # Calculate pseudo inverse
  pseudo = solve(t(x) %*% x) %*% t(x)
  w = pseudo %*% y
  error = sum(y != sign(t(w) %*% t(x)))
  list(w = w, error = error)
}

# Test one
fit <- linearRegression(x, y)

plot(x[,1], x[,2], pch = y+2, col = y+2)
abline(-fit$w[1]/fit$w[3], -fit$w[2]/fit$w[3], lty=4)

# Run many simulations
N <- 100
# N2 <- 1000
# iterations = data.frame()
iterations <- vector()
for (runs in seq(1000)) {
  # Create linearly separable data
  # Choose a line in X^2 {-1, 1}
  f <- matrix(c(runif(2, min = -1, max = 1), -1, 1),
              nrow = 2)
  # Get uniformly distributed points in {-1, 1}
  x <- matrix(runif(2 * N, min=-1, max=1), ncol=2)
  # Figure out which side of our line they are on
  y <- apply(x, 1, function(z) chooseSide(f, z))
  
#   x_out <- matrix(runif(2 * N2, min=-1, max=1), ncol=2)
#   y_out <- apply(x, 1, function(z) chooseSide(f, z))
#   
  # Fit linear model
  fit <- linearRegression(x, y)
  
  # Feed into PLA
  fit.pla <- PLA(x, y,maxit=10000, w = t(fit$w))
  
  # Out of sample error
#   x_out1 <- cbind(bias = rep(1, nrow(x_out)), x_out)
#   Eout <- sum(y_out != sign(t(fit$w) %*% t(x_out1)))
  
#   iterations <- rbind(iterations, cbind(Ein = fit$error/N, Eout = Eout/N2))
  iterations <- c(iterations, fit.pla$iter)
}

summary(iterations)
# Ein               Eout       
# Min.   :0.00000   Min.   :0.0000  
# 1st Qu.:0.02000   1st Qu.:0.3800  
# Median :0.03000   Median :0.4625  

# Mean   :0.03703   Mean   :0.4085  

# 3rd Qu.:0.05000   3rd Qu.:0.4890  
# Max.   :0.16000   Max.   :0.5470


# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.000   1.000   1.000   5.612   3.000 475.000 
# 1.000   1.000   1.000   5.363   1.000 584.000


N <- 1000
f <- function(x) sign(x[,1] ^2 + x[,2] ^2 - 0.6)
phi <- function(x) cbind(x[,1], x[,2], x[,1] * x[,2], x[,1]^2, x[,2]^2)

iterations <- data.frame()
iterations <- vector()
for(i in seq(1000)) {
  # Generate data
  x <- matrix(runif(2 * N, min=-1, max=1), ncol=2)
  y <- f(x)
  # Add Noise
  sampIdx <- sample(N, .1 * N)
  y[sampIdx] <- -y[sampIdx]

  # Get new data
  x_out <- matrix(runif(2 * N, min=-1, max=1), ncol=2)
  y_out <- f(x)
  # Add Noise
  sampIdx <- sample(N, .1 * N)
  y_out[sampIdx] <- -y_out[sampIdx]
  
  fit.lin <- linearRegression(x,y)
  fit.transformed <- linearRegression(phi(x), y)
  
  # Out of sample error
    x_out1 <- cbind(bias = rep(1, nrow(x_out)), phi(x_out))
    Eout <- sum(y_out != sign(t(fit.transformed$w) %*% t(x_out1)))
  
  iterations <- c(iterations, Eout/N)
  
#   iterations <- rbind(iterations, t(fit.transformed$w))
#   iterations <- c(iterations, fit.lin$error/N)
}

summary(iterations)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.3840  0.4690  0.5040  0.5056  0.5440  0.6010 


# bias               V2                   V3                   V4                   V5       
# Min.   :-1.0996   Min.   :-1.154e-01   Min.   :-1.451e-01   Min.   :-0.2121800   Min.   :1.313  
# 1st Qu.:-1.0182   1st Qu.:-2.843e-02   1st Qu.:-3.053e-02   1st Qu.:-0.0512053   1st Qu.:1.511  
# Median :-0.9922   Median : 7.166e-05   Median :-3.502e-04   Median : 0.0006909   Median :1.558  

# Mean   :-0.9921   Mean   : 5.774e-04   Mean   :-9.418e-05   Mean   : 0.0012421   Mean   :1.558  

# 3rd Qu.:-0.9679   3rd Qu.: 2.880e-02   3rd Qu.: 2.883e-02   3rd Qu.: 0.0546515   3rd Qu.:1.608  
# Max.   :-0.8448   Max.   : 1.512e-01   Max.   : 1.219e-01   Max.   : 0.2652424   Max.   :1.786  
# V6       
# Min.   :1.353  
# 1st Qu.:1.509  
# Median :1.561  

# Mean   :1.558  

# 3rd Qu.:1.605  
# Max.   :1.803  


# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.4550  0.4900  0.5010  0.5005  0.5100  0.5460 

# Wrongs
# 3(d) 4(e)
# 6(e)10(d)


