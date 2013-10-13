##############################
# Helper functions
##############################

initTarget <- function(n=2, min=-1, max=1){
  x1 <- runif(n, min, max)
  x2 <- runif(n, min, max)
  b = -(x2[1] - x1[1])
  a = x2[2] - x1[2]
  c = -a*x1[1] - b*x1[2]
  w = c(a, b)
  w0 = c
  return(list(w0=w0, w=w))
}
distance <- function(z, w0, w) {
  return(w0 + sum(z*w))
}
classify.linear <- function(X, w0, w) {
  distances = apply(as.data.frame(X), 1, distance, w0 = w0, w = w)
  return(ifelse(distances < 0, -1, +1))
}
pickPoint <- function(Class, tmpClass, method = c("first", "random")){
  switch(method,
         first = {idx = which(tmpClass != Class)[1]},
         random = {idx = sample(which(tmpClass != Class), 1)})
  return(idx)
}
addPolygon <- function(m0, m1){
  f <- function(x, m0, m1){return(m1*x+m0)}
  polygon(x = c(-2,-2,2,2), y = c(-2, f(-2, m0, m1), f(2, m0, m1), -2), border = 'blue', col = rgb(0,0,1,0.1))
  polygon(x = c(-2,-2,2,2), y = c(f(-2, m0, m1), 2, 2, f(2, m0, m1)), border = 'red', col = rgb(1,0,0,0.1))
}
perceptron <- function(X, Class, w0 = 0, w = rep(0, ncol(X)), pick = "random",
                       rate = 1, Plot = FALSE){
  iter = 0
  missClass = TRUE
  while (missClass & iter<1000){
    missClass = FALSE
    iter = iter  + 1
    tmpClass <- classify.linear(X, w0, w)
    if(any(tmpClass != Class)){
      missClass = TRUE
        # use method "first" or "random" to get
        #  the first misclassified point or a random one, resp.
      i <- pickPoint(Class, tmpClass, pick)
      w0 <- w0 + Class[i]*rate
      w <- w + Class[i]*X[i,]*rate
        # Keep track of the slope and intercept for graphics.
      m0 <- -w0/w[2]
      m1 <- -w[1]/w[2]
    }
  }
  if(Plot){
    plot(X, pch = ifelse(Class<0, '-', '+'),
         col = ifelse(Class<0, 'blue', 'red3'), cex = 1.5,
         xlim = range(-1, 1), ylim = range(-1, 1))
    addPolygon(m0, m1)
    abline(m0, m1, lwd = 3)
  }
  if(iter >= 1000) iter <- NA
  return(iter)
}
##############################

# Choose 2 points to define the classifier
# Generate some random points and assign then a group with respect to the line
# run the perceptron:
N = 1000
Iter <- sapply(1:N, function(i){
  if(i%%(N/10) == 0) cat(i, 'runs\t')
  init <- initTarget()
  w0 <- init$w0
  w <- init$w
  n = 10
  X = matrix(runif(n*2, -1, 1), n, 2)
  Class <- classify.linear(X, w0, w)
  perceptron(X, Class)
})
# mean iterations:
mean(Iter, na.rm = TRUE)
# PLA faied:
sum(is.na(Iter))