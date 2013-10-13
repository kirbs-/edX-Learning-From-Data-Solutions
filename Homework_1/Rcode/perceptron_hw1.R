##############################
# Helper functions
##############################

initTarget <- function(n=2, min=-1, max=1){
  # Create a target function (a line in 2D) by picking 2 random points
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
  # Compute the distance of point z from a line,
  # given the components (w0, w) of a vector normal to the line
  return(w0 + sum(z*w))
}
classify.linear <- function(X, w0, w) {
  #Assign points to classes according to the target function defined by w0, w
  distances = apply(as.data.frame(X), 1, distance, w0 = w0, w = w)
  return(ifelse(distances < 0, -1, +1))
}
pickPoint <- function(Class, tmpClass, method = c("first", "random")){
  # What method to use to pick a misclassifed point: first point or random
  # Class: the true class
  # tmpClass: the temporary class at a given step of the optimization
  # method: "first" (first misclassified point), or "random" (any misclassifed point)
  switch(method,
         first = {idx = which(tmpClass != Class)[1]},
         random = {idx = sample(which(tmpClass != Class), 1)})
  return(idx)
}
addPolygon <- function(m0, m1){
  # Add colored areas
  f <- function(x, m0, m1){return(m1*x+m0)}
  polygon(x = c(-2,-2,2,2), y = c(-2, f(-2, m0, m1), f(2, m0, m1), -2), border = 'blue', col = rgb(0,0,1,0.1))
  polygon(x = c(-2,-2,2,2), y = c(f(-2, m0, m1), 2, 2, f(2, m0, m1)), border = 'red', col = rgb(1,0,0,0.1))
}
perceptron <- function(X, Class, w0 = 1, w = rep(1, ncol(X)), pick = "random",
                       rate = 1, Plot = FALSE){
  # inputs: - the points to classify, as a np matrix
  #         - The real class
  # output: the number of iterations
  
  # X: data matrix
  # Class: original class (from target function)
  # w0, w: initial weights to use.
  #        Initializing to 1 avoid errors if there are only few points to classify.
  # pick: what method to use to pick the misclassified points: "first", "random"
  # rate: learning rate
  # Plot: if TRUE, display the graph
  iter = 0
  missClass = TRUE
  while (missClass & iter<1000){
    missClass = FALSE
    iter = iter  + 1
    tmpClass <- classify.linear(X, w0, w)
    if(any(tmpClass != Class)){
      missClass = TRUE
        # use method "first" or "random" to pick a misclassified point.
      i <- pickPoint(Class, tmpClass, pick)
      w0 <- w0 + Class[i]*rate
      w <- w + Class[i]*X[i,]*rate
    }
  }
  if(Plot){
    plot(X, pch = ifelse(Class<0, '-', '+'),
         col = ifelse(Class<0, 'blue', 'red3'), cex = 1.5,
         xlim = range(-1, 1), ylim = range(-1, 1))
    cat(w0, w)
    m0 <- -w0/w[2]
    m1 <- -w[1]/w[2]
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
# PLA failed:
sum(is.na(Iter))
