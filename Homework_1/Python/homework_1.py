#! /usr/bin/python
#
# This is the answer code for the course "Learning from Data" on edX.org
# https://www.edx.org/course/caltechx/cs1156x/learning-data/1120
#
# The software is intended for course usage, no guarantee whatsoever.
# Date: 10/4/2013
# Created by: kirbs
# See notes at bottom for further details.

import sys
import os
import random
import pylab
import scipy
import numpy as np

#############################################################################
#############################################################################

# Returns a list of points with y (indicating 1/-1) as the last element
# and the x,y coordinates for the two points separating line.
# Returns a list of points; each point is a list in the following format.
# [x0, x1, x2, y] i.e. [dummy 1 to represent threshold, x1 value, x2 value, sample points correct sign (+1/-1)]
def generatePoints(numberOfPoints):
##    random.seed(1) # used for testing
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    points = []

    for i in range (0,numberOfPoints - 1):
##        random.seed(1) # used for testing
        x = random.uniform (-1, 1)
        y = random.uniform (-1, 1)
        points.append([1, x, y, targetFunction(x1, y1, x2, y2, x, y)]) # add 1/-1 indicator to the end of each point list
    return x1, y1, x2, y2, points

# This function determines the cross product between a line and a given point.
# Returns 1 if above the line and -1 if below the line.
def targetFunction(x1,y1,x2,y2,x3,y3):
    u = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    if u >= 0:
        return 1
    elif u < 0:
        return -1

# Simple sign function
def sign(y):
    if y >= 0:
        return 1
    elif y < 0:
        return -1

# a.k.a dot product
def perceptronCalc(x, w):
    return x[0]*w[0] + x[1]*w[1] + x[2]*w[2]

def train(training_points, iterationLimit):
    w = [0.0,0.0,0.0] # initialize weights for w[0], w[1], w[2]
    learned = False
    iterations = 0 # keep track of the iteration count
    
    # This method is the primary PLA implentation. 
    # It returns True when all sample points are corectly classfied by the hypothesis.
    # Returns False if there was a misclassified point and the weight vector changed.
    def updateWeights():
        random.shuffle(training_points) # randomize training points
        for point in training_points:
            result = sign(perceptronCalc(point,w)) # caclulate point and determine its sign.
            if point[3] != result: # does sample point's result match our calculated result?
                # Use line below to watch the perceptron's weights change
                # print str(iterations) + " " + str(w) + " " + str(result) + " " + str(point) + " " + str(perceptronCalc(point))
                
                # if not update weights by sample point's result
                w[0] += point[0]*point[3]
                w[1] += point[1]*point[3]
                w[2] += point[2]*point[3]


                return False # break out of loop and return
        return True # if the loop reaches this point all calculated points in the training points match their expected y's


    while not learned:
        iterations += 1
        noErrors = updateWeights() 
        if iterations == iterationLimit or noErrors:
            learned = True
            break

    return iterations, w

# Calculates approximate probability of hypothesis function returns a result
# that is different from the target function.
def findErrorProbability(x1,y1,x2,y2, weights, numberOfPointsToTest):
    numberOfErrors = 0
    for i in range(0, numberOfPointsToTest-1):
        #generate random test points
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)

        #compare results from target function and hypothesis function
        if targetFunction(x1,y1,x2,y2,x,y) != sign(perceptronCalc([1,x,y], weights)):
            numberOfErrors += 1 # keep track of errors
    return numberOfErrors/float(numberOfPointsToTest)


# Runs runTrial specified number of times.
# Returns average iterations, average error probability, and a histogram of trial iteration count.
def runSimulation(numberOfTrials, numberOfTestPoints, iterationLimit):
    interations = []
    probability = []
    for t in range(1,numberOfTrials+1):
        iteration_count, w, error_probability = runTrial(numberOfTestPoints, iterationLimit)
        interations.append(iteration_count)
        probability.append(error_probability)

    print "Avg. iterations: " + str(np.mean(interations)) + " : Avg. error probability: " + str(np.mean(probability))
    pylab.hist(interations)
    pylab.show()

# Runs one trial based on the number of test points desired and an iteration limit to cap run time.
# If showChart is set to True, this function with also return a chart of the points, target function and hypothesis.
# Returns the number of iterations perceptron took to converge, final weights, and the error probability.
def runTrial(numberOfTestPoints, iterationLimit, showChart = False):
    x1, y1, x2, y2, points = generatePoints(numberOfTestPoints)
    iterations, w = train(points, iterationLimit)
    errorProb = findErrorProbability(x1,y1,x2,y2,w, 10000)

    if showChart:
        if iterations == iterationLimit:
            print "No solution found in " + str(iterations) + " iterations!"
        print "Iterations: " + str(iterations) + ' | Weights: ' + str(w)

        # plot points above(green) and below(blue) the target function.
        green_x = []
        green_y = []
        blue_x = []
        blue_y = []
        for x in points:
            if x[3] == 1:
                green_x.append(x[1])
                green_y.append(x[2])
            else:
                blue_x.append(x[1])
                blue_y.append(x[2])
        pylab.plot(green_x, green_y, 'go')
        pylab.plot(blue_x, blue_y, 'bo')

        # plot target function(black) and hypothesis function(red) lines
        x = np.array( [-1,1] )
        slope = (y2-y1)/(x2-x1)
        intercept = y2 - slope * x2
        pylab.plot(x, slope*x + intercept, 'k--')
        pylab.plot( x, -w[1]/w[2] * x - w[0] / w[2] , 'r' ) # this will throw an error if w[2] == 0
        pylab.ylim([-1,1])
        pylab.xlim([-1,1])
        pylab.show()

    return iterations, w, errorProb

########################################################################
############################----NOTES----###############################
########################################################################
# Uncomment one line below and reload the script in your favorite Python
# environment. Or load the script and type the method with requireed
# paramaters you want to execute.


########################################################################
########################################################################
# runSimulation takes 3 arguments, number of trials to run, number of test points, and interation limit.
# The higher you set each parameter, the longer this method takes to run.
# This will return the average number of iterations the perceptron took to converge
# and the average error probability.

# Question 7/8
# runSimulation(1000, 10, 100)

# Question 9/10
# runSimulation(1000, 100, 1000)

#########################################################################
#########################################################################
# runTrial takes 3 arguments, number of points, iteration limit, and boolean if a chart should be shown.
# This method returns the number of iteration perceptron took to converge, the final
# weights vector, and the error probability.

# runTrial(10, 100, True) # Show graph of one trial with points, hypothesis (red line), and target funtion (black line).
# runTrial(10, 100) # No chart
# runTrial(10, 100, False) # No chart


