#! /usr/bin/python
#
# This is the answer code for the course "Learning from Data" on edX.org
# https://www.edx.org/course/caltechx/cs1156x/learning-data/1120
#
# The software is intended for course usage, no guarantee whatsoever.
# Date: 10/9/2013
# Created by: kirbs
# See notes at bottom for further details.

import numpy
import random
import pylab
import imp




"""
Generate random coin flips and return a result list
"""
def generateCoinFlipData(numberOfCoins, numberOfFlips):
    flippedCoinResults = []
    for i in range(numberOfCoins):
        coin = []
        for i in range(numberOfFlips):
            coin.append(random.randint(0, 1))
        flippedCoinResults.append(coin)

    return flippedCoinResults


def runQ1CoinSim(numberOfTrials, numberOfCoins, numberOfFlips):
    minCoins = []
    firstCoins = []
    randCoins = []
    for i in range(numberOfTrials):
        coinResults = generateFlipData(numberOfCoins, numberOfFlips)
        currentMin = 10
        minCoin = []
        for coin in coinResults:
            if sum(coin) < currentMin:
                currentMin = sum(coin)
                minCoin = coin
            if currentMin == 0:
                break
        minCoins.append(currentMin/10.0)
        firstCoins.append(sum(coinResults[0])/10.0)
        randCoins.append(sum(coinResults[random.randint(0,999)])/10.0)
    print "Average minimum coin " + str(numpy.mean(minCoins))
    print "Average first coin " + str(numpy.mean(firstCoins))
    print "Average random coin " + str(numpy.mean(randCoins))



"""
Helper function to visualize 2D data in [-1,1]x[-1,1] plane.
samplePoints is required; all other parameters are optional.
weights takes a list of weights and plots a line.
x1, y1, x2, y2 represents two points in a 2D target function;
this also plots the line.
"""
def plot(samplePoints, weights = None, x1 = None, y1 = None, x2 = None, y2 = None):
    red_x = []
    red_y = []
    blue_x = []
    blue_y = []

    for point in samplePoints:
        if point[3] == -1.0:
            red_x.append(point[1])
            red_y.append(point[2])
        else:
            blue_x.append(point[1])
            blue_y.append(point[2])

    pylab.plot(red_x, red_y, 'ro', label = '-1\'s')
    pylab.plot(blue_x, blue_y , 'bo', label = '1\'s')
    x = numpy.array( [-1,1] )
    if x1 is not None:
        # plot target function(black) and hypothesis function(red) lines
        slope = (y2-y1)/(x2-x1)
        intercept = y2 - slope * x2
        pylab.plot(x, slope*x + intercept, 'r')
    if weights is not None:
        pylab.plot( x, -weights[1]/weights[2] * x - weights[0] / weights[2] , linewidth = 2, c ='g', label = 'g') # this will throw an error if w[2] == 0
    pylab.ylim([-1,1])
    pylab.xlim([-1,1])
    pylab.legend()
    pylab.show()


"""
Calculate weights using linear regression.
Return list of weights.
"""
def linearRegression(samplePoints):
    X = []
    y = []
    y_location = len(samplePoints[0]) -1 # y's location is assumed to be the last element in the list

    # Construct X space and split y values out
    for point in samplePoints:
        X.append(numpy.array(point[:y_location]))
        y.append(point[y_location])

    X = numpy.array(X)
    y = numpy.array(y)
    X_inverse = numpy.linalg.pinv(X)

    return numpy.dot(X_inverse, y)

# ########################################
# Perceptron helper functions from HW 1 ##
# ########################################
def generatePoints(numberOfPoints):
##    random.seed(1) #used for testing
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    points = []

    for i in range (numberOfPoints):
##        random.seed(1)
        x = random.uniform (-1, 1)
        y = random.uniform (-1, 1)
        points.append([1, x, y, hw1TargetFunction(x1, y1, x2, y2, x, y)]) # add 1/-1 indicator to the end of each point list
    return x1, y1, x2, y2, points

def hw1TargetFunction(x1,y1,x2,y2,x3,y3):
    u = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    if u >= 0:
        return 1
    elif u < 0:
        return -1

# ##########################################

"""
Returns E_in error percentage for given weights and sample points.
Assumes samplePoints is a list of lists, and the last element in given list
is the y value.
"""
def Ein(weights, samplePoints):
    errorCount = 0
    y_location = len(samplePoints[0]) - 1
    for point in samplePoints:
        if numpy.sign(numpy.dot(weights,point[:y_location])) != point[y_location]:
            errorCount += 1

    return errorCount/float(len(samplePoints))


"""
Calculates the average E_in error of desired number of trials, using a new
set of sample points each time.
Returns average in sample error.
"""
def runQ5EinSimulation(numberOfTrials, numberOfPoints):
    ein_results = []
    for i in range(numberOfTrials):
        x1, y1, x2, y2, points = generatePoints(numberOfPoints)
        ein_results.append(Ein(linearRegression(points), points))

    return numpy.mean(ein_results)


"""
Calculates the average E_out error of desired number of trials, using a new
set of sample points each time and selecting a number of random points defined
in numberOfPoints parameter.
Returns average out of sample error.
"""
def runQ6EoutSimulation(numberOfTrials, numberOfPoints):
    eout_results = []
    for i in range(numberOfTrials):
        errorCount = 0
        x1, y1, x2, y2, points = generatePoints(numberOfPoints)
        weights = linearRegression(points)

        for i in range(numberOfPoints):
            point = [1, random.uniform(-1,1), random.uniform(-1,1)]
            if numpy.sign(numpy.dot(point, weights)) != numpy.sign(hw1TargetFunction(x1, y1, x2, y2, point[1], point[2])):
                errorCount += 1

        eout_results.append(errorCount/float(numberOfPoints))

    return numpy.mean(eout_results)



# ###############################
# Question 7 helper methods  ####
# ###############################
"""
Runs one trial based on the number of test points desired and an iteration limit to cap run time.
If showChart is set to True, this function with also return a chart of the points, target function and hypothesis.
Returns the number of iterations perceptron took to converge, final weights, and the error probability.
"""
def runPerceptron(numberOfTestPoints, iterationLimit, x1, y1, x2, y2, points, weights, showChart = False):
##    x1, y1, x2, y2, points = generatePoints(numberOfTestPoints)
    iterations, w = train(points, iterationLimit, weights)

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
        x = numpy.array( [-1,1] )
        slope = (y2-y1)/(x2-x1)
        intercept = y2 - slope * x2
        pylab.plot(x, slope*x + intercept, 'k', label = 'f(x)')
        pylab.plot( x, -w[1]/w[2] * x - w[0] / w[2] ,lw=3.0, c='r', ls='-.', label = 'f(g) starting') # this will throw an error if w[2] == 0
        pylab.plot( x, -weights[1]/weights[2] * x - weights[0] / weights[2] , 'm--', label = 'f(g) final') # this will throw an error if w[2] == 0
        pylab.ylim([-1,1])
        pylab.xlim([-1,1])
        pylab.legend()
        pylab.show()

    return iterations

"""
Plots sample points, and two lines based on weight lists.
Useful in showing how the perceptron is updating its weights during each
iteration.
"""
def showPlot(samplePoints, w1, w2):
    green_x = []
    green_y = []
    blue_x = []
    blue_y = []
    for x in samplePoints:
        if x[3] == 1:
            green_x.append(x[1])
            green_y.append(x[2])
        else:
            blue_x.append(x[1])
            blue_y.append(x[2])
    pylab.plot(green_x, green_y, 'go')
    pylab.plot(blue_x, blue_y, 'bo')

    # plot target function(black) and hypothesis function(red) lines
    x = numpy.array( [-1,1] )
    pylab.plot( x, -w1[1]/w1[2] * x - w1[0] / w1[2] ,lw=3.0, c='r', ls='-.', label = 'Before update') # this will throw an error if w[2] == 0
    pylab.plot( x, -w2[1]/w2[2] * x - w2[0] / w2[2] , 'm--', label = 'After update weights') # this will throw an error if w[2] == 0
    pylab.legend()
    pylab.show()

"""
Primary perceptron method.
Returns iteration count and final weight list.
"""
def train(training_points, iterationLimit, weights):
    w = weights.copy() # initialize weights for w[0], w[1], w[2]
    learned = False
    iterations = 0

    def updateWeights():
        random.shuffle(training_points) #randomize training points
        for point in training_points:
            res = numpy.sign(numpy.dot(w, point[:3])) #caclulate point
            if point[3] != res: # does point's y match our calculated y?
                #if not update weights
                w[0] += point[0]*point[3]
                w[1] += point[1]*point[3]
                w[2] += point[2]*point[3]
##                showPlot(training_points, weights, w)

                return False # break out of loop and return
        return True # if the loop reaches this point all calculated points in the training points match their expected y's


    while not learned:
        noErrors = updateWeights()
        if iterations == iterationLimit or noErrors:
            learned = True
            break
        if iterations >1:
            i = 0
        iterations += 1

    return iterations, w

# #############################################

def q7Simulation(numberOfTrials, numberOfPoints, showCharts = False):
    iterations = []
    for i in range(numberOfTrials):
        x1, y1, x2, y2, points = generatePoints(numberOfPoints)
        weights = linearRegression(points)
        iters = runPerceptron(numberOfPoints, 100, x1, y1, x2, y2, points, weights, showCharts)
        iterations.append(iters)
    pylab.hist(iterations)
    pylab.title('# of iterations')
    pylab.show()
    print "Average # of iterations " + str(numpy.mean(iterations))




# ####################################
# Question 8-10 helper methods     ###
# ####################################
"""
Questions 8-10 target function helper.
"""
def targetFunction(x1, x2):
    return numpy.sign(x1**2 + x2**2 - .6)

"""
Adds noise to samplePoints parameter based on numberToNoisify.
Returns noisified data.
"""
def noisify(numberToNoisify, samplePoints):
    random.shuffle(samplePoints) # randomize list
    out = []
    cnt = 0

    for point in samplePoints:
        if cnt < numberToNoisify:
            point[3] *= -1
        out.append(point)
        cnt += 1
    return out


def generateNoisyPoints(numberOfPoints):
    samplePoints = []
    for i in range(numberOfPoints):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)

        samplePoints.append([1, x1, x2, targetFunction(x1,x2)])

    return noisify(numberOfPoints/10, samplePoints)

# #########################################

def q8Simulation(numberOfTrials, numberOfPoints):
    results = []
    for i in range(numberOfTrials):
        points = generateNoisyPoints(numberOfPoints)
        weights = linearRegression(points)
        results.append(Ein(weights, points))

    print numpy.mean(results)



# ##########################################
# ##   Question 9-10 helper methods  #######
# ##########################################
def transform(point):
    return [1, point[1], point[2], point[1]*point[2], point[1]**2, point[2]**2, point[3]]


def transformPoints(samplePoints):
    out = []
    for point in samplePoints:
        out.append(transform(point))
    return out

# ##########################################

def runQ9Simulation():
    g_s = {
        "a": [-1.0, -.05, .08, .13, 1.5, 1.5],
        "b": [-1.0, -.05, .08, .13, 1.5, 15.0],
        "c": [-1.0, -.05, .08, .13, 15.0, 1.5],
        "d": [-1.0, -1.5, .08, .13, .05, .05],
        "e": [-1.0, -.05, .08, 1.5, .15, .15]
        }
    points = transformPoints(generateNoisyPoints(1000))
    weights = linearRegression(points)
    for key, value in g_s.items():

        errorCount = 0
        for point in points:
            if numpy.sign(numpy.dot(weights, point[:6])) != numpy.sign(numpy.dot(value, point[:6])):
                errorCount += 1
        print key +" agreement: " + str(1 - errorCount/float(len(points)))


def runQ10Simulation(numberOfTrials, numberOfPoints):
    error_perc = []
    for i in range(numberOfTrials):
        errorCount = 0
        points = transformPoints( generateNoisyPoints(numberOfPoints))
        weights = linearRegression(points)
        error_perc.append(Ein(weights, points))

    print numpy.mean(error_perc)

# #########################################################

# #########################################################
# ##############      Basic usage   #######################
# #########################################################

# Questions 1 & 2
# runCoinSim(numberOfTrials, numberOfCoins, numberOfFlips)
#runCoinSim(100000, 1000, 10)

# Question 5
# runQ5EinSimulation(numberOfTrials, numberOfPoints)
#print runQ5EinSimulation(1000, 100)

# Question 6
# runQ5EoutSimulation(numberOfTrials, numberOfPoints)
#print runQ6EoutSimulation(1000, 100)

# Question 7
# runQ7Simulation(numberOfTrials, numberOfPoints)
#q7Simulation(1000, 10)
# runQ7Simulation(numberOfTrials, numberOfPoints, showChart)
#q7Simulation(10, 10, True) # show chart of each trial.

# Question 8
# q8Simulation(numberOfTrials, numberOfPoints)
#q8Simulation(1000, 1000)

# Question 9
#runQ9Simulation()

# Question 10
# runQ10Simulation(numberOfTrials, numberOfPoints)
#runQ10Simulation(1000, 1000)



