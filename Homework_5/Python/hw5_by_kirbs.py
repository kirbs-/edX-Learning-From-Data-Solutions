#-------------------------------------------------------------------------------
# Name:        homework 5
# Purpose:
#
# Author:      kirbs
#-----------------------
#!/usr/bin/env python

import numpy
import pylab
import random


def q1(variance, d):
    answers = {"a": 10,"b": 25, "c": 100, "d": 500, "e": 1000}

    for key, val in answers.iteritems():
        print key + ": " + str(variance**2 * (1 - ((d+1)/float(val))))


def gradientError(u, v):
    return (u*numpy.exp(v) - 2*v*numpy.exp(-u))**2

def dE_du(u, v):
    '''
    returns partial derivative of E with respect to u.
    '''
    return 2 * ( u*numpy.exp(v) - 2*v*numpy.exp(-u) ) * ( numpy.exp(v) + 2*v*numpy.exp(-u) )

def dE_dv(u,v):
    '''
    returns partial derivative of E with respect to v.
    '''
    return  2 * ( u * numpy.exp(v) - 2*v*numpy.exp(-u) ) * ( u*numpy.exp(v) - 2*numpy.exp(-u))

def updateWeights(weights, n):
    '''
    new_weight = old_weight - (learning rate) * (partial derivative w.r.t. u or v)
    '''
    u = weights[0]
    v = weights[1]
    weights[0] -= n * dE_du(u, v)
    weights[1] -= n * dE_dv(u, v)
    return weights


def q5():
    n = .1
    threshold = 10**-14
    weights = [1.0,1.0]
    iterations = 0

    while True:
        error = gradientError(weights[0], weights[1])
        iterations += 1

        if error < threshold or iterations >= 10000:
            print iterations
            break
        else:
            weights = updateWeights(weights, n)

    print gradientError(weights[0], weights[1])
    return weights


def q6():
    weights = q5()
    answers = {"a": (1.0, 1.0), "b": (.713, .045), "c": (.016, .112), "d": (-.083, .029), "e": (.045, .024)}

    for key, point in answers.iteritems():
        print key + ": " + str(numpy.sqrt((weights[0]-point[0])**2 + (weights[1]-point[1])**2))


def updateU(weights, n, error):
    weights[0] -= n
    return weights


def updateV(weights, n, error):
    weights[1] -= n
    return weights


def q7():
    n = .1
    weights = [1.0,1.0]
    iterations = 15

    for i in range(1,iterations):
        weights = updateU(weights, n, None)
        weights = updateV(weights, n, None)

    print gradientError(weights[0], weights[1])
    return weights


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
# #########################################


def stochasticLogisticRegression(threshold, weights, points, n):
    increment = 1.0
    iterations = 0

    while numpy.any(increment >= threshold):
        iterations += 1
        random.shuffle(points)
        oldWeights = list(weights)
        for point in points:
            ein = stochasticGradient(weights, point)
            weights -= (n*ein)

        increment = numpy.abs(oldWeights - weights)


    return weights, iterations



def crossEntErr(weights, points):
    vals = []
    for point in points:
        vals.append(numpy.log(1.0 + numpy.exp(-point[3] * numpy.dot(weights, point[:3]))))

    return numpy.mean(vals)


def stochasticGradient(weights, point):
    return (numpy.array(point[:3]) * point[3])/(1.0 + numpy.exp(point[3]*numpy.dot(weights, point[:3]))) * - 1


def q8(numberOfTrials, numberOfPoints):
    eouts = []
    iters = []
    threshold = .01
    n = .01
    testPoints = 1000

    for i in range(numberOfTrials):
        weights = numpy.array([0.0,0.0,0.0])
        x1, y1, x2, y2, points = generatePoints(numberOfPoints)
        weights, iterations = stochasticLogisticRegression(threshold, weights, points, n)
        errorCnt = 0

        # generate points for e_out
        x = []
        for i in range(testPoints):
            x_ = random.uniform(-1,1)
            y_ = random.uniform(-1,1)
            x.append([1, x_, y_, hw1TargetFunction(x1, y1, x2, y2, x_, y_)])

        eouts.append(crossEntErr(weights, x))
        iters.append(iterations)
    return numpy.mean(eouts), numpy.mean(iters)


print q8(1000,100)












