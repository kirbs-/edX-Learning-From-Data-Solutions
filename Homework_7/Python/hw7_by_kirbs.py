#-------------------------------------------------------------------------------
# Name:        homework 7
# Author:      kirbs#
# Created:     11/16/2013
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import numpy
import urllib
import random
import sys
from sklearn import svm
from sklearn.grid_search import GridSearchCV

# ###################################################
# ##################  Helpers  ######################
# ###################################################

def in_dta():
    fpin = urllib.urlopen("http://work.caltech.edu/data/in.dta")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def out_dta():
    fpin = urllib.urlopen("http://work.caltech.edu/data/out.dta")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def transformPoint(p):
    return [1, p[0], p[1], p[0]**2, p[1]**2, p[0]*p[1], abs(p[0]-p[1]), abs(p[0]+p[1]), p[2]]

def transformPoints(points, slicePosition):
    transformedPoints = []
    for point in points:
        out = transformPoint(point)[:slicePosition + 1]
        out.append(point[-1])
        transformedPoints.append(out)
    return transformedPoints

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

def linRegWithRegularization(samplePoints, l):
    X = []
    y = []
    y_location = len(samplePoints[0]) -1 # y's location is assumed to be the last element in the list

    # Construct X space and split y values out
    for point in samplePoints:
        X.append(numpy.array(point[:y_location]))
        y.append(point[y_location])

    weights = linearRegression(samplePoints)
    X = numpy.array(X)
    X_inverse = numpy.linalg.pinv(X + numpy.array(l/len(samplePoints)*numpy.dot(weights, weights)))
    return numpy.dot(X_inverse, y)

def eVal(validationPoints, weights):
    y_location = len(validationPoints[0]) - 1
    e_val = 0

    for p in validationPoints:
        if numpy.sign(numpy.dot(weights, p[:y_location])) != numpy.sign(p[y_location]):
            e_val += 1

    return e_val/float(len(validationPoints))


# ##############################################################################



def q1():
    data = in_dta()

    for k in range(3,8):
        weights = linearRegression(transformPoints(data[:25], k))
        e_val = eVal(transformPoints(data[25:],k), weights)
        print "k={}, e={}".format(k, e_val)

#q1()


def q2():
    data = in_dta()
    testData = out_dta()

    for k in range(3,8):
        weights = linearRegression(transformPoints(data[:25], k))
        e_val = eVal(transformPoints(testData[25:],k), weights)
        print "k={}, e={}".format(k, e_val)


#q2()


def q3():
    data = in_dta()

    for k in range(3,8):
        weights = linearRegression(transformPoints(data[25:], k))
        e_val = eVal(transformPoints(data[:25],k), weights)
        print "k={}, e={}".format(k, e_val)

#q3()

def q4():
    data = in_dta()
    testData = out_dta()

    for k in range(3,8):
        weights = linearRegression(transformPoints(data[25:], k))
        e_val = eVal(transformPoints(testData[:25],k), weights)
        print "k={}, e={}".format(k, e_val)


#q4()




def q6():
    e = []
    e1 = []
    e2 = []
    for i in range(1000000):
        _e1 = random.uniform(0,1)
        _e2 = random.uniform(0,1)
        e.append(min(_e1, _e2))
        e1.append(_e1)
        e2.append(_e2)


    print "e_1={}, e_2={}, e={}".format(numpy.mean(e1), numpy.mean(e2), numpy.mean(e))


#q6()


def q7():

    def getPoints(val):
        return [[1, -1,0],[1, val,1],[1, 1,0]]

    def linear(points):
        return numpy.linalg.lstsq(points)

    answers = {"a": (3**.5+4)**.5, "b":(3**.5-1)**.5, "c": (3+4*6**.5)**.5, "d":(9-6**.5)**.5}
    e_cv = {}
    for key, ans in answers.iteritems():
        e_constant = []
        e_linear = []
        for i in range(3):
            points = getPoints(ans)
            del points[i]
            weights = linearRegression(points)

            # squared error
            for p in points:
                e_linear.append((numpy.dot(weights, p[:2]) - p[2])**2)
                e_constant.append((weights[0] - p[2])**2)
        print "ans={}, e_constant={}, e_linear={}".format(key, numpy.mean(e_constant), numpy.mean(e_linear))




#q7()





# #####################################################

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

# ###############################
# ######### Perceptron  #########
# ###############################

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
    w = weights[:] # initialize weights for w[0], w[1], w[2]
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

"""
Calculates the average E_out error of desired number of trials, using a new
set of sample points each time and selecting a number of random points defined
in numberOfPoints parameter.
Returns average out of sample error.
"""
def eOutPLA(testPoints, weights, x1, y1, x2, y2):
    errorCount = 0
    for point in testPoints:
        if numpy.sign(numpy.dot(point[:3], weights)) != numpy.sign(hw1TargetFunction(x1, y1, x2, y2, point[1], point[2])):
            errorCount += 1

    return errorCount/float(len(testPoints))

def eOutSVM(testPoints, svm, x1, y1, x2, y2):
    errorCount = 0
    for point in testPoints:
        if svm.predict([point[:3]])[0] != numpy.sign(hw1TargetFunction(x1, y1, x2, y2, point[1], point[2])):
            errorCount += 1
    return errorCount/float(len(testPoints))

def machinery(points, c):
    X = []
    y = []
    y_location = len(points[0]) -1 # y's location is assumed to be the last element in the list

    for point in points:
        X.append(numpy.array(point[:y_location]))
        y.append(point[y_location])

    machine = svm.SVC(kernel = 'linear', C=c)
    return machine.fit(X, y)

def estimator(points):
    X = []
    y = []
    y_location = len(points[0]) -1 # y's location is assumed to be the last element in the list

    for point in points:
        X.append(numpy.array(point[:y_location]))
        y.append(numpy.array(point[y_location]))

    params = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

    machine = GridSearchCV(svm.SVC(), params)
    machine.fit(X, numpy.array(y))
    return machine.best_estimator

def validPoints(points):
    has_0 = False
    has_1 = False

    for p in points:
        if p[-1] == -1:
            has_0 = True
        else:
            has_1 = True
        if has_0 and has_1:
            return True

    return False

# #############################################



def plaVSsvm(numOfTrials, numOfPoints):
    svm_better_cnt = 0
    sv_count = []
    for i in range(numOfTrials):
        x1, y1, x2, y2, points = generatePoints(numOfPoints)
        iterations, perc_weights = train(points, 100, [0,0,0])
        if(validPoints(points)):
            a,b,c,d, testPoints = generatePoints(10000)
            pla_e_out = eOutPLA(testPoints, perc_weights, x1, y1, x2, y2)
            machine = machinery(points, 1.0e6)
            svm_e_out = eOutSVM(testPoints, machine, x1, y1, x2, y2)
##            print "PLA {}, SVM {}, SVM better? {}".format(pla_e_out, svm_e_out, svm_e_out < pla_e_out)
            if svm_e_out < pla_e_out:
                svm_better_cnt += 1
                sv_count.append(numpy.sum(machine.n_support_))
##                print machine.n_support_

    return svm_better_cnt/float(numOfTrials), numpy.mean(sv_count)

# Question 8
#print plaVSsvm(1000,10)

# Question 9/10
#print plaVSsvm(1000, 100, 100.0)




















