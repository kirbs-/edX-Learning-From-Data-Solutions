#-------------------------------------------------------------------------------
# Name:        homework 6
# Author:      kirbs
# Created:     11/9/2013
#-------------------------------------------------------------------------------
#!/usr/bin/env python


import urllib
import numpy


# ###################################################
# ##################Question 2-6 Helpers  #############
# ###################################################

def in_dta():
    fpin = urllib.urlopen("http://work.caltech.edu/data/in.dta")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def out_dta():
    fpin = urllib.urlopen("http://work.caltech.edu/data/out.dta")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def transform(point):
    return [1, point[0], point[1], point[0]**2, point[1]**2, point[0]* point[1],abs(point[0] - point[1]), abs(point[0] + point[1]), point[2]]

def transformPoints(points):
    transformedPoints = []
    for point in points:
        transformedPoints.append(transform(point))
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

    for point in samplePoints:
        X.append(numpy.array(point[:y_location]))
        y.append(point[y_location])

    weights = linearRegression(samplePoints)
    X = numpy.array(X)
    X_inverse = numpy.linalg.pinv(X + numpy.array(l/len(samplePoints)*numpy.dot(weights, weights)))
    return numpy.dot(X_inverse, y)

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
# ##################################################################

"""
Print in and out of sample errors.
"""
def q2():  
    points = in_dta()
    testPoints = out_dta()
    transformedPoints = transformPoints(points)
    transformedTestPoints = transformPoints(testPoints)
    weights = linearRegression(transformedPoints)
    print "E_in: {}, E_out: {}".format(Ein(weights, transformedPoints), Ein(weights, transformedTestPoints))


def q3(l):
    points = in_dta()
    testPoints = out_dta()
    transformedPoints = transformPoints(points)
    transformedTestPoints = transformPoints(testPoints)
    weights = linRegWithRegularization(transformedPoints, l)
    print "E_in: {}, E_out: {}".format(Ein(weights, transformedPoints), Ein(weights, transformedTestPoints))


# Question 3
#q3(10**-3)

# Question 4
#q3(10**3)

# Question 5
def q5(start, end):
    points = in_dta()
    testPoints = out_dta()
    transformedPoints = transformPoints(points)
    transformedTestPoints = transformPoints(testPoints)
    smallest_k = -2

    for i in range(start, end + 1):
        e_out = Ein(linRegWithRegularization(transformedPoints, 10**i), transformedTestPoints)
        print "k={}, E_out={}".format(i, e_out)

# Question 5
#q5(-2, 2)

# Question 6
#q5(-20, 20)





























