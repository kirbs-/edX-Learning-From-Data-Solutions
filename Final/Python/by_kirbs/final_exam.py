#-------------------------------------------------------------------------------
# Name:        final exam
# Author:      kirbs
# Created:     12/5/2013
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import urllib
import numpy
import pylab
from sklearn import svm
import math
import random
from sklearn import cluster

# ###################################################
# ##################Question 2-6 Helpers  #############
# ###################################################

def in_dta():
    fpin = urllib.urlopen("http://www.amlbook.com/data/zip/features.train")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def out_dta():
    fpin = urllib.urlopen("http://www.amlbook.com/data/zip/features.test")
    return ([map(float,(line.strip('\n').split('\r')[0].split())) for line in fpin])

def transform(point):
    return [point[0], point[1], point[2], point[1] * point[2], point[1]**2, point[2]**2, point[3]]

def transformPoints(points):
    transformedPoints = []
    for point in points:
        transformedPoints.append(transform(point))
    return transformedPoints

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
##    print X.T
##    print numpy.dot(X.T,X)
##    print (l/len(samplePoints)*numpy.dot(weights, weights))
##    X_inverse = numpy.linalg.pinv(numpy.dot(X.T,X) + numpy.array(l/len(samplePoints)*numpy.dot(weights, weights)))
    X_inverse = numpy.linalg.pinv(X + numpy.array(l/len(samplePoints)*numpy.dot(weights, weights)))
##    print X_inverse.T
##    print y
##    print X_inverse
    return numpy.dot(X_inverse, y)

def d_vs_all(points, d):
    out = []
    for point in points:
        classified = 1 if point[0] == d else -1
        out.append([1, point[1], point[2], classified])
    return out

def d_vs_dd(points, d, dd):
    out = []
    for point in points:
        if point[0] == d or point[0] == dd:
            classified = 1 if point[0] == d else -1
            out.append([1, point[1], point[2], classified])
    return out


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


def q7():
    trainData = in_dta()
    for i in range(5,10):
        points = d_vs_all(trainData, i)
        weights = linRegWithRegularization(points, 1)
        print Ein(weights, points)



##q7()

def q8():
    trainData = in_dta()
    testData = out_dta()
    for i in range(5):
        points = transformPoints(d_vs_all(trainData, i))
        testPoints = transformPoints(d_vs_all(testData, i))
        weights = linRegWithRegularization(points, 1)
        print Ein(weights, testPoints)


##q8()


def q9():
    trainData = in_dta()
    testData = out_dta()
    for i in range(10):
        points = d_vs_all(trainData, i)
        transPoints = transformPoints(d_vs_all(trainData, i))
        testPoints = d_vs_all(testData, i)
        transTestPoints = transformPoints(d_vs_all(testData, i))
        weights = linRegWithRegularization(points, 1)
        transWeights = linRegWithRegularization(transPoints, 1)
        eout_norm = Ein(weights, testPoints)
        eout_trans = Ein(transWeights, transTestPoints)
        print "i: {} vs all, Enorm: {}, Etrans: {}, var:{}".format(i, eout_norm, eout_trans, eout_trans-eout_norm)

##q9()


def q10():
    trainData = in_dta()
    testData = out_dta()

    points = transformPoints(d_vs_dd(trainData, 1, 5))
    testPoints = transformPoints(d_vs_dd(testData, 1, 5))

    weights_001 = linRegWithRegularization(points, .01)
    weights_1 = linRegWithRegularization(points, 1)

    ein_001 = Ein(weights_001, points)
    ein_1 = Ein(weights_1, points)

    eout_001 = Ein(weights_001, testPoints)
    eout_1 = Ein(weights_1, testPoints)

    print "lamda: 1.00, e_in: {}, e_out: {}".format(ein_1, eout_1)
    print "lamda: 0.01, e_in: {}, e_out: {}".format(ein_001, eout_001)


##q10()

def transform_q11(point):
    z1 = point[1]**2 - 2*point[0] - 1
    z2 = point[1]**2 - 2*point[0] + 1
    return [z1, z2, point[2]]

def q11():
    data = [
        [1,0,-1],
        [0,1,-1],
        [0,-1,-1],
        [-1,0,1],
        [0,2,1],
        [0,-2,1],
        [-2,0,1]
    ]
    red_x = []
    red_y = []
    blue_x = []
    blue_y = []
    points = []

    for d in data:
        points.append(transform_q11(d))

    for point in points:
        if point[2] == -1:
            red_x.append(point[0])
            red_y.append(point[1])
        else:
            blue_x.append(point[0])
            blue_y.append(point[1])

    pylab.plot(red_x, red_y, 'ro', label = '-1\'s')
    pylab.plot(blue_x, blue_y , 'bo', label = '1\'s')
    pylab.xlim([-10,10])
    pylab.ylim([-10,10])
    pylab.show()


##q11()

def split(samplePoints):
    X = []
    y = []
    y_location = len(samplePoints[0]) -1 # y's location is assumed to be the last element in the list

    # Construct X space and split y values out
    for point in samplePoints:
        X.append(numpy.array(point[:y_location]))
        y.append(numpy.array(point[y_location]))

    return X, numpy.array(y)


def q12():
    data = [
        [1,0,-1],
        [0,1,-1],
        [0,-1,-1],
        [-1,0,1],
        [0,2,1],
        [0,-2,1],
        [-2,0,1]
    ]

    machine = svm.SVC(kernel='poly', degree = 2, coef0 = 1, C=100000, gamma = 1)
    X, y = split(data)

    machine.fit(X, y)
    print machine.n_support_

##q12()

def q13Target(x1, x2):
    return numpy.sign(x2 - x1 + .25 * math.sin(math.pi * x1))

def generatePoints(numOfPoints):
    out = []
    for i in range(numOfPoints):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        out.append([1, x1, x2, q13Target(x1, x2)])
    return out

def ein_svm(machine, points):
    errorCnt = 0
    for point in points:
        if point[3] != machine.predict(point[:3])[0]:
            errorCnt += 1
    return errorCnt/float(len(points))

def rbf_kernel():
    ein = 0
    for i in range(1000):
        points = generatePoints(100)

        machine = svm.SVC(kernel='rbf', C=1000000, gamma = 1.5)
        X, y = split(points)
        machine.fit(X, y)

        if ein_svm(machine, points) != 0.0:
            ein += 1

    print ein / float(100)

##rbf_kernel()

def lloyd(weights, X, U, gamma):
    out = 0.0

    for x in X:
        i = 0
        out = 0.0
        for k in U:
            out += (weights[i] * math.exp(-gamma * abs(x - k)**2))
            i += 1
    return numpy.sign(out)

def rbf_cluster(k_val):
    ein = 0
    gamma = 1.5
    for i in range(100):
        points = generatePoints(100)
        X, y = split(points)

        machine = cluster.KMeans(k = k_val, init = 'k-means++', n_init = 1)
        machine.fit(X)

        testPoints = generatePoints(100)
        X_test, y_test = split(testPoints)
        clusterResults = machine.predict(numpy.array(X_test))
        clusterPoints = machine.cluster_centers_

        matrix = []

        for point in X_test:
            temp = []
            for k in clusterResults:
                temp.append(math.exp(-gamma * abs(numpy.array(point) - numpy.array(k))**2))
            matrix.append(temp)


        clusterInverse = numpy.linalg.pinv(matrix)
        weights = numpy.dot(matrix, y_test)
        U = machine.cluster_centers_

        for point in testPoints:
            if point[3] != lloyd(weights, X_test, U, 1.5):
                ein += 1

    print ein /float(100)

rbf_cluster(9)










