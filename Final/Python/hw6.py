'''
Created on 

@author: Mark



'''
import numpy as np
import string
import scipy.linalg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
# from cvxopt import matrix

def readIn(name):
  return np.fromfile(name, np.float64, sep=' ').reshape((-1, 3))

def dataSet(d):
  def f(x):
    x1, x2 = x[0], x[1]
    return [1, x1, x2, x1*x1, x2*x2, x1*x2, abs(x1-x2), abs(x1+x2)] 
  return np.apply_along_axis(f, axis=1, arr=d), d[:,-1:]

def regression((X, Y)):
  return scipy.linalg.pinv(X).dot(Y)


def weightDecayRegression((X, Y), l=0.):
  I = np.identity(len(X[0]))
  inv = scipy.linalg.inv(X.T.dot(X) + l*I)
  return inv.dot(X.T).dot(Y)
  


def sign(array):
  array[array >= 0] = 1.
  array[array < 0] = -1.
  return array

def plotData(data, w):
  x = readIn('c://users//mark//out.dta')
  z = sign(data[0].dot(w))
  mis = np.equal(z, -data[1]).T
  pos = np.equal(z, data[1]).T
#   mis = np.equal(z, -data[1]).T
  
  p = x[mis[0,:]]
  plt.plot(p[:,0], p[:,1], 'ob')
  p = x[pos[0,:]]
  plt.plot(p[:,0], p[:,1], 'og')
#   pidx = [i for i, x in enumerate(idx[0])]
#   plt.plot(p[:,1], p[:,2], 'or')
  plt.show()
#   return np.sum(np.equal(sign(z), -data[1])) / float(len(data[0]))
#   return np.sum(np.equal(sign(z), -data[1])) / float(len(data[0]))

def classificationError(data, w):
  return np.sum(np.equal(sign(data[0].dot(w)), -data[1])) / float(len(data[0]))


def findClosest(options, computed):
  def f(a):
    return euclidean(a, computed)
  distances = np.apply_along_axis(f, axis=1, arr=options)
  idx = np.argmin(distances)
  return options[idx], distances[idx]

def problem2_6():
  name = ['c://users//mark//in.dta', 'c://users//mark//out.dta']
  train = dataSet(readIn(name[0]))
  test = dataSet(readIn(name[1]))
  w = weightDecayRegression(train, 0)
#   print w
  computed2 = [classificationError(train, w), classificationError(test, w)]
  print 'Without:', computed2
  for k in range(-5, 4):
    w = weightDecayRegression(train, 10 ** k)
    print k, classificationError(train, w), classificationError(test, w)#, w
    if k == -3:
      computed3 = [classificationError(train, w), classificationError(test, w)]
    if k == 3:
      computed4 = [classificationError(train, w), classificationError(test, w)]

  options2 = [[.03, .08], [.03, .1], [.04, .09], [.04, .11], [.05, .1]]
  options3 = [[.01, .02], [.02, .04], [.02, .06], [.03, .08], [.03, .1]]
  options4 = [[.2, .2], [.2, .3], [.3, .3], [.3, .4], [.4, .4]]
  print 'Problem 2', computed2, findClosest(options2, computed2)
  print 'Problem 3', computed3, findClosest(options3, computed3)
  print 'Problem 4', computed4, findClosest(options4, computed4)
  print 'Problem 5', -1, .056
  
# problem2_6()


def multiplyX(v):
  return np.insert(v[0:-1], 0, 0)


def legendre(P, n):
  return 1. / (n + 1) * ((2. * n + 1) * multiplyX(P[n]) - n * P[n - 1])

def init(N):
  P = np.zeros((N+1, N+1))
  P[0,0] = 1.
  P[1,1] = 1.
  for n in range(1,N):
    P[n+1] = legendre(P, n)
  return P
  

def createSet(P, Q, C, Q_o):
#   H = np.empty_like(['*',P[1]])
  H = np.zeros((0, 12))
#   print H.shape
#   print P[1].shape
#   np.reshape(H, (0, 11))
  for q in range(Q+1):
    symbol = 99
    if q >= Q_o:
      symbol = C
    if symbol != 0:
      v = np.insert(P[q][:], 0, symbol)
  #     print v.shape
      H = scipy.append(H, [v], axis=0)
  return H
    
def problem7():
  '''
  P(n+1) = 1/(n+1) * ((2*n+1)*x*P(n) - n*P(n-1))
  '''
  P = init(10)
  H = createSet(P, 10, 1, 3)
  print 'H(10,1,3)'
  print np.sum(H, axis=0)
  H = createSet(P, 10, 1, 4)
  print 'H(10,1,4)'
  print np.sum(H, axis=0)
  H = createSet(P, 1, 99, 5)
#   print H.shape
  print 'H(3,99,5)'
  print np.sum(H, axis=0)
#   print P
#   for n in range(10):
#     print P[n].dot(P[n+1])
  
  
  
# problem7()

def problem8():
  ''' Number operations in backpropagation '''
  L = 2
  d = [5, 3, 1]
  N = 0
  ''' deltas computation '''
  for l in range(L, 0, -1):
    for i in range (d[l-1]):
      ''' delta_i computation '''
      for j in range(1, d[l]+1):
        ''' sum delta_j * w_ij '''
        N += 1
  print N
  print N * 3
      
# problem8()

def weigthNumber(D):
  ''' 10 inputs and 1 output '''
  n = 0 
  d_prev = 10
  for d in D:
    n += d_prev * (d - 1)
    d_prev = d
  n += d_prev
  return n

def problem9_10():
  print weigthNumber([36])
  line18 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  print sum(line18)
  print weigthNumber(line18)
  return
  print weigthNumber([18, 18])
  print weigthNumber([17, 19])
  print weigthNumber([19, 17]) # 501
  print weigthNumber([20, 16]) # 506
  print weigthNumber([21, 15]) # 509
  print weigthNumber([22, 14]) # 510
  print weigthNumber([23, 13]) # 509
  print weigthNumber([22, 12, 2]) # 466
  print weigthNumber([21, 13, 2]) # 467
  print weigthNumber([20, 14, 2]) # 466
  return
  print weigthNumber([12, 12, 12])
  print
  print weigthNumber([11, 13, 12])
  print weigthNumber([11, 12, 13])
  
  print weigthNumber([12, 11, 13])
  print weigthNumber([13, 11, 12])
  
  print weigthNumber([13, 12, 11])
  print weigthNumber([12, 13, 11])

  print weigthNumber([11, 14, 11])
  print weigthNumber([12, 14, 10])
  print weigthNumber([12, 15, 9]) # 407
  print weigthNumber([12, 16, 8]) # 410
  print weigthNumber([13, 15, 8]) # 415
  print weigthNumber([14, 15, 7]) # 423
  print weigthNumber([13, 16, 7]) # 418
  print weigthNumber([15, 15, 6]) # 431 
  print weigthNumber([14, 16, 6]) # 426 
  
# problem9_10()
def test():
  a = matrix([[1,1], [0,0]])
  print a
  
# test()