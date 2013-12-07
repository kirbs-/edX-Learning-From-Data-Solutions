'''
Created on 

@author: Mark
'''

from hw6 import dataSet, readIn, regression, classificationError
from cPickle import dump, load
import sklearn

def problem1():
  name = ['c://users//mark//in.dta', 'c://users//mark//out.dta']
  in_data = dataSet(readIn(name[0]))
  out_data = dataSet(readIn(name[1]))
  X, Y = in_data
  print 'Problem 1, 2'
  for k in (3, 4, 5, 6, 7):
    X_t, Y_t = X[:25, :k+1], Y[:25]
    X_v, Y_v = X[25:, :k+1], Y[25:]
    w = regression((X_t, Y_t))
    print k, classificationError((X_v, Y_v), w), classificationError((out_data[0][:, :k+1], out_data[1]) , w)
  print 'Problem 3, 4'
  for k in (3, 4, 5, 6, 7):
    X_t, Y_t = X[:25, :k+1], Y[:25]
    X_v, Y_v = X[25:, :k+1], Y[25:]
    w = regression((X_v, Y_v))
    print k, classificationError((X_t, Y_t), w), classificationError((out_data[0][:, :k+1], out_data[1]) , w)
#     print w.shape
#     break
  
# problem1()

import numpy as np

def problem6():
  N = 10000000
  e1 = np.random.uniform(size=N)
  e2 = np.random.uniform(size=N)
  e = np.minimum(e1, e2)
#   print e1
#   print e2
#   print e
  print np.mean(e1), np.mean(e2), np.mean(e)
  
# problem6()

def regression(train):
      a = train[:,:-1]
      b = train[:,-1]
      w = np.linalg.lstsq(a, b)[0]
#       print 'T', train
#       print '\tW', w
#       print 'Res', a.dot(w1), b
      return w


def problem7():
  RO = [(3**.5 + 4)**.5,
        (3**.5 - 1)**.5,
        (9 + 4 * 6 **.5)**.5,
        (9 - 6 **.5)**.5,
        ]
  for i, ro in enumerate(RO):
    points = [(-1.,0.), (ro, 1.), (1., 0.)]
    points = np.array(points)
    e0, e1 = 0, 0
    for row in range(3):
      val = points[row]
      train = points[points[:,0] != val[0],:]
      val = np.insert(val,  0, 1, 0)
      train = np.insert(train,  0, 1, 1)
      train0 = np.copy(train)
      train0[:,1] = 0
      w0 = regression(train0)
      w1 = regression(train)
      e0 += (val[:-1].dot(w0) - val[-1])**2
      e1 += (val[:-1].dot(w1) - val[-1])**2
    print i, ro, e0, e1
#       print train
#       print w0, w1
#       print val, e0, e1
#     break
    
# problem7()

# a = np.float64(1000.2)
# b = np.float64(1000.1)
# print '{:.30f}'.format(a/b) 


from hw2 import Dataset, VSet
import matplotlib.pyplot as plt
from sklearn import svm

def paint(v, w):
  X = v.X
  Y = np.array(v.Y)
  plt.plot(X[Y > 0,1], X[Y > 0,2], 'og')
  plt.plot(X[Y < 0,1], X[Y < 0,2], 'or')
#   plt.plot(x, np.sin(np.pi * x))
#   plt.plot(x, w[0] + w[1]*x*x)
#   plt.plot([x1, x2], [y1,y2], 'o') 
#   plt.plot([0,0], [0, 0], 'o')
  plt.show()
  
def writeV(N):
  v = VSet(Dataset(N))
  with open('v.dump', 'w') as f:
    dump(v, f)

def readV():
  with open('v.dump', 'r') as f:
    return load(f)

def problem8():
  points = 30
#   v = VSet(Dataset(points))
#   w, n = v.perceptron()
#   paint(v, w)
#   print v.outOfSampleError(w, 1000)
#   writeV(points)
  v = readV()
  X = v.X[:,1:]
#   X = sklearn.preprocessing.normalize(X)
#   print X_norm
  Y = v.Y
#   Y = [(y+1)/2 for y in Y]
#   sum()
  clf = svm.SVC(kernel='linear', verbose=True, tol=1e-9, C=5.)
#   clf = svm.NuSVC(kernel='linear', verbose=True, tol=1e-9, nu=1e-6)
#   clf = svm.LinearSVC(verbose=True, tol=1e-9)
#   clf = svm.SVC(kernel='rbf', verbose=True, tol=1e-3)
  clf.fit(X, Y)
#   print Y
  print len(clf.support_vectors_)
#   print clf.support_vectors_
  print clf.dual_coef_
  # get the separating hyperplane
  w = clf.coef_[0]
  a = -w[0]/w[1]
  xx = np.linspace(-1, 1)
  yy = a*xx - (clf.intercept_[0])/w[1]
  # plot the parallels to the separating hyperplane that pass through the
  # support vectors
  b = clf.support_vectors_[0]
  yy_down = a*xx + (b[1] - a*b[0])
  b = clf.support_vectors_[3]
  yy_up = a*xx + (b[1] - a*b[0])
  # plot the line, the points, and the nearest vectors to the plane
  plt.axis([-1.,1,-1.,1.])
#   plt.set_cmap(plt.cm.Paired)
  plt.plot(xx, yy, 'k-')
  plt.plot(xx, yy_down, 'k--')
  plt.plot(xx, yy_up, 'k--')
  plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
  s=80, facecolors='none')
  C = ['r' if y==1 else 'g' for y in Y]
  plt.scatter(X[:,0], X[:,1], c=C)
#   plt.axis('tight')
#   print clf.
  plt.show()
  
# problem8()

def test():
  a = np.arange(9).reshape((3, 3))
  np.insert
  print a
  print
  print a.T*a
  
# test()