'''
Created on 

@author: Mark
'''
import numpy as np
from hw8 import getData, prepareOneVsOne, prepareOneVsAll
from hw6 import weightDecayRegression, classificationError
from hw2 import Dataset, VSet, Function
from hw7 import writeV, readV
import hw2
import math
import hw8
import matplotlib
import warnings

def problem1():
  ''' HW5 problem 3 '''
  N = 10
  print ((N+1)*(N+2)/2 - 1)

# problem1()

def paint((X, Y), W, points=None):
  import matplotlib.pyplot as plt
  plt.plot(X[Y > 0,1], X[Y > 0,2], 'og')
  plt.plot(X[Y < 0,1], X[Y < 0,2], 'or')
#   x = np.linspace(0, .6)
  x = np.linspace(-3, 5)
  for w in W:
    if w[2] != 0:
      plt.plot(x, -(w[0] + w[1]*x)/w[2])
    else:
      plt.plot(-(w[0] + w[2]*x)/w[1], x)
  if points is not None:
    plt.scatter(points[:,0], points[:,1], c='yellow', alpha=.2, s=1)
  plt.grid()
#   plt.axis('equal')
  plt.show()
  
def paintLin((X, Y), W):
  import matplotlib.pyplot as plt
  plt.plot(X[Y > 0,1], X[Y > 0,2], 'og')
  plt.plot(X[Y < 0,1], X[Y < 0,2], 'or')
#   x = np.linspace(0, .6)
  x = np.linspace(-3, 5)
  for w in W:
    if w[2] != 0:
      plt.plot(x, -(w[0] + w[1]*x)/w[2])
    else:
      plt.plot(-(w[0] + w[2]*x)/w[1], x)
#   plt.grid()
#   plt.axis([-2.,7.,-4.,7.])
  plt.show()
  

def transform(X, param):
  if param == 'linear':
    return np.insert(X, 0, 1, 1)
  def f(x):
    x1, x2 = x[0], x[1]
    return [1, x1, x2, x1*x2, x1*x1, x2*x2]
  if param == 'nonlinear':
    return np.apply_along_axis(f, axis=1, arr=X)
  def g(x):
    x1, x2 = x[0], x[1]
    return [x2*x2 -2*x1 -1, x1*x1 - 2*x2 + 1]
  if param == 'problem11':
    return np.apply_along_axis(g, axis=1, arr=X)
  def z(x):
    x1, x2 = x[0], x[1]
    sqr2 = math.sqrt(2)
    return [x1*x1, x2*x2,  sqr2*x1, sqr2*x2, sqr2*x1*x2]
  if param == 'problem12':
    return np.apply_along_axis(z, axis=1, arr=X)



def regression(X, Y, X_out, Y_out, l, ts='linear'):
  X = transform(X, ts)
  X_out = transform(X_out, ts)
  w = weightDecayRegression((X, Y), l)
#   print w
  return classificationError((X, Y), w), classificationError((X_out, Y_out), w)

def problem7_9():
  train, test = getData()
  l = 1.
  class2 = 1
  for class1 in range(10):
    X, Y = prepareOneVsAll(train, class1)
    X_out, Y_out = prepareOneVsAll(test, class1)
    print '{} E_in {:} E_out {:}'.format(class1, *regression(X, Y, X_out, Y_out, l))
    print '{} E_in {:} E_out {:}'.format(class1, *regression(X, Y, X_out, Y_out, l, 'nonlinear'))
#     print w
#     paint((X, Y), [w])
  
#problem7_9()

def problem10():
  train, test = getData()
  class1 = 3
  class2 = 7
  for  l in [.01, 1.]:
    X, Y = prepareOneVsOne(train, class1, class2)
    X_out, Y_out = prepareOneVsOne(test, class1, class2)
    print 'lambda {} E_in {:} E_out {:}'.format(l, *regression(X, Y, X_out, Y_out, l, 'nonlinear'))
  
#problem10()

'''
x1 = (1; 0); y1 = -1 x2 = (0; 1); y2 = -1 x3 = (0;-1); y3 = -1
x4 = (-1; 0); y4 = +1 x5 = (0; 2); y5 = +1 x6 = (0;-2); y6 = +1
x7 = (-2; 0); y7 = +1
'''
trainSVM = np.array([
            [1, 0, -1], [0, 1, -1], [0, -1, -1],
            [-1, 0, 1], [0, 2, 1], [0, -2, 1],
            [-2, 0, 1]
            ])

def problem11():
  X = trainSVM[:,:-1]
  Y = trainSVM[:,-1]
  X = transform(X, 'problem11')
#   X = transform(X, 'problem12')
  print X
  ''' w1 = 1, w2 = 0, b = 0.5 '''
  w = np.array([-.5, 1, 0])
  paint((np.insert(X, 0, 1, 1), Y), [w])

#problem11()

from sklearn.svm import SVC


def predict(XX, model):
  def sign(x):
    return 1. if x >=0 else -1.
  def K(x0, x1):
    return (1 + x0.T.dot(x1))**2
  Y = []
  for x in XX:
    sum = 0.
    for alpha, sv in zip(model.dual_coef_[0], model.support_vectors_):
#       print K(x, sv)
#       print alpha
      sum += alpha*K(x, sv)
    sum -= model.intercept_
    Y.append(sign(sum))
  return np.array(Y)
#   return np.array([sign(np.sum([alpha*((xx.T.dot(x) + 1)**2) for alpha, x in 
#                                 zip(model.dual_coef_[0], model.support_vectors_)]) + model.intercept_) for xx in XX])


def problem12():
  X = trainSVM[:,:-1]
  Y = trainSVM[:,-1]
  model = SVC(C=1, kernel='poly', degree=2, gamma=1., coef0=1., verbose=True)
  model.fit(X, Y)
#   print 'xx', model.
  print model.dual_coef_, model.support_vectors_
  
  def K(x0, x1):
    return (1 + x0.T.dot(x1))**2
  def _b(A, X):
    return 1 - np.sum(A*np.array([K(x, X[A>0,:]) for x in X]))
  b = _b(model.dual_coef_, model.support_vectors_)
#   print X
  print transform(model.support_vectors_, 'problem12')
  print 'dual_coeff', model.dual_coef_
#   print 'mul', transform(model.support_vectors_, 'problem12')*model.dual_coef_
  w = transform(model.support_vectors_, 'problem12').dot(model.dual_coef_.T)
  x1 = X[:,0] 
#   x1 = linspace(np.min(x1),np.max(x1), 100)
  x1 = np.linspace(-2, 3, 100)
  x2 = X[:,1] 
  x2 = np.linspace(np.min(x2),np.max(x2), 100)
  XX = np.array([[x01,x02] for x01 in x1 for x02 in x2])
#   print 'computed w', w, b
#   print 'sum alphas', np.sum(model.dual_coef_)
#   b -= 1.2
#   w = np.array([[2.6*.320547], [1.5*.43822], [1.9*-.6411], [.0], [.0]])
#   print 'w', w, b
#   print transform(XX, 'problem12').shape, w[:,0].shape
  YY = predict(XX, model)
#   print YY
#   YY = model.predict(XX)
#   print XX.shape, YY.shape
  print len(XX[YY > 0, :])
  paint((np.insert(X, 0, 1, 1), Y), [], XX[YY > 0, :])
  
#   print
#   print model.support_vectors_
#   print transform(model.support_vectors_, 'problem11')


#problem12()  

class SimFunction(Function):
  def apply(self, X):
    return np.array([hw2.sign(x[2] - x[1] + .25*math.sin(math.pi*x[1])) for x in X])

def problem13():
  runs = 1000
  notSep = 0
  func = SimFunction()
  for run in range(runs):
    points = 100
    v = VSet(Dataset(points), func)
  #   writeV(points)
  #   v = readV()
    X = v.X[:,1:]
    Y = v.Y
    clf = SVC(kernel='rbf', C=1e6, gamma=1.5, coef0=1., verbose=False)
    clf.fit(X, Y)
#     hw8.plotGraph(X, Y, clf, func)
    e_in = hw8.classificationError(Y, clf.predict(X))
    if e_in > 0.0:
      print e_in
      notSep += 1
  print notSep, notSep/float(runs)
  
#problem13()

def drawGraph(X, U, S, k, Q):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'black']
  X = np.vstack((X, U))
  S = np.vstack((S, np.eye(len(U), dtype=bool)))
  for i, s in enumerate(S.T):
    plt.scatter(X[s,0], X[s,1], c=colors[i])
  plt.scatter(U[:,0], U[:,1], s=80, facecolors='none') #, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, hold)
  plt.scatter(Q[:,0], Q[:,1], c='red', s=60)
  plt.axis('tight')
  plt.show()
  
def dist2(D):
  return D.T.dot(D)
  
def cluster(X, U):
  def minDistanse(x):
    d = np.apply_along_axis(dist2, 1, U-x)
    return np.equal(d, min(d))
  return np.apply_along_axis(minDistanse, 1, X)

def meanU(X, S):
  def mean(s):
    return np.mean(X[s], 0)
  return np.apply_along_axis(mean, 0, S).T


def k_neighbour(X, k, rand=False):
  S = None
  for _i in range(100):
    v = VSet(Dataset(k))
    U = v.X[:,1:]
    if not rand:
      U = X[np.random.choice(np.arange(len(X)), k, replace=False)]
  
    for _ in range(10000):
      S = cluster(X, U)
#       print S.shape
#       print np.sum(S, 0)
#       print np.all(np.sum(S, 0), 0)
      if not np.all(np.sum(S, 0), 0):
#         print 'break'
        S = None
        break
      U_ = U
      U = meanU(X, S)
      if np.all(U_ == U):
        return U, S
    
  return

from sklearn.cluster import KMeans
import scipy.linalg as ln


def rbfTransform(X, U, gamma):
  Fi = np.array([[math.exp(-gamma * dist2(x - u)) for u in U] for x in X])
  return np.insert(Fi, 0, 1, 1)

def problem14():
  runs = 100
  notSep = 0
  bit = 0
  error = 0
  func = SimFunction()
  k = 9
  gamma = 1.5
  e_out = 0
  e_out_k = 0
  w0 = 0.0
  for run in range(runs):
    points = 100
    v = VSet(Dataset(points), func)
    X = v.X[:,1:]
    Y = v.Y
    try:
      U, S = k_neighbour(X, k, True)
#     clf = KMeans(k)
#     clf.fit(X)
#     U = clf.cluster_centers_
      Fi = rbfTransform(X, U, gamma)
      w = ln.lstsq(Fi, Y)[0]
      w0 = max(w0, np.max(np.abs(w)))
#       print w
      clf = SVC(C=1e6, kernel='rbf', gamma=gamma, coef0=1.)
      clf.fit(X, Y)
      ''' Generate E_out '''
      v = VSet(Dataset(1000), func)
      X = v.X[:,1:]
      Y = v.Y
      yy = rbfTransform(X, U, gamma).dot(w)
      yy[yy>=0]=1
      yy[yy<0]=-1
      rbf = hw8.classificationError(Y, yy)
      e_out += rbf
      kernel = hw8.classificationError(Y, clf.predict(X))
      e_out_k += kernel
      if kernel < rbf: bit += 1
    except:
      error += 1
  print bit, error, e_out, e_out_k, w0
#       print U
#     clf = KMeans(k)
#     clf.fit(X)
#       print clf.cluster_centers_
#     drawGraph(X, U, S, k, clf.cluster_centers_)
         
      
#     print S, step
    
#problem14()

def rbfClassifier(v, v0, k, gamma):
  X = v.X[:,1:]
  Y = v.Y
  U, S = k_neighbour(X, k, True)
  Fi = rbfTransform(X, U, gamma)
  w = ln.lstsq(Fi, Y)[0]
  yy = Fi.dot(w)
  yy[yy>=0]=1
  yy[yy<0]=-1
  e_in = hw8.classificationError(Y, yy)
  ''' Generate E_out '''
  X = v0.X[:,1:]
  Y = v0.Y
  yy = rbfTransform(X, U, gamma).dot(w)
  yy[yy>=0]=1
  yy[yy<0]=-1
  e_out = hw8.classificationError(Y, yy)
  return e_in, e_out

def problem16():
  runs = 100
  notSep = 0
  bit = 0
  error = 0
  func = SimFunction()
#   k = 12
  gamma = 1.5
  results = np.zeros(5)
  e = np.zeros(4)
  for run in range(runs):
    points = 100
    v = VSet(Dataset(points), func)
    v0 = VSet(Dataset(10), func)
    try:
      e_in9, e_out9 = rbfClassifier(v, v0, 9, 1.5)
      e_in12, e_out12 = rbfClassifier(v, v0, 9, 2.)
      e[0] += e_in9
      e[1] += e_out9
      e[3] += e_in12
      e[4] += e_out12
#       if e_in9 > e_in12 and e_out9 < e_out12: results[0] += 1
#       if e_in9 < e_in12 and e_out9 > e_out12: results[1] += 1
#       if e_in9 < e_in12 and e_out9 < e_out12: results[2] += 1
#       if e_in9 > e_in12 and e_out9 > e_out12: results[3] += 1
#       if e_in9 == e_in12 and e_out9 == e_out12: results[4] += 1
      if e_in9 < 0.001: results[4] += 1
    except:
      error += 1
  print bit, error, results, e

#problem16()

def paint2((X, Y), W):
  import matplotlib.pyplot as plt
  plt.plot(X, Y, 'og')
#   plt.plot(X[Y < 0,0], X[Y < 0,1], 'or')
#   x = np.linspace(0, .6)
  x = np.linspace(-1, 1)
  plt.plot(x, np.sin(np.pi*x))
  for w in W:
#     if w[1] != 0:
      plt.plot(x, w[0] + w[1]*x)
#     else:
# #       plt.plot(-(w[0] + w[2]*x)/w[1], x)
#       plt.plot(x, -(w[0] + w[1]*x))

  plt.grid()
#   plt.axis('equal')
  plt.show()
  
def problem20():
  def mySin(X):
    return np.sin(np.pi*X)
  def kPoints(k):
    return np.random.uniform(-1., 1., k).reshape(-1,1)
  def meanSquareError(X, Y, w):
    return ((X.dot(w)-Y)**2).mean()
  
  prob = np.zeros(4)
  e = np.zeros(2)
  for _ in range(100):
    for k in range(2,20):
      X = kPoints(k)
      Y = mySin(X)
      w1 = ln.lstsq(np.insert(X, 0, 1, 1), Y)[0]
#       print X
      X0 = np.apply_along_axis(lambda x: [1, x, x**2], 1, X)
#       print X0
      w2 = ln.lstsq(X0, Y)[0]
#       print w1, w2
#       paint2((X, Y), [w1, w2])
      sample = 100
      X_out = kPoints(sample)
      Y_out = mySin(X_out)
#       X_out = np.insert(X_out, 0, 1, 1)
      E_out1 = meanSquareError(np.insert(X_out, 0, 1, 1),Y_out, w1)
#       print E_out1
      E_out2 = meanSquareError(np.apply_along_axis(lambda x: [1, x, x**2], 1, X_out),Y_out, w2)
#       print E_out2
#       print np.insert(w1,2, 0, 0), w2
      w = .5*(np.insert(w1,2, 0, 0)+w2)
    #   print w1, w2, w
      E_out = meanSquareError(np.apply_along_axis(lambda x: [1, x, x**2], 1, X_out),Y_out, w)
    #   print E_out
#       print 'E1 {} E2 {} 1/2(E1+E2) {} E {}'.format(E_out1, E_out2, .5*(E_out1+E_out2), E_out)
      
      if (E_out < min(E_out1, E_out2)): prob[0] +=1
      if (E_out > max(E_out1, E_out2)): prob[1] +=1
      if (E_out < .5*(E_out1+E_out2)): prob[2] +=1
      if (E_out >= .5*(E_out1+E_out2)): prob[3] +=1
  print prob
     
  
#problem20()

def avg():
  def sigmoid(x, a, b):
    arg = a * x + b
    return np.exp(arg)/(1+np.exp(arg))
  import matplotlib.pyplot as plt
  x = np.linspace(-10, 10)
  a, b, c, d = 1, 1, -1, 2
  plt.plot(x, sigmoid(x, a, b))
  plt.plot(x, sigmoid(x, c, d))
  plt.plot(x, (sigmoid(x, a, b) + sigmoid(x, c, d))*.5, 'r')
  plt.grid()
#   plt.axis('equal')
  plt.show()

#avg()

def problem19():
  N = 10
  dist = np.zeros(N)
  dist[:] = 1/float(N)
  for h in range(N):
    dist[h] = h/float(N) * dist[h]
  print dist
  
#problem19()

# print math.exp(-10)
# print math.exp(-100)
