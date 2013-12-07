'''
Created on 

@author: Mark
'''
import numpy as np
import sklearn.svm as svm
from hw6 import readIn
import random
from sklearn.cross_validation import KFold
from ctypes.test.test_bitfields import func

def getData():
  return readIn('features.train'), readIn('features.test')
  

def classificationError(Y, Y_in):
  return sum(Y != Y_in) / float(len(Y))

def prepareOneVsAll(data, class1):
    return data[:,1:], (data[:,0] == class1)*2. - 1.

def prepareOneVsOne(data, class1, class2):
    sel = np.logical_or(data[:,0] == class1, data[:,0] == class2)
    return data[sel,1:], (data[sel,0] == class1)*2. - 1.

def polySVC(X, Y, C=.01, Q=2, coef0=1, verbose=False):
  clf = svm.SVC(kernel='poly', C=C, degree=Q, coef0=coef0, gamma=1., verbose=verbose)
  clf.fit(X, Y)
  return classificationError(Y, clf.predict(X)), len(clf.support_vectors_), clf

def problem2_4():
  train, test = getData()
  for class1 in (1, 3, 5, 7, 9):
#   for class1 in (0, 2, 4, 6, 8):
    res = polySVC(*prepareOneVsAll(train, class1))
    print '{} vs ALL E_in {:.4f} #SV {}' .format(class1, res[0], res[1])
#     print len(res[2].dual_coef_[0]), res[2].dual_coef_

# problem2_4()

def problem5_6():
    train, test = getData()
    class1 = 1
    class2 = 5
    X, Y = prepareOneVsOne(train, class1, class2)
    X_test, Y_test = prepareOneVsOne(test, class1, class2)
    for C in (.0001, .001, .01, 1):
      for Q in (2, 5):
        E_in, SV, clf = polySVC(X, Y, C=C, Q=Q)
        E_out = classificationError(Y_test, clf.predict(X_test))
        print 'C={:5} Q={} E_in {} E_out {} #SV {}' .format(C, Q, E_in, E_out, SV)

# problem5_6()

def problem7_8():
    train, test_out = getData()
    class1 = 1
    class2 = 5
    options = [.0001, .001, .01, .1, 1.]
    folds = 10
    X, Y = prepareOneVsOne(train, class1, class2)
    numberExperiments = 100
    C_choosen = np.zeros(len(options))
    E_cv_mean = np.zeros_like(C_choosen)
    for experiment in range(numberExperiments):
      kf = KFold(len(Y), folds, indices=False, shuffle=True)
      E_cv = np.zeros_like(E_cv_mean)
      for train, test in kf:
        X_t = X[train,:]
        Y_t = Y[train]
        X_v = X[test,:]
        Y_v = Y[test]
        for i, C in enumerate(options):
          clf = polySVC(X_t, Y_t, C=C)[2]
          E_cv[i] += classificationError(Y_v, clf.predict(X_v))
      C_choosen[np.argmin(E_cv)] += 1
      E_cv_mean += E_cv/folds
    print C_choosen, E_cv_mean/numberExperiments
    C = options[np.argmax(C_choosen)]
    X_v, Y_v = prepareOneVsOne(test_out, class1, class2)
    print 'C', C, 'E_out', classificationError(Y_v, polySVC(X, Y, C=C)[2].predict(X_v))
         
# problem7_8()

def plotGraph(X, Y, clf, func=None):
  import matplotlib.pyplot as plt
  xx1 = np.linspace(-1., 1., 100)
  xx2 = np.linspace(-1., 1., 100)
  S = np.array([[x1, x2] for x2 in xx2 for x1 in xx1])
  if func:
    Y_s = func.apply(np.insert(S, 0, 1, 1))
    Y = clf.predict(X)
  else:
    Y_s = clf.predict(S)
  color = np.empty_like(Y_s, dtype=np.str_)
  color[:] = 'white'
  color[ Y_s > 0 ] = 'yellow'
  pcolor = np.empty_like(Y, dtype=np.str_)
  pcolor[:] = 'green'
  pcolor[ Y > 0 ] = 'red'
  plt.scatter(S[:,0], S[:,1], c=color, alpha=.3) #, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, hold)
  plt.scatter(X[:,0], X[:,1], c=pcolor) #, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, hold)
  plt.axis('tight')
  plt.show()  

def problem9_10():
    train, test_out = getData()
    class1 = 1
    class2 = 5
    X, Y = prepareOneVsOne(train, class1, class2)
    X_v, Y_v = prepareOneVsOne(test_out, class1, class2)
    for C in ( .01, 1., 100., 1e4, 1e6):
      clf = svm.SVC(kernel='rbf', C=C, gamma=1., verbose=False)
      clf.fit(X, Y)
#       plotGraph(X, Y, clf)
      print 'C={:9} E_in {}\t E_out {}'.format(C,
              classificationError(Y, clf.predict(X)), classificationError(Y_v, clf.predict(X_v)))

              
# problem9_10()
    
def problemXX():
    train, test_out = getData()
    X, Y = train[:,1:], train[:,0]
    X_v, Y_v = test_out[:,1:], test_out[:,0]
    for C in (100.,):# .01, 1., 100., 1e4, 1e6):
      clf = svm.SVC(kernel='rbf', C=C, gamma=10., verbose=True)
      clf.fit(X, Y)
      plotGraph(X, Y, clf)
      print 'C={:9} E_in {}\t E_out {}'.format(C,
              classificationError(Y, clf.predict(X)), classificationError(Y_v, clf.predict(X_v)))

              
# problemXX()

def cv():
#   train, test = getData()
#   class1 = 1
#   class2 = 5
#   X, Y = prepareOneVsOne(train, class1, class2)
#   YY = 10*np.arange(10)
  from sklearn.cross_validation import KFold
  for i in range(10):
#     prm = np.random.permutation(10)
#     print a
#     Y = YY[a]
#     print YY
#     print Y
    print
#     kf = KFold(len(Y), 3, indices=False, shuffle=True)
#     for train, test in kf:
#       print Y[train]
#       print Y[test]
#       break
#   print kf
#   first = np.array([[train, test] for train, test in kf])
#   kf = KFold(len(Y), 3, indices=False, shuffle=True)
#   for train, test in kf:
#     print Y[train]
#     print Y[test]
#     break
#   second = np.array([[train,test] for train, test in kf])
#   print np.all(np.equal(first, second))
  
# cv()

def writeCSV():
  p = getData()[0]
  with open('out.csv', 'w') as f:
    for s in p:
#       str = ''
      for ss in s:
        f.write(str(ss)+',')
      f.write('\n')
  print p
    
# writeCSV()

