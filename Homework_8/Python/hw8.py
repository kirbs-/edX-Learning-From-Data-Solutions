import os
import random
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn import cross_validation
#
# points_in = number of learning points
# points_out = number of test points
#
#read from file: thanks to vbipin
def read_data( filename ) :
    with open(filename, "r") as f:
        data = []
        for line in f:
            if line.strip() : #some empty lines are skipped if present. 
                y, x1, x2 = line.split()
                data.append( [ int(float(y)), float(x1), float(x2)] ) #data will be list of lists.
    return data

#eg: If you want the y and x of say 1_vs_all call 
#   y,x = d_vs_all( 1, data )
def d_vs_all(d, data) :
    #take out the y and x from data.
    y = [data[i][0] for i in range(len(data)) ]  #simple list
    x = [[data[i][1], data[i][2]] for i in range(len(data)) ]  #list of list

    #we need to put d as 1 and rest as -1 in y
    y_new = []
    contd = 0
    for i in range(len(y)) :
                if y[i] == d :
        #if abs( float( y[i] ) - d ) < 0.01 : 
            y_new.append( 1.0 )
            contd += 1

        else :
            y_new.append( -1.0 )

    #we do not want the np array.
    return y_new, x

#if you want 1 vs 5 call 
# y,x = d_vs_d( 1, 5, data)
def d_vs_d(d1, d2, data) :
    #take out the y and x from data.
    y = [data[i][0] for i in range(len(data)) ]  #simple list
    x = [[data[i][1], data[i][2]] for i in range(len(data)) ]  #list of list

    #we need to put d as 1 and rest as -1 in y
    y_new = []
    x_new = []
    for i in range(len(y)) :

        if y[i]  == d1 :
        #if abs( float( y[i] ) - d1 ) < 0.01 :
            y_new.append( 1.0 )
            x_new.append( x[i] )

        if  y[i]  == d2 :
        #if abs( float( y[i] ) - d1 ) < 0.01 :
            y_new.append( -1.0 )
            x_new.append( x[i] )

    return y_new, x_new
#for P2 : start = 0 stop = 10
#for P3-P4 : start = 1, stop = 11
#for P5-P10: start = 0, stop = 0
start = 0
stop = 0    
filename = "/Users/danielamaurizio/Documents/EDX-courses/CS1155x/Exercises/features.train"
data = read_data(filename)
print data[:20]
points_in = len(data)
filenameout = "/Users/danielamaurizio/Documents/EDX-courses/CS1155x/Exercises/features.test"
dataout = read_data(filenameout)
print dataout[:20]
points_out = len(dataout)
bestEin = 100.0
bestNSV = 0
jbest = -1
worstEin = 0.0
worstNSV = 0
jworst = -1
#following code is for P2-4
for j in range(start, stop, 2):
   coorxpos = []
   coorypos = []
   coorxneg = []
   cooryneg = []
   yn = []
   x = []
   yn, x = d_vs_all(j, data)
   #print yn[:20]
   #print x[:20]
   X = np.array(x)
   YN = np.array(yn)
   countj = 0
   for i in range(len(yn)):
       if yn[i] == 1.0:
           countj +=1
   print j, countj
   #insert code for quadratic programming SVM
   # fit the model
   clf = SVC(C = 0.01, kernel = 'poly', degree = 2, gamma = 1.0, coef0 = 1.0)
   clf.fit(X, YN)
   #print 'dual_coef_ holds the product yn_i * alpha_i'
   lagcoef = list(np.array(clf.dual_coef_).reshape(-1,))
   #print lagcoef
   #print 'w from svc'
   #print clf.coef_
   print 'support vectors from SLSQP'
   supvec = clf.support_vectors_
   #print supvec
   print 'indexes of support vectors and number of SV'
   ind = clf.support_
   nsv = len(ind)
   print ind, nsv
   alfacalc = []
   #calculate alfas
   for i in range(len(ind)):
       m = ind[i]
       alfacalc.append(-lagcoef[i]/yn[m])
   #print 'alfas'
   #print alfacalc
   #calculate vector w and b from qp results
   wvecsvm = np.zeros(shape = (1, 2))
   for i in range(nsv):
       wvecsvm += alfacalc[i]*yn[ind[i]]*supvec[i]
   print 'calculated vectors from alfas'
   wsvm = list(np.array(wvecsvm).reshape(-1,))
   print wsvm
   b = []
   averageb = 0.0
   for i in range(nsv):
       temp = 1.0/yn[ind[i]] - np.matrix.item(np.dot(supvec[i],wvecsvm.T))
       b.append(temp)
       averageb += temp
   averageb /= nsv
   print 'b average'
   print averageb
   y_pred_in = clf.predict(X) 
   Ein = 0.0
   for n in range(points_in):
       if y_pred_in[n] <> YN[n]:
           Ein += 1.0
   Ein /= float(points_in)
   if Ein < bestEin:
       bestEin = Ein
       bestNSV = nsv
       jbest = j
   if Ein > worstEin:
       worstEin = Ein
       worstNSV = nsv
       jworst = j
   sample_out = []
   yside = []
   yside, sample_out = d_vs_all(j, dataout)
   for i in range(len(yside)):
      if yside[i] == 1:
         coorxpos.append(sample_out[i][0])
         coorypos.append(sample_out[i][1])
      else:
         coorxneg.append(sample_out[i][0])
         cooryneg.append(sample_out[i][1])
   xqp_out = np.array(sample_out)
   print 'out of sample data for svm test'
   print xqp_out[:20], yside[:20]
   y_pred = clf.predict(xqp_out)
   print y_pred[:20]
   Eout = 0.0
   for i in range(points_out):
       if y_pred[i] <> yside[i]:
           Eout += 1.0
   Eout /= float(points_out)
   print j, Ein, Eout

   figXR = plt.figure()
   ax2 = figXR.add_subplot(111)
   plt.plot(coorxpos, coorypos, 'b*', label = 'positive')
   plt.plot(coorxneg, cooryneg, 'ro', label = 'negative')
   #plt.ylim(-1.0, 1.0)
   plt.xlim(-1.0, 1.0)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('digit ='+str(j))
   plt.legend(loc=1)
   plt.grid(True)
   plt.draw()
   #plt.show()
print 'j, NSV for best Ein'
print jbest, bestNSV, bestEin
print 'j, NSV for worst Ein'
print jworst, worstNSV, worstEin

#following code is for P5 & P6
print '\n start p5 P6 '
yn = []
x = []
#if kval = 0 no validation test
kval = 10
yn, x = d_vs_d(1, 5, data)
print yn[:10]
print yn[10:20]
print yn[20:30]
print x[:10]
print x[10:20]
print x[20:30]
if kval == 0:
    X = np.array(x)
    YN = np.array(yn)
    count1 = 0
    count5 = 0
    points_in = len(yn)
    for i in range(len(yn)):
        if yn[i] == 1.0:
            count1 += 1
        else:
            count5 += 1
    print ' number of 1s and 5s'
    print ('sample_in = ' + str(points_in))
    print count1, count5
    #insert code for quadratic programming SVM
    # fit the model for Q= 2 and Q=5 then for C in [0.001, 0.01, 0.1, 1]
    Q = [2, 5]
    uplim = [0.0001, 0.001, 0.01, 0.1, 1.0]
    for deg in Q:
        print ('polynomial degree: ' +str(deg))
        for ul in uplim:
            print ('value of C: ' + str(ul))
            clf = SVC(C = ul, kernel = 'poly', degree = deg, gamma = 1.0, coef0 = 1.0)
            clf.fit(X, YN)
            #print 'support vectors from SLSQP'
            supvec = clf.support_vectors_
            #print supvec
            #print 'indexes of support vectors and number of SV'
            ind = clf.support_
            nsv = len(ind)
            #print ind, nsv
            y_pred_in = clf.predict(X) 
            Ein = 0.0
            for n in range(points_in):
                if y_pred_in[n] <> YN[n]:
                    Ein += 1.0
            Ein /= float(points_in)
            sample_out = []
            yside = []
            yside, sample_out = d_vs_d(1, 5, dataout)
            xqp_out = np.array(sample_out)
            points_out = len(yside)
            #print 'out of sample data for svm test'
            #print xqp_out[:20], yside[:20]
            print ('sample_out = ' + str(points_out))
            y_pred = clf.predict(xqp_out)
            #print y_pred[:20]
            Eout = 0.0
            for i in range(points_out):
                if y_pred[i] <> yside[i]:
                    Eout += 1.0
            Eout /= float(points_out)
            print nsv, Ein, Eout
else:
    ntrials = 100
    X = np.array(x)
    YN = np.array(yn)
    averageEbest = [0.0, 0.0, 0.0, 0.0, 0.0]
    countbest = [0, 0, 0, 0, 0]
    for i in range(ntrials):
        bestEval = 1.0
        bestUL = 0.0 
        #shuffle data
        kf = cross_validation.KFold(len(YN), kval, indices=False, shuffle=True)
        for train, test in kf:
            X_t = X[train,:]
            Y_t = YN[train]
            X_v = X[test,:]
            Y_v = YN[test]
        #insert code for quadratic programming SVM
        # fit the model for Q= 2 then for C in [0.0001, 0.001, 0.01, 0.1, 1]
        Q = [2]
        uplim = [0.0001, 0.001, 0.01, 0.1, 1.0]        
        for deg in Q:
            #print ('polynomial degree: ' +str(deg))
            for ul in uplim:
                #print ('value of C: ' + str(ul))
                clf = SVC(C = ul, kernel = 'poly', degree = deg, gamma = 1.0, coef0 = 1.0)
                #cv = cross_validation.StratifiedShuffleSplit(YN, n_iter=10, test_size=0.1)
                #cv = cross_validation.ShuffleSplit(len(YN), n_iter=10, test_size=0.1)
                #scores = cross_validation.cross_val_score(clf, X, YN, cv=cv, scoring='accuracy')
                clf.fit(X_t, Y_t)
                scores = clf.score(X_v, Y_v)
                Eval = 1-scores.mean()
                if Eval < bestEval:
                    bestEval = Eval
                    bestUL = ul
                if Eval == bestEval:
                    bestUL = min(bestUL, ul)
        #print 'rsults for best Eval'
        #print i, bestUL, bestEval
        bestindex = uplim.index(bestUL)
        #print 'index ' + str(bestindex)
        countbest[bestindex] += 1
        averageEbest[bestindex] += bestEval
    for k in range(len(averageEbest)):
        averageEbest[k] /= float(ntrials)
    print 'results for validation run'
    print countbest, max(countbest), countbest.index(max(countbest)), uplim[countbest.index(max(countbest))]
    print 'average Eval ' + str(averageEbest)

#for problems 9-10
#data are from line 215
print '\n ****************************'
print 'start rbf model'
X = np.array(x)
YN = np.array(yn)
count1 = 0
count5 = 0
points_in = len(yn)
for i in range(len(yn)):
    if yn[i] == 1.0:
        count1 += 1
    else:
        count5 += 1
print ' number of 1s and 5s'
print ('sample_in = ' + str(points_in))
print count1, count5
#insert code for quadratic programming SVM
# fit the model for Q= 2 and Q=5 then for C in [0.001, 0.01, 0.1, 1]
uplim = [0.01, 1.0, 100.0, 10**4, 10**6]
for ul in uplim:
    print ('value of C: ' + str(ul))
    clf = SVC(C = ul, kernel = 'rbf', degree = 0, gamma = 1.0, coef0 = 1.0)
    clf.fit(X, YN)
    #print 'support vectors from SLSQP'
    supvec = clf.support_vectors_
    #print supvec
    #print 'indexes of support vectors and number of SV'
    ind = clf.support_
    nsv = len(ind)
    #print ind, nsv
    y_pred_in = clf.predict(X) 
    Ein = 0.0
    for n in range(points_in):
        if y_pred_in[n] <> YN[n]:
            Ein += 1.0
    Ein /= float(points_in)
    sample_out = []
    yside = []
    yside, sample_out = d_vs_d(1, 5, dataout)
    xqp_out = np.array(sample_out)
    points_out = len(yside)
    #print 'out of sample data for svm test'
    #print xqp_out[:20], yside[:20]
    print ('sample_out = ' + str(points_out))
    y_pred = clf.predict(xqp_out)
    #print y_pred[:20]
    Eout = 0.0
    for i in range(points_out):
        if y_pred[i] <> yside[i]:
            Eout += 1.0
    Eout /= float(points_out)
    print nsv, Ein, Eout