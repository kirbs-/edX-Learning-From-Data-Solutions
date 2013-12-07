'''
Created on 

@author: Mark
'''
import random
import numpy
from scipy import linalg

def uniformPoints(size):
  return [(random.uniform(-1., 1.), random.uniform(-1., 1.)) for i in range(size)]

class Coin(object):
  
  def __init__(self, p=.5):
    self.p = p
    
  def flip(self):
    return random.random() >= self.p
  
  def flipN(self, n):
    return sum([self.flip() for i in range(n)])/float(n)
  
class Dataset(object):
  
  def __init__(self, points):
    self.points = uniformPoints(points)
    self.line = uniformPoints(2)
    
  def sign (self, point):
    (x1, y1), (x2, y2) = self.line
    res = ((x2 - x1) * (point[1] - y1)) - ((y2 - y1) * (point[0] - x1))
    if res >= 0:
        return 1
    else:
        return -1
      
class LinearRegression(object):
  
  def __init__(self, dataset):
    self.set(dataset)
    
  def set(self, dataset):
    self.dataset = dataset
    
  def getw(self):
    X = numpy.array([[1, x, y] for (x, y) in self.dataset.points])
    Y = numpy.array([[self.dataset.sign(point)] for point in self.dataset.points])
    return linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
  
  def err(self):
    X = numpy.array([[1, x, y] for (x, y) in self.dataset.points])
    Y = numpy.array([[self.dataset.sign(point)] for point in self.dataset.points])
    return X.dot(self.getw()), Y
  
  def outerr(self, w):
    X = numpy.array([[1, x, y] for (x, y) in self.dataset.points])
    Y = numpy.array([[self.dataset.sign(point)] for point in self.dataset.points])
    return X.dot(w), Y
  
  def misplaced(self, w):
    X = numpy.array([[1, x, y] for (x, y) in self.dataset.points])
    Y = numpy.array([[self.dataset.sign(point)] for point in self.dataset.points])
    return numpy.array([(point, y) for point, h, y in zip(X, X.dot(w), Y) if h >= 0 and y < 0 or h < 0 and y >= 0])
  
def sign(a, b=0):
    if a>=0 and b<0 or a<0 and b>=0:
        return -1
    return 1

class Function(object):
    def apply(self, X):
        pass

class LineFunction(Function):
    
    def __init__(self, line):
        self.f = numpy.cross([1, line[0][0], line[0][1]], [1, line[1][0], line[1][1]])
        
    def  apply(self, X):
        return [sign(x) for x in X.dot(self.f)]

class NonLinearFunction(Function):
  
  def prob(self, p):
    return -1 if random.random() < p else 1
  
  def comp(self, x0, x1, x2):
    return x1*x1 + x2*x2 - 0.6

  def apply(self, X):
    return [self.prob(.1) * self.comp(*x) for x in X]

class VSet(object):

    def __init__(self, dataset, function=None):
        if function == None:
            function = LineFunction(dataset.line)
        self.function = function
        self.X = numpy.array([[1, x1, x2] for (x1, x2) in dataset.points])
        self.Y = self.function.apply(self.X)
        
    def regression(self):
        X = self.X; Y = self.Y
        return linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
      
    def misplaced(self, w):
      return [(x, y) for x, y in zip(self.X, self.Y) if sign(x.dot(w), y) < 0]

    def perceptron(self, w=numpy.zeros(3), max=1000):
#         print w/w[0]
        X = self.X; Y = self.Y
        for n in range(max):
            incorrect = self.misplaced(w)
            if len(incorrect) == 0:
              return w, n+1
            (x, y) = random.choice(incorrect)
            w += y * x
        return w, max
      
    def inSampleError(self, g):
      return len(self.misplaced(g))/float(len(self.X))

    def outOfSampleError(self, g, n):
      v = VSet(Dataset(n),self.function)
      return v.inSampleError(g)
    
    def transform(self):
      self.X = numpy.array([[1, x1, x2, x1*x2, x1*x1, x2*x2] for (x0, x1, x2) in self.X])

class NonlinearDataset(Dataset):
    
  def sign (self, point):
    x1, x2 = point
    res = (x1*x1 + x2*x2 - 0.6)
    if res >= 0:
        return 1
    else:
        return -1
  

  
def test(number):
  v = VSet(Dataset(1000), NonLinearFunction())
  w = v.regression()
#   x = v.X[number]
  return number, v.inSampleError(w), number