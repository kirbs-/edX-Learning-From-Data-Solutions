#
#
#
# Created by: BartoszKP
#

import random
import math

def sign(x):
    if abs(x) < 0.000000001:
        return 0

    return x > 0 and 1 or -1

class Perceptron:
    def __init__(self, dimensionality):
        self.dimensionality = dimensionality
        self.weights = [0 for x in range(dimensionality + 1)]

    def setWeights(self, weights):
        if len(weights) != dimensionality + 1:
            raise Exception()

        self.weights = weights

    def combine(self, inputs):
        if len(inputs) != self.dimensionality:
            raise Exception()

        return sum([i*w for (i,w) in zip([1] + inputs, self.weights)])

    def classify(self, inputs):
        return sign(self.combine(inputs))

    def learn(self, point, label):
        self.weights = [w + label*x for (w,x) in zip(self.weights, [1] + point)]

    def asLine(self):
        return Line(-self.weights[1]/self.weights[2], -self.weights[0]/self.weights[2])

class Point:
    @staticmethod
    def random():
        return Point(random.uniform(-1, 1), random.uniform(-1, 1))

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def asVector(self):
        return [self.x, self.y]

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

class Line:
    @staticmethod
    def random():
        p1 = Point.random()
        p2 = Point.random()

        a = (p2.y - p1.y) / (p2.x - p1.x)
        b = p1.y - a * p1.x

        return Line(a, b)

    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def discriminate(self, point):
        return sign(self.a * point.x - point.y + self.b) or 1

    def __str__(self):
        return "y = " + str(self.a) + " x + " + str(self.b)

def discriminationDifference(line1, line2):
    diff = 0
    total = 10000
    for i in range(total):
        p = Point.random()
        if line1.discriminate(p) != line2.discriminate(p):
            diff += 1
    return 1.0 * diff / total

def weightsFromLine(line):
    w0 = -line.b
    w1 = -line.a
    w2 = 1

    return [w0, w1, w2]
    
def run(N):
    target = Line.random()

    trainingData = [(point, target.discriminate(point)) \
                    for point in [Point.random() for i in range(N)]]

    perceptron = Perceptron(2)
    
    it = 0
    misclassified = [t for t in trainingData if perceptron.classify(t[0].asVector()) != t[1]]
    while misclassified:
        m = random.choice(misclassified)
        perceptron.learn(m[0].asVector(), m[1])
        it += 1
        misclassified = [t for t in trainingData if perceptron.classify(t[0].asVector()) != t[1]]

    return (perceptron, trainingData, target, it)

itsum = 0
p = 0
total = 1000
for i in range(total):
    perceptron,data,target,it = run(100)
    p += discriminationDifference(perceptron.asLine(), target)
    itsum += it

print "Average it:", 1.0 * itsum / total
print "Average p:", 1.0 * p / total