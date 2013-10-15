'''
CaltechX: CS1156x Learning From Data (introductory Machine Learning course)
Fall 2013
Homework 2

Q5
number_of_training_points = 100 # N
Ein = 0.02

Q6
Eout = 0.02

Q7
1


Created on Oct 10, 2013

Run time: less than a minute

@author: Franck Dernoncourt <franck.dernoncourt@gmail.com>
'''

import random
import numpy as np # for pseudo-inverse of matrix

class Point(object):
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def shuffle(self):
        self.x = random.uniform(-1.0, 1.0)
        self.y = random.uniform(-1.0, 1.0)
        return self
        
    def print_object(self):
        print(self.x, self.y)

class Line(object):
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0    

    def __init__(self, w0, w1, w2):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        
        
    def print_object(self):
        print(self.w0, self.w1, self.w2)

def slope (x1, y1, x2, y2):
    '''
    http://stackoverflow.com/questions/8640316/figuring-out-y-intercept-with-a-given-slope-in-python
    '''
    return float(y2 - y1)/(x2 - x1)

def intercept(x1, y1, x2, y2):
    '''
    http://stackoverflow.com/questions/8640316/figuring-out-y-intercept-with-a-given-slope-in-python
    '''
    return y2 - (slope(x1,y1,x2,y2)*x2)

def generate_random_line():
    x1, y1, x2, y2 = [random.uniform(-1.0, 1.0) for i in range(4)]
    return Line(-1*intercept(x1, y1, x2, y2), -1*slope (x1, y1, x2, y2), 1)
    #return Line(-0.5, 1, 0.5)
    #return Line(0, 1, -1) # Useful for debugging

def generate_point():
    return Point(0.0, 0.0).shuffle()

def generate_points(number_of_points):
    return [generate_point() for i in range(number_of_points)]

def estimate_eout(hypothesis_line, target_line, number_of_points):
    return len(find_misclassified_points(hypothesis_line, target_line, generate_points(number_of_points)))/ float(number_of_points)

def estimate_ein(hypothesis_line, target_line, training_points):
    return len(find_misclassified_points(hypothesis_line, target_line, training_points))/ float(len(training_points))

def find_class(line, point):
    return (1 if (line.w1 * point.x + line.w2 * point.y + line.w0 > 0) else -1)

def generate_target_vector(line, points):
    return [find_class(line, point) for point in points]
            
def is_misclassified_point(hypothesis_line, target_line, point):
    return (find_class(hypothesis_line, point) != find_class(target_line, point))

def find_misclassified_points(hypothesis_line, target_line, points):
    return [point for point in points if is_misclassified_point(hypothesis_line, target_line, point)]
    
def aggregate_points_into_a_matrix(points):
    x = np.empty(shape=(len(points),3)) 
    count = 0 
    for point in points:
        x[count] = [1, point.x, point.y]
        count += 1        
    return x

def run_linear_regression(training_points, target_line):
    '''
    Use the normal equation: http://en.wikipedia.org/wiki/Ordinary_least_squares
    w = ((t(X)X)^-1)t(X)Y
    '''
    x = aggregate_points_into_a_matrix(training_points)
    xdagger = np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)), np.transpose(x))
    target_vector = generate_target_vector(target_line, training_points) # target_vector, aka y
    w = np.dot(xdagger, target_vector)
    return Line(w[0], w[1], w[2])

def experiment():        
    max_number_of_iterations = 10000  
    number_of_runs = 1
    number_of_training_points = 100 # N
    learning_rate = 1.0
    total_number_of_iterations = 0
    total_generalization_error = 0  
    for run_number in range(number_of_runs):
        target_line = generate_random_line() # Target 
        training_points = generate_points(number_of_training_points)
        target_line.print_object()
        hypothesis_line = run_linear_regression(training_points, target_line)
        hypothesis_line.print_object()
        ein = estimate_ein(hypothesis_line, target_line, training_points)
        print ein
        eout = estimate_eout(hypothesis_line, target_line, 1000)
        print eout
        

def main():
    experiment()

if __name__ == "__main__":    
    # Do some tests
    assert(intercept(1, 6, 3, 12) == 3.0)
    assert(intercept(6, 1, 1, 6) == 7.0)
    assert(intercept(4, 6, 12, 8) == 5.0)
    
    # Start program
    main()
    

    

