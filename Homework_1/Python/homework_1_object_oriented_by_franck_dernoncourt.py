'''
CaltechX: CS1156x Learning From Data (introductory Machine Learning course)
Fall 2013
Homework 1

Q7-Q8:
number_of_training_points = 10 # N
('average_number_of_iterations: ', 10.555)
('average_generalization_error: ', 0.10651809999999991)

Q9-Q10:
number_of_training_points = 100 # N
('average_number_of_iterations: ', 103.353)
('average_generalization_error: ', 0.013376299999999994)

Created on Oct 3, 2013

Run time: less than a minute

@author: Franck Dernoncourt <franck.dernoncourt@gmail.com>
'''

import random

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
        print(self.w1, self.w2, self.b)

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

def estimate_disagreement(hypothesis_line, target_line, number_of_points):
    return len(find_misclassified_points(hypothesis_line, target_line, generate_points(number_of_points)))/ float(number_of_points)

def find_class(line, point):
    return (1 if (line.w1 * point.x + line.w2 * point.y + line.w0 > 0) else -1)

def is_misclassified_point(hypothesis_line, target_line, point):
    return (find_class(hypothesis_line, point) != find_class(target_line, point))

def find_misclassified_points(hypothesis_line, target_line, points):
    return [point for point in points if is_misclassified_point(hypothesis_line, target_line, point)]

def pick_random_misclassified_point(hypothesis_line, target_line, points):
    return random.choice(find_misclassified_points(hypothesis_line, target_line, points))

def update_hypothesis(hypothesis_line, misclassified_point, target_line, learning_rate):
    sign = find_class(target_line, misclassified_point)
    #print sign
    hypothesis_line.w0 += learning_rate * sign
    hypothesis_line.w1 += learning_rate * (misclassified_point.x * sign)
    hypothesis_line.w2 += learning_rate * (misclassified_point.y * sign)    
    return hypothesis_line

def run_pla(target_line, hypothesis_line, training_points, max_number_of_iterations, learning_rate):
    number_of_iterations = 0
    for i in range(max_number_of_iterations):
        #print "iteration"
        number_of_iterations += 1       
        #hypothesis_line.print_object() 
        if len(find_misclassified_points(hypothesis_line, target_line, training_points)) == 0: break
        #print len(find_misclassified_points(hypothesis_line, target_line, training_points))
        misclassified_point = pick_random_misclassified_point(hypothesis_line, target_line, training_points)
        #misclassified_point.print_object()
        hypothesis_line = update_hypothesis(hypothesis_line, misclassified_point, target_line, learning_rate)
    return hypothesis_line, number_of_iterations
    

def experiment():        
    max_number_of_iterations = 10000  
    number_of_runs = 1000
    number_of_training_points = 10 # N
    learning_rate = 1.0
    total_number_of_iterations = 0
    total_generalization_error = 0  
    for run_number in range(number_of_runs):
        hypothesis_line = Line(0, 0, 0) # Hypothesis  
        target_line = generate_random_line() # Target 
        training_points = generate_points(number_of_training_points) 
        #[training_point.print_object() for training_point in training_points] 
        #target_line.print_object()     
        hypothesis_line, number_of_iterations = run_pla(target_line, hypothesis_line, training_points, max_number_of_iterations, learning_rate)
        total_number_of_iterations += number_of_iterations
        total_generalization_error += estimate_disagreement(hypothesis_line, target_line, 10000)
        #hypothesis_line.print_object()
    average_number_of_iterations = total_number_of_iterations / float(number_of_runs)
    average_generalization_error = total_generalization_error / float(number_of_runs)
    print('average_number_of_iterations: ', average_number_of_iterations)
    print('average_generalization_error: ', average_generalization_error)

def main():
    experiment()

if __name__ == "__main__":    
    # Do some tests
    assert(intercept(1, 6, 3, 12) == 3.0)
    assert(intercept(6, 1, 1, 6) == 7.0)
    assert(intercept(4, 6, 12, 8) == 5.0)
    
    # Start program
    main()
    

    

