function iterations = linearToPerceptron()

[X, y, soln] = genData(10);
pseudoInv = inv(X' * X) * X';
weights = pseudoInv * y;

[weights, iterations] = perceptron(weights, X, y);

end