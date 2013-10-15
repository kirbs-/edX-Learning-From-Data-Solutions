function [weights] = linearRegression(X, y)

pseudoInv = inv(X' * X) * X';
weights = pseudoInv * y;

end