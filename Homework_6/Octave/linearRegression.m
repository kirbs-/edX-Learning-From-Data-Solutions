function [weights] = linearRegression(X, y, lambda)

pseudoInv = inv(X' * X + lambda * eye(size(X, 2))) * X';
weights = pseudoInv * y;

end