function [X, y, soln] = genData(N)
    
% Generate random points
X = unifrnd(-1, 1, N, 2);
p1 = unifrnd(-1, 1, 1, 2);
p2 = unifrnd(-1, 1, 1, 2);
X = sortrows(X);
[m, n] = size(X);
X = [ones(m, 1) X];

% Classify points based on random line
slope = slope(p1, p2);
b = p1(2) - slope*p1(1);
soln = [b ; slope ; -1];
y = sign(X * soln);

end