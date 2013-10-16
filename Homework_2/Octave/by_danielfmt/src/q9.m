addpath ("./common");

% X/Y range where the examples will be plotted
spc = [-1 1];

% Number of dimensions (excluding the synthetic dimension x0, which
% will be added later)
d = 2;

% Number of examples. Change this according to the question
N = 1000;

% N random examples
X = unifrnd (spc(1), spc(2), N, d);

% Uses the target function to set the desired labels
y = arrayfun (@target, X(:,1), X(:,2));

% Introduces noise
y = addnoise (y, 0.1);

% Introduces the synthetic dimension x0
X = [ones(N,1) X];

% Introduces the new features
X = addfeatures (X);

% Apply linear regression
w = linearreg (X, y);

% Labels assigned by w for each X
hy = sign (X * w);

% Functions to compare with our own g

function y = g1 (x1, x2)
  y = sign (-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
            + 1.5 * x1 * x1 + 1.5 * x2 * x2);
end

function y = g2 (x1, x2)
  y = sign (-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
            + 1.5 * x1 * x1 + 15 * x2 * x2);
end

function y = g3 (x1, x2)
  y = sign (-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
            + 15 * x1 * x1 + 1.5 * x2 * x2);
end

function y = g4 (x1, x2)
  y = sign (-1 - 1.5 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
            + 0.05 * x1 * x1 + 0.05 * x2 * x2);
end

function y = g5 (x1, x2)
  y = sign (-1 - 0.05 * x1 + 0.08 * x2 + 1.5 * x1 * x2 ...
            + 0.15 * x1 * x1 + 0.15 * x2 * x2);
end

% Pick manually the one that agrees with most with our hypothesis
results = [];

gy = arrayfun (@ (x1, x2) g1 (x1, x2) , X(:,2), X(:,3));
results(1,1) = mean (gy != hy);

gy = arrayfun (@ (x1, x2) g2 (x1, x2) , X(:,2), X(:,3));
results(1,2) = mean (gy != hy);

gy = arrayfun (@ (x1, x2) g3 (x1, x2) , X(:,2), X(:,3));
results(1,3) = mean (gy != hy);

gy = arrayfun (@ (x1, x2) g4 (x1, x2) , X(:,2), X(:,3));
results(1,4) = mean (gy != hy);

gy = arrayfun (@ (x1, x2) g5 (x1, x2) , X(:,2), X(:,3));
results(1,5) = mean (gy != hy);

printf ("Ein for g(1)...g(5) with respect to hypothesis w: [%f %f %f %f %f]\n", results);
