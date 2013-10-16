addpath ("./common");

% Number of times the algorithm will run in order to calculate the
% average number of steps taken
iters = 1000;

% X/Y range where the examples will be plotted
spc = [-1 1];

% Number of dimensions (excluding the synthetic dimension x0, which
% will be added later)
d = 2;

% Number of examples. Change this according to the question
N = 1000;

errors = zeros (iters, 1);

for i = 1:iters

  % N random examples
  X = unifrnd (spc(1), spc(2), N, d);

  % Uses the target function to set the desired labels
  y = arrayfun (@target, X(:,1), X(:,2));

  % Introduces noise
  y = addnoise (y, 0.1);

  % Introduces the synthetic dimension x0
  X = [ones(N, 1) X];

  % Introduces the new features
  X = addfeatures (X);

  % Apply linear regression
  w = linearreg (X, y);

  errors(i) = eout (N, d, spc, @target, 0, w, 0.1);
end

fprintf ("Eout(X) = %f\n", mean (errors));
