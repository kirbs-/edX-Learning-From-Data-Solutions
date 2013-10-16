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
N = 10;

itercount = zeros (iters, 1);

for i = 1:iters

  % Loops until we find negative and positive examples
  while (1)

    % Two random points used in target function f
    fp1 = unifrnd (spc(1), spc(2), 2, 1);
    fp2 = unifrnd (spc(1), spc(2), 2, 1);

    % N random examples
    X = unifrnd (spc(1), spc(2), N, d);

    % Introduces the synthetic dimension x0
    X = [ones(N, 1) X];

    % Weight vector that represents the target function f
    wf = unifrnd (spc(1), spc(2), size (X, 2), 1);

    % Uses the target function to set the desired labels
    y = sign (X * wf);

    pos = find (y > 0);
    neg = find (y < 0);

    % Make sure we have both positive and negative examples
    if (any (pos) && any (neg))
      break;
    end
  end

  % Maximum number of iterations
  maxiter = 10000;

  % Weight vector w after linear regression
  w = linearreg (X, y);

  % Number of iterations needed to converge
  iters(i) = pla (X, y, w, maxiter, 1);
end

fprintf ("Average number of iterations: %f\n", mean (iters));
