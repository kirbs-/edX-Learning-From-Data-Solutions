% Runs stochastic gradient descent
function [w, epoch] = sgd (X, y, wp, eta, gradfunc, threshold, maxiters)

  % Number of examples in the given set
  N = size (X, 1);

  % Number of dimensions in the given set
  d = size (X, 2);

  % Initial weights
  w = zeros (d, 1);

  % Epoch number
  epoch = 0;

  while (1)
    epoch++;

    % Shuffle examples
    perm = randperm (length (X));
    Xs = X(perm,:);
    ys = y(perm,:);

    for i = 1:N
      % Updates the weights according to the gradient on point n
      % with respect to w
      w += -eta * gradfunc (Xs(i,:), ys(i), w);
    end

    % Termination criterion
    if ((norm (wp - w) < threshold) || (maxiters != 0 && epoch >= maxiters))
      break;
    end

    wp = w;
  end
end
