% Adds some other features to X
function nX = addfeatures (X)
  N = size (X, 1);
  d = size (X, 2);

  nX = X;

  % Introduces the proposed nonlinear features
  nX(:,d+1) = arrayfun (@ (x1, x2) x1 * x2, X(:,2), X(:,3));
  nX(:,d+2) = arrayfun (@ (x1) x1 * x1, X(:,2));
  nX(:,d+3) = arrayfun (@ (x2) x2 * x2, X(:,3));
end
