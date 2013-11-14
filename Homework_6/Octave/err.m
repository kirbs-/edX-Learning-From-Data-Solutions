function E = err(weights, X, y)
  N = size(X, 1);
  hyp = sign(X * weights);
  err = sum(hyp == y);
  E = 1 - err / N;
end