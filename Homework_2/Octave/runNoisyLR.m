function [weights, E_in] = runNoisyLR(N)

  X = unifrnd(-1, 1, N, 2);
  X = sortrows(X);
  [m, n] = size(X);

  y = sign(X(:, 1).^2 + X(:, 2).^2 - 0.6);
  noise_i = randperm(N)(1:N / 10);
  y(noise_i) = -y(noise_i);
  X = transform(X);

  weights = linearRegression(X, y);
  hyp = sign(X * weights);
  err = sum(hyp == y);
  E_in = 1 - err / N;

end

function new_X = transform(X)
  [m, n] = size(X);
  new_X = [ones(m, 1) X X(:, 1).*X(:, 2) X(:, 1).^2 X(:, 2).^2];
end