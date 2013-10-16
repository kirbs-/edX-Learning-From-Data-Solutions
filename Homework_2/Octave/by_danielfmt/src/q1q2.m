% Number of coins to flip
N = 1000;

% Number of times to flip each coin
d = 10;

% Number of iterations
iter = 100000;

% v1, vrand, and vmin for each iteration
vs = zeros (iter, 3);

for i = 1:iter
  % Flips all coins (0 = heads ; 1 = tails)
  coins = rand (N, d) >= 0.5;

  % First coin
  c1 = coins(1,:);

  % Random coin
  crand = coins(fix (unifrnd (1, N)),:);

  % Coin which had the minimum frequency of heads
  tails = sum (coins, 2);
  minheads = tails == max (tails);
  cmin = coins(find (minheads)(1),:);

  % Fraction of heads for c1
  vs(i,1) = sum (!c1) / length (c1);

  % Fraction of heads for crand
  vs(i,2) = sum (!crand) / length (crand);

  % Fraction of heads for cmin
  vs(i,3) = sum (!cmin) / length (cmin);
end

% Compute the mean for each v1, vrand, and vmin
fprintf ("Fraction of heads for c1, crand and cmin is %f, %f, and %f\n", mean (vs));
