1;

% Solving this exercise using a closed-form expression
% ----------------------------------------------------
%
% P(B) = 1/2 * 1 + 1/2 * 1/2
%      = 1/2 + 1/4
%      = 3/4
%
% For the 2nd ball to be black, then I must have picked the 1st bag, so:
%
% P(1st bag | B) = P(1st bag ^ B) / P(B)
%                = (1/2 * 1) / (3/4)
%                = 4/6
%                = 2/3

% ... or solving by running a little experiment for a little while :-P

% Number of times the experiment will run in order to average the probability
iters = 9999;

% Number of marbles on each bin
nmarbles = 2;

% Two bins containing white marbles (0) and black marbles (1)
bins = [ [0 1] ; [1 1] ];

% Result of each iteration
results = zeros (iters, 1);

for i = 1:iters

  % Picks a random bin
  bin = bins(randperm (length (bins)), :)(1, :);

  % Shuffles the marbles
  marbles = bin(:, randperm (nmarbles));

  % This experiment should not be considered if the first marble is not black
  if marbles(1) != 1
    results(i) = 2;
    continue;
  end

  % Picks the remaining marble from the bin
  results(i) = marbles(2);
end

% Removes the invalid experiments
results = results(find (results != 2));

printf("Probability of picking two black marbles from a random bag: %.6f\n", mean (results));