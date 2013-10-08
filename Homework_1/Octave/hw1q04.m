1;

function b = noRedMarble (N, mu)
  b = length (find (rand (N, 1) <= mu)) == 0;
end

% Number of times the experiment will run in order to average the probability
iters = 99999;

% Number of marbles per sample
N = 10;

% Probability of picking a red marble
mu = 0.55;

% Result of each iteration
results = arrayfun (@ (_) noRedMarble (N, mu), zeros (iters, 1));

printf("Probability of picking no red marbles on one sample: %e\n", mean (results));