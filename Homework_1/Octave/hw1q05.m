1;

function b = noRedMarble (N, mu)
  b = length (find (rand (N, 1) <= mu)) == 0;
end

function b = hasSampleWithNoRedMarbles (N, mu, samples)
  results = arrayfun (@ (_) noRedMarble (N, mu), zeros (samples, 1));
  b = length (find (results == 1)) > 0;
end

% Number of times the experiment will run in order to average the probability
iters = 10000;

% Number of samples
samples = 1000;

% Number of marbles per sample
N = 10;

% Probability of picking a red marble
mu = 0.55;

% Result of each iteration
results = arrayfun (@ (_) hasSampleWithNoRedMarbles (N, mu, samples), zeros (iters, 1));

printf("Probability of picking no red marbles: %.3f\n", sum (results) / length (results));