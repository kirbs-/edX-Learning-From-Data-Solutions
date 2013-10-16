function [weights, iterations] = perceptron(initial_weights, X, y)

weights = initial_weights;
iterations = 0;
[is_good, bad_point] = goal_reached(weights, X, y);

while (not(is_good) & iterations < 50000)
  weights += X(bad_point, :)' * y(bad_point);
  iterations++;
  [is_good, bad_point] = goal_reached(weights, X, y);
end

end

function [is_good, bad_point] = goal_reached(weights, X, y)

hyp = sign(X * weights);
if find(hyp == 0)
  hyp = hyp(find(hyp == 0)) + 1;
end
res = hyp == y;

is_good = all(res);
bad_points = find(res == 0);
bad_point = 0;
if bad_points
  i = floor(rand * numel(bad_points) + 1);
  bad_point = bad_points(i);
end

end