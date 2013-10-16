function [weights, E_in] = runLinearRegression(N)

[X, y, soln] = genData(N);
weights = linearRegression(X, y);

hyp = sign(X * weights);
err = sum(hyp == y);
E_in = 1 - err / N;

% [new_X] = genData(1000);
% new_y = sign(new_X * soln);
% new_hyp = sign(new_X * weights);
% E_out = 1 - sum(new_hyp == new_y) / 1000;

end