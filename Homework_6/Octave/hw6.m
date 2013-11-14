
data = load('in.dta');
X = data(:, [1, 2]); y = data(:, 3);
% plotData(X, y);
X = transform(X);

k = -1;
lambda = 10 ^ k;
weights = linearRegression(X, y, lambda);
E_in = err(weights, X, y);

outdata = load('out.dta');
Xout = outdata(:, [1, 2]); yout = outdata(:, 3);
Xout = transform(Xout);
E_out = err(weights, Xout, yout);
E_in, E_out