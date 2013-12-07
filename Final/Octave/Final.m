pkg load statistics;

function Z = polytrans(X)
  x1 = X(:, 1);
  x2 = X(:, 2);
  s = sqrt(2);
  n = size(X, 1);
  Z = [ones(n, 1) (s * x1) (s * x2) (x1 .^2) (x2 .^ 2) (s * (x1 .* x2))];
end

function Xt = addones(X)
  n = size(X, 1);
  Xt = [ones(n, 1) X];
end

function Xt = nonlinear(X)
  x1 = X(:, 1);
  x2 = X(:, 2);
  n = size(X, 1);
  Xt = [ones(n, 1) x1 x2 (x1 .* x2) (x1 .^ 2) (x2 .^ 2)];
end

function [ein, eout] = least_squares(X, Y, Xtest, Ytest, lambda)
  % X is in R^{n x m}
  [n, m] = size(X);
  [r, _] = size(Xtest);
  wreg = inv(X' * X + lambda * eye(m)) * X' * Y;
  ein = sum(sign(X * wreg) ~= Y) / n;
  eout = sum(sign(Xtest * wreg) ~= Ytest) / r;
end

function [X, Y] = k_vs_all(A, k)
  B = A((A(:, 1) == k), [2, 3]);
  C = A((A(:, 1) != k), [2, 3]);
  X = [B; C];
  Y = [ones(size(B, 1), 1); -ones(size(C, 1), 1)];
end

function [ein, eout] = run_k_vs_all(A, B, k, lambda, transform)
  [X, Y] = k_vs_all(A, k);
  [Xtest, Ytest] = k_vs_all(B, k);
  [ein, eout] = least_squares(transform(X), Y, transform(Xtest), Ytest, lambda);
end

function [X, Y] = n_vs_m(A, n, m)
  B = A((A(:,1) == n), [2, 3]);
  C = A((A(:,1) == m), [2, 3]);
  X = [B; C];
  Y = [ones(size(B, 1), 1);-ones(size(C, 1), 1)];
end

function [ein, eout] = run_n_vs_m(A, B, n, m, lambda, transform)
  [X, Y] = n_vs_m(A, n, m);
  [Xtest, Ytest] = n_vs_m(B, n, m);
  [ein, eout] = least_squares(transform(X), Y, transform(Xtest), Ytest, lambda);
end

function M = margin(alphas, Q)
  M = 1/sqrt(alphas' * Q * alphas);
end

function s = show(A)
  s = substr(sprintf("%.5f, ", A), 1, -2);
end

function [X, Y] = generate_sample(n)
  X = 1 - 2 * rand(n, 2);
  x1 = X(:, 1);
  x2 = X(:, 2);
  Y = sign(x2 - x1 + sin(pi * x1) / 4);
end

function result = run_rbf_kernel(X, Y, g, Xtest, Ytest)
  model = svmtrain(Y, X, cstrcat("-q -s 0 -t 2 -c 10000000 -g ", num2str(g)));
  [Y_, stats, _] = svmpredict(Y, X, model, "-q");
  ein = 1 - stats(1) / 100;
  [Y_, stats, _] = svmpredict(Ytest, Xtest, model, "-q");
  eout = 1 - stats(1) / 100;
  result.ein = ein;
  result.eout = eout;
  result.failed = ein != 0;
end

function plot_svm_error()
  while true
	  [X, Y] = generate_sample(100);
	  model = svmtrain(Y, X, "-q -s 0 -t 2 -c 1000000 -g 1.5");
	  Y_ = svmpredict(Y, X, model, "-q");
	  errors = X((Y_ ~= Y), :);
	  if (size(errors, 1) > 0)
		  B = X((Y == 1), :);
		  R = X((Y == -1), :);
		  hold on;
		  plot(B(:, 1), B(:, 2), "b+");
		  plot(R(:, 1), R(:, 2), "rx");
		  plot(errors(:, 1), errors(:, 2), "ok", "markersize", 30, "markerfacecolor", "k");
		  hold off;
		  break;
		else
  	  printf(".");
		end
	end
end


function result = run_rbf_kmeans(X, Y, k, g, Xtest, Ytest)
	try
    [idx, centers] = kmeans(X, k);
  catch
    result.failed = 1;
    result.ein = inf;
    result.eout = inf;
    return;
  end_try_catch
  n = size(X, 1);
	M = zeros(n, k);
	for i = [1:n]
	 	v = X(i, :);
	 	for j = [1:k]
	 		M(i, j) = exp(- g * norm(v - centers(j, :))^2);
	  end
	end
	M = [ones(n, 1) M];
  w = pinv(M'*M) * M' * Y;
  Y_ = sign(M * w);
  result.ein = sum(Y_~=Y)/n;

  n_ = size(Xtest, 1);
  for i = [1:n_]
  	v = Xtest(i, :);
  	for j = [1:k]
  		V(i, j) = exp(-g * norm(v - centers(j, :))^2);
  	end
  end
  V = [ones(n_, 1) V];
  YV = sign(V * w);
  result.eout = sum(YV~=Ytest)/n_;

  result.failed = 0;
end

function plot_kmeans(k)
	[X, Y] = generate_sample(100);
  [idx, centers] = kmeans(X, k);
  B = X((Y == 1), :);
	R = X((Y == -1), :);
	hold on;
	plot(B(:, 1), B(:, 2), "b+");
	plot(R(:, 1), R(:, 2), "rx");
	plot(centers(:, 1), centers(:, 2), "mo", "markersize", 20, "markerfacecolor", "m");
	hold off;
end

function rbf_result = run_rbf(g, k, run_svm = true)
  [X, Y] = generate_sample(100);
  [Xtest, Ytest] = generate_sample(1000);
  if run_svm
  	rbf_result.svm = run_rbf_kernel(X, Y, g, Xtest, Ytest);
  end
  if k != 0
    rbf_result.kmeans = run_rbf_kmeans(X, Y, k, g, Xtest, Ytest);
  end
end

function c = kernel_beats_regular(g, k, total_runs)
	kernel_beat_regular = 0;
	i = 0;
	h = waitbar(0, "Running RBF-SVM versus RBF k-means regression...");
	while i < total_runs
    result = run_rbf(g, k);
    if (result.svm.failed || result.kmeans.failed)
    	continue;
    end
    kernel_beat_regular += result.svm.eout < result.kmeans.eout;
    i++;
  	waitbar(i / total_runs, h);
	end
  c = (kernel_beat_regular / total_runs) * 100;
end

function [counters, nochange] = contrast_kmeans(total_runs, rbf1, rbf2)
  i = 0;
  h = waitbar(0, "Contrasting RBF k-means regressions...");
  counters = zeros(2, 2);
  nochange = 0;
  % both down, both up
  % (ein up, eout down), (ein down, eout up)
  while i < total_runs
  	result1 = rbf1();
  	if (result1.kmeans.failed)
  		continue
  	end
  	result2 = rbf2();
  	if (result2.kmeans.failed)
  		continue
  	end
    a = result1.kmeans.ein;
    b = result1.kmeans.eout;
    c = result2.kmeans.ein;
    d = result2.kmeans.eout;
    if (a == c && b == d)
    	nochange++;
    elseif (a < c && b < d)
    	counters(1, 2)++;
    elseif (a < c && b > d)
    	counters(2, 1)++;
    elseif (a > c && b < d)
    	counters(2, 2)++;
    elseif (a > c && b > d)
    	counters(1, 1)++;
    end
    	waitbar(i / total_runs, h);
    i++;
  end
  counters /= total_runs;
  counters *= 100;
end

function show_contrast(counters, nochange)
  printf("\nE_in and E_out both go down %.1f%% of the time.\n", counters(1, 1));
  printf("E_in and E_out both go up %.1f%% of the time.\n", counters(1, 2));
  printf("E_in goes down, E_out goes up %.1f%% of the time.\n", counters(2, 2));
  printf("E_in goes up, E_out goes down %.1f%% of the time.\n", counters(2, 1));
  printf("There was no change %.1f%% of the time.\n", nochange);
end

function exercise_7(A, B)
  printf("7)\n");
  for n = [5:9]
  	[ein, eout] = run_k_vs_all(A, B, n, 1, @addones);
    printf("n = %d: E_in = %.5f, E_out = %.5f\n", n, ein, eout);
  end
end

function exercise_8(A, B)
  printf("8)\n");
  for n = [0:4]
  	[ein, eout] = run_k_vs_all(A, B, n, 1, @nonlinear);
    printf("n = %d: E_in = %.5f, E_out = %.5f\n", n, ein, eout);
  end
end

function exercise_9(A, B)
  printf("9)\n");
  printf("For each line, the format is (E_in, E_out).\n");
  for n = [0:9]
  	[ein0, eout0] = run_k_vs_all(A, B, n, 1, @addones);  	
  	[ein1, eout1] = run_k_vs_all(A, B, n, 1, @nonlinear);
  	printf("n = %d:\n", n);
    printf("\tWithout transform: (%.10f, %.5f)\n", ein0, eout0);
    printf("\tWith transform: (%.10f, %.5f)\n", ein1, eout1);
    overfit = ifelse(ein1 < ein0 && eout1 > eout0, "Yes", "No");
    printf("\tOverfitting: %s.\n", overfit);
    improved = ifelse(eout1 <= 0.95 * eout0, "Yes", "No");
    printf("\tImproved out-of-sample performance by at least 5%%: %s.\n", improved);
  end
end

function exercise_10(A, B)
  printf("10)\n");
  [ein0, eout0] = run_n_vs_m(A, B, 1, 5, 0.01, @nonlinear);  	
  [ein1, eout1] = run_n_vs_m(A, B, 1, 5, 1, @nonlinear); 
  printf("With lambda = 0.01:\n\tE_in = %.5f, E_out = %.5f\n", ein0, eout0);
  printf("With lambda = 1:\n\tE_in = %.5f, E_out = %.5f\n", ein1, eout1);
end

function exercise_11()
  % For exercise 11, I plotted the following in Mathematica:
  %ListPlot[{{{-3, 2}, {0, -1}, {0, 3}}, {{1, 2}, {3, -3}, {3, 5}, {3, 
  %  5}}},  PlotStyle -> {Directive[PointSize[Large], Red], 
  % Directive[PointSize[Large ], Blue]}]
  % Where 
  % λ: let f(x1, x2) = (x2^2 - 2*x1 - 1, x1^2 - 2*x2 + 1)
  % λ: let points = [(1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), (-2, 0)]
  % λ: map f points
  % [(-3,2),(0,-1),(0,3),(1,2),(3,-3),(3,5),(3,5)]
end

function exercise_12(debugging = false)
  printf("12)\n");
  X = [1, 0; 0, 1; 0, -1; -1, 0; 0, 2; 0, -2; -2, 0];
  Y = [-1; -1; -1; 1; 1; 1; 1];

  n = size(X, 1);
  T = (1 + X * X') .^ 2;
  Q = (Y * Y') .* T;
  q = -ones(n, 1);
  x0 = rand(n, 1);
  lb = zeros(n, 1);
  epsilon = 0.00000001;
  alpha1 = qp([], Q, q, Y', 0, lb, []);
  printf("With quadratic programming (1): %d (%s).\n", sum(abs(alpha1) > epsilon), show(alpha1));
  printf("\tMargin: %.5f\n", margin(alpha1 .* Y, T));

  model = svmtrain(Y, X, "-q -s 0 -t 1 -d 2 -g 1 -r 1 -c 10000000 -h 1");
  alpha2 = zeros(7, 1);
  for i = [1:size(model.sv_indices,1)]
  	alpha2(model.sv_indices(i)) = model.sv_coef(i);
  end
  printf("With libsvm polykernel (2): %d (%s).\n", size(model.SVs, 1), show(alpha2));
  printf("\tMargin: %.5f\n", margin(alpha2, T));

  kernel = [(1:n)' T];
  model = svmtrain(Y, kernel, "-s 0 -t 4 -c 10000000 -h 1 -q");
  alpha3 = zeros(7, 1);
  for i = [1:size(model.sv_indices,1)]
  	alpha3(model.sv_indices(i)) = model.sv_coef(i);
  end
  printf("With precomputed kernel (3): %d (%s).\n", size(model.SVs, 1), show(alpha2));
  printf("\tMargin: %.5f\n", margin(alpha3, T));

  Z = polytrans(X);
  model = svmtrain(Y, Z, "-q -s 0 -t 0 -c 10000000 -h 1");
  alpha4 = zeros(7, 1);
  for i = [1:size(model.sv_indices,1)]
  	alpha4(model.sv_indices(i)) = model.sv_coef(i);
  end
  w4 = model.SVs' * model.sv_coef;
  b4 = -model.rho;
  if (model.Label(1) == -1)
      w4 = -w4;
      b4 = -b4;
  end
  printf("With computation of phi (4): %d (%s).\n", size(model.SVs, 1), show(alpha4));
  printf("\tMargin: %.5f\n", margin(alpha3, T));

  if (debugging)
	  w1 = ((alpha1 .* Y)' * Z)';
	  L = Y - (Z * w1);
	  k1 = find(alpha1, 1); % pick a support vector
	  b1 = Y(k1) - (Z(k1, :) * w1);
    printf("1: w = (%s), b = %.5f\n", show(w1), b1);

	  w2 = (alpha2' * Z)';
	  L = Y - Z * w2;
	  k2 = find(alpha2, 1);
	  b2 = Y(k2) - (Z(k2, :) * w2);
    printf("2: w = (%s), b = %.5f\n", show(w2), b2);

	  w3 = (alpha3' * Z)';
	  L = Y - Z*w3;
	  k3 = find(alpha3, 1);
	  b3 = Y(k3) - (Z(k3, :) * w3);
    printf("3: w = (%s), b = %.5f\n", show(w3), b3);

	  printf("4: w = (%s), b = %.5f\n", show(w4), b4);
	end
end

function exercise_13(total_runs = 1000)
  printf("13)\n");
  linearly_inseparables = 0;
  h = waitbar(0, "Running RBF-SVM, checking for separability...");
  for i = [1:total_runs]
  	linearly_inseparables += run_rbf(1.5, 0).svm.failed;
  	waitbar(i / total_runs, h);
  end
  printf("\nPercentage of inseparables: %.1f\n", (linearly_inseparables / total_runs) * 100);
end

function exercise_14(total_runs = 200)
  printf("14)\n");
  c = kernel_beats_regular(1.5, 9, total_runs);
	printf("\nPercentage of time kernel RBF beats k-means RBF, k = 9: %.1f\n", c);
end

function exercise_15(total_runs = 200)
	printf("15)\n");
	c = kernel_beats_regular(1.5, 12, total_runs);
	printf("\nPercentage of time kernel RBF beats k-means RBF, k = 12: %.1f\n", c);
end

function exercise_16(total_runs = 200)
	printf("16)\n");
  [counters, nochange] = contrast_kmeans(total_runs, @() run_rbf(1.5, 9, false), @() run_rbf(1.5, 12, false));
  show_contrast(counters, nochange);  
end

function exercise_17(total_runs = 200)
  printf("17)\n");
  [counters, nochange] = contrast_kmeans(total_runs, @() run_rbf(1.5, 9, false), @() run_rbf(2, 9, false));
  show_contrast(counters, nochange);
end

function exercise_18(total_runs = 200)
  printf("18)\n");
  zero_ein = 0;
  i = 0;
  h = waitbar(0, "Running k-means RBF with k = 9, checking for E_{in} = 0...");
  while i < total_runs
  	result = run_rbf(1.5, 9, false);
  	if result.kmeans.failed
  		break;
  	end
  	zero_ein += result.kmeans.ein == 0;
  	i++;
  	waitbar(i / total_runs, h);
  end
  c = (zero_ein / total_runs) * 100;
  printf("\nRBF k-means with k = 9 achieves E_{in} = 0 in %.1f%% of the cases.\n", c);
end

function main()
  A = dlmread("features.train");
  B = dlmread("features.test");
  more off;

  exercise_7(A, B);
  exercise_8(A, B);
  exercise_9(A, B);
  exercise_10(A, B);
  exercise_11();
  exercise_12(true);
  exercise_13();
  exercise_14();
  exercise_15();
  exercise_16();
  exercise_17();
  exercise_18();
end