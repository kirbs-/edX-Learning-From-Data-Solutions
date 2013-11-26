function [ein, eout] = rbf(X, Y, Xtest, Ytest, c)
  options = cstrcat("-q -s 0 -t 2 -g 1 -h 0 -c ", num2str(c));
  model = svmtrain(Y, X, options);
  [Y_, stats, _] = svmpredict(Y, X, model, "-q");
  ein = 1 - stats(1) / 100;
  [Y_, stats, _] = svmpredict(Ytest, Xtest, model, "-q");
  eout = 1 - stats(1) / 100;
end

function e_val = cross_validate(X, Y, k, C, Q)
  options = cstrcat("-q -s 0 -t 1 -g 1 -r 1 -d ", num2str(Q), " -v ", num2str(k), " -c ", num2str(C));
  e_val = 1 - svmtrain(Y, X, options) / 100;
end

function [X, Y] = k_vs_all(A, k)
  B = A((A(:, 1) == k), [2, 3]);
  C = A((A(:, 1) != k), [2, 3]);
  X = [B; C];
  Y = [ones(size(B, 1), 1); -ones(size(C, 1), 1)];
end

function [ein, alphas] = run_k_vs_all(A, k)
  [X, Y] = k_vs_all(A, k);
  [alphas, model] = polykernel_svm(X, Y, 0.01, 2);
  [Y_, stats, _] = svmpredict(Y, X, model, "-q");
  ein = 1 - stats(1) / 100;
end

function [X, Y] = n_vs_m(A, n, m)
  B = A((A(:,1) == n), [2, 3]);
  C = A((A(:,1) == m), [2, 3]);
  X = [B; C];
  Y = [ones(size(B, 1), 1);-ones(size(C, 1), 1)];
end

function [ein, alphas, eout] = run_1_vs_5(A, V, c, q)
  [X, Y] = n_vs_m(A, 1, 5);
  [alphas, model] = polykernel_svm(X, Y, c, q);
  [Y_, stats, _] = svmpredict(Y, X, model, "-q");
  ein = 1 - stats(1) / 100;

  [Xval, Yval] = n_vs_m(V, 1, 5);
  [Y_, stats, _] = svmpredict(Yval, Xval, model, "-q");
  eout = 1 - stats(1) / 100;
end

function [alphas, model] = polykernel_svm(X, Y, C, Q)
  kernel = cstrcat("-q -h 0 -s 0 -c ", num2str(C), " -t 1 -g 1 -r 1 -d ", num2str(Q));
  model = svmtrain(Y, X, kernel);
  alphas = size(model.SVs, 1);
end

function run_experiment()
  A = dlmread("features.train");
  V = dlmread("features.test");

  more off;

  printf("2-4)\n");
  for v = [(0:2:8);(1:2:9)]'
    for k = v'
      [ein, alphas] = run_k_vs_all(A, k);
      printf("%d-vs-all: E_in = %.5f, SVs = %d\n", k, ein, alphas);
    end
  end

  printf("5)\n");
  for c = [0.001, 0.01, 0.1, 1]
    [ein, alphas, eout] = run_1_vs_5(A, V, c, 2);
    printf("c = %.3f: ein = %.10f, alphas = %d, eout = %.10f \n", c, ein, alphas, eout);
  end

  printf("6)\n");
  for c = [0.0001, 0.001, 0.01, 1]
    printf("c = %.4f\n", c);
    for q = [2, 5]
      printf("\tq = %d\n", q);
      [ein, alphas, eout] = run_1_vs_5(A, V, c, q);
      printf("\t\tein = %.5f, alphas = %d, eout = %.5f \n", ein, alphas, eout);
    end
  end

  [X, Y] = n_vs_m(A, 1, 5);
  total_e_vals = zeros(5, 1);
  wins = zeros(5, 1);
  Cs = [0.0001, 0.001, 0.01, 0.1, 1];

  for _ = [1:100]
    permut = randperm(size(X, 1));
    X = X(permut, :);
    Y = Y(permut, :);
    e_vals = zeros(5, 1);
    for i = [1:5]
      C = Cs(i);
      e_vals(i) += cross_validate(X, Y, 10, C, 2);
    end
    [val, idx] = min(e_vals);
    wins(idx)++;
    total_e_vals += e_vals;
  end

  printf("7-8)\n");
  for i = [1:5]
    printf("wins for %.5f: %d. E[e_val] = %.5f\n", Cs(i), wins(i), total_e_vals(i) / 100);
  end

  [X, Y] = n_vs_m(A, 1, 5);
  [Xtest, Ytest] = n_vs_m(V, 1, 5);
  Cs = [0.01, 1, 100, 10^4 , 10^6];
  printf("9-10)\n");
  for c = Cs
    [ein, eout] = rbf(X, Y, Xtest, Ytest, c);
    printf("c = %.5f: ein = %.5f, eout = %.5f\n", c, ein, eout);
  end
end