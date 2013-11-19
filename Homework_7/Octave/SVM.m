warning("on", "backtrace")

function [X, Y] = generateSample(n)
  points = rand(2);
  p = points(1,:);
  q = points(2,:);
  X = rand(n, 2);
  Z = X - repmat(p, n, 1);
  a = (q(1) - p(1));
  b = (q(2) - p(2));
  Y = sign(Z(:,2) * a - Z(:,1) * b);
endfunction

function show_experiment(Xtrain, Ytrain, Xval, Yval)
  figure
  pos = find(Yval == 1);
  neg = find(Yval == -1);
  pt = find(Ytrain == 1);
  nt = find(Ytrain == -1);
  plot(Xtrain(pt,1), Xtrain(pt,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 20); hold on;
  plot(Xtrain(nt,1), Xtrain(nt,2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 20); hold on;
  plot(Xval(pos,1), Xval(pos,2), 'ko', 'MarkerFaceColor', 'b'); hold on;
  plot(Xval(neg,1), Xval(neg,2), 'ko', 'MarkerFaceColor', 'r'); hold on;
endfunction

function show_boundary(X, w, b)
  % Plot the decision boundary
  C = 10000;
  plot_x = linspace(min(X(:,1)), max(X(:,1)), 30);
  plot_y = (-1/w(2))*(w(1)*plot_x + b);
  plot(plot_x, plot_y, 'k-', 'LineWidth', 2); hold on;

  title(sprintf('SVM Linear Classifier with C = %g', C), 'FontSize', 14)
end

function [w_, b] = pla(X,Y)
  w = zeros(3, 1);
  n = size(X, 1);
  X_ = [ones(n, 1) X];
  err = 1;
  iters = 0;
  while err > 0 && iters < 100
    for ii = 1 : n
      if sign(X_(ii,:) * w) ~= Y(ii)
        w = w + X_(ii,:)' * Y(ii);
      end
    end
    iters += 1;
    err = sum(sign(X_*w)~=Y)/n;
  end
  b = w(1);
  w_ = w(2:end);
endfunction

function [w, b, alphas] = svm(X,Y)
  model = svmtrain(Y, X, '-s 0 -t 0 -c 10000 -q');
  w = model.SVs' * model.sv_coef;
  b = -model.rho;
  if (model.Label(1) == -1)
      w = -w;
      b = -b;
  end
  alphas = size(model.SVs, 1);
endfunction

function [x, alphas] = run_experiment(n)
  k = 10000;
  [X, Y] = generateSample(n + k);
  Xtrain = X([1:n],:);
  Ytrain = Y([1:n]);
  if Ytrain == ones(n, 1) || Ytrain == -ones(n, 1)
    [x, alphas] = run_experiment(n);
    return
  end
  Xval = X([n+1:end],:);
  Yval = Y([n+1:end]);

  [w, b] = pla(Xtrain, Ytrain);
  [w_, b_, alphas] = svm(Xtrain, Ytrain);

  pla_error = sum(sign(Xval*w+b)~=Yval)/k;
  svm_error = sum(sign(Xval*w_+b_)~=Yval)/k;

  %show_experiment(Xtrain, Ytrain, Xval, Yval);
  %show_boundary(Xtrain, w, b);
  %show_boundary(Xtrain, w_, b_);

  x = svm_error < pla_error;
endfunction

function [acum, bcum] = exercise(n)
  acum = bcum = 0;
  iterations = 10;
  for i = [1:iterations]
    [t, alphas] = run_experiment(n);
    acum += t;
    bcum += alphas;
    i
    fflush(stdout);
  end
  acum /= iterations;
  bcum /= iterations;
endfunction
