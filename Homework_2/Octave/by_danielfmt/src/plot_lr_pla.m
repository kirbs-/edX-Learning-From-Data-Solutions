addpath ("./common");

clear all;
close all;
clc;

% Number of dimensions (excluding the synthetic dimension x0, which
% will be added later)
d = 2;

% Number of examples. Change this according to the question
N = 100;

h = figure (1);
hold on;

title ("PLA w/ Linear Regression");
xlabel ("x values");
ylabel ("y values");

% X/Y range where t0e examples will be plotted
spc = [-1 1];

% Loops until we find negative and positive examples
while (1)

  % Two random points used in target function f
  fp1 = unifrnd (spc(1), spc(2), 2, 1);
  fp2 = unifrnd (spc(1), spc(2), 2, 1);

  % N random examples
  X = unifrnd (spc(1), spc(2), N, d);

  % Introduces the synthetic dimension x0
  X = [ones(N, 1) X];

  % Weight vector that represents the target function f
  wf = unifrnd (spc(1), spc(2), size (X, 2), 1);

  % Uses the target function to set the desired labels
  y = sign (X * wf);

  pos = find (y > 0);
  neg = find (y < 0);

  % Make sure we have both positive and negative examples
  if (any (pos) && any (neg))
    break;
  end
end

% Plot limits
limits = cat (2, spc, spc);

% Plots the equation created from the two points
plotboundary (wf, limits, "m-");

% Plots the examples
plot (X(pos,2), X(pos,3), "ko", "MarkerSize", 3, "MarkerFaceColor", "r");
plot (X(neg,2), X(neg,3), "ko", "MarkerSize", 3, "MarkerFaceColor", "b");

% Maximum number of PLA iterations
maxiter = 10000;

% Initial weight vector w after linear regression
w = linearreg (X, y);

% Plots the decision boundary based on the weight vector
plotboundary (w, limits, "y-");

% Tune the weight vector w with the perceptron learning algorithm
w = pla (X, y, w,  maxiter, 0);

% Plots the final decision boundary
plotboundary (w, limits, "g");
legend ("hide");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h, 4, 3, "../img/plot_lr_pla.png")
