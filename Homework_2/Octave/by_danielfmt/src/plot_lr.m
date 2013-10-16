addpath ("./common");

clear all;
close all;
clc;

% Number of dimensions (before adding x0)
d = 1;

% Number of examples
N = 10;

% N random examples
X = unifrnd (-1, 1, N, d);
y = unifrnd (-1, 1, N, 1);

% Introduces the synthetic dimension x0
nX = [ones(N, 1), X];

% Weight vector w after linear regression
wlin = linearreg (nX, y);

% Function that plots a weight vector w as a line
wf = @ (x, w) x * w(2) + w(1);

% Function that calculates the squared error for some weight vector w
sqerr = @(xn, yn, w) (wf (xn, w) - yn)^2;

% Function that calculates the in-sample error for some weight vector w
ein = @(w) mean (arrayfun (@ (xn, yn) sqerr (xn, yn, w), X, y));

printf ("Values of w that minimizes the error: [%.2f %.2f]\n", wlin);
printf ("In-sample squared error measure: %.2f\n", ein (wlin));

% Plots the line represented by wlin with the data points
h1 = figure (1);
hold on;

title ("Linear Regression Demo");
xlabel ("x");
ylabel ("y");

plot (X, y, "ko", "MarkerSize", 3, "MarkerFaceColor", "b");
fplot (@ (x) wf (x, wlin), [-1, 1], "r-");
legend ("hide");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h1, 4, 3, "../img/plot_lr_01.png")

% Plots the error surface and a point representing the global minimum
h2 = figure (2);
hold on;

grid on;

title ("Squared Error Surface");
xlabel ("w_0");
ylabel ("w_1");

p = linspace (-1, 1, 50);
[w0, w1] = meshgrid (p, p);

plot (wlin(1), wlin(2), "ko", "MarkerSize", 3, "MarkerFaceColor", "g");
contour (w0, w1, arrayfun (@ (w0, w1) ein ([w0; w1]), w0, w1));
legend ("hide");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h2, 4, 3, "../img/plot_lr_02.png")
