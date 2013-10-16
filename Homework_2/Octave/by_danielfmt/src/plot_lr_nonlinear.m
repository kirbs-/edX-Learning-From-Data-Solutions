addpath ("./common");

clear all;
close all;
clc;

% X/Y range where the examples will be plotted
spc = [-1 1];

% Number of dimensions (excluding the synthetic dimension x0, which
% will be added later)
d = 2;

% Number of examples. Change this according to the question
N = 1000;

% N random examples
X = unifrnd (spc(1), spc(2), N, d);

% Uses the target function to set the desired labels
y = arrayfun (@(x1, x2) target (x1, x2), X(:,1), X(:,2));

% Introduces noise
y = addnoise (y, 0.1);

pos = find (y > 0);
neg = find (y < 0);

% Introduces the synthetic dimension x0
X = [ones(N,1), X];

% Introduces the new features
X = addfeatures (X);

% Apply linear regression
w = linearreg (X, y);

% Labels assigned by w for each X
hy = sign (X * w);

h = figure (1);
hold on;

title ("Nonlinear Hypothesis Demo");
xlabel ("x_1");
ylabel ("x_2");

plot (X(pos,2), X(pos,3), "ko", "MarkerSize", 3, "MarkerFaceColor", "r");
plot (X(neg,2), X(neg,3), "ko", "MarkerSize", 3, "MarkerFaceColor", "b");

p = linspace (spc(1), spc(2), 100);
[x1, x2] = meshgrid (p, p);

% Hypothesis function
hf = @ (x1, x2) sign (addfeatures ([1 x1 x2]) * w);
contour (x1, x2, arrayfun (hf, x1, x2), "g-");

% Target function
contour (x1, x2, arrayfun (@target, x1, x2), "m-");

legend ("hide");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h, 4, 3, "../img/plot_lr_nonlinear.png")
