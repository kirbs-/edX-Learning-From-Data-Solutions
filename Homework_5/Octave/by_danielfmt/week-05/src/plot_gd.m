1;

addpath ("./common");

clear all;
close all;
clc;

h = figure (1);

hold on;

% Learning rate
eta = 0.1;

% Error tolerance
errprec = 10^-21;

% Plot limits
limits = [-2 2];

% Initial weight
w = ones (2, 1);

% Maximum number of GD iterations
iters = 15;

% Plots the error surface
ploterrfunc (limits, @errfunc, "Gradient Descent");

% Weights for each epoch
ws = [w'];

% Same as gd function, but keeps track of all changes in w
for epoch = 1:iters

  % Updates the weights
  w += -eta * gradfunc (w);

  ws(epoch+1,:) = w';

  % termination criterion
  if (errfunc (w) < errprec)
     break;
  end
end

axis ([-2 2 -2 2]);
plot (ws(:,1), ws(:,2), "m-x", "MarkerSize", 3);
legend ("", "Iterations");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h, 4, 3, "../img/plot_gd.png");
