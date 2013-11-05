addpath ("./common");

1;

clear all;
close all;
clc;

% Axis limits
limits = [-1 1];

% Plots each proposed error surface

% h = figure (1);
% ploterrfunc (limits, @ (w) w(2)^2, "w_1 = 0, w_2 > 0");

% h = figure (2);
% ploterrfunc (limits, @ (w) w(1)^2, "w_1 > 0, w_2 = 0");

% h = figure (3);
% ploterrfunc (limits, @ (w) w(1)^2 + w(2)^2,  "w_1 > 0, w_2 > 0");

h = figure (4);
ploterrfunc (limits, @ (w) -w(1)^2 + w(2)^2, "w_1 < 0, w_2 > 0");

% h = figure (5);
% ploterrfunc (limits, @ (w) w(1)^2 + -w(2)^2, "w_1 > 0, w_2 < 0");

% Uncomment the following line in order to save the plot to a PNG file
% saveplot (h, 4, 3, "../img/plot_error_surface.png");
