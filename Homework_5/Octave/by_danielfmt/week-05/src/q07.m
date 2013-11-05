1;

addpath ("./common");

clear all;
clc;

% Learning rate
eta = 0.1;

% Error tolerance
errprec = 10^-21;

% Maximum number of GD iterations
iters = 15;

% Initial weights
w= ones (2, 1);

% Runs coordinate descent
[w, epoch] = cgd (w, eta, @errfunc, @gradfunc, errprec, iters);

fprintf ("Ein: %f\n", errfunc (w));
