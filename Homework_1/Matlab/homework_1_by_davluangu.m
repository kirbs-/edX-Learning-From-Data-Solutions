clc
close all;
clear all;
NMONTECARLO     = 10000;
% DIMENSIONALITY  = 2;
DIMENSIONALITY  = 3;
N               = 20;

for m = 1:1000
  % generate a random hyperplane inside the [-1 1]^N space
  linePoints    = [2*rand(DIMENSIONALITY,DIMENSIONALITY)-1];
  w_target      = [1; linePoints\ones(DIMENSIONALITY,1)];
  
  % generate random data inside [-1 1]^N space
  X             = 2*rand(DIMENSIONALITY,N)-1; % uniform distribution between [-1 and 1]
  X_augmented   = [ones(1,N); X];
  
  % classify data using the random hyperplane generated
  Y             = sign(w_target'*X_augmented);
  
  % execute PLA
  w_hypothesis  = zeros(DIMENSIONALITY+1,1);
  Y_hyp         = sign(w_hypothesis'*X_augmented);
  iter          = 0;
  while(~all(Y_hyp == Y))
    misclassified       = find(Y_hyp ~= Y);
    randomMisclassified = misclassified(round(rand*(numel(misclassified)-1)+1));
    w_hypothesis        = w_hypothesis + Y(randomMisclassified)*X_augmented(:,randomMisclassified);
    Y_hyp               = sign(w_hypothesis'*X_augmented);
    iter                = iter + 1;
  end
  iterations(m) = iter;
  
  % monte carlo: difference in classficiation
  XMC                   = 2*rand(DIMENSIONALITY,NMONTECARLO)-1;
  XMC_augmented         = [ones(1,NMONTECARLO); XMC];
  YMC                   = sign(w_target'     * XMC_augmented);
  YMC_hyp               = sign(w_hypothesis' * XMC_augmented);
  DISAGREE              = [YMC ~= YMC_hyp];
  percentDisagree(m)   = sum(DISAGREE)/NMONTECARLO;
end
meanIter = mean(iterations);
meanDisagreement = mean(percentDisagree);
fprintf('N = %d\n', N);
fprintf('\t mean(iterations) = %f\n', meanIter);
fprintf('\t mean(P(disagreement)) = %f\n', meanDisagreement);

%% the rest of the code is just plotting for eye candy, it is a mess and not specifically relevant to PLA
hold on
plot3(X(1,Y>0), X(2,Y>0), X(3,Y>0), 'ro');
plot3(X(1,Y<0), X(2,Y<0), X(3,Y<0), 'bo');
corners = [1 1; -1 1; -1 -1; 1 -1]';
z       = -(w_target(1) + w_target(2:end-1)'*corners)/w_target(end);
fill3(corners(1,:), corners(2,:), z, 'g');
alpha(0.3);
z       = -(w_hypothesis(1) + w_hypothesis(2:end-1)'*corners)/w_hypothesis(end);
fill3(corners(1,:), corners(2,:), z, 'm');
alpha(0.3);
bin1 = [ 1 1 1; -1 1 1; -1 -1 1; 1 -1 1; 1 1  1];
bin2 =  flipud(bin1);
bin2(:,3) = -1;
bin3 = [1 -1 -1; 1 -1 1; -1 -1 1; -1 -1 -1; -1 1 -1; -1 1 1];
bin =[bin1; bin2; bin3];
plot3(bin(:,1), bin(:,2), bin(:,3), ':k')
hold off
title(sprintf('N = %d\ngreen plane = target\npink plane = PLA hypothesis', N))
xlabel('x')
ylabel('y')
zlabel('z')
axis equal
axis(2*[-1 1 -1 1 -1 1])
view(3)