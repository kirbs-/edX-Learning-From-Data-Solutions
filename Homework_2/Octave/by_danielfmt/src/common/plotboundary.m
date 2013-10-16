% Plots the decision boundary based on the weight vector
function plotboundary (w, limits, plotstr)
  f = @ (x) (-1 ./ w(3)) .* (w(2) .* x + w(1));
  fplot (f, limits, plotstr);
end
