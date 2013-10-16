% Target function
function y = target (x1, x2)
  y = sign (x1 * x1 + x2 * x2 - 0.6);
end
