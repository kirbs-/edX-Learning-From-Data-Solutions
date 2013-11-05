% Gradient of the cross-entropy error measure
function g = xentropygrad (Xn, yn, w)
  g = -(yn * Xn') / (1 + exp (yn * (w' * Xn')));
end
