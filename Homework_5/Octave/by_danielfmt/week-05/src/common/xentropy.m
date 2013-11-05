% Cross-entropy error measure
function err = xentropy (X, y, w)
  func = @(n) log (1 + exp (-y(n) * (w' * X(n,:)')));
  err = mean (arrayfun (func, 1:length (X)));
end
