% Calculates the linear regression
function w = linearreg (X, y)
  w = pinv (X) * y;
end
