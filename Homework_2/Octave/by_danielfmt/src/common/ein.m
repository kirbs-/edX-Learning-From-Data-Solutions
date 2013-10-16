% Returns the in-sample error (Ein)
function retval = ein (X, y, w)
  retval = length (find (sign (X*w) != y)) / length (X);
end
