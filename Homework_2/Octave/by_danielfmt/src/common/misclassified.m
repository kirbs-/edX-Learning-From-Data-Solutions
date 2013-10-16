% Returns the indexes of the misclassified examples, if they exist
function retval = misclassified (X, y, w)
  retval = find (bsxfun (@(n, y) sign (X(n,:) * w) != y, [1:size(X)]', y));
end
