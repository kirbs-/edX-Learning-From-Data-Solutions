% Adds noise to y by flipping the sign of some examples
function ny = addnoise (y, prob)
  ny = y;
  N = size (y, 1);

  % Number of training examples to be changed
  noisecount = fix (prob * length (y));

  for i = 1:noisecount

    % Picks a random y and flips it
    n = fix (unifrnd (1, N));
    ny(n) = y(n) * -1;
  end
end
