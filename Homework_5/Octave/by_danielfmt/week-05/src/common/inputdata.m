function [X y w] = inputdata (spc, N, d)

  % Loops until we find negative and positive examples
  while (1)

    % N random examples
    X = unifrnd (spc(1), spc(2), N, d);

    % Introduces the synthetic dimension x0
    X = [ones(N, 1) X];

    % Weight vector that represents the target function f
    w = unifrnd (spc(1), spc(2), size (X, 2), 1);

    % Uses the target function to set the desired labels
    y = sign (X * w);

    pos = find (y > 0);
    neg = find (y < 0);

    % Make sure we have both positive and negative examples
    if (any (pos) && any (neg))
      break;
    end
  end
end
