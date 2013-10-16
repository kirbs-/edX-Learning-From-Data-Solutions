% Returns the out-of-sample error (Eout)
function diff = eout (N, d, spc, f, wf, w, nprob)

  % Generates n more examples, where n = count
  X = unifrnd (spc(1), spc(2), N, d);

  % Introduces the synthetic dimension x0
  X = [ones(N, 1) X];

  % Calculates the correct value for y using the target function
  if wf
    y = sign (X * wf);
  else
    y = arrayfun (f, X(:,2), X(:,3));
  end

  if (nprob)
    % Introduces noise
    y = addnoise (y, nprob);

    % Introduces the new features, including x0
    X = addfeatures (X);
  end

  % Calculates Eout
  diff = ein (X, y, w);
end
