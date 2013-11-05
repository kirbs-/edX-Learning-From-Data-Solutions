function err = eout (spc, N, d, wt, wh, errfunc)

  % N random examples
  X = unifrnd (spc(1), spc(2), N, d);

  % Introduces the synthetic dimension x0
  X = [ones(N, 1) X];

  % Target labels
  y = sign (X * wt);

  err = errfunc (X, y, wh);
end
