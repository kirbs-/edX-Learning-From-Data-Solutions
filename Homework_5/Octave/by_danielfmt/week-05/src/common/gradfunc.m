% Function that computes the gradient for the error surface
function nw = gradfunc (w)
  u = w(1);
  v = w(2);

  % Partial derivative with respect to u and v, respectively
  du = 2 * exp (-2*u) * (u*exp (u+v) - 2*v) * (exp (u+v) + 2*v);
  dv = 2 * exp (-2*u) * (u*exp (u+v) - 2) * (u*exp (u+v) - 2*v);

  nw = [du; dv];
end
