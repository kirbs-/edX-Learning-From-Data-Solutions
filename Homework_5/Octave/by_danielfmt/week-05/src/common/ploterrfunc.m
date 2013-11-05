% Plots a given nonlinear transformation
function ploterrfunc (spc, f, tit)
  x = y = linspace (spc(1), spc(2));

  [xx, yy] = meshgrid (x', y');

  contour (xx, yy, arrayfun (@ (u, v) f ([u;v]), xx, yy), 50);
  title (tit);

  xlabel ("w_1");
  ylabel ("w_2");
end

