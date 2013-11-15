function X = transform(oldX)
  [m, n] = size(oldX);
  x_1 = oldX(:, 1); x_2 = oldX(:, 2);
  X = [ones(m, 1) oldX x_1.^2 x_2.^2 x_1.*x_2 abs(x_1 - x_2) abs(x_1 + x_2)];
end