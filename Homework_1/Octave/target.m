% Target function
function y = target (fp1, fp2, x)
  m = (fp2(2) - fp1(2)) / (fp2(1) - fp1(1));
  b = fp1(2) - m * fp1(1);

  y = m * x + b;
end