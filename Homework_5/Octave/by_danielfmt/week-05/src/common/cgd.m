% Runs coordinate Descent until termination criteria is met
function [w, epoch] = cgd (w, eta, errfunc, errgradfunc, threshold, maxiters)
  for epoch = 1:maxiters

    w(1) += -eta * (errgradfunc (w))(1);
    w(2) += -eta * (errgradfunc (w))(2);

    % termination criterion
    if (errfunc (w) < threshold)
      return;
    end
  end
end
