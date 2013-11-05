% Runs gradient descent until termination criteria is met
function [w, epoch] = gd (w, eta, errfunc, errgradfunc, threshold, maxiters)
  for epoch = 1:maxiters

    % Updates the weights
    w += -eta * errgradfunc (w);

    % termination criterion
    if (errfunc (w) < threshold)
      return;
    end
  end
end
