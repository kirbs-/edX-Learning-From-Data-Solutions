window.Quadratic = function() {
}

// f(x) = ax^2 + bx + c
window.Quadratic.solve = function(a,b,c) {
  var discriminant = b*b - 4*a*c;
  if(discriminant < 0) return undefined; //We don't care imaginary root;
  var sqrtOfDis = Math.sqrt(discriminant);
  return [(0-b+sqrtOfDis)/(2*a), (0-b-sqrtOfDis)/(2*a)];
}
