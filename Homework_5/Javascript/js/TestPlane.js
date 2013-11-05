Math.sign = function(x) {
  return x ? x < 0 ? -1 : 1 : 0;
}

window.TestPlane = function(x_min, x_max, y_min, y_max) {
  this._x_min = x_min;
  this._x_max = x_max;
  this._y_min = y_min;
  this._y_max = y_max;
  this._line = [this.randomPoint(), this.randomPoint()];
}

window.TestPlane.prototype.getOutput = function(point) {
  var output;
  var Ax = this._line[0][0];
  var Ay = this._line[0][1];
  var Bx = this._line[1][0];
  var By = this._line[1][1];
  output = Math.sign((Bx-Ax) * (point[1]-Ay) - (By-Ay) * (point[0]-Ax));
  return output;
}

window.TestPlane.prototype.lineToWeights = function() {
  var weights = [];
  var Ax = this._line[0][0];
  var Ay = this._line[0][1];
  var Bx = this._line[1][0];
  var By = this._line[1][1];

  weights[0] = Ax*(By-Ay) - Ay*(Bx-Ax);
  weights[1] = -1 * (By-Ay);
  weights[2] = (Bx-Ax);

  return weights;
}

window.TestPlane.prototype.randomPoint = function() {
  var x =  this._x_min + Math.random() * (this._x_max - this._x_min);
  var y =  this._y_min + Math.random() * (this._y_max - this._y_min);
  return [x,y];
}

