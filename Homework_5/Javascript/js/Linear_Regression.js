window.Linear_Regression = function() {
  this._weights = [];
}

window.Linear_Regression.prototype.regression = function(dataSet, completeCallback) {
  var array_X = [];
  var array_Y = [];
  for(var i in dataSet) {
    array_X.push(dataSet[i].point);
    array_Y.push([dataSet[i].output]);
  }

  var matrix_X = math.matrix(array_X);
  var matrix_Y = math.matrix(array_Y);
  var matrix_X_traspose = math.transpose(matrix_X);
  var matrix_X_pseudo_inverse = math.multiply(math.inv(math.multiply(matrix_X_traspose, matrix_X)), matrix_X_traspose)
  var weights_matrix = math.multiply(matrix_X_pseudo_inverse, matrix_Y).toArray();
  this._weights = [].concat.apply([], weights_matrix);
  if(completeCallback) completeCallback(this._weights);
}

window.Linear_Regression.prototype.getOutput = function(point) {
  if(point.length !== this._weights.length) throw "Dimension of point is not correct"

  var output = 0;
  for(var i = 0; i < point.length; i++) {
    output += this._weights[i] * point[i];
  }

  return output;
}