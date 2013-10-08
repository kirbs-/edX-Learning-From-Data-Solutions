Math.sign = function(x) {
  return x ? x < 0 ? -1 : 1 : 0;
}

window.PLA = function(weights, modifier) {
  this._weights = weights
  this._modifier = modifier
  this._iteration = 0;
}

window.PLA.prototype.train = function(data) {
  if(data.output === this.getOutput(data.point)) return false;

  this._weights[0] += data.output;
  for(var i = 0; i < data.point.length; i++) {
    this._weights[i+1] += data.output * data.point[i];
  }

  this._iteration += 1;
  return true;
}

// iterationCallback(weightCurrent, weightsBefore, iteration, trainedPoint)
// completeCallback(weight, iteration)
window.PLA.prototype.trainSet = function(dataSet, singleIteration, iterationCallback, completeCallback) {
  var misclassifiedDataSet;
  do {
    // Check training complete
    misclassifiedDataSet = [];
    for(var i in dataSet) {
      if(dataSet[i].output !== this.getOutput(dataSet[i].point)) {
        misclassifiedDataSet.push(dataSet[i]);
      }
    }

    if(misclassifiedDataSet.length !== 0) {
      var misclassifiedData = misclassifiedDataSet[Math.floor(Math.random() * misclassifiedDataSet.length)];
      var weightsBefore = $.extend([],this._weights);
      this.train(misclassifiedData);
      if(iterationCallback) iterationCallback(this._weights, weightsBefore, this._iteration, misclassifiedData.point);
    }
  } while(misclassifiedDataSet.length !== 0 && !singleIteration)

  if(misclassifiedDataSet.length === 0 && completeCallback) completeCallback(this._weights, this._iteration);
}

window.PLA.prototype.getOutput = function(point) {
  if(point.length !== this._weights.length - 1) throw "Dimension of point is not correct"

  var output;
  output = this._weights[0];
  for(var i = 0; i < point.length; i++) {
    output += this._weights[i+1] * point[i];
  }
  
  if(typeof(this._modifier) === 'function') output = this._modifier(output);
  return output;
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

