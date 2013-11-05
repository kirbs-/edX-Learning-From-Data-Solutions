window.Logistic_Regression = function() {
  this._weights = [];
  this._weights_history = [];
  this._norm_history = [0];
  this._training_dataSet;
  this._training_completed = false;
  this._eta = 0.01;
  this._ending_norm = 0.01;
  this._epoch = 0;
  this._current_epoch_dataSet;
  this.stochasticGradientDescentStep = this.stochasticGradientDescentStep.bind(this);
}

window.Logistic_Regression.prototype.initializeLearning = function(dataSet, startingWeights, eta, endingNorm) {
  this._weights = startingWeights;
  this._weights_history = [this._weights];
  this._norm_history = [0];
  this._training_dataSet = dataSet;
  this._training_completed = false;
  this._eta = eta;
  this._ending_norm = endingNorm;
  this._current_epoch_dataSet;
  this._epoch = 0;
}

window.Logistic_Regression.prototype.stochasticGradientDescent = function(completeCallback) {
  this.stochasticGradientDescentStep(null, null, completeCallback);
  while(!this._training_completed) {
    this.stochasticGradientDescentStep(null, null, completeCallback);
  }
}

window.Logistic_Regression.prototype.stochasticGradientDescentEpochStep = function(epochCallback, completeCallback) {
  this.stochasticGradientDescentStep(null, epochCallback, completeCallback);
  while(this._current_epoch_dataSet.length > 0) {
    this.stochasticGradientDescentStep(null, epochCallback, completeCallback);
  }
}

window.Logistic_Regression.prototype.stochasticGradientDescentStep = function(stepCallback, epochCallback, completeCallback) {
  // Randomize the order of training dataset before each epoch
  if(!this._current_epoch_dataSet || this._current_epoch_dataSet.length === 0) {
    this._current_epoch_dataSet = this.shuffle(this._training_dataSet);
  }

  var data = this._current_epoch_dataSet.pop();
  var col_vector_w = math.transpose(math.matrix([this._weights]));
  var col_vector_x = math.transpose(math.matrix([data.point]));

  var matrix_delta_e = this.get_delta_e(col_vector_w, col_vector_x, data.output);
  var result_matrix_w = math.subtract(col_vector_w, math.emultiply(this._eta, matrix_delta_e));
  this._weights = result_matrix_w.toVector();

  var norm = this.get_norm(this._weights_history[this._weights_history.length - 1], this._weights);

  // All points in dataset is trained in this epoch
  if(this._current_epoch_dataSet.length === 0) {
    this._weights_history.push(this._weights);
    this._norm_history.push(norm);
    this._epoch += 1;

    if(norm < this._ending_norm) {
      this._training_completed = true;
      if(completeCallback) completeCallback(this._norm_history, this._weights_history, this._epoch);
      return;
    }
    if(epochCallback) epochCallback(this._norm_history, this._weights_history, this._epoch);
  }

  if(stepCallback) stepCallback(data,
                                norm, 
                                this._weights,
                                this._norm_history,
                                this._weights_history,
                                this._epoch,
                                this._current_epoch_dataSet.length);
}

window.Logistic_Regression.prototype.get_norm = function(w1, w2) {
  var delta_weights = math.subtract(w1, w2);

  var sum = 0;
  for(var i = 0; i < delta_weights.length; i++) {
    sum += Math.pow(delta_weights[i], 2);
  }

  return Math.sqrt(sum);
}

window.Logistic_Regression.prototype.get_delta_e = function(w, x, y) {
  var y_wT_x = this.get_y_wT_x(w, x, y);
  var result = math.multiply(math.multiply((-1 * y), x), 1/(1+Math.exp(y_wT_x)));
  return result;
}

window.Logistic_Regression.prototype.Eout = function(dataSet) {
  var col_vector_w = math.transpose(math.matrix([this._weights]));
  var total_E = 0;
  for(var i in dataSet) {
    var col_vector_x = math.transpose(math.matrix([dataSet[i].point]));
    var y_wT_x = this.get_y_wT_x(col_vector_w, col_vector_x, dataSet[i].output);
    var E = Math.log(1 + Math.exp(-1 * y_wT_x));
    total_E += E;
  }
  return total_E / dataSet.length;
}

window.Logistic_Regression.prototype.get_y_wT_x = function(w, x, y) {
  return math.multiply(y, math.multiply(math.transpose(w), x)).toScalar();
}

window.Logistic_Regression.prototype.shuffle = function (a){
  var o = a.slice(0);
  for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
  return o;
};