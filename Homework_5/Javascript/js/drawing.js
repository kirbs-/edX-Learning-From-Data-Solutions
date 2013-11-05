
  window.red = new jsColor('red');
  window.redPen = new jsPen(red,1);
  window.blue = new jsColor('blue');
  window.bluePen = new jsPen(blue,1);
  window.black = new jsColor('black');
  window.blackPen = new jsPen(black,1);
  window.orange = new jsColor('#FFCC33');
  window.orangePen = new jsPen(orange,4);
  window.green = new jsColor('#00AA00');
  window.greenPen = new jsPen(green,2);
  window.pink = new jsColor('#F19CBB');
  window.pinkPen = new jsPen(pink,1);
  window.pinkPen2 = new jsPen(pink,2);

  window.smallFont = new jsFont('sans-serif', 'normal', 'x-small');

  window.initCanvas = function() {
    if(window.gr && window.gr.clear) window.gr.clear();
    window.gr = new jsGraphics(document.getElementById('canvas'));
    window.gr.setOrigin(new jsPoint(250,250));
    window.gr.setScale(200);
    window.gr.setCoordinateSystem('cartecian');
    window.gr.showGrid(1);
  }

  window.plotLine = function(points, pen) {
    window.gr.drawLine(pen ? pen : blackPen,
    new jsPoint(points[0][0],points[0][1]),
    new jsPoint(points[1][0],points[1][1])); 
  }

  window.plotPoint = function(point, color) {
    window.gr.fillCircle(color,new jsPoint(point[0],point[1]),0.01);
  }

  window.plotCircle = function(point, pen) {
    window.gr.drawCircle(pen, new jsPoint(point[0],point[1]), 0.02);
  }

  window.drawTargetLine = function(plane) {
    var targetLine = window.weightsToLine(plane.lineToWeights(), plane);
    window.plotLine(targetLine);
  }

  window.drawDataSet = function(dataSet) {
    for(var i = 0; i < dataSet.length; i++) {
      window.plotPoint(dataSet[i].point, (dataSet[i].output < 0 ? red : blue));
    }
  }

  window.weightsToLine = function(weights, plane) {
    var y_on_x_min = -1 * (weights[1] * plane._x_min + weights[0]) / weights[2];
    var y_on_x_max = -1 * (weights[1] * plane._x_max + weights[0]) / weights[2];
    var x_on_y_min = -1 * (weights[2] * plane._y_min + weights[0]) / weights[1];
    var x_on_y_max = -1 * (weights[2] * plane._y_max + weights[0]) / weights[1];

    var result = [];

    if(y_on_x_min >= plane._y_min && y_on_x_min <= plane._y_max) {
      result.push([plane._x_min, y_on_x_min]);
    }
    if(y_on_x_max >= plane._y_min && y_on_x_max <= plane._y_max) {
      result.push([plane._x_max, y_on_x_max]);
    }
    if(x_on_y_min >= plane._x_min && x_on_y_min <= plane._x_max) {
      result.push([x_on_y_min, plane._y_min]);
    }
    if(x_on_y_max >= plane._x_min && x_on_y_max <= plane._x_max) {
      result.push([x_on_y_max, plane._y_max]);
    }
  if(result.length !== 2) result = [[0,0,],[0,0]];
  return result;
  }

  window.joinWithRounding = function(input, decimalPoint) {
    var result = [];
    for(var i in input) {
      result.push(input[i].toFixed(decimalPoint));
    }
    return result.join(', ');
  }
