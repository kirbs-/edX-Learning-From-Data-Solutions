function saveplot (h, width, height, filename)
  set (h, "PaperUnits", "inches");
  set (h, "PaperOrientation", "portrait");
  set (h, "PaperSize", [height width]);
  set (h, "PaperPosition", [0 0 width height]);

  print (h, filename, "-dpng", "-FHelvetica:7");
end
