function saveplot (h, width, height, filename)
  set (h, "PaperUnits", "inches");
  set (h, "PaperOrientation", "portrait");
  set (h, "PaperSize", [height width]);
  set (h, "PaperPosition", [0 0 width height]);

  FN = findall (h, "-property", "FontName");
  set (FN, "FontName", "Helvetica");

  FS = findall (h, "-property", "FontSize");
  set(FS, "FontSize", 8);

  print (h, filename, "-dpng");
end
