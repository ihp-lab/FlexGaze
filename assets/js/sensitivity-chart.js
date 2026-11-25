// Sensitivity chart: crop-sensitivity curves from window.FLEXGAZE_SENSITIVITY
// (assets/data/sensitivity.js). The legend and every panel are drawn here; each panel is drawn at
// its own pixel width, so type and marks keep their size in any layout.
(function () {
  var data = window.FLEXGAZE_SENSITIVITY;
  var chart = document.querySelector(".sensitivity-chart");
  var NS = "http://www.w3.org/2000/svg";

  if (!data) {
    var message = document.createElement("p");
    message.className = "sensitivity-message";
    message.textContent = "The chart data could not be loaded.";
    chart.appendChild(message);
    return;
  }

  // In legend order; drawn in reverse, so FlexGaze ends up on top.
  var SERIES = [
    { key: "flexgaze", name: "FlexGaze-B", marker: "square", legend: { tag: "rect", attrs: { x: 10, y: 2.5, width: 7, height: 7 } } },
    { key: "unigaze", name: "UniGaze-H", marker: "diamond", legend: { tag: "path", attrs: { d: "M14 1.6 18.4 6 14 10.4 9.6 6Z" } } },
    { key: "gat", name: "GaT + ST-WSGE", marker: "circle", legend: { tag: "circle", attrs: { cx: 14, cy: 6, r: 3.9 } } }
  ];
  var DRAW_ORDER = SERIES.slice().reverse();
  var CROP_TICKS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];
  var X_TITLE = { "crop-scale": "Eye distance / crop side", "center-shift": "Center shift / eye distance" };
  var FAMILY = { "crop-scale": "crop scale", "center-shift": "center shift" };
  var METHODS = SERIES.map(function (s) { return s.name; });
  // Plot area width over height, as in one panel of the paper's figure.
  var ASPECT = 1.7;

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, value, cls, anchor) {
    var node = el("text", { x: x, y: y, "class": cls, "text-anchor": anchor || "middle" }, parent);
    node.textContent = value;
    return node;
  }

  function legend() {
    var list = chart.querySelector(".sensitivity-legend");
    list.setAttribute("aria-label", "Methods");
    SERIES.forEach(function (s) {
      var item = document.createElement("li");
      var icon = el("svg", { viewBox: "0 0 28 12", "aria-hidden": "true" }, item);
      el("line", { x1: 1, y1: 6, x2: 27, y2: 6, "class": "sensitivity-line series-" + s.key }, icon);
      var attrs = { "class": "sensitivity-marker series-" + s.key };
      for (var key in s.legend.attrs) attrs[key] = s.legend.attrs[key];
      el(s.legend.tag, attrs, icon);
      item.appendChild(document.createTextNode(s.name));
      list.appendChild(item);
    });
  }

  // The fewest ticks of a 1-2-2.5-5 step that still leave two to three inside the range.
  function niceTicks(low, high) {
    var steps = [];
    [1, 2, 2.5, 5].forEach(function (base) {
      for (var power = -2; power <= 3; power++) steps.push(base * Math.pow(10, power));
    });
    steps.sort(function (a, b) { return a - b; });
    for (var i = 0; i < steps.length; i++) {
      var step = steps[i], ticks = [];
      for (var t = Math.ceil(low / step) * step; t <= high + 1e-9; t += step) ticks.push(+t.toFixed(6));
      if (ticks.length <= 3 && ticks.length >= 2) return ticks;
    }
    return [low, high];
  }

  function marker(group, s, x, y) {
    var cls = "sensitivity-marker series-" + s.key;
    if (s.marker === "square") return el("rect", { x: x - 2.7, y: y - 2.7, width: 5.4, height: 5.4, "class": cls }, group);
    if (s.marker === "diamond") return el("path", { d: "M" + x + " " + (y - 3.6) + "L" + (x + 3.6) + " " + y + "L" + x + " " + (y + 3.6) + "L" + (x - 3.6) + " " + y + "Z", "class": cls }, group);
    return el("circle", { cx: x, cy: y, r: 3.0, "class": cls }, group);
  }

  function draw(holder, firstInRow) {
    while (holder.firstChild) holder.removeChild(holder.firstChild);
    var W = holder.clientWidth;
    if (!W) return;
    var M = { left: firstInRow ? 46 : 32, right: 8, top: 22, bottom: 36 };
    var pw = W - M.left - M.right, ph = Math.round((W - 40) / ASPECT);
    var H = M.top + ph + M.bottom;
    var family = holder.getAttribute("data-family");
    var panel = data[holder.getAttribute("data-dataset")][family];
    var crop = family === "crop-scale";
    var all = [];
    SERIES.forEach(function (s) { all = all.concat(panel[s.key]); });
    var xs = all.map(function (p) { return crop ? Math.log(p[0]) : p[0]; });
    var ys = all.map(function (p) { return p[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var mx = (x1 - x0) * 0.04, my = (y1 - y0) * 0.08;
    x0 -= mx; x1 += mx; y0 -= my; y1 += my;
    function sx(v) { return M.left + ((crop ? Math.log(v) : v) - x0) / (x1 - x0) * pw; }
    function sy(v) { return M.top + (y1 - v) / (y1 - y0) * ph; }

    var label = holder.getAttribute("data-label");
    var svg = el("svg", {
      width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Panel " + label + ", " + FAMILY[family] + ": mean angular error of " +
        METHODS.slice(0, -1).join(", ") + " and " + METHODS[METHODS.length - 1] + "."
    }, holder);

    text(svg, M.left, 14, label + ", " + FAMILY[family], "sensitivity-panel-title", "start");
    niceTicks(y0, y1).forEach(function (t) {
      el("line", { x1: M.left, x2: W - M.right, y1: sy(t), y2: sy(t), "class": "sensitivity-gridline" }, svg);
      text(svg, M.left - 5, sy(t) + 4, String(t), "sensitivity-tick", "end");
    });
    var xticks = crop
      ? CROP_TICKS.filter(function (t) { var l = Math.log(t); return l >= x0 && l <= x1; })
      : [0, 0.25, 0.5, 0.75];
    if (crop && pw < 200) xticks = xticks.filter(function (t) { return [0.15, 0.2, 0.3, 0.4, 0.6].indexOf(t) >= 0; });
    xticks.forEach(function (t) {
      var x = sx(t);
      el("line", { x1: x, x2: x, y1: M.top + ph, y2: M.top + ph + 3, "class": "sensitivity-axis" }, svg);
      text(svg, x, M.top + ph + 15, crop ? String(t) : t.toFixed(2), "sensitivity-tick");
    });
    if (!crop) el("line", { x1: sx(0), x2: sx(0), y1: M.top, y2: M.top + ph, "class": "sensitivity-gridline" }, svg);
    el("line", { x1: M.left, x2: M.left, y1: M.top, y2: M.top + ph, "class": "sensitivity-axis" }, svg);
    el("line", { x1: M.left, x2: W - M.right, y1: M.top + ph, y2: M.top + ph, "class": "sensitivity-axis" }, svg);
    text(svg, M.left + pw / 2, H - 4, X_TITLE[family], "sensitivity-axis-title");
    if (firstInRow) {
      var yTitle = text(svg, 0, 0, "Mean angular error (°)", "sensitivity-axis-title");
      yTitle.setAttribute("transform", "translate(11 " + (M.top + ph / 2) + ") rotate(-90)");
    }

    DRAW_ORDER.forEach(function (s) {
      var points = panel[s.key];
      var group = el("g", {}, svg);
      el("polyline", {
        points: points.map(function (p) { return sx(p[0]).toFixed(2) + "," + sy(p[1]).toFixed(2); }).join(" "),
        "class": "sensitivity-line series-" + s.key
      }, group);
      points.forEach(function (p) {
        var mark = marker(group, s, +sx(p[0]).toFixed(2), +sy(p[1]).toFixed(2));
        var tip = el("title", {}, mark);
        tip.textContent = s.name + ": " + X_TITLE[family].toLowerCase() + " " + p[0].toFixed(3) + ", error " + p[1].toFixed(2) + "°";
      });
    });
  }

  legend();
  var grid = chart.querySelector(".sensitivity-grid");
  var holders = Array.prototype.slice.call(grid.querySelectorAll(".sensitivity-panel"));
  var lastWidths = "";
  function drawAll() {
    var widths = holders.map(function (h) { return h.clientWidth + "@" + h.offsetLeft; }).join(",");
    if (widths === lastWidths) return;
    lastWidths = widths;
    var left = Math.min.apply(null, holders.map(function (h) { return h.offsetLeft; }));
    holders.forEach(function (h) { draw(h, Math.abs(h.offsetLeft - left) < 2); });
  }
  drawAll();
  if (window.ResizeObserver) new ResizeObserver(drawAll).observe(grid);
  else window.addEventListener("resize", drawAll);
})();
