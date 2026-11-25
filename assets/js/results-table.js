// Results table alignment, after the paper's table layout.
// Per score column: M = widest mean, S = widest SD suffix, H = head width.
// Over all score columns: L = max(H/2, M/2) less 3 pt (never below M/2), R = max(H/2, M/2 + S).
// Every score cell holds a block L + R wide; its mean box (M wide) is centered L from the block's
// left edge and its SD suffix follows; the head is centered over the block.
// A mean is split at its decimal point into two boxes sized by the widest in the column, bold
// included, so bold and regular digits cannot move the point.
(function () {
  var table = document.querySelector(".results-table");

  function width(node) {
    return node ? node.getBoundingClientRect().width : 0;
  }

  function align() {
    var heads = table.tHead.rows[0].cells;
    var rows = [];
    Array.prototype.forEach.call(table.tBodies, function (body) {
      Array.prototype.forEach.call(body.rows, function (row) {
        if (row.cells.length === heads.length) rows.push(row);
      });
    });
    var parts = table.querySelectorAll(".results-score-block, .results-mean, .results-mean-whole, .results-mean-fraction, .results-sd, .results-head, .results-params-value");
    Array.prototype.forEach.call(parts, function (node) {
      node.style.width = "";
      node.style.minWidth = "0";
      node.style.paddingLeft = "";
    });
    var columns = [], L = 0, R = 0, halfMean = 0;
    // 3 pt at the paper's 8 pt table type is 0.375 em.
    var fontSize = parseFloat(getComputedStyle(table).fontSize);
    var trim = 0.375 * fontSize;
    // Widths are written in em, so they follow the text if it is resized after loading.
    function em(px) { return px / fontSize + "em"; }
    for (var c = 0; c < heads.length; c++) {
      if (!heads[c].classList.contains("results-score")) continue;
      var I = 0, F = 0, S = 0;
      rows.forEach(function (row) {
        var cell = row.cells[c];
        I = Math.max(I, width(cell.querySelector(".results-mean-whole")));
        F = Math.max(F, width(cell.querySelector(".results-mean-fraction")));
        S = Math.max(S, width(cell.querySelector(".results-sd")));
      });
      var M = I + F;
      var H = width(heads[c].querySelector(".results-head"));
      columns.push({ index: c, M: M, I: I, F: F });
      L = Math.max(L, H / 2, M / 2);
      R = Math.max(R, H / 2, M / 2 + S);
      halfMean = Math.max(halfMean, M / 2);
    }
    L = Math.max(L - trim, halfMean);
    var W = L + R;
    columns.forEach(function (column) {
      heads[column.index].querySelector(".results-head").style.width = em(W);
      rows.forEach(function (row) {
        var cell = row.cells[column.index];
        var block = cell.querySelector(".results-score-block");
        block.style.width = em(W);
        block.style.paddingLeft = em(L - column.M / 2);
        cell.querySelector(".results-mean").style.width = em(column.M);
        cell.querySelector(".results-mean-whole").style.width = em(column.I);
        cell.querySelector(".results-mean-fraction").style.width = em(column.F);
      });
    });
    // Params: numbers right-aligned in one box as wide as the widest, the box centered in the column.
    var params = table.querySelectorAll(".results-params-value"), P = 0;
    Array.prototype.forEach.call(params, function (node) { P = Math.max(P, width(node)); });
    Array.prototype.forEach.call(params, function (node) { node.style.width = em(P); });
  }

  align();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(align);
  window.addEventListener("load", align);
})();
