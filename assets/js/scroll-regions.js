// Sideways scroll boxes become named, focusable regions only while their content overflows,
// so keyboard users get a stop exactly where there is something to scroll.
(function () {
  var boxes = Array.prototype.slice.call(document.querySelectorAll(".scroll-x[data-scroll-label]"));

  function update(box) {
    if (box.scrollWidth > box.clientWidth + 1) {
      box.setAttribute("role", "region");
      box.setAttribute("aria-label", box.getAttribute("data-scroll-label"));
      box.setAttribute("tabindex", "0");
    } else {
      box.removeAttribute("role");
      box.removeAttribute("aria-label");
      box.removeAttribute("tabindex");
    }
  }

  function updateAll() {
    boxes.forEach(update);
  }

  updateAll();
  window.addEventListener("load", updateAll);
  if (window.ResizeObserver) {
    var observer = new ResizeObserver(updateAll);
    boxes.forEach(function (box) {
      observer.observe(box);
      if (box.firstElementChild) observer.observe(box.firstElementChild);
    });
  } else {
    window.addEventListener("resize", updateAll);
  }
})();
