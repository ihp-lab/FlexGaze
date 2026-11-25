// Theme switch: system -> light -> dark. The root's data-theme is the only state: the inline
// script in <head> sets it from storage before the first frame, CSS shows the matching icon and
// label, and this script only changes it and stores the choice. "system" means no data-theme.
(function () {
  var STORAGE_KEY = "flexgaze-theme";
  var ORDER = ["system", "light", "dark"];
  var root = document.documentElement;
  var button = document.getElementById("theme-switch");

  function current() {
    return root.getAttribute("data-theme") || "system";
  }

  function next(mode) {
    return ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
  }

  function describe(mode) {
    button.setAttribute("aria-label", "Color theme: " + mode + ". Switch to " + next(mode) + ".");
  }

  describe(current());
  button.addEventListener("click", function () {
    var mode = next(current());
    if (mode === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
    try { localStorage.setItem(STORAGE_KEY, mode); } catch (e) {}
    describe(mode);
  });
})();
