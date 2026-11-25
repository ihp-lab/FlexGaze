// Copy BibTeX button. The outcome shows on the button for a moment; the button keeps its name
// (aria-label), so only the polite live region announces it.
(function () {
  var button = document.getElementById("citation-copy");
  var status = document.getElementById("citation-copy-status");
  var bibtex = document.getElementById("citation-bibtex");
  var label = button.textContent;
  var timer = null;

  function done(message) {
    button.textContent = message;
    status.textContent = message;
    clearTimeout(timer);
    timer = setTimeout(function () {
      button.textContent = label;
      status.textContent = "";
    }, 3000);
  }

  button.addEventListener("click", function () {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(bibtex.textContent).then(function () { done("Copied"); }, function () { done("Copy failed"); });
      return;
    }
    var range = document.createRange();
    range.selectNodeContents(bibtex);
    var selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    done("Selected, copy with your keyboard");
  });
})();
