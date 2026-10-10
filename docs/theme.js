// Light / dark for the Flint site. The visitor's system theme applies by default; the nav
// toggle overrides it and the choice is remembered in this browser. Loaded in <head>, before
// first paint, so a remembered choice never flashes the other theme.
(function () {
  var KEY = "flint.theme";
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (e) { /* storage may be unavailable */ }

  function system() { return window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"; }
  function current() { return root.dataset.theme || system(); }
  function paint() {
    var b = document.getElementById("theme-toggle");
    if (!b) return;
    var dark = current() === "dark";
    b.textContent = dark ? "☀" : "☾";
    var label = dark ? "Switch to light mode" : "Switch to dark mode";
    b.setAttribute("aria-label", label);
    b.title = label;
  }
  document.addEventListener("DOMContentLoaded", function () {
    var b = document.getElementById("theme-toggle");
    if (b) b.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
      paint();
    });
    paint();
    if (window.matchMedia) matchMedia("(prefers-color-scheme: light)").addEventListener("change", paint);
  });
})();
