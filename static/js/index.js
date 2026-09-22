(function () {
  "use strict";

  /* ---- hero video: skip the download on small screens and reduced motion ---- */
  var hero = document.getElementById("hero-video");
  if (hero) {
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var small = window.matchMedia("(max-width: 768px)").matches;
    if (reduce || small) {
      hero.removeAttribute("autoplay");
      hero.pause();
      hero.removeAttribute("src");
      while (hero.firstChild) hero.removeChild(hero.firstChild);
      hero.load();
    } else {
      var p = hero.play();
      if (p && typeof p.catch === "function") p.catch(function () { /* autoplay blocked */ });
    }
  }

  /* ---- demo player: swap clips ---- */
  var player = document.getElementById("demo-video");
  var source = document.getElementById("demo-source");
  var caption = document.getElementById("demo-caption");
  var buttons = Array.prototype.slice.call(document.querySelectorAll(".switch button"));

  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      buttons.forEach(function (b) { b.setAttribute("aria-selected", "false"); });
      btn.setAttribute("aria-selected", "true");
      source.setAttribute("src", btn.dataset.src);
      if (caption) caption.textContent = btn.dataset.caption;
      player.load();
      player.play().catch(function () { /* ignore */ });
    });
  });

  /* ---- scroll to top ---- */
  var top = document.getElementById("to-top");
  if (top) {
    window.addEventListener("scroll", function () {
      top.classList.toggle("show", window.scrollY > 600);
    }, { passive: true });
    top.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
})();
