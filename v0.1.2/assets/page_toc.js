document.addEventListener("DOMContentLoaded", function () {
  var page = document.getElementById("documenter-page");
  if (!page) return;
  var headings = page.querySelectorAll("h2[id]");
  if (headings.length < 2) return;

  var nav = document.createElement("nav");
  nav.className = "page-toc";
  var title = document.createElement("p");
  title.className = "page-toc-title";
  title.textContent = "On this page";
  nav.appendChild(title);

  var list = document.createElement("ul");
  var links = [];
  headings.forEach(function (h) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.replace(/¬|permalink/gi, "").trim();
    li.appendChild(a);
    list.appendChild(li);
    links.push({ heading: h, link: a });
  });
  nav.appendChild(list);
  document.body.appendChild(nav);

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        var match = links.find(function (l) {
          return l.heading === entry.target;
        });
        if (!match) return;
        if (entry.isIntersecting) {
          links.forEach(function (l) {
            l.link.classList.remove("active");
          });
          match.link.classList.add("active");
        }
      });
    },
    { rootMargin: "0px 0px -70% 0px" }
  );
  headings.forEach(function (h) {
    observer.observe(h);
  });
});
