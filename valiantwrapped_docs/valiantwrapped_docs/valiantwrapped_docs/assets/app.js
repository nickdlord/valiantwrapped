async function loadAuthors() {
  const res = await fetch("data/authors.json");
  if (!res.ok) throw new Error("Failed to load data/authors.json");
  return await res.json();
}

function uniq(arr) {
  return Array.from(new Set(arr)).sort((a,b) => a.localeCompare(b));
}

function norm(s) {
  return (s || "").toString().toLowerCase();
}

function matches(author, q, theme) {
  if (theme && !(author.themes || []).includes(theme)) return false;
  if (!q) return true;

  const hay = [
    author.display_name,
    author.author_label,
    author.artist_name,
    author.album_title,
    (author.themes || []).join(" "),
    author.top_journal,
    author.top_paper
  ].join(" ");

  return norm(hay).includes(norm(q));
}

function cardHTML(a) {
  const tags = (a.themes || []).slice(0, 3).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("");
  const sub = [
    a.artist_name ? `Artist: ${escapeHtml(a.artist_name)}` : "",
    a.album_title ? `Album: ${escapeHtml(a.album_title)}` : ""
  ].filter(Boolean).join(" • ");

  const cover = a.cover_url
    ? `<img src="${a.cover_url}" alt="" loading="lazy" />`
    : "";

  return `
    <a class="card" href="${a.profile_url}">
      <div class="cover">${cover}</div>
      <div class="meta">
        <div class="name">${escapeHtml(a.display_name || a.author_label)}</div>
        <div class="sub">${sub || "&nbsp;"}</div>
        <div class="tags">${tags}</div>
      </div>
    </a>
  `;
}

function escapeHtml(str) {
  return (str || "").toString()
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function render(authors, q, theme) {
  const grid = document.getElementById("grid");
  const empty = document.getElementById("empty");

  const filtered = authors.filter(a => matches(a, q, theme));
  grid.innerHTML = filtered.map(cardHTML).join("");

  empty.classList.toggle("hidden", filtered.length !== 0);
}

function fillThemeFilter(authors) {
  const sel = document.getElementById("themeFilter");
  const allThemes = uniq(authors.flatMap(a => a.themes || []).filter(Boolean));
  if (allThemes.length === 0) {
    // hide filter if no themes exist
    sel.classList.add("hidden");
    return;
  }
  for (const t of allThemes) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  }
}

(async function init() {
  const authors = await loadAuthors();
  fillThemeFilter(authors);

  const search = document.getElementById("search");
  const themeFilter = document.getElementById("themeFilter");

  const rerender = () => render(authors, search.value, themeFilter.value);

  search.addEventListener("input", rerender);
  themeFilter.addEventListener("change", rerender);

  rerender();
})();
