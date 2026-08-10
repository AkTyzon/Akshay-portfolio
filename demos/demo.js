/* Shared front end for the precomputed recommenders.
   CFG is defined per page before this script loads. */
(function () {
  var DATA = null, byId = {}, sel = -1, filtered = [];
  var $ = function (id) { return document.getElementById(id); };
  var input = $('q'), sugg = $('sugg'), out = $('out'), seed = $('seed');

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  fetch(CFG.data)
    .then(function (r) { return r.json(); })
    .then(function (d) {
      DATA = d;
      d.items.forEach(function (i) { byId[String(i.id)] = i; });
      renderMetrics(d.meta);
      show(String(CFG.initial(d)));
    })
    .catch(function () {
      out.innerHTML = '<p style="color:#a53d3d">Could not load the model data file.</p>';
    });

  function renderMetrics(meta) {
    var el = $('metrics');
    if (!el) return;
    el.innerHTML = CFG.metrics(meta).map(function (m) {
      return '<div class="metric' + (m.muted ? ' muted' : '') + '"><span class="k">' +
        esc(m.k) + '</span><span class="v">' + esc(m.v) + '</span></div>';
    }).join('');
  }

  function show(id) {
    var it = byId[id];
    if (!it) return;
    seed.innerHTML = '<div class="lab">Showing recommendations for</div>' +
      '<div class="name">' + esc(it.t) + '</div>' +
      '<div class="meta">' + CFG.seedMeta(it) + '</div>';

    var nbs = (DATA.neighbours[id] || []);
    if (!nbs.length) {
      out.innerHTML = '<p style="color:var(--faint)">No close neighbours above the similarity threshold.</p>';
      return;
    }
    var max = nbs[0].s || 1;
    out.innerHTML = nbs.map(function (n) {
      var o = byId[String(n.id)];
      if (!o) return '';
      return '<div class="rec" data-id="' + esc(n.id) + '" tabindex="0" role="button">' +
        '<div class="t">' + esc(o.t) + '</div>' +
        '<div class="m">' + CFG.recMeta(o, n) + '</div>' +
        '<div class="bar"><i style="width:' + Math.round((n.s / max) * 100) + '%"></i></div>' +
        '</div>';
    }).join('');

    Array.prototype.forEach.call(out.querySelectorAll('.rec'), function (el) {
      var go = function () { show(el.dataset.id); window.scrollTo({ top: seed.offsetTop - 90, behavior: 'smooth' }); };
      el.addEventListener('click', go);
      el.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
  }

  function closeSugg() { sugg.className = 'sugg'; sel = -1; }

  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    if (q.length < 2) { closeSugg(); return; }
    filtered = DATA.items.filter(function (i) {
      return i.t.toLowerCase().indexOf(q) !== -1;
    }).slice(0, 20);
    if (!filtered.length) { closeSugg(); return; }
    sugg.innerHTML = filtered.map(function (i, k) {
      return '<button data-id="' + esc(i.id) + '" data-k="' + k + '">' + esc(i.t) +
        '<span class="sub">' + CFG.suggMeta(i) + '</span></button>';
    }).join('');
    sugg.className = 'sugg on';
    Array.prototype.forEach.call(sugg.querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () {
        show(b.dataset.id); input.value = ''; closeSugg();
      });
    });
  });

  input.addEventListener('keydown', function (e) {
    var btns = sugg.querySelectorAll('button');
    if (!btns.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      sel += (e.key === 'ArrowDown' ? 1 : -1);
      if (sel < 0) sel = btns.length - 1;
      if (sel >= btns.length) sel = 0;
      Array.prototype.forEach.call(btns, function (b, i) { b.className = i === sel ? 'active' : ''; });
      btns[sel].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && sel >= 0) {
      e.preventDefault(); btns[sel].click();
    } else if (e.key === 'Escape') { closeSugg(); }
  });

  document.addEventListener('click', function (e) {
    if (!sugg.contains(e.target) && e.target !== input) closeSugg();
  });
})();
