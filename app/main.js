const pages = [...document.querySelectorAll('.page')];
const pageIds = pages.map((p) => p.id);
const navPrev = document.getElementById('nav-prev');
const navNext = document.getElementById('nav-next');
const navPos = document.getElementById('nav-pos');

// Flatiron logo on every slide.
pages.forEach((p) => {
  const img = document.createElement('img');
  img.className = 'logo';
  img.alt = 'Flatiron Institute';
  img.src = 'assets/flatiron-logo.png';
  p.appendChild(img);
});

// Table of contents: the slide counter in the nav bar opens a list of slide
// titles; T toggles it and Escape closes it.
const toc = document.getElementById('toc');
const tocList = document.getElementById('toc-list');

pages.forEach((p) => {
  const li = document.createElement('li');
  const b = document.createElement('button');
  b.textContent = p.querySelector('h1, h2').textContent;
  b.addEventListener('click', () => {
    location.hash = p.id;
    setTocOpen(false);
  });
  li.appendChild(b);
  tocList.appendChild(li);
});

function setTocOpen(open) {
  toc.hidden = !open;
  navPos.setAttribute('aria-expanded', String(open));
  if (open) tocList.querySelector('.current')?.scrollIntoView({ block: 'nearest' });
}

navPos.addEventListener('click', (e) => {
  e.stopPropagation();
  setTocOpen(toc.hidden);
});
document.addEventListener('click', (e) => {
  if (!toc.hidden && !toc.contains(e.target)) setTocOpen(false);
});

// Called when a slide becomes visible or the deck is rescaled, keyed by slide id.
const onShow = {};

function show(id) {
  pages.forEach((p) => p.classList.toggle('active', p.id === id));
  const i = pageIds.indexOf(id);
  const prev = pages[i - 1];
  const next = pages[i + 1];
  navPrev.textContent = prev ? `← ${prev.dataset.hint}` : '←';
  navNext.textContent = next ? `${next.dataset.hint} →` : '→';
  navPrev.disabled = !prev;
  navNext.disabled = !next;
  navPos.textContent = `${i + 1} / ${pages.length} \u2630`;
  tocList.querySelectorAll('button').forEach((b, j) => b.classList.toggle('current', j === i));
  onShow[id]?.();
}

// Scale the 1920 x 1080 deck to the largest size that fits, centered.
const SLIDE_W = 1920;
const SLIDE_H = 1080;
const MARGIN = 16;
const stage = document.querySelector('main');
const deck = document.getElementById('deck');
let deckScale = 1;

function fitDeck() {
  stage.style.bottom = `${document.querySelector('.pager').offsetHeight}px`;
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  deckScale = Math.min((w - 2 * MARGIN) / SLIDE_W, (h - 2 * MARGIN) / SLIDE_H);
  deck.style.transform = `scale(${deckScale})`;
  deck.style.left = `${(w - SLIDE_W * deckScale) / 2}px`;
  deck.style.top = `${(h - SLIDE_H * deckScale) / 2}px`;
  const active = document.querySelector('.page.active');
  if (active) onShow[active.id]?.();
}

window.addEventListener('resize', fitDeck);

// Size a canvas's backing store to its on-screen pixels (slide size x deck
// scale x device pixel ratio) and return a context that draws in slide pixels.
function fitCanvas(canvas) {
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  const k = deckScale * (window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(w * k));
  canvas.height = Math.max(1, Math.round(h * k));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(k, 0, 0, k, 0, 0);
  return { ctx, w, h };
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Why Figpack: a simulated multichannel recording, shown as a static figure
// and as an interactive viewer that loads one-second chunks on demand.

const rec = {
  fs: 10000,          // samples per second
  nCh: 4,
  chunkSec: 1,
  durationSec: 600,   // 10 minutes
  binSize: 10,        // samples per min/max bin, for zoomed-out drawing
};
rec.chunkLen = rec.fs * rec.chunkSec;
rec.nChunks = rec.durationSec / rec.chunkSec;

// Spiking units: peak channel, trough amplitude (uV), mean rate (Hz), and a
// slow rate modulation so zoomed-out views show structure.
const units = [
  { ch: 0.2, amp: 140, rate: 9, period: 47, phase: 0.0 },
  { ch: 1.3, amp: 95, rate: 15, period: 31, phase: 1.3 },
  { ch: 1.9, amp: 180, rate: 4, period: 73, phase: 2.1 },
  { ch: 2.9, amp: 110, rate: 12, period: 59, phase: 4.0 },
];

// One color per channel. The static figure always uses the light-theme ones.
const staticChannelColors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd'];
const channelColors = () => [1, 2, 3, 4].map((i) => cssVar(`--ch${i}`));

const templateLen = Math.round(0.002 * rec.fs);
const template = new Float32Array(templateLen).map((_, i) => {
  const tau = i / rec.fs;
  return -Math.exp(-(((tau - 0.0005) / 0.00015) ** 2)) + 0.35 * Math.exp(-(((tau - 0.001) / 0.00035) ** 2));
});

function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

// Deterministic, so a chunk looks the same however often it is generated.
function generateChunk(k) {
  const rng = mulberry32(k * 7919 + 17);
  const { fs, nCh, chunkLen, binSize } = rec;
  const t0 = k * rec.chunkSec;
  const data = [];
  for (let c = 0; c < nCh; c++) {
    const x = new Float32Array(chunkLen);
    let ar = 0;
    for (let i = 0; i < chunkLen; i++) {
      const t = t0 + i / fs;
      ar = 0.6 * ar + 0.8 * gaussian(rng);
      const lfp = 25 * (1 - 0.12 * c) * Math.sin(2 * Math.PI * 7 * t) * (0.6 + 0.4 * Math.sin(2 * Math.PI * t / 40));
      x[i] = 7 * ar + lfp;
    }
    data.push(x);
  }
  for (const u of units) {
    const peak = 1.9 * u.rate;
    let t = t0;
    for (;;) {
      t += -Math.log(1 - rng()) / peak;
      if (t >= t0 + rec.chunkSec) break;
      const rate = u.rate * (1 + 0.9 * Math.sin(2 * Math.PI * t / u.period + u.phase));
      if (rng() * peak > rate) continue;
      const i0 = Math.floor((t - t0) * fs);
      for (let c = 0; c < nCh; c++) {
        const a = u.amp * Math.exp(-(((c - u.ch) / 1.1) ** 2));
        if (a < 2) continue;
        const x = data[c];
        for (let j = 0; j < templateLen && i0 + j < chunkLen; j++) x[i0 + j] += a * template[j];
      }
    }
  }
  const nBins = chunkLen / binSize;
  const mins = [];
  const maxs = [];
  for (const x of data) {
    const lo = new Float32Array(nBins);
    const hi = new Float32Array(nBins);
    for (let b = 0; b < nBins; b++) {
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = b * binSize; i < (b + 1) * binSize; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
      }
      lo[b] = mn;
      hi[b] = mx;
    }
    mins.push(lo);
    maxs.push(hi);
  }
  return { data, mins, maxs };
}

// Min and max of channel c over global samples [i0, i1), or null if a chunk
// in that range isn't loaded. Uses the bins when the range is wide.
function sampleRange(chunks, c, i0, i1) {
  const { chunkLen, binSize } = rec;
  let mn = Infinity;
  let mx = -Infinity;
  let i = i0;
  while (i < i1) {
    const k = Math.floor(i / chunkLen);
    const chunk = chunks.get(k);
    if (!chunk) return null;
    const end = Math.min(i1, (k + 1) * chunkLen);
    let j = i - k * chunkLen;
    const jEnd = end - k * chunkLen;
    const x = chunk.data[c];
    while (j < jEnd) {
      if (j % binSize === 0 && j + binSize <= jEnd) {
        const b = j / binSize;
        if (chunk.mins[c][b] < mn) mn = chunk.mins[c][b];
        if (chunk.maxs[c][b] > mx) mx = chunk.maxs[c][b];
        j += binSize;
      } else {
        if (x[j] < mn) mn = x[j];
        if (x[j] > mx) mx = x[j];
        j++;
      }
    }
    i = end;
  }
  return [mn, mx];
}

function sampleAt(chunks, c, i) {
  const chunk = chunks.get(Math.floor(i / rec.chunkLen));
  return chunk ? chunk.data[c][i % rec.chunkLen] : null;
}

// Draw channels as stacked traces over [t0, t1] into a w x h box.
// Returns false if some needed chunk was missing.
function drawTraces(ctx, chunks, channels, t0, t1, w, h, uvPerLane, colors) {
  const laneH = h / channels.length;
  const { fs } = rec;
  const samplesPerPx = ((t1 - t0) * fs) / w;
  const yScale = laneH / uvPerLane;
  let complete = true;
  channels.forEach((c, lane) => {
    const yMid = laneH * (lane + 0.5);
    ctx.strokeStyle = colors[c];
    ctx.beginPath();
    if (samplesPerPx > 1) {
      for (let px = 0; px < w; px++) {
        const i0 = Math.max(0, Math.floor((t0 * fs) + px * samplesPerPx));
        const i1 = Math.min(rec.durationSec * fs, Math.floor((t0 * fs) + (px + 1) * samplesPerPx));
        if (i1 <= i0) continue;
        const r = sampleRange(chunks, c, i0, i1);
        if (!r) { complete = false; continue; }
        ctx.moveTo(px + 0.5, yMid - r[1] * yScale);
        ctx.lineTo(px + 0.5, yMid - r[0] * yScale + 0.5);
      }
    } else {
      const i0 = Math.max(0, Math.floor(t0 * fs));
      const i1 = Math.min(rec.durationSec * fs - 1, Math.ceil(t1 * fs));
      let pen = false;
      for (let i = i0; i <= i1; i++) {
        const v = sampleAt(chunks, c, i);
        if (v === null) { complete = false; pen = false; continue; }
        const x = ((i / fs - t0) / (t1 - t0)) * w;
        const y = yMid - v * yScale;
        if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        pen = true;
      }
    }
    ctx.stroke();
  });
  return complete;
}

// The static figure: half a second of four channels, rendered once to a PNG.
(function makeStaticFigure() {
  const t0 = 12.2;
  const t1 = 12.7;
  const channels = [0, 1, 2, 3];
  const W = 1920;
  const H = 540;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);
  const chunks = new Map([[12, generateChunk(12)]]);
  const left = 150;
  const right = 50;
  const top = 20;
  const bottom = 110;
  const pw = W - left - right;
  const ph = H - top - bottom;
  ctx.save();
  ctx.translate(left, top);
  ctx.lineWidth = 1.5;
  drawTraces(ctx, chunks, channels, t0, t1, pw, ph, 260, staticChannelColors);
  ctx.restore();
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, pw, ph);
  ctx.fillStyle = '#000';
  ctx.font = '40px sans-serif';
  ctx.textAlign = 'center';
  for (let k = 0; k <= 5; k++) {
    const t = t0 + (k * (t1 - t0)) / 5;
    const x = left + (k / 5) * pw;
    ctx.beginPath();
    ctx.moveTo(x, top + ph);
    ctx.lineTo(x, top + ph + 10);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), x, top + ph + 50);
  }
  ctx.fillText('Time (s)', left + pw / 2, H - 6);
  ctx.textAlign = 'right';
  channels.forEach((c, lane) => {
    ctx.fillText(`Ch ${c + 1}`, left - 16, top + (ph / channels.length) * (lane + 0.5) + 14);
  });
  document.getElementById('why-static').src = canvas.toDataURL('image/png');
})();

// The interactive viewer.
const viewer = (() => {
  const traces = document.getElementById('why-traces');
  const timeline = document.getElementById('why-timeline');
  const readout = document.getElementById('why-readout');
  const chunks = new Map();
  const pending = new Set();
  const channels = [...Array(rec.nCh).keys()];
  const minSpan = 0.02;
  const maxSpan = 20;
  const labelW = 64;  // left margin of the trace canvas, for channel labels
  // Starts on the static figure's window.
  let t0 = 12.2;
  let t1 = 12.7;
  let current = null;  // the current time point, set by clicking

  // Stand-in for fetching a chunk over the network: generated on demand, with
  // a short delay so the loading is visible.
  function request(k) {
    if (k < 0 || k >= rec.nChunks || chunks.has(k) || pending.has(k)) return;
    pending.add(k);
    setTimeout(() => {
      chunks.set(k, generateChunk(k));
      pending.delete(k);
      draw();
    }, 80);
  }

  function setView(a, b) {
    let span = Math.min(maxSpan, Math.max(minSpan, b - a));
    const mid = (a + b) / 2;
    a = Math.max(0, Math.min(rec.durationSec - span, mid - span / 2));
    t0 = a;
    t1 = a + span;
    draw();
  }

  function draw() {
    if (!traces.offsetWidth) return;
    for (let k = Math.floor(t0 / rec.chunkSec); k < Math.ceil(t1 / rec.chunkSec); k++) request(k);

    const { ctx, w, h } = fitCanvas(traces);
    ctx.clearRect(0, 0, w, h);
    const fg = cssVar('--fg');
    const colors = channelColors();
    const pw = w - labelW;
    ctx.save();
    ctx.translate(labelW, 0);
    const complete = drawTraces(ctx, chunks, channels, t0, t1, pw, h, 260, colors);
    if (current !== null && current >= t0 && current <= t1) {
      const x = ((current - t0) / (t1 - t0)) * pw;
      ctx.strokeStyle = cssVar('--cursor');
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillStyle = cssVar('--cursor');
      ctx.font = `20px ${cssVar('--font-body')}`;
      ctx.textAlign = x > pw - 140 ? 'right' : 'left';
      ctx.fillText(`${current.toFixed(4)} s`, x + (x > pw - 140 ? -8 : 8), 20);
    }
    ctx.restore();
    ctx.font = `20px ${cssVar('--font-body')}`;
    ctx.textAlign = 'left';
    channels.forEach((c, lane) => {
      ctx.fillStyle = colors[c];
      ctx.fillText(`Ch ${c + 1}`, 0, (h / channels.length) * (lane + 0.5) + 7);
    });
    if (!complete) {
      ctx.fillStyle = fg;
      ctx.globalAlpha = 0.6;
      ctx.font = `22px ${cssVar('--font-body')}`;
      ctx.textAlign = 'right';
      ctx.fillText('loading…', w - 8, 24);
      ctx.globalAlpha = 1;
    }

    const tl = fitCanvas(timeline);
    const accent = cssVar('--accent');
    tl.ctx.clearRect(0, 0, tl.w, tl.h);
    tl.ctx.fillStyle = fg;
    tl.ctx.globalAlpha = 0.1;
    tl.ctx.fillRect(0, 8, tl.w, tl.h - 16);
    tl.ctx.globalAlpha = 0.5;
    tl.ctx.fillStyle = accent;
    const kw = tl.w / rec.nChunks;
    for (const k of chunks.keys()) tl.ctx.fillRect(k * kw, 8, Math.max(kw, 1), tl.h - 16);
    tl.ctx.globalAlpha = 1;
    tl.ctx.strokeStyle = accent;
    tl.ctx.lineWidth = 2;
    const x0 = (t0 / rec.durationSec) * tl.w;
    const x1 = (t1 / rec.durationSec) * tl.w;
    tl.ctx.strokeRect(x0 - 1, 2, Math.max(x1 - x0, 2) + 2, tl.h - 4);

    const mb = (n) => (n * rec.nCh * rec.chunkLen * 4) / 1e6;
    readout.textContent = `${t0.toFixed(2)}–${t1.toFixed(2)} s · ` +
      `loaded ${chunks.size}/${rec.nChunks} chunks, ${mb(chunks.size).toFixed(1)}/${mb(rec.nChunks).toFixed(0)} MB`;
  }

  function timeAt(canvas, clientX) {
    const r = canvas.getBoundingClientRect();
    return (clientX - r.left) / r.width;
  }

  // Time at a pointer position over the traces (right of the label margin).
  function traceTimeAt(clientX) {
    const r = traces.getBoundingClientRect();
    const k = r.width / traces.offsetWidth;  // screen pixels per slide pixel
    const f = (clientX - r.left - labelW * k) / (r.width - labelW * k);
    return t0 + f * (t1 - t0);
  }

  traces.addEventListener('wheel', (e) => {
    e.preventDefault();
    const t = Math.min(t1, Math.max(t0, traceTimeAt(e.clientX)));
    const factor = Math.exp(e.deltaY * 0.0015);
    setView(t - (t - t0) * factor, t + (t1 - t) * factor);
  }, { passive: false });

  // Drag to pan; a click without dragging sets the current time.
  let drag = null;
  traces.addEventListener('pointerdown', (e) => {
    const r = traces.getBoundingClientRect();
    drag = { x: e.clientX, t0, t1, width: r.width * (1 - labelW / traces.offsetWidth), moved: false };
    traces.setPointerCapture(e.pointerId);
  });
  traces.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag.x) > 4) {
      drag.moved = true;
      traces.classList.add('dragging');
    }
    if (!drag.moved) return;
    const dt = ((e.clientX - drag.x) / drag.width) * (drag.t1 - drag.t0);
    setView(drag.t0 - dt, drag.t1 - dt);
  });
  traces.addEventListener('pointerup', (e) => {
    if (drag && !drag.moved) {
      const t = traceTimeAt(e.clientX);
      if (t >= t0 && t <= t1) {
        current = t;
        draw();
      }
    }
    drag = null;
    traces.classList.remove('dragging');
  });
  traces.addEventListener('pointercancel', () => {
    drag = null;
    traces.classList.remove('dragging');
  });

  timeline.addEventListener('click', (e) => {
    const t = timeAt(timeline, e.clientX) * rec.durationSec;
    const span = t1 - t0;
    setView(t - span / 2, t + span / 2);
  });

  const zoom = (factor) => {
    const mid = (t0 + t1) / 2;
    const span = (t1 - t0) * factor;
    setView(mid - span / 2, mid + span / 2);
  };
  const pan = (fraction) => {
    const dt = (t1 - t0) * fraction;
    setView(t0 + dt, t1 + dt);
  };
  document.getElementById('why-zoom-in').addEventListener('click', () => zoom(0.5));
  document.getElementById('why-zoom-out').addEventListener('click', () => zoom(2));
  document.getElementById('why-pan-left').addEventListener('click', () => pan(-0.5));
  document.getElementById('why-pan-right').addEventListener('click', () => pan(0.5));
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

  return { draw };
})();

onShow.why = viewer.draw;

// ---------------------------------------------------------------------------
// Figpack figure example: show the script that made the embedded figure.

fetch('example_figure.py')
  .then((r) => r.text())
  .then((src) => {
    const code = document.getElementById('example-code');
    code.textContent = src.trimEnd();
    hljs.highlightElement(code);
    // Circle the parameter that makes the figure shareable.
    code.innerHTML = code.innerHTML.replace(
      /upload=<span class="hljs-literal">True<\/span>/,
      '<span class="circled">$&</span>',
    );
  });

// ---------------------------------------------------------------------------
// Zarr layout: the arrays, chunks and consolidated file of the example figure,
// from its data.zarr/.zmetadata.

(function makeZarrDiagram() {
  const arrays = [
    { name: 'data', length: 600000, chunk: 262144, color: '--ch1' },
    { name: 'data_ds_4', length: 150000, chunk: 131072, color: '--ch2' },
    { name: 'data_ds_16', length: 37500, chunk: 37500, color: '--ch3' },
    { name: 'data_ds_64', length: 9375, chunk: 9375, color: '--ch4' },
    { name: 'data_ds_256', length: 2344, chunk: 2344, color: '--ch5' },
  ];
  // refs: chunk key -> [file, byte offset, byte length], in file order.
  const refs = [
    ['ephys/data_ds_4/0.0.0', 0, 3702964],
    ['ephys/data/1.0', 3702964, 3685148],
    ['ephys/data/0.0', 7388112, 3683890],
    ['ephys/data/2.0', 11072002, 1078070],
    ['ephys/data_ds_16/0.0.0', 12150072, 1060209],
    ['ephys/data_ds_4/1.0.0', 13210281, 550493],
    ['ephys/data_ds_64/0.0.0', 13760774, 261514],
    ['ephys/data_ds_256/0.0.0', 14022288, 66589],
  ];
  const fileSize = 14088877;
  const chunkKey = (a, i) => `ephys/${a.name}/${i}.0${a.name === 'data' ? '' : '.0'}`;

  const W = 960;
  const x0 = 200;            // left edge of bars
  const barW = W - x0 - 20;  // full-resolution array spans this width
  const rowY = (i) => 80 + i * 44;
  const rowH = 30;
  const fileY = 450;
  const fileH = 50;
  const minW = 14;           // so the smallest arrays stay visible

  const parts = [];
  const chunkPos = {};  // key -> bottom center of the chunk in its array row
  parts.push(`<text x="0" y="24" font-size="24" font-weight="600">Arrays: full resolution and downsampled, split into chunks</text>`);
  parts.push(`<text class="muted" x="0" y="54" font-size="19">data_ds_N holds the min and max of every N samples</text>`);
  arrays.forEach((a, r) => {
    const y = rowY(r);
    const scale = barW / arrays[0].length;
    const total = Math.max(minW, a.length * scale);
    parts.push(`<text class="mono" x="${x0 - 14}" y="${y + 21}" font-size="19" text-anchor="end">${a.name}</text>`);
    const n = Math.ceil(a.length / a.chunk);
    for (let i = 0; i < n; i++) {
      const s = (i * a.chunk) / a.length;
      const e = Math.min(1, ((i + 1) * a.chunk) / a.length);
      const key = chunkKey(a, i);
      const x = x0 + s * total;
      const w = (e - s) * total;
      parts.push(`<rect class="chunk" data-key="${key}" x="${x}" y="${y}" width="${w}" height="${rowH}" fill="var(${a.color})"/>`);
      chunkPos[key] = { x: x + w / 2, y: y + rowH, color: a.color };
    }
  });

  parts.push(`<text class="mono" x="${x0 - 14}" y="${fileY + 31}" font-size="16" text-anchor="end">_consolidated_0.dat</text>`);
  for (const [key, offset, length] of refs) {
    const x = x0 + (offset / fileSize) * barW;
    const w = Math.max(3, (length / fileSize) * barW);
    const c = chunkPos[key];
    parts.push(`<path class="link" data-key="${key}" stroke="var(${c.color})" d="M ${c.x} ${c.y} C ${c.x} ${c.y + 90}, ${x + w / 2} ${fileY - 90}, ${x + w / 2} ${fileY}"/>`);
    parts.push(`<rect class="chunk" data-key="${key}" x="${x}" y="${fileY}" width="${w}" height="${fileH}" fill="var(${c.color})"/>`);
  }
  parts.push(`<text class="muted mono" x="${x0}" y="${fileY + fileH + 26}" font-size="17">0</text>`);
  parts.push(`<text class="muted mono" x="${x0 + barW}" y="${fileY + fileH + 26}" font-size="17" text-anchor="end">${(fileSize / 1e6).toFixed(1)} MB</text>`);
  parts.push(`<text class="muted" x="${x0 + barW / 2}" y="${fileY + fileH + 26}" font-size="19" text-anchor="middle">each chunk compressed (blosc), then packed end to end</text>`);

  const svgH = fileY + fileH + 36;
  const svg = `<svg viewBox="0 0 ${W} ${svgH}" xmlns="http://www.w3.org/2000/svg">${parts.join('')}</svg>`;

  const lines = refs.map(([key, offset, length]) => {
    const color = chunkPos[key].color;
    return `<div data-key="${key}"><span class="swatch" style="background: var(${color})"></span>"${key}": ["_consolidated_0.dat", ${offset}, ${length}]</div>`;
  });
  const refsHtml = `<div class="zrefs"><div class="hdr">.zmetadata refs: chunk → [file, offset, length]</div>${lines.join('')}</div>`;

  const root = document.getElementById('zarr-diagram');
  root.innerHTML = svg + refsHtml;

  root.addEventListener('mouseover', (e) => {
    const el = e.target.closest('[data-key]');
    root.querySelectorAll('.hl').forEach((n) => n.classList.remove('hl'));
    if (!el) {
      root.classList.remove('hovering');
      return;
    }
    root.classList.add('hovering');
    root.querySelectorAll(`[data-key="${el.dataset.key}"]`).forEach((n) => n.classList.add('hl'));
  });
  root.addEventListener('mouseleave', () => {
    root.classList.remove('hovering');
    root.querySelectorAll('.hl').forEach((n) => n.classList.remove('hl'));
  });
})();

// ---------------------------------------------------------------------------
// Lifetime tiers: how long a figure stays available, on a log time axis.
// Durations from figpack-api-cf (ephemeral 6 h, default 24 h, renew to 7 days).

(function makeLifetimeDiagram() {
  const HOUR = 1;
  const DAY = 24 * HOUR;
  const YEAR = 365 * DAY;
  const groups = [
    {
      label: 'on figpack.org',
      color: '--accent',
      tiers: [
        { name: 'Ephemeral', until: 6 * HOUR, word: '6 hours' },
        { name: 'Default', until: DAY, word: '24 hours' },
        { name: 'Renewed', until: 7 * DAY, word: '7 days' },
        { name: 'Pinned', until: 3 * YEAR, open: true, word: 'indefinite' },
      ],
    },
    {
      label: 'in your own bucket',
      color: '--ch3',
      tiers: [{ name: 'Own bucket', until: 10 * YEAR, open: true, word: 'yours' }],
    },
    {
      label: 'a copy hosted elsewhere',
      color: '--ch4',
      tiers: [
        { name: 'GitHub Pages', until: 10 * YEAR, open: true, word: 'self-hosted' },
        { name: 'Zenodo', until: 30 * YEAR, arrow: true, word: 'citable, long term' },
      ],
    },
  ];

  const W = 844;
  const xa = 190;        // left edge of the bars: the moment of upload
  const xb = W - 24;     // right edge: 30 years
  const tMin = HOUR;
  const tMax = 30 * YEAR;
  const xAt = (t) => xa + (Math.log(t / tMin) / Math.log(tMax / tMin)) * (xb - xa);
  const barH = 44;
  const rowGap = 16;
  const groupGap = 26;
  const labelH = 30;

  const parts = [];
  const defs = [];
  let y = 0;
  const ticks = [
    [HOUR, '1 hour'], [DAY, '1 day'], [7 * DAY, '1 week'],
    [YEAR, '1 year'], [10 * YEAR, '10 years'],
  ];

  groups.forEach((g, gi) => {
    y += gi === 0 ? 0 : groupGap;
    parts.push(`<text class="muted" x="${xa}" y="${y + 22}" font-size="21" font-style="italic">${g.label}</text>`);
    y += labelH;
    for (const t of g.tiers) {
      const x1 = xAt(t.until);
      let fill = `var(${g.color})`;
      if (t.open) {
        // Open-ended: fade out, since there's no fixed end date.
        const id = `fade-${t.name.replace(/\W/g, '')}`;
        defs.push(`<linearGradient id="${id}" x1="0" x2="1" y1="0" y2="0">` +
          `<stop offset="0.55" style="stop-color: var(${g.color}); stop-opacity: 1"/>` +
          `<stop offset="1" style="stop-color: var(${g.color}); stop-opacity: 0"/></linearGradient>`);
        fill = `url(#${id})`;
      }
      parts.push(`<text x="${xa - 16}" y="${y + barH / 2 + 8}" font-size="23" text-anchor="end">${t.name}</text>`);
      parts.push(`<rect x="${xa}" y="${y}" width="${x1 - xa}" height="${barH}" rx="6" fill="${fill}"/>`);
      if (t.arrow) {
        parts.push(`<path d="M ${x1} ${y - 6} L ${x1 + 18} ${y + barH / 2} L ${x1} ${y + barH + 6} Z" fill="var(${g.color})"/>`);
      }
      // The word goes inside long bars and just past the end of short ones.
      const inside = x1 - xa > 300;
      const tx = inside ? xa + 14 : x1 + 12;
      const style = inside ? 'fill: var(--bg); font-weight: 600' : 'fill: var(--fg)';
      parts.push(`<text x="${tx}" y="${y + barH / 2 + 8}" font-size="22" style="${style}">${t.word}</text>`);
      y += barH + rowGap;
    }
  });

  const axisY = y + 10;
  for (const [t, label] of ticks) {
    const x = xAt(t);
    parts.unshift(`<line class="grid" x1="${x}" x2="${x}" y1="0" y2="${axisY}"/>`);
    parts.push(`<text class="muted" x="${x}" y="${axisY + 28}" font-size="19" text-anchor="middle">${label}</text>`);
  }
  parts.push(`<text class="muted" x="${(xa + xb) / 2}" y="${axisY + 62}" font-size="19" text-anchor="middle">time after upload (log scale)</text>`);

  const H = axisY + 72;
  document.getElementById('lifetime-diagram').innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"><defs>${defs.join('')}</defs>${parts.join('')}</svg>`;
})();

// ---------------------------------------------------------------------------
// Gallery: example figures built and hosted with the figpack documentation.
// Names are the headings they appear under in the docs.

(function makeGallery() {
  const docs = 'https://flatironinstitute.github.io/figpack';
  const figures = [
    ['spike_sorting_tutorial_dashboard', 'Spike sorting: combined layout', 'Spike sorting'],
    ['spike_sorting_tutorial_sorting_curation', 'Sorting curation', 'Spike sorting'],
    ['spike_sorting_tutorial_raster_plot', 'Raster plot', 'Spike sorting'],
    ['spike_sorting_tutorial_average_waveforms', 'Average waveforms', 'Spike sorting'],
    ['spike_sorting_tutorial_unit_locations', 'Unit locations', 'Spike sorting'],
    ['nwb_tutorial_plane_segmentation', 'Plane segmentation', 'NWB'],
    ['nwb_tutorial_pose_estimation', 'Pose estimation', 'NWB'],
    ['tutorial_multichannel_timeseries', 'Multichannel timeseries', 'Basic views'],
    ['tutorial_spectrogram_example', 'Spectrogram', 'Basic views'],
    ['tutorial_complex_layout', 'Complex nested layout', 'Basic views'],
    ['tutorial_plotly_example', 'Plotly integration', 'Basic views'],
    ['tutorial_dataframe_example', 'DataFrame view', 'Basic views'],
    ['tutorial_3d_scene_example', '3D scene', 'Other views'],
    ['tutorial_force_graph_example', 'Force graph', 'Other views'],
    ['tutorial_sphere_embedding_example', 'Sphere embedding', 'Other views'],
    ['tutorial_fmri_bold_example', 'fMRI BOLD', 'Other views'],
    ['tutorial_track_animation_example', 'Track animation', 'Other views'],
    ['tutorial_linear_decode_example', 'Linear decode', 'Other views'],
  ];
  const list = document.getElementById('gallery-list');
  const frame = document.getElementById('gallery-frame');
  const link = document.getElementById('gallery-url');

  function select(i) {
    const url = `${docs}/${figures[i][0]}/index.html`;
    frame.src = `${url}?embedded=1`;
    link.href = url;
    link.textContent = url;
    list.querySelectorAll('button').forEach((b, j) => b.classList.toggle('selected', j === i));
  }

  figures.forEach(([, name, category], i) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.innerHTML = `${name}<span class="cat">${category}</span>`;
    button.addEventListener('click', () => select(i));
    li.appendChild(button);
    list.appendChild(li);
  });
  select(0);
})();

// ---------------------------------------------------------------------------

// Replay of how recipe.md was written, from app/crafted-link.js (npm run crafted-link).
if (window.CRAFTED_LINK) {
  const link = document.getElementById('crafted-link');
  link.href = window.CRAFTED_LINK;
  link.hidden = false;
}

// The current slide lives in the URL hash, so reloading stays put.
function showFromHash() {
  const id = location.hash.slice(1);
  show(pageIds.includes(id) ? id : pageIds[0]);
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', showFromHash);
fitDeck();
showFromHash();

function goToIndex(i) {
  location.hash = pageIds[Math.max(0, Math.min(pageIds.length - 1, i))];
}

function currentIndex() {
  return pageIds.indexOf(document.querySelector('.page.active').id);
}

navPrev.addEventListener('click', () => goToIndex(currentIndex() - 1));
navNext.addEventListener('click', () => goToIndex(currentIndex() + 1));

document.addEventListener('keydown', (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.target.closest?.('input, select, textarea')) return;
  if (e.key === 't' || e.key === 'T') {
    setTocOpen(toc.hidden);
    return;
  }
  if (e.key === 'Escape' && !toc.hidden) {
    setTocOpen(false);
    return;
  }
  const target = {
    ArrowLeft: currentIndex() - 1,
    ArrowRight: currentIndex() + 1,
    Home: 0,
    End: pageIds.length - 1,
  }[e.key];
  if (target === undefined) return;
  e.preventDefault();
  goToIndex(target);
});
