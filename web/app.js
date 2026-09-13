/* Palette completion in the browser: Oklab color math, the trained set-transformer denoiser,
   flow-matching Euler sampler, gamut mapping, scorer rerank and diverse selection.  */
"use strict";

// ---------- color math (mirrors palette/color.py) ----------
const M1 = [[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]];
const M2 = [[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]];
const M2I = [[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728], [1.0, -0.0894841775, -1.2914855480]];
const M1I = [[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965], [-0.0041960863, -0.7034186147, 1.7076147010]];
const mv = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const s2l = c => c <= 0.04045 ? c / 12.92 : Math.sign(c) * Math.pow((Math.abs(c) + 0.055) / 1.055, 2.4);
const l2s = c => c <= 0.0031308 ? 12.92 * c : Math.sign(c) * (1.055 * Math.pow(Math.abs(c), 1 / 2.4) - 0.055);
const cbrt = Math.cbrt;
function srgbToOklab(rgb) { const lms = mv(M1, rgb.map(s2l)).map(cbrt); return mv(M2, lms); }
function oklabToLinear(lab) { const l = mv(M2I, lab).map(v => v * v * v); return mv(M1I, l); }
function oklabToSrgb(lab) { return oklabToLinear(lab).map(l2s); }
function inGamut(lab, tol = 1e-4) { return oklabToLinear(lab).every(v => v >= -tol && v <= 1 + tol); }
function gamutMap(lab) {
  const out = [Math.min(1, Math.max(0, lab[0])), lab[1], lab[2]];
  if (inGamut(out)) return out;
  let lo = 0, hi = 1;
  for (let i = 0; i < 32; i++) { const mid = (lo + hi) / 2; if (inGamut([out[0], lab[1] * mid, lab[2] * mid])) lo = mid; else hi = mid; }
  return [out[0], lab[1] * lo, lab[2] * lo];
}
function hexToSrgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255); }
function srgbToHex(rgb) { return "#" + rgb.map(v => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0")).join(""); }
const hexToOklab = h => srgbToOklab(hexToSrgb(h));
const oklabToHex = lab => srgbToHex(oklabToSrgb(lab));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// ---------- weights ----------
let W = null, MAN = null;
function f16(u) { const s = (u >> 15) & 1, e = (u >> 10) & 31, f = u & 1023; let v;
  if (e === 0) v = f * Math.pow(2, -24); else if (e === 31) v = f ? NaN : Infinity; else v = (1 + f / 1024) * Math.pow(2, e - 15);
  return s ? -v : v; }
async function loadModel() {
  MAN = await (await fetch("model.json")).json();
  const bin = atob(MAN.weights_b64), bytes = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const dv = new DataView(bytes.buffer);
  W = {};
  for (const [k, t] of Object.entries(MAN.tensors)) { const a = new Float32Array(t.n); for (let i = 0; i < t.n; i++) a[i] = f16(dv.getUint16(t.offset + 2 * i, true)); W[k] = { a, shape: t.shape }; }
  return MAN;
}
// y = x W^T + b, x: Float32Array(n_in), W: [n_out, n_in]
function linear(x, w, b) { const [no, ni] = w.shape, y = new Float32Array(no); const a = w.a;
  for (let o = 0; o < no; o++) { let s = b ? b.a[o] : 0; const r = o * ni; for (let i = 0; i < ni; i++) s += a[r + i] * x[i]; y[o] = s; } return y; }
function layerNorm(x, g, b) { const n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n; let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = (x[i] - m) * inv * g.a[i] + b.a[i]; return y; }
const silu = x => x.map(v => v / (1 + Math.exp(-v)));
function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
const gelu = x => x.map(v => 0.5 * v * (1 + erf(v / Math.SQRT2)));
function timeEmb(t, dim) { const half = dim / 2, e = new Float32Array(dim); for (let i = 0; i < half; i++) { const f = Math.exp(-Math.log(10000) * i / half); e[i] = Math.cos(t * f); e[half + i] = Math.sin(t * f); } return e; }

// ---------- denoiser forward: nodes = [{x:[3], fixed:bool}], t index → velocities [N][3] ----------
function denoise(nodes, t) {
  const cfg = MAN.model, D = cfg.dim || 128, H = cfg.heads || 4, dh = D / H, N = nodes.length;
  const nCtx = nodes.filter(n => n.fixed).length / 5, mTgt = (N - nodes.filter(n => n.fixed).length) / 5, te = timeEmb(t, 64);
  let h = nodes.map(n => { const f = new Float32Array(3 + 2 + 64 + 2); f.set(n.x, 0); f[3] = f[4] = n.fixed ? 1 : 0; f.set(te, 5); f[69] = nCtx; f[70] = mTgt;
    return linear(silu(linear(f, W["model.inp.0.weight"], W["model.inp.0.bias"])), W["model.inp.2.weight"], W["model.inp.2.bias"]); });
  for (let b = 0; b < (cfg.depth || 4); b++) {
    const p = `model.blocks.${b}.`;
    const xn = h.map(v => layerNorm(v, W[p + "n1.weight"], W[p + "n1.bias"]));
    const qkv = xn.map(v => linear(v, W[p + "attn.in_proj_weight"], W[p + "attn.in_proj_bias"]));
    const att = h.map(() => new Float32Array(D));
    for (let hd = 0; hd < H; hd++) { const o = hd * dh;
      for (let i = 0; i < N; i++) { const sc = new Float64Array(N); let mx = -1e9;
        for (let j = 0; j < N; j++) { let s = 0; for (let d = 0; d < dh; d++) s += qkv[i][o + d] * qkv[j][D + o + d]; sc[j] = s / Math.sqrt(dh); mx = Math.max(mx, sc[j]); }
        let z = 0; for (let j = 0; j < N; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
        for (let j = 0; j < N; j++) { const w = sc[j] / z; for (let d = 0; d < dh; d++) att[i][o + d] += w * qkv[j][2 * D + o + d]; } } }
    h = h.map((v, i) => { const a = linear(att[i], W[p + "attn.out_proj.weight"], W[p + "attn.out_proj.bias"]); const r = new Float32Array(D); for (let k = 0; k < D; k++) r[k] = v[k] + a[k]; return r; });
    h = h.map(v => { const m = linear(gelu(linear(layerNorm(v, W[p + "n2.weight"], W[p + "n2.bias"]), W[p + "mlp.0.weight"], W[p + "mlp.0.bias"])), W[p + "mlp.2.weight"], W[p + "mlp.2.bias"]);
      const r = new Float32Array(D); for (let k = 0; k < D; k++) r[k] = v[k] + m[k]; return r; });
  }
  return h.map(v => Array.from(linear(layerNorm(v, W["model.out_norm.weight"], W["model.out_norm.bias"]), W["model.out.weight"], W["model.out.bias"])));
}
function score(palNorm) { // DeepSets scorer on normalized Oklab rows
  const acc = new Float32Array(128);
  for (const x of palNorm) { const e = silu(linear(silu(linear(Float32Array.from(x), W["scorer.phi.0.weight"], W["scorer.phi.0.bias"])), W["scorer.phi.2.weight"], W["scorer.phi.2.bias"])); for (let k = 0; k < 128; k++) acc[k] += e[k]; }
  return linear(silu(linear(acc, W["scorer.rho.0.weight"], W["scorer.rho.0.bias"])), W["scorer.rho.2.weight"], W["scorer.rho.2.bias"])[0];
}
const norm = lab => [(lab[0] - MAN.norm.L_mean) / MAN.norm.L_std, lab[1] / MAN.norm.ab_scale, lab[2] / MAN.norm.ab_scale];
const denorm = z => [z[0] * MAN.norm.L_std + MAN.norm.L_mean, z[1] * MAN.norm.ab_scale, z[2] * MAN.norm.ab_scale];

// ---------- deterministic RNG ----------
function rng(seed) { let s = seed >>> 0 || 1; const u = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  return { normal() { const a = Math.max(u(), 1e-12), b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); } }; }

// ---------- sampler: flow Euler (or VP DDIM) over the targets only ----------
function sampleOne(ctxLab, m, steps, r) {
  const T = MAN.diffusion.T || 1000, isFlow = MAN.diffusion.kind === "flow";
  const nodes = ctxLab.map(l => ({ x: norm(l), fixed: true }));
  for (let i = 0; i < m; i++) nodes.push({ x: [r.normal(), r.normal(), r.normal()], fixed: false });
  if (isFlow) {
    for (let s = 0; s < steps; s++) { const tau = 1 - s / steps, tauN = 1 - (s + 1) / steps, dt = tauN - tau;
      const t = Math.min(T - 1, Math.max(0, Math.round(tau * T - 0.5))); const v = denoise(nodes, t);
      nodes.forEach((n, i) => { if (!n.fixed) n.x = n.x.map((c, k) => c + dt * v[i][k]); }); }
  } else { // DDIM on the cosine VP schedule
    const ac = tt => { const f = x => Math.cos((x + 0.008) / 1.008 * Math.PI / 2) ** 2; return Math.max(1e-5, f((tt + 1) / T) / f(0)); };
    const ts = Array.from({ length: steps }, (_, i) => Math.round((T - 1) * (1 - i / (steps - 1))));
    for (let s = 0; s < steps; s++) { const a = ac(ts[s]), an = s + 1 < steps ? ac(ts[s + 1]) : 1; const e = denoise(nodes, ts[s]);
      nodes.forEach((n, i) => { if (!n.fixed) n.x = n.x.map((c, k) => { const x0 = (c - Math.sqrt(1 - a) * e[i][k]) / Math.sqrt(a); return Math.sqrt(an) * x0 + Math.sqrt(1 - an) * e[i][k]; }); }); }
  }
  return nodes.filter(n => !n.fixed).map(n => denorm(n.x));
}

// ---------- pipeline (mirrors palette/api.py): reject → map → rerank → diverse top-k ----------
function permutations(n) { if (n === 1) return [[0]]; const out = []; for (const p of permutations(n - 1)) for (let i = 0; i <= p.length; i++) out.push([...p.slice(0, i), n - 1, ...p.slice(i)]); return out; }
function matchDist(A, B) { let best = Infinity; for (const p of permutations(A.length)) { let s = 0; for (let i = 0; i < A.length; i++) s += dist(A[i], B[p[i]]); best = Math.min(best, s); } return best / A.length; }
function complete(ctxLab, m, k, { steps = 10, seed = 1, oversample = 32, dupThreshold = 0.03, diversityWeight = 1.0 } = {}) {
  const r = rng(seed * 7919 + 17), pool = [], rejected = []; const stats = { raw: 0, inGamut: 0, dup: 0, mapped: 0 };
  const minPair = comp => { let d = Infinity; for (let i = 0; i < comp.length; i++) { for (let j = i + 1; j < comp.length; j++) d = Math.min(d, dist(comp[i], comp[j])); for (const c of ctxLab) d = Math.min(d, dist(comp[i], c)); } return d; };
  for (let round = 0; round < 3 && pool.length < Math.max(k, oversample / 2); round++) {
    for (let i = 0; i < oversample; i++) { const comp = sampleOne(ctxLab, m, steps, r); stats.raw++;
      const ok = comp.every(c => inGamut(c)), nd = minPair(comp) >= dupThreshold; if (ok) stats.inGamut++; if (ok && !nd) stats.dup++;
      if (ok && nd) pool.push(comp); else if (!ok && nd) rejected.push(comp); }
  }
  if (pool.length < k) { for (const c of rejected) { const mp = c.map(gamutMap); if (minPair(mp) >= dupThreshold) { pool.push(mp); stats.mapped++; } } }
  if (pool.length < k) throw new Error("Could not produce enough valid completions; try fewer colors or a different seed.");
  const scored = pool.map(comp => ({ comp, s: score([...ctxLab, ...comp].map(norm)) })).sort((a, b) => b.s - a.s);
  const smin = scored[scored.length - 1].s, smax = scored[0].s, sn = scored.map(o => (o.s - smin) / (smax - smin + 1e-8));
  const chosen = [0];
  while (chosen.length < k) { let bi = -1, bo = -Infinity; const dmax = Math.max(...scored.map((o, i) => Math.min(...chosen.map(c => matchDist(o.comp, scored[c].comp)))));
    scored.forEach((o, i) => { if (chosen.includes(i)) return; const d = Math.min(...chosen.map(c => matchDist(o.comp, scored[c].comp))); const obj = sn[i] + diversityWeight * d / (dmax + 1e-8); if (obj > bo) { bo = obj; bi = i; } });
    chosen.push(bi); }
  return { palettes: chosen.map(i => ({ colors: scored[i].comp, score: scored[i].s, rank: i + 1 })), stats, poolSize: pool.length };
}

function selfCheck() { const ref = MAN.reference; const out = denoise(ref.x.map((x, i) => ({ x, fixed: ref.is_fixed[i] })), ref.t);
  let err = 0; for (let i = 2; i < 4; i++) for (let k = 0; k < 3; k++) err = Math.max(err, Math.abs(out[i][k] - ref.out[i][k]));
  const se = Math.abs(score(ref.x) - ref.score); return { err, scoreErr: se }; }

window.Palette = { loadModel, complete, selfCheck, hexToOklab, oklabToHex, oklabToSrgb, srgbToOklab, inGamut, gamutMap, denoise, score, norm };
