let state = {
  p: null,
  q: null,
  r: null,
  mask: null,
};

function randMatrix(rows, cols, rng) {
  const m = [];
  for (let i = 0; i < rows; i += 1) {
    const row = [];
    for (let j = 0; j < cols; j += 1) {
      row.push(rng());
    }
    m.push(row);
  }
  return m;
}

function matmul(a, b) {
  const rows = a.length;
  const cols = b[0].length;
  const k = b.length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      let sum = 0;
      for (let t = 0; t < k; t += 1) {
        sum += a[i][t] * b[t][j];
      }
      out[i][j] = sum;
    }
  }
  return out;
}

function transpose(a) {
  const rows = a.length;
  const cols = a[0].length;
  const out = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      out[j][i] = a[i][j];
    }
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function lcg(seed) {
  let state = seed >>> 0;
  return function rng() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(rng, mean = 0, std = 1) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * std + mean;
}

function totalSquaredError(r, p, q, mask) {
  let sum = 0;
  const qT = transpose(q);
  const rHat = matmul(p, qT);
  for (let i = 0; i < r.length; i += 1) {
    for (let j = 0; j < r[0].length; j += 1) {
      if (mask && !mask[i][j]) continue;
      const diff = r[i][j] - rHat[i][j];
      sum += diff * diff;
    }
  }
  return { sum, rHat };
}

function sgdStep(r, p, q, alpha, mask) {
  for (let i = 0; i < r.length; i += 1) {
    for (let j = 0; j < r[0].length; j += 1) {
      if (mask && !mask[i][j]) continue;
      const eij = r[i][j] - dot(p[i], q[j]);
      const pOld = [...p[i]];
      for (let k = 0; k < p[i].length; k += 1) {
        p[i][k] = p[i][k] + 2 * alpha * eij * q[j][k];
        q[j][k] = q[j][k] + 2 * alpha * eij * pOld[k];
      }
    }
  }
}

function sampleMask(rows, cols, density, rng) {
  if (density >= 1) return null;
  const mask = [];
  for (let i = 0; i < rows; i += 1) {
    const row = [];
    for (let j = 0; j < cols; j += 1) {
      row.push(rng() < density);
    }
    mask.push(row);
  }
  return mask;
}

function renderMatrix(m, maxRows = 4, maxCols = 5) {
  const rows = Math.min(m.length, maxRows);
  const cols = Math.min(m[0].length, maxCols);
  const view = [];
  for (let i = 0; i < rows; i += 1) {
    view.push(
      m[i]
        .slice(0, cols)
        .map((v) => v.toFixed(3))
        .join("  ")
    );
  }
  if (m.length > rows || m[0].length > cols) view.push("...");
  return view.join("\n");
}

function run() {
  const users = Number(document.getElementById("users").value);
  const items = Number(document.getElementById("items").value);
  const factors = Number(document.getElementById("factors").value);
  const seed = Number(document.getElementById("seed").value || 0);
  const maskDensity = Number(document.getElementById("mask").value);
  const noise = Number(document.getElementById("noise").value);

  const rng = lcg(seed);
  const p = randMatrix(users, factors, rng);
  const q = randMatrix(items, factors, rng);
  const rHat = matmul(p, transpose(q));
  const r = rHat.map((row) =>
    row.map((val) => val + gaussian(rng, 0, noise))
  );
  const mask = sampleMask(users, items, maskDensity, rng);

  state = { p, q, r, mask };
  const { sum } = totalSquaredError(r, p, q, mask);
  document.getElementById("error").textContent = sum.toFixed(4);
  document.getElementById("rhat").textContent = renderMatrix(rHat);
}

function step() {
  if (!state.p) run();
  sgdStep(state.r, state.p, state.q, 0.01, state.mask);
  const { sum, rHat } = totalSquaredError(state.r, state.p, state.q, state.mask);
  document.getElementById("error").textContent = sum.toFixed(4);
  document.getElementById("rhat").textContent = renderMatrix(rHat);
}

document.getElementById("run").addEventListener("click", run);
document.getElementById("sgd").addEventListener("click", step);

run();
