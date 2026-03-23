const EXPLORER_DATA_PATH = '../static/data/oneocean_public_currents_subset.json';

const explorerState = {
  dataset: null,
  controls: {},
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function formatSigned(value, digits = 2) {
  const numeric = Number(value);
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(digits)}`;
}

function formatDateLabel(isoText) {
  const date = new Date(isoText);
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function burgerSetup() {
  const burgers = Array.from(document.querySelectorAll('.navbar-burger'));
  burgers.forEach((burger) => {
    burger.addEventListener('click', () => {
      const target = burger.dataset.target;
      const menu = document.getElementById(target);
      burger.classList.toggle('is-active');
      menu.classList.toggle('is-active');
    });
  });
}

function colorForBathymetry(elevation, isLand) {
  if (isLand) return '#e7eef6';
  const depth = Math.max(0, Math.min(5500, -elevation));
  const t = depth / 5500;
  const shallow = [192, 222, 238];
  const mid = [79, 126, 167];
  const deep = [14, 41, 68];
  const low = t < 0.55 ? shallow : mid;
  const high = t < 0.55 ? mid : deep;
  const localT = t < 0.55 ? t / 0.55 : (t - 0.55) / 0.45;
  const rgb = low.map((value, idx) => Math.round(value + (high[idx] - value) * localT));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function colorForSpeed(speed, maxSpeed) {
  const safeMax = Math.max(0.1, maxSpeed);
  const t = clamp(speed / safeMax, 0, 1);
  const c0 = [31, 228, 255];
  const c1 = [255, 224, 130];
  const rgb = c0.map((value, idx) => Math.round(value + (c1[idx] - value) * t));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(width * 0.6));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function buildIndexOptions(select, labels) {
  select.innerHTML = labels
    .map((label, index) => `<option value="${index}">${label}</option>`)
    .join('');
}

function initializeControls(dataset) {
  const controls = {
    timeStart: document.getElementById('timeStart'),
    timeEnd: document.getElementById('timeEnd'),
    depthSelect: document.getElementById('depthSelect'),
    latMin: document.getElementById('latMin'),
    latMax: document.getElementById('latMax'),
    lonMin: document.getElementById('lonMin'),
    lonMax: document.getElementById('lonMax'),
    vectorDensity: document.getElementById('vectorDensity'),
    latRangeValue: document.getElementById('latRangeValue'),
    lonRangeValue: document.getElementById('lonRangeValue'),
    vectorDensityValue: document.getElementById('vectorDensityValue'),
    reset: document.getElementById('explorerReset'),
    stats: document.querySelectorAll('#explorerStats .platform-stat-v'),
    narrative: document.getElementById('explorerNarrative'),
    meta: document.getElementById('explorerMeta'),
    canvas: document.getElementById('flowFieldCanvas'),
  };

  buildIndexOptions(controls.timeStart, dataset.time.map(formatDateLabel));
  buildIndexOptions(controls.timeEnd, dataset.time.map(formatDateLabel));
  buildIndexOptions(
    controls.depthSelect,
    dataset.depth.map((value) => `${formatNumber(value, 1)} m`)
  );

  const latMinValue = dataset.latitude[0];
  const latMaxValue = dataset.latitude[dataset.latitude.length - 1];
  const lonMinValue = dataset.longitude[0];
  const lonMaxValue = dataset.longitude[dataset.longitude.length - 1];

  [controls.latMin, controls.latMax].forEach((element) => {
    element.min = latMinValue;
    element.max = latMaxValue;
  });
  [controls.lonMin, controls.lonMax].forEach((element) => {
    element.min = lonMinValue;
    element.max = lonMaxValue;
  });

  controls.timeStart.value = '2';
  controls.timeEnd.value = String(dataset.time.length - 1);
  controls.depthSelect.value = '0';
  controls.latMin.value = String(latMinValue + 1.0);
  controls.latMax.value = String(latMaxValue - 1.0);
  controls.lonMin.value = String(lonMinValue + 1.0);
  controls.lonMax.value = String(lonMaxValue - 1.0);
  controls.vectorDensity.value = '2';

  const rerender = () => {
    normalizeRanges(controls);
    updateRangeLabels(controls);
    renderExplorer();
  };

  [
    controls.timeStart,
    controls.timeEnd,
    controls.depthSelect,
    controls.latMin,
    controls.latMax,
    controls.lonMin,
    controls.lonMax,
    controls.vectorDensity,
  ].forEach((element) => {
    element.addEventListener('input', rerender);
    element.addEventListener('change', rerender);
  });

  controls.reset.addEventListener('click', () => {
    controls.timeStart.value = '2';
    controls.timeEnd.value = String(dataset.time.length - 1);
    controls.depthSelect.value = '0';
    controls.latMin.value = String(latMinValue + 1.0);
    controls.latMax.value = String(latMaxValue - 1.0);
    controls.lonMin.value = String(lonMinValue + 1.0);
    controls.lonMax.value = String(lonMaxValue - 1.0);
    controls.vectorDensity.value = '2';
    rerender();
  });

  window.addEventListener('resize', renderExplorer);
  updateRangeLabels(controls);
  return controls;
}

function normalizeRanges(controls) {
  if (Number(controls.timeStart.value) > Number(controls.timeEnd.value)) {
    controls.timeEnd.value = controls.timeStart.value;
  }
  if (Number(controls.latMin.value) > Number(controls.latMax.value)) {
    controls.latMax.value = controls.latMin.value;
  }
  if (Number(controls.lonMin.value) > Number(controls.lonMax.value)) {
    controls.lonMax.value = controls.lonMin.value;
  }
}

function updateRangeLabels(controls) {
  controls.latRangeValue.textContent = `${formatNumber(controls.latMin.value)}° to ${formatNumber(controls.latMax.value)}°`;
  controls.lonRangeValue.textContent = `${formatNumber(controls.lonMin.value)}° to ${formatNumber(controls.lonMax.value)}°`;
  controls.vectorDensityValue.textContent = ['dense', 'balanced', 'light', 'sparse'][Number(controls.vectorDensity.value) - 1];
}

function activeSubsetIndices(dataset, controls) {
  const timeStart = Number(controls.timeStart.value);
  const timeEnd = Number(controls.timeEnd.value);
  const depthIndex = Number(controls.depthSelect.value);
  const latMin = Number(controls.latMin.value);
  const latMax = Number(controls.latMax.value);
  const lonMin = Number(controls.lonMin.value);
  const lonMax = Number(controls.lonMax.value);

  const latIndices = dataset.latitude
    .map((value, index) => (value >= latMin && value <= latMax ? index : -1))
    .filter((index) => index >= 0);
  const lonIndices = dataset.longitude
    .map((value, index) => (value >= lonMin && value <= lonMax ? index : -1))
    .filter((index) => index >= 0);

  return { timeStart, timeEnd, depthIndex, latIndices, lonIndices, latMin, latMax, lonMin, lonMax };
}

function summarizeSubset(dataset, subset) {
  const { timeStart, timeEnd, depthIndex, latIndices, lonIndices } = subset;
  const field = [];
  let speedSum = 0;
  let speedCount = 0;
  let maxSpeed = 0;
  let minElevation = Number.POSITIVE_INFINITY;
  let maxElevation = Number.NEGATIVE_INFINITY;

  latIndices.forEach((latIndex) => {
    const row = [];
    lonIndices.forEach((lonIndex) => {
      let u = 0;
      let v = 0;
      let count = 0;
      for (let timeIndex = timeStart; timeIndex <= timeEnd; timeIndex += 1) {
        u += dataset.u[timeIndex][depthIndex][latIndex][lonIndex];
        v += dataset.v[timeIndex][depthIndex][latIndex][lonIndex];
        count += 1;
      }
      const meanU = u / count;
      const meanV = v / count;
      const speed = Math.sqrt(meanU * meanU + meanV * meanV);
      speedSum += speed;
      speedCount += 1;
      maxSpeed = Math.max(maxSpeed, speed);
      const elevation = dataset.elevation[latIndex][lonIndex];
      minElevation = Math.min(minElevation, elevation);
      maxElevation = Math.max(maxElevation, elevation);
      row.push({
        latIndex,
        lonIndex,
        meanU,
        meanV,
        speed,
        elevation,
        land: dataset.land_mask[latIndex][lonIndex] > 0.5,
      });
    });
    field.push(row);
  });

  return {
    field,
    meanSpeed: speedCount ? speedSum / speedCount : 0,
    maxSpeed,
    minElevation,
    maxElevation,
    cellCount: speedCount,
  };
}

function drawAxes(ctx, width, height, dataset, subset, margins) {
  const { left, right, top, bottom } = margins;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width - left - right, height - top - bottom);

  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.font = '12px Inter, sans-serif';
  const lonLabels = [subset.lonMin, (subset.lonMin + subset.lonMax) / 2, subset.lonMax];
  const latLabels = [subset.latMin, (subset.latMin + subset.latMax) / 2, subset.latMax];
  lonLabels.forEach((value, idx) => {
    const x = left + ((width - left - right) * idx) / (lonLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, x - 18, height - bottom + 22);
  });
  latLabels.forEach((value, idx) => {
    const y = height - bottom - ((height - top - bottom) * idx) / (latLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, 8, y + 4);
  });
  ctx.fillText('longitude', width / 2 - 28, height - 8);
  ctx.save();
  ctx.translate(16, height / 2 + 24);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('latitude', 0, 0);
  ctx.restore();
  ctx.restore();
}

function renderExplorer() {
  const { dataset, controls } = explorerState;
  if (!dataset || !controls) return;

  const subset = activeSubsetIndices(dataset, controls);
  if (!subset.latIndices.length || !subset.lonIndices.length) return;
  const summary = summarizeSubset(dataset, subset);

  const { ctx, width, height } = setupCanvas(controls.canvas);
  const margins = { left: 52, right: 28, top: 20, bottom: 44 };
  const drawWidth = width - margins.left - margins.right;
  const drawHeight = height - margins.top - margins.bottom;
  const cols = subset.lonIndices.length;
  const rows = subset.latIndices.length;
  const cellWidth = drawWidth / Math.max(1, cols);
  const cellHeight = drawHeight / Math.max(1, rows);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#16324e';
  ctx.fillRect(0, 0, width, height);

  summary.field.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      const x = margins.left + colIndex * cellWidth;
      const y = margins.top + (rows - rowIndex - 1) * cellHeight;
      ctx.fillStyle = colorForBathymetry(cell.elevation, cell.land);
      ctx.fillRect(x, y, Math.ceil(cellWidth + 1), Math.ceil(cellHeight + 1));
    });
  });

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 0.5;
  for (let row = 0; row <= rows; row += 1) {
    const y = margins.top + row * cellHeight;
    ctx.beginPath();
    ctx.moveTo(margins.left, y);
    ctx.lineTo(width - margins.right, y);
    ctx.stroke();
  }
  for (let col = 0; col <= cols; col += 1) {
    const x = margins.left + col * cellWidth;
    ctx.beginPath();
    ctx.moveTo(x, margins.top);
    ctx.lineTo(x, height - margins.bottom);
    ctx.stroke();
  }
  ctx.restore();

  const densityStep = Number(controls.vectorDensity.value);
  summary.field.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      if (cell.land) return;
      if (rowIndex % densityStep !== 0 || colIndex % densityStep !== 0) return;
      const centerX = margins.left + colIndex * cellWidth + cellWidth / 2;
      const centerY = margins.top + (rows - rowIndex - 1) * cellHeight + cellHeight / 2;
      const scale = Math.min(cellWidth, cellHeight) * 0.55;
      const norm = Math.max(0.12, summary.maxSpeed);
      const dx = (cell.meanU / norm) * scale;
      const dy = (-cell.meanV / norm) * scale;
      const endX = centerX + dx;
      const endY = centerY + dy;

      ctx.strokeStyle = colorForSpeed(cell.speed, summary.maxSpeed);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      const angle = Math.atan2(dy, dx);
      const arrowSize = 5;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - arrowSize * Math.cos(angle - Math.PI / 6), endY - arrowSize * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(endX - arrowSize * Math.cos(angle + Math.PI / 6), endY - arrowSize * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    });
  });

  drawAxes(ctx, width, height, dataset, subset, margins);

  controls.stats[0].textContent = `${formatNumber(summary.meanSpeed)} m/s`;
  controls.stats[1].textContent = `${formatNumber(summary.maxSpeed)} m/s`;
  controls.stats[2].textContent = String(summary.cellCount);
  controls.stats[3].textContent = `${Math.round(summary.minElevation)} to ${Math.round(summary.maxElevation)} m`;

  controls.narrative.textContent =
    `Averaging ${formatDateLabel(dataset.time[subset.timeStart])} to ${formatDateLabel(dataset.time[subset.timeEnd])} at ${formatNumber(dataset.depth[subset.depthIndex], 1)} m shows how the grounded current field changes across the selected footprint while remaining lightweight enough for static deployment.`;

  controls.meta.innerHTML = [
    `Time window: ${formatDateLabel(dataset.time[subset.timeStart])} → ${formatDateLabel(dataset.time[subset.timeEnd])}`,
    `Depth: ${formatNumber(dataset.depth[subset.depthIndex], 1)} m`,
    `Lat: ${formatSigned(subset.latMin)}° → ${formatSigned(subset.latMax)}°`,
    `Lon: ${formatSigned(subset.lonMin)}° → ${formatSigned(subset.lonMax)}°`,
  ]
    .map((text) => `<span class="platform-meta-pill">${text}</span>`)
    .join('');
}

async function loadDataset() {
  const response = await fetch(EXPLORER_DATA_PATH);
  if (!response.ok) {
    throw new Error(`Failed to load explorer dataset: ${response.status}`);
  }
  return response.json();
}

async function bootstrapExplorer() {
  burgerSetup();
  try {
    const dataset = await loadDataset();
    explorerState.dataset = dataset;
    explorerState.controls = initializeControls(dataset);
    renderExplorer();
  } catch (error) {
    const narrative = document.getElementById('explorerNarrative');
    if (narrative) {
      narrative.textContent = 'The web subset failed to load. Rebuild `docs/static/data/oneocean_public_currents_subset.json` and refresh the page.';
    }
    console.error(error);
  }
}

document.addEventListener('DOMContentLoaded', bootstrapExplorer);
