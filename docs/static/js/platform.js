const EXPLORER_DATA_PATH = '../static/data/oneocean_public_currents_subset.json';

const explorerState = {
  dataset: null,
  controls: {},
};

const FIELD_CONFIG = {
  speed: {
    label: 'current speed',
    legend: 'current-speed tint',
    narrative: 'Current-speed tint highlights the magnitude of the grounded flow field over a bathymetry-informed background.',
    primaryLabel: 'Mean speed',
    secondaryLabel: 'Peak speed',
    units: 'm/s',
  },
  temperature: {
    label: 'temperature',
    legend: 'temperature tint',
    narrative: 'Temperature tint highlights thermal structure within the selected depth layer.',
    primaryLabel: 'Mean temperature',
    secondaryLabel: 'Peak temperature',
    units: '°C',
  },
  salinity: {
    label: 'salinity',
    legend: 'salinity tint',
    narrative: 'Salinity tint highlights spatial changes in water-mass composition.',
    primaryLabel: 'Mean salinity',
    secondaryLabel: 'Peak salinity',
    units: 'psu',
  },
  ssh: {
    label: 'sea surface height',
    legend: 'sea-surface-height tint',
    narrative: 'Sea-surface-height tint highlights the surface elevation field sampled from the same public subset.',
    primaryLabel: 'Mean SSH',
    secondaryLabel: 'Peak |SSH|',
    units: 'm',
  },
  bathymetry: {
    label: 'bathymetry',
    legend: 'bathymetry shading',
    narrative: 'Bathymetry shading emphasizes the seabed geometry beneath the selected current field.',
    primaryLabel: 'Mean depth',
    secondaryLabel: 'Max depth',
    units: 'm',
  },
  u: {
    label: 'zonal current (u)',
    legend: 'zonal-current tint',
    narrative: 'Zonal-current tint separates westward and eastward flow components.',
    primaryLabel: 'Mean u',
    secondaryLabel: 'Peak |u|',
    units: 'm/s',
  },
  v: {
    label: 'meridional current (v)',
    legend: 'meridional-current tint',
    narrative: 'Meridional-current tint separates southward and northward flow components.',
    primaryLabel: 'Mean v',
    secondaryLabel: 'Peak |v|',
    units: 'm/s',
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mixColor(c0, c1, t) {
  return c0.map((value, idx) => Math.round(lerp(value, c1[idx], t)));
}

function toCss(rgb, alpha = 1) {
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
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

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(width * 0.58));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function buildIndexOptions(select, labels) {
  select.innerHTML = labels.map((label, index) => `<option value="${index}">${label}</option>`).join('');
}

function initializeControls(dataset) {
  const controls = {
    timeStart: document.getElementById('timeStart'),
    timeEnd: document.getElementById('timeEnd'),
    depthSelect: document.getElementById('depthSelect'),
    fieldMode: document.getElementById('fieldMode'),
    vectorOverlay: document.getElementById('vectorOverlay'),
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
    statPrimaryLabel: document.getElementById('statPrimaryLabel'),
    statSecondaryLabel: document.getElementById('statSecondaryLabel'),
    narrative: document.getElementById('explorerNarrative'),
    meta: document.getElementById('explorerMeta'),
    legendText: document.getElementById('explorerLegendText'),
    surfaceLegendLabel: document.getElementById('surfaceLegendLabel'),
    surfaceLegendSwatch: document.getElementById('surfaceLegendSwatch'),
    canvas: document.getElementById('flowFieldCanvas'),
  };

  buildIndexOptions(controls.timeStart, dataset.time.map(formatDateLabel));
  buildIndexOptions(controls.timeEnd, dataset.time.map(formatDateLabel));
  buildIndexOptions(controls.depthSelect, dataset.depth.map((value) => `${formatNumber(value, 1)} m`));

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
  controls.fieldMode.value = 'speed';
  controls.vectorOverlay.value = 'on';
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
    controls.fieldMode,
    controls.vectorOverlay,
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
    controls.fieldMode.value = 'speed';
    controls.vectorOverlay.value = 'on';
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

  const latIndices = dataset.latitude.map((value, index) => (value >= latMin && value <= latMax ? index : -1)).filter((index) => index >= 0);
  const lonIndices = dataset.longitude.map((value, index) => (value >= lonMin && value <= lonMax ? index : -1)).filter((index) => index >= 0);

  return { timeStart, timeEnd, depthIndex, latIndices, lonIndices, latMin, latMax, lonMin, lonMax };
}

function summarizeSubset(dataset, subset) {
  const { timeStart, timeEnd, depthIndex, latIndices, lonIndices } = subset;
  const field = [];
  let speedSum = 0;
  let tempSum = 0;
  let salinitySum = 0;
  let sshSum = 0;
  let cellCount = 0;
  let maxSpeed = 0;
  let maxTemperature = Number.NEGATIVE_INFINITY;
  let maxSalinity = Number.NEGATIVE_INFINITY;
  let peakAbsSSH = 0;
  let uSum = 0;
  let vSum = 0;
  let peakAbsU = 0;
  let peakAbsV = 0;
  let elevationSum = 0;
  let minElevation = Number.POSITIVE_INFINITY;
  let maxElevation = Number.NEGATIVE_INFINITY;
  let minTemperature = Number.POSITIVE_INFINITY;
  let minSalinity = Number.POSITIVE_INFINITY;
  let minSSH = Number.POSITIVE_INFINITY;
  let maxSSH = Number.NEGATIVE_INFINITY;

  latIndices.forEach((latIndex) => {
    const row = [];
    lonIndices.forEach((lonIndex) => {
      let u = 0;
      let v = 0;
      let temperature = 0;
      let salinity = 0;
      let ssh = 0;
      let count = 0;
      for (let timeIndex = timeStart; timeIndex <= timeEnd; timeIndex += 1) {
        u += dataset.u[timeIndex][depthIndex][latIndex][lonIndex];
        v += dataset.v[timeIndex][depthIndex][latIndex][lonIndex];
        temperature += dataset.temperature[timeIndex][depthIndex][latIndex][lonIndex];
        salinity += dataset.salinity[timeIndex][depthIndex][latIndex][lonIndex];
        ssh += dataset.ssh[timeIndex][latIndex][lonIndex];
        count += 1;
      }
      const meanU = u / count;
      const meanV = v / count;
      const meanTemperature = temperature / count;
      const meanSalinity = salinity / count;
      const meanSSH = ssh / count;
      const speed = Math.sqrt(meanU * meanU + meanV * meanV);
      const elevation = dataset.elevation[latIndex][lonIndex];
      const land = dataset.land_mask[latIndex][lonIndex] > 0.5;

      speedSum += speed;
      tempSum += meanTemperature;
      salinitySum += meanSalinity;
      sshSum += meanSSH;
      uSum += meanU;
      vSum += meanV;
      elevationSum += elevation;
      cellCount += 1;
      maxSpeed = Math.max(maxSpeed, speed);
      peakAbsU = Math.max(peakAbsU, Math.abs(meanU));
      peakAbsV = Math.max(peakAbsV, Math.abs(meanV));
      maxTemperature = Math.max(maxTemperature, meanTemperature);
      minTemperature = Math.min(minTemperature, meanTemperature);
      maxSalinity = Math.max(maxSalinity, meanSalinity);
      minSalinity = Math.min(minSalinity, meanSalinity);
      peakAbsSSH = Math.max(peakAbsSSH, Math.abs(meanSSH));
      minSSH = Math.min(minSSH, meanSSH);
      maxSSH = Math.max(maxSSH, meanSSH);
      minElevation = Math.min(minElevation, elevation);
      maxElevation = Math.max(maxElevation, elevation);

      row.push({
        latIndex,
        lonIndex,
        meanU,
        meanV,
        meanTemperature,
        meanSalinity,
        meanSSH,
        speed,
        elevation,
        land,
      });
    });
    field.push(row);
  });

  return {
    field,
    meanSpeed: cellCount ? speedSum / cellCount : 0,
    maxSpeed,
    meanTemperature: cellCount ? tempSum / cellCount : 0,
    maxTemperature,
    minTemperature,
    meanSalinity: cellCount ? salinitySum / cellCount : 0,
    maxSalinity,
    minSalinity,
    meanSSH: cellCount ? sshSum / cellCount : 0,
    peakAbsSSH,
    minSSH,
    maxSSH,
    meanU: cellCount ? uSum / cellCount : 0,
    meanV: cellCount ? vSum / cellCount : 0,
    peakAbsU,
    peakAbsV,
    meanElevation: cellCount ? elevationSum / cellCount : 0,
    minElevation,
    maxElevation,
    cellCount,
  };
}

function valueForField(cell, mode) {
  if (mode === 'speed') return cell.speed;
  if (mode === 'temperature') return cell.meanTemperature;
  if (mode === 'salinity') return cell.meanSalinity;
  if (mode === 'ssh') return cell.meanSSH;
  if (mode === 'u') return cell.meanU;
  if (mode === 'v') return cell.meanV;
  return cell.elevation;
}

function rangeForField(summary, mode) {
  if (mode === 'speed') return { min: 0, max: Math.max(summary.maxSpeed, 0.1) };
  if (mode === 'temperature') return { min: summary.minTemperature, max: summary.maxTemperature };
  if (mode === 'salinity') return { min: summary.minSalinity, max: summary.maxSalinity };
  if (mode === 'ssh') return { min: -Math.max(summary.peakAbsSSH, 0.05), max: Math.max(summary.peakAbsSSH, 0.05) };
  if (mode === 'u') return { min: -Math.max(summary.peakAbsU, 0.1), max: Math.max(summary.peakAbsU, 0.1) };
  if (mode === 'v') return { min: -Math.max(summary.peakAbsV, 0.1), max: Math.max(summary.peakAbsV, 0.1) };
  return { min: summary.minElevation, max: summary.maxElevation };
}

function colorForBathymetry(elevation, land) {
  if (land) {
    const h = clamp((elevation + 5) / 150, 0, 1);
    return mixColor([194, 183, 151], [116, 105, 78], h);
  }
  const depth = clamp(Math.abs(elevation) / 5500, 0, 1);
  if (depth < 0.25) return mixColor([33, 88, 125], [56, 140, 176], depth / 0.25);
  if (depth < 0.65) return mixColor([56, 140, 176], [18, 73, 112], (depth - 0.25) / 0.40);
  return mixColor([18, 73, 112], [8, 34, 62], (depth - 0.65) / 0.35);
}

function fieldTint(mode, value, range) {
  const span = Math.max(1e-6, range.max - range.min);
  if (mode === 'speed') {
    const t = clamp((value - range.min) / span, 0, 1);
    return mixColor([11, 91, 150], [255, 227, 127], t);
  }
  if (mode === 'temperature') {
    const t = clamp((value - range.min) / span, 0, 1);
    return mixColor([35, 111, 184], [244, 119, 62], t);
  }
  if (mode === 'salinity') {
    const t = clamp((value - range.min) / span, 0, 1);
    return mixColor([47, 132, 123], [230, 238, 120], t);
  }
  if (mode === 'bathymetry') {
    return colorForBathymetry(value, value >= 0);
  }
  const t = clamp((value - range.min) / span, 0, 1);
  return t < 0.5
    ? mixColor([230, 128, 72], [235, 241, 246], t / 0.5)
    : mixColor([235, 241, 246], [22, 103, 184], (t - 0.5) / 0.5);
}

function bilinearCell(summary, gx, gy) {
  const rows = summary.field.length;
  const cols = summary.field[0].length;
  const x = clamp(gx, 0, cols - 1);
  const y = clamp(gy, 0, rows - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(cols - 1, x0 + 1);
  const y1 = Math.min(rows - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const c00 = summary.field[y0][x0];
  const c10 = summary.field[y0][x1];
  const c01 = summary.field[y1][x0];
  const c11 = summary.field[y1][x1];

  const blend = (a, b, c, d) => lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
  return {
    speed: blend(c00.speed, c10.speed, c01.speed, c11.speed),
    meanTemperature: blend(c00.meanTemperature, c10.meanTemperature, c01.meanTemperature, c11.meanTemperature),
    meanSalinity: blend(c00.meanSalinity, c10.meanSalinity, c01.meanSalinity, c11.meanSalinity),
    meanSSH: blend(c00.meanSSH, c10.meanSSH, c01.meanSSH, c11.meanSSH),
    meanU: blend(c00.meanU, c10.meanU, c01.meanU, c11.meanU),
    meanV: blend(c00.meanV, c10.meanV, c01.meanV, c11.meanV),
    elevation: blend(c00.elevation, c10.elevation, c01.elevation, c11.elevation),
    land: blend(c00.land ? 1 : 0, c10.land ? 1 : 0, c01.land ? 1 : 0, c11.land ? 1 : 0) > 0.35,
  };
}

function drawMapBackground(ctx, width, height, summary, fieldMode, range, margins) {
  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;
  const rows = summary.field.length;
  const cols = summary.field[0].length;
  const image = ctx.createImageData(plotWidth, plotHeight);
  let offset = 0;

  for (let py = 0; py < plotHeight; py += 1) {
    const gy = ((plotHeight - py - 1) / Math.max(1, plotHeight - 1)) * Math.max(1, rows - 1);
    for (let px = 0; px < plotWidth; px += 1) {
      const gx = (px / Math.max(1, plotWidth - 1)) * Math.max(1, cols - 1);
      const cell = bilinearCell(summary, gx, gy);
      const base = colorForBathymetry(cell.elevation, cell.land);
      const overlay = fieldTint(fieldMode, valueForField(cell, fieldMode), range);
      const blendRatio = fieldMode === 'bathymetry' ? 0.22 : fieldMode === 'ssh' ? 0.44 : 0.36;
      let rgb = mixColor(base, overlay, blendRatio);

      const ridge = clamp((Math.abs(cell.elevation) % 220) / 220, 0, 1);
      const light = cell.land ? 0.05 : 0.08 * (1 - clamp(Math.abs(cell.elevation) / 5200, 0, 1));
      rgb = rgb.map((v) => Math.round(clamp(v + ridge * 8 + light * 255, 0, 255)));

      image.data[offset] = rgb[0];
      image.data[offset + 1] = rgb[1];
      image.data[offset + 2] = rgb[2];
      image.data[offset + 3] = 255;
      offset += 4;
    }
  }

  ctx.putImageData(image, margins.left, margins.top);

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= 4; x += 1) {
    const px = margins.left + (plotWidth * x) / 4;
    ctx.beginPath();
    ctx.moveTo(px, margins.top);
    ctx.lineTo(px, height - margins.bottom);
    ctx.stroke();
  }
  for (let y = 0; y <= 3; y += 1) {
    const py = margins.top + (plotHeight * y) / 3;
    ctx.beginPath();
    ctx.moveTo(margins.left, py);
    ctx.lineTo(width - margins.right, py);
    ctx.stroke();
  }
  ctx.restore();
}

function drawAxes(ctx, width, height, subset, margins) {
  const { left, right, top, bottom } = margins;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.38)';
  ctx.lineWidth = 1.2;
  ctx.strokeRect(left, top, plotWidth, plotHeight);

  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.font = '12px Inter, sans-serif';

  const lonLabels = [subset.lonMin, (subset.lonMin + subset.lonMax) / 2, subset.lonMax];
  const latLabels = [subset.latMin, (subset.latMin + subset.latMax) / 2, subset.latMax];

  lonLabels.forEach((value, idx) => {
    const x = left + (plotWidth * idx) / (lonLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, x - 20, height - bottom + 24);
  });

  latLabels.forEach((value, idx) => {
    const y = height - bottom - (plotHeight * idx) / (latLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, 20, y + (idx === latLabels.length - 1 ? -8 : 4));
  });

  ctx.fillText('longitude', width / 2 - 30, height - 8);
  ctx.save();
  ctx.translate(12, height / 2 + 10);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('latitude', 0, 0);
  ctx.restore();
  ctx.restore();
}

function drawVectors(ctx, width, height, summary, controls, margins) {
  const densityStep = Number(controls.vectorDensity.value);
  const rows = summary.field.length;
  const cols = summary.field[0].length;
  const drawWidth = width - margins.left - margins.right;
  const drawHeight = height - margins.top - margins.bottom;
  const cellWidth = drawWidth / Math.max(1, cols);
  const cellHeight = drawHeight / Math.max(1, rows);
  let vectorCount = 0;

  summary.field.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      if (cell.land) return;
      if (rowIndex % densityStep !== 0 || colIndex % densityStep !== 0) return;
      vectorCount += 1;
      const centerX = margins.left + colIndex * cellWidth + cellWidth / 2;
      const centerY = margins.top + (rows - rowIndex - 1) * cellHeight + cellHeight / 2;
      const scale = Math.min(cellWidth, cellHeight) * 0.95;
      const norm = Math.max(0.12, summary.maxSpeed);
      const dx = (cell.meanU / norm) * scale;
      const dy = (-cell.meanV / norm) * scale;
      const endX = centerX + dx;
      const endY = centerY + dy;

      ctx.strokeStyle = 'rgba(6, 15, 23, 0.85)';
      ctx.lineWidth = 4.2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(228, 247, 255, 0.92)';
      ctx.lineWidth = 2.3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      const angle = Math.atan2(dy, dx);
      const arrowSize = 7;
      ctx.fillStyle = 'rgba(228, 247, 255, 0.92)';
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - arrowSize * Math.cos(angle - Math.PI / 6), endY - arrowSize * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(endX - arrowSize * Math.cos(angle + Math.PI / 6), endY - arrowSize * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    });
  });

  controls.stats[2].textContent = String(vectorCount);
}

function fieldSummaryValues(summary, fieldMode) {
  if (fieldMode === 'bathymetry') {
    return {
      primary: `${Math.round(Math.abs(summary.meanElevation))} m`,
      secondary: `${Math.round(Math.abs(summary.minElevation))} m`,
    };
  }
  if (fieldMode === 'temperature') {
    return {
      primary: `${formatNumber(summary.meanTemperature)} °C`,
      secondary: `${formatNumber(summary.maxTemperature)} °C`,
    };
  }
  if (fieldMode === 'salinity') {
    return {
      primary: `${formatNumber(summary.meanSalinity)} psu`,
      secondary: `${formatNumber(summary.maxSalinity)} psu`,
    };
  }
  if (fieldMode === 'ssh') {
    return {
      primary: `${formatSigned(summary.meanSSH, 3)} m`,
      secondary: `${formatNumber(summary.peakAbsSSH, 3)} m`,
    };
  }
  if (fieldMode === 'u') {
    return {
      primary: `${formatSigned(summary.meanU)} m/s`,
      secondary: `${formatNumber(summary.peakAbsU)} m/s`,
    };
  }
  if (fieldMode === 'v') {
    return {
      primary: `${formatSigned(summary.meanV)} m/s`,
      secondary: `${formatNumber(summary.peakAbsV)} m/s`,
    };
  }
  return {
    primary: `${formatNumber(summary.meanSpeed)} m/s`,
    secondary: `${formatNumber(summary.maxSpeed)} m/s`,
  };
}

function updateLegend(controls, fieldMode) {
  controls.surfaceLegendLabel.textContent = FIELD_CONFIG[fieldMode].legend;
  if (fieldMode === 'bathymetry') {
    controls.surfaceLegendSwatch.className = 'legend-swatch legend-bathy';
  } else if (fieldMode === 'speed') {
    controls.surfaceLegendSwatch.className = 'legend-swatch legend-speed';
  } else {
    controls.surfaceLegendSwatch.className = 'legend-swatch legend-current';
  }
}

function renderExplorer() {
  const { dataset, controls } = explorerState;
  if (!dataset || !controls) return;

  const subset = activeSubsetIndices(dataset, controls);
  if (!subset.latIndices.length || !subset.lonIndices.length) return;
  const summary = summarizeSubset(dataset, subset);
  const fieldMode = controls.fieldMode.value;
  const fieldConfig = FIELD_CONFIG[fieldMode];
  const showVectors = controls.vectorOverlay.value === 'on';

  const { ctx, width, height } = setupCanvas(controls.canvas);
  const margins = { left: 86, right: 34, top: 30, bottom: 58 };
  const fieldRange = rangeForField(summary, fieldMode);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d2740';
  ctx.fillRect(0, 0, width, height);
  drawMapBackground(ctx, width, height, summary, fieldMode, fieldRange, margins);
  if (showVectors) {
    drawVectors(ctx, width, height, summary, controls, margins);
  } else {
    controls.stats[2].textContent = '0';
  }
  drawAxes(ctx, width, height, subset, margins);

  const values = fieldSummaryValues(summary, fieldMode);
  controls.statPrimaryLabel.textContent = fieldConfig.primaryLabel;
  controls.statSecondaryLabel.textContent = fieldConfig.secondaryLabel;
  controls.stats[0].textContent = values.primary;
  controls.stats[1].textContent = values.secondary;
  controls.stats[3].textContent = `${Math.round(summary.minElevation)} to ${Math.round(summary.maxElevation)} m`;
  controls.legendText.textContent = `${fieldConfig.narrative} ${showVectors ? 'Vectors show averaged current direction.' : 'Vector overlay is currently hidden.'}`;
  updateLegend(controls, fieldMode);

  controls.narrative.textContent =
    `This panel shows only ${dataset.metadata.grid_shape[0]} × ${dataset.metadata.grid_shape[1]} cells, ${dataset.metadata.depth_count} sampled depth levels, and ${dataset.time.length} sampled dates from the public release. Averaging ${formatDateLabel(dataset.time[subset.timeStart])} to ${formatDateLabel(dataset.time[subset.timeEnd])} keeps the browser example lightweight while still exposing real data variation.`;

  controls.meta.innerHTML = [
    `Time: ${formatDateLabel(dataset.time[subset.timeStart])} → ${formatDateLabel(dataset.time[subset.timeEnd])}`,
    `Depth: ${formatNumber(dataset.depth[subset.depthIndex], 1)} m`,
    `Display: ${fieldConfig.label}`,
    `Lat: ${formatSigned(subset.latMin)}° → ${formatSigned(subset.latMax)}°`,
    `Lon: ${formatSigned(subset.lonMin)}° → ${formatSigned(subset.lonMax)}°`,
  ].map((text) => `<span class="platform-meta-pill">${text}</span>`).join('');
}

async function loadDataset() {
  const response = await fetch(EXPLORER_DATA_PATH);
  if (!response.ok) {
    throw new Error(`Failed to load explorer dataset: ${response.status}`);
  }
  return response.json();
}

async function bootstrapExplorer() {
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
