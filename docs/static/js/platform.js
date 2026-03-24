const EXPLORER_DATA_PATH = '../static/data/oneocean_public_currents_subset.json';

const explorerState = {
  dataset: null,
  controls: {},
};

const FIELD_CONFIG = {
  speed: {
    label: 'current speed',
    legend: 'current-speed tint',
    narrative: 'Current speed is visualized on top of a synthetic ocean-relief base map generated from bathymetry and smooth noise.',
    primaryLabel: 'Mean speed',
    secondaryLabel: 'Peak speed',
    units: 'm/s',
  },
  temperature: {
    label: 'temperature',
    legend: 'temperature tint',
    narrative: 'Temperature tint shows thermal variation for the selected depth layer.',
    primaryLabel: 'Mean temperature',
    secondaryLabel: 'Peak temperature',
    units: '°C',
  },
  salinity: {
    label: 'salinity',
    legend: 'salinity tint',
    narrative: 'Salinity tint shows spatial differences in water-mass composition.',
    primaryLabel: 'Mean salinity',
    secondaryLabel: 'Peak salinity',
    units: 'psu',
  },
  ssh: {
    label: 'sea surface height',
    legend: 'sea-surface-height tint',
    narrative: 'Sea-surface-height tint visualizes surface elevation sampled from the same subset.',
    primaryLabel: 'Mean SSH',
    secondaryLabel: 'Peak |SSH|',
    units: 'm',
  },
  u: {
    label: 'zonal current (u)',
    legend: 'zonal-current tint',
    narrative: 'Zonal-current tint shows east-west flow variation.',
    primaryLabel: 'Mean u',
    secondaryLabel: 'Peak |u|',
    units: 'm/s',
  },
  v: {
    label: 'meridional current (v)',
    legend: 'meridional-current tint',
    narrative: 'Meridional-current tint shows north-south flow variation.',
    primaryLabel: 'Mean v',
    secondaryLabel: 'Peak |v|',
    units: 'm/s',
  },
  bathymetry: {
    label: 'bathymetry',
    legend: 'bathymetry shading',
    narrative: 'Bathymetry shading emphasizes seabed structure in the subset footprint.',
    primaryLabel: 'Mean depth',
    secondaryLabel: 'Max depth',
    units: 'm',
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

function buildIndexOptions(select, labels) {
  select.innerHTML = labels.map((label, index) => `<option value="${index}">${label}</option>`).join('');
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(width * 0.56));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function wireCopyButtons() {
  const buttons = document.querySelectorAll('.platform-copy-btn');
  buttons.forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        const before = button.textContent;
        button.textContent = 'Copied';
        window.setTimeout(() => {
          button.textContent = before;
        }, 1200);
      } catch {
        button.textContent = 'Failed';
        window.setTimeout(() => {
          button.textContent = 'Copy';
        }, 1200);
      }
    });
  });
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
      minElevation = Math.min(minElevation, elevation);
      maxElevation = Math.max(maxElevation, elevation);

      row.push({
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

function scalarRange(summary, fieldMode) {
  if (fieldMode === 'speed') return { min: 0, max: Math.max(summary.maxSpeed, 0.1) };
  if (fieldMode === 'temperature') return { min: summary.minTemperature, max: summary.maxTemperature };
  if (fieldMode === 'salinity') return { min: summary.minSalinity, max: summary.maxSalinity };
  if (fieldMode === 'ssh') return { min: -Math.max(summary.peakAbsSSH, 0.05), max: Math.max(summary.peakAbsSSH, 0.05) };
  if (fieldMode === 'u') return { min: -Math.max(summary.peakAbsU, 0.1), max: Math.max(summary.peakAbsU, 0.1) };
  if (fieldMode === 'v') return { min: -Math.max(summary.peakAbsV, 0.1), max: Math.max(summary.peakAbsV, 0.1) };
  return { min: summary.minElevation, max: summary.maxElevation };
}

function valueForMode(cell, fieldMode) {
  if (fieldMode === 'speed') return cell.speed;
  if (fieldMode === 'temperature') return cell.meanTemperature;
  if (fieldMode === 'salinity') return cell.meanSalinity;
  if (fieldMode === 'ssh') return cell.meanSSH;
  if (fieldMode === 'u') return cell.meanU;
  if (fieldMode === 'v') return cell.meanV;
  return cell.elevation;
}

function colorForRelief(elevation, land, nx, ny) {
  const depthNorm = clamp(Math.abs(elevation) / 5600, 0, 1);
  let rgb;
  if (land) {
    rgb = mixColor([210, 196, 158], [123, 110, 79], clamp((elevation + 10) / 210, 0, 1));
  } else if (depthNorm < 0.2) {
    rgb = mixColor([71, 147, 184], [34, 104, 153], depthNorm / 0.2);
  } else if (depthNorm < 0.65) {
    rgb = mixColor([34, 104, 153], [15, 63, 104], (depthNorm - 0.2) / 0.45);
  } else {
    rgb = mixColor([15, 63, 104], [8, 31, 60], (depthNorm - 0.65) / 0.35);
  }

  const wave = 0.04 * Math.sin(nx * 24.0 + ny * 8.0) + 0.03 * Math.cos(nx * 10.0 - ny * 18.0);
  const light = 0.12 * (1.0 - ny) + wave;
  return rgb.map((value) => Math.round(clamp(value + light * 255, 0, 255)));
}

function colorForField(fieldMode, value, range) {
  const span = Math.max(1e-6, range.max - range.min);
  const t = clamp((value - range.min) / span, 0, 1);
  if (fieldMode === 'speed') return mixColor([20, 98, 153], [255, 225, 126], t);
  if (fieldMode === 'temperature') return mixColor([37, 111, 186], [245, 121, 61], t);
  if (fieldMode === 'salinity') return mixColor([39, 130, 116], [225, 240, 122], t);
  if (fieldMode === 'bathymetry') return mixColor([22, 82, 124], [196, 214, 232], t);
  if (t < 0.5) return mixColor([232, 130, 74], [236, 243, 247], t / 0.5);
  return mixColor([236, 243, 247], [22, 103, 185], (t - 0.5) / 0.5);
}

function bilinear(field, gx, gy) {
  const rows = field.length;
  const cols = field[0].length;
  const x = clamp(gx, 0, cols - 1);
  const y = clamp(gy, 0, rows - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(cols - 1, x0 + 1);
  const y1 = Math.min(rows - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const c00 = field[y0][x0];
  const c10 = field[y0][x1];
  const c01 = field[y1][x0];
  const c11 = field[y1][x1];

  const blend = (a, b, c, d) => lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
  return {
    speed: blend(c00.speed, c10.speed, c01.speed, c11.speed),
    meanTemperature: blend(c00.meanTemperature, c10.meanTemperature, c01.meanTemperature, c11.meanTemperature),
    meanSalinity: blend(c00.meanSalinity, c10.meanSalinity, c01.meanSalinity, c11.meanSalinity),
    meanSSH: blend(c00.meanSSH, c10.meanSSH, c01.meanSSH, c11.meanSSH),
    meanU: blend(c00.meanU, c10.meanU, c01.meanU, c11.meanU),
    meanV: blend(c00.meanV, c10.meanV, c01.meanV, c11.meanV),
    elevation: blend(c00.elevation, c10.elevation, c01.elevation, c11.elevation),
    land: blend(c00.land ? 1 : 0, c10.land ? 1 : 0, c01.land ? 1 : 0, c11.land ? 1 : 0) > 0.45,
  };
}

function drawOceanMap(ctx, width, height, summary, fieldMode, range, subset) {
  const margin = { left: 64, right: 14, top: 14, bottom: 46 };
  const mapWidth = Math.max(1, width - margin.left - margin.right);
  const mapHeight = Math.max(1, height - margin.top - margin.bottom);
  const rows = summary.field.length;
  const cols = summary.field[0].length;

  const image = ctx.createImageData(mapWidth, mapHeight);
  let p = 0;
  for (let y = 0; y < mapHeight; y += 1) {
    const ny = y / Math.max(1, mapHeight - 1);
    const gy = (1 - ny) * Math.max(1, rows - 1);
    for (let x = 0; x < mapWidth; x += 1) {
      const nx = x / Math.max(1, mapWidth - 1);
      const gx = nx * Math.max(1, cols - 1);
      const cell = bilinear(summary.field, gx, gy);
      const relief = colorForRelief(cell.elevation, cell.land, nx, ny);
      const fieldColor = colorForField(fieldMode, valueForMode(cell, fieldMode), range);
      const blend = fieldMode === 'bathymetry' ? 0.2 : 0.42;
      const rgb = mixColor(relief, fieldColor, blend);
      image.data[p] = rgb[0];
      image.data[p + 1] = rgb[1];
      image.data[p + 2] = rgb[2];
      image.data[p + 3] = 255;
      p += 4;
    }
  }

  ctx.putImageData(image, margin.left, margin.top);

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.lineWidth = 1.1;
  ctx.strokeRect(margin.left, margin.top, mapWidth, mapHeight);

  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.font = '12px Inter, sans-serif';

  const lonLabels = [subset.lonMin, (subset.lonMin + subset.lonMax) / 2, subset.lonMax];
  const latLabels = [subset.latMin, (subset.latMin + subset.latMax) / 2, subset.latMax];

  ctx.textAlign = 'center';
  lonLabels.forEach((value, idx) => {
    const x = margin.left + (mapWidth * idx) / (lonLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, x, height - 18);
  });

  ctx.textAlign = 'right';
  latLabels.forEach((value, idx) => {
    const y = margin.top + mapHeight - (mapHeight * idx) / (latLabels.length - 1);
    ctx.fillText(`${formatNumber(value)}°`, margin.left - 10, y + (idx === 2 ? -8 : 4));
  });

  ctx.textAlign = 'center';
  ctx.fillText('longitude', margin.left + mapWidth / 2, height - 4);
  ctx.save();
  ctx.translate(14, margin.top + mapHeight / 2 + 6);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('latitude', 0, 0);
  ctx.restore();
  ctx.restore();

  return { margin, mapWidth, mapHeight };
}

function drawVectorOverlay(ctx, summary, rangeSpeed, controls, mapLayout) {
  const step = Number(controls.vectorDensity.value);
  const rows = summary.field.length;
  const cols = summary.field[0].length;
  const cellWidth = mapLayout.mapWidth / Math.max(1, cols);
  const cellHeight = mapLayout.mapHeight / Math.max(1, rows);
  let count = 0;

  summary.field.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      if (cell.land) return;
      if (rowIndex % step !== 0 || colIndex % step !== 0) return;

      const cx = mapLayout.margin.left + colIndex * cellWidth + cellWidth / 2;
      const cy = mapLayout.margin.top + (rows - rowIndex - 1) * cellHeight + cellHeight / 2;
      const scale = Math.min(cellWidth, cellHeight) * 0.88;
      const norm = Math.max(0.08, rangeSpeed.max);
      const dx = (cell.meanU / norm) * scale;
      const dy = (-cell.meanV / norm) * scale;
      const ex = cx + dx;
      const ey = cy + dy;

      ctx.strokeStyle = 'rgba(6, 18, 26, 0.82)';
      ctx.lineWidth = 3.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(236, 247, 255, 0.95)';
      ctx.lineWidth = 2.1;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      const angle = Math.atan2(dy, dx);
      const arrow = 6.3;
      ctx.fillStyle = 'rgba(236, 247, 255, 0.95)';
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - arrow * Math.cos(angle - Math.PI / 6), ey - arrow * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(ex - arrow * Math.cos(angle + Math.PI / 6), ey - arrow * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      count += 1;
    });
  });

  return count;
}

function summaryValues(summary, fieldMode) {
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
  const range = scalarRange(summary, fieldMode);
  const rangeSpeed = scalarRange(summary, 'speed');

  const { ctx, width, height } = setupCanvas(controls.canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0a2238';
  ctx.fillRect(0, 0, width, height);

  const mapLayout = drawOceanMap(ctx, width, height, summary, fieldMode, range, subset);
  const vectorCount = showVectors ? drawVectorOverlay(ctx, summary, rangeSpeed, controls, mapLayout) : 0;
  controls.stats[2].textContent = String(vectorCount);

  const values = summaryValues(summary, fieldMode);
  controls.statPrimaryLabel.textContent = fieldConfig.primaryLabel;
  controls.statSecondaryLabel.textContent = fieldConfig.secondaryLabel;
  controls.stats[0].textContent = values.primary;
  controls.stats[1].textContent = values.secondary;
  controls.stats[3].textContent = `${Math.round(summary.minElevation)} to ${Math.round(summary.maxElevation)} m`;

  controls.legendText.textContent = `${fieldConfig.narrative} ${showVectors ? 'Vectors show averaged current direction.' : 'Vectors are hidden in this view.'}`;
  updateLegend(controls, fieldMode);

  controls.narrative.textContent =
    `This demo renders a small static subset (${dataset.metadata.grid_shape[0]} × ${dataset.metadata.grid_shape[1]}, ${dataset.metadata.depth_count} depth levels, ${dataset.time.length} dates) to keep GitHub Pages responsive while preserving real ocean variability.`;

  controls.meta.innerHTML = [
    `Time: ${formatDateLabel(dataset.time[subset.timeStart])} -> ${formatDateLabel(dataset.time[subset.timeEnd])}`,
    `Depth: ${formatNumber(dataset.depth[subset.depthIndex], 1)} m`,
    `Variable: ${fieldConfig.label}`,
    `Lat: ${formatSigned(subset.latMin)}° -> ${formatSigned(subset.latMax)}°`,
    `Lon: ${formatSigned(subset.lonMin)}° -> ${formatSigned(subset.lonMax)}°`,
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
  wireCopyButtons();
  try {
    const dataset = await loadDataset();
    explorerState.dataset = dataset;
    explorerState.controls = initializeControls(dataset);
    renderExplorer();
  } catch (error) {
    const narrative = document.getElementById('explorerNarrative');
    if (narrative) {
      narrative.textContent = 'The web subset failed to load. Rebuild docs/static/data/oneocean_public_currents_subset.json and refresh the page.';
    }
    console.error(error);
  }
}

document.addEventListener('DOMContentLoaded', bootstrapExplorer);
