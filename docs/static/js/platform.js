const EXPLORER_DATA_PATH = '../static/data/oneocean_public_currents_subset.json';

const explorerState = {
  dataset: null,
  controls: {},
};

const FIELD_CONFIG = {
  speed: {
    label: 'current speed',
    legend: 'current-speed tint',
    narrative: 'Current speed is rendered on top of a synthetic ocean-relief base map.',
    primaryLabel: 'Mean speed',
    secondaryLabel: 'Peak speed',
  },
  temperature: {
    label: 'temperature',
    legend: 'temperature tint',
    narrative: 'Temperature tint highlights thermal variation in the selected layer.',
    primaryLabel: 'Mean temperature',
    secondaryLabel: 'Peak temperature',
  },
  salinity: {
    label: 'salinity',
    legend: 'salinity tint',
    narrative: 'Salinity tint highlights water-mass composition changes.',
    primaryLabel: 'Mean salinity',
    secondaryLabel: 'Peak salinity',
  },
  ssh: {
    label: 'sea surface height',
    legend: 'sea-surface-height tint',
    narrative: 'Sea-surface-height tint visualizes the sampled surface-elevation field.',
    primaryLabel: 'Mean SSH',
    secondaryLabel: 'Peak |SSH|',
  },
  u: {
    label: 'zonal current (u)',
    legend: 'zonal-current tint',
    narrative: 'Zonal-current tint visualizes east-west flow variation.',
    primaryLabel: 'Mean u',
    secondaryLabel: 'Peak |u|',
  },
  v: {
    label: 'meridional current (v)',
    legend: 'meridional-current tint',
    narrative: 'Meridional-current tint visualizes north-south flow variation.',
    primaryLabel: 'Mean v',
    secondaryLabel: 'Peak |v|',
  },
  bathymetry: {
    label: 'bathymetry',
    legend: 'bathymetry shading',
    narrative: 'Bathymetry shading visualizes seafloor geometry in the selected region.',
    primaryLabel: 'Mean depth',
    secondaryLabel: 'Max depth',
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(start, end, ratio) {
  return start + (end - start) * ratio;
}

function mixColor(colorA, colorB, ratio) {
  return colorA.map((value, index) => Math.round(lerp(value, colorB[index], ratio)));
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
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width));
  const fallbackHeight = Math.round(width * 0.56);
  const height = Math.max(1, Math.round(bounds.height) || fallbackHeight);

  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = '100%';
  canvas.style.height = '100%';

  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function wireCopyButtons() {
  const buttons = document.querySelectorAll('.platform-copy-btn');
  buttons.forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent || '');
        const previous = button.textContent;
        button.textContent = 'Copied';
        window.setTimeout(() => {
          button.textContent = previous;
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

  const minLat = dataset.latitude[0];
  const maxLat = dataset.latitude[dataset.latitude.length - 1];
  const minLon = dataset.longitude[0];
  const maxLon = dataset.longitude[dataset.longitude.length - 1];

  [controls.latMin, controls.latMax].forEach((element) => {
    element.min = minLat;
    element.max = maxLat;
  });
  [controls.lonMin, controls.lonMax].forEach((element) => {
    element.min = minLon;
    element.max = maxLon;
  });

  controls.timeStart.value = '2';
  controls.timeEnd.value = String(dataset.time.length - 1);
  controls.depthSelect.value = '0';
  controls.fieldMode.value = 'speed';
  controls.vectorOverlay.value = 'on';
  controls.latMin.value = String(minLat + 1.0);
  controls.latMax.value = String(maxLat - 1.0);
  controls.lonMin.value = String(minLon + 1.0);
  controls.lonMax.value = String(maxLon - 1.0);
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
    controls.latMin.value = String(minLat + 1.0);
    controls.latMax.value = String(maxLat - 1.0);
    controls.lonMin.value = String(minLon + 1.0);
    controls.lonMax.value = String(maxLon - 1.0);
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
  let tempSum = 0;
  let salinitySum = 0;
  let sshSum = 0;
  let elevationSum = 0;
  let cellCount = 0;

  let maxSpeed = 0;
  let maxTemperature = Number.NEGATIVE_INFINITY;
  let minTemperature = Number.POSITIVE_INFINITY;
  let maxSalinity = Number.NEGATIVE_INFINITY;
  let minSalinity = Number.POSITIVE_INFINITY;
  let peakAbsSSH = 0;
  let peakAbsU = 0;
  let peakAbsV = 0;
  let minElevation = Number.POSITIVE_INFINITY;
  let maxElevation = Number.NEGATIVE_INFINITY;

  let meanUSum = 0;
  let meanVSum = 0;

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
      elevationSum += elevation;
      meanUSum += meanU;
      meanVSum += meanV;
      cellCount += 1;

      maxSpeed = Math.max(maxSpeed, speed);
      maxTemperature = Math.max(maxTemperature, meanTemperature);
      minTemperature = Math.min(minTemperature, meanTemperature);
      maxSalinity = Math.max(maxSalinity, meanSalinity);
      minSalinity = Math.min(minSalinity, meanSalinity);
      peakAbsSSH = Math.max(peakAbsSSH, Math.abs(meanSSH));
      peakAbsU = Math.max(peakAbsU, Math.abs(meanU));
      peakAbsV = Math.max(peakAbsV, Math.abs(meanV));
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
    meanU: cellCount ? meanUSum / cellCount : 0,
    meanV: cellCount ? meanVSum / cellCount : 0,
    peakAbsU,
    peakAbsV,
    meanElevation: cellCount ? elevationSum / cellCount : 0,
    minElevation,
    maxElevation,
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
    rgb = mixColor([197, 188, 155], [120, 108, 77], clamp((elevation + 15) / 240, 0, 1));
  } else if (depthNorm < 0.22) {
    rgb = mixColor([78, 153, 190], [41, 112, 160], depthNorm / 0.22);
  } else if (depthNorm < 0.68) {
    rgb = mixColor([41, 112, 160], [15, 66, 108], (depthNorm - 0.22) / 0.46);
  } else {
    rgb = mixColor([15, 66, 108], [6, 27, 54], (depthNorm - 0.68) / 0.32);
  }

  const wave = 0.038 * Math.sin(nx * 22.0 + ny * 8.0) + 0.025 * Math.cos(nx * 11.0 - ny * 17.0);
  const sheen = 0.11 * (1.0 - ny);
  const light = wave + sheen;
  return rgb.map((value) => Math.round(clamp(value + light * 255, 0, 255)));
}

function colorForField(fieldMode, value, range) {
  const span = Math.max(1e-6, range.max - range.min);
  const ratio = clamp((value - range.min) / span, 0, 1);

  if (fieldMode === 'speed') return mixColor([23, 101, 156], [255, 224, 126], ratio);
  if (fieldMode === 'temperature') return mixColor([42, 117, 191], [248, 121, 56], ratio);
  if (fieldMode === 'salinity') return mixColor([43, 131, 120], [225, 241, 120], ratio);
  if (fieldMode === 'bathymetry') return mixColor([22, 82, 126], [197, 216, 232], ratio);
  if (ratio < 0.5) return mixColor([233, 132, 75], [237, 244, 248], ratio / 0.5);
  return mixColor([237, 244, 248], [23, 106, 188], (ratio - 0.5) / 0.5);
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
    meanU: blend(c00.meanU, c10.meanU, c01.meanU, c11.meanU),
    meanV: blend(c00.meanV, c10.meanV, c01.meanV, c11.meanV),
    meanTemperature: blend(c00.meanTemperature, c10.meanTemperature, c01.meanTemperature, c11.meanTemperature),
    meanSalinity: blend(c00.meanSalinity, c10.meanSalinity, c01.meanSalinity, c11.meanSalinity),
    meanSSH: blend(c00.meanSSH, c10.meanSSH, c01.meanSSH, c11.meanSSH),
    speed: blend(c00.speed, c10.speed, c01.speed, c11.speed),
    elevation: blend(c00.elevation, c10.elevation, c01.elevation, c11.elevation),
    land: blend(c00.land ? 1 : 0, c10.land ? 1 : 0, c01.land ? 1 : 0, c11.land ? 1 : 0) > 0.45,
  };
}

function drawOceanMap(context, width, height, summary, fieldMode, range, subset) {
  const margin = { left: 66, right: 12, top: 12, bottom: 44 };
  const mapWidth = Math.max(1, width - margin.left - margin.right);
  const mapHeight = Math.max(1, height - margin.top - margin.bottom);

  const rows = summary.field.length;
  const cols = summary.field[0].length;

  const image = context.createImageData(mapWidth, mapHeight);
  let pixelOffset = 0;

  for (let py = 0; py < mapHeight; py += 1) {
    const ny = py / Math.max(1, mapHeight - 1);
    const gy = (1 - ny) * Math.max(1, rows - 1);
    for (let px = 0; px < mapWidth; px += 1) {
      const nx = px / Math.max(1, mapWidth - 1);
      const gx = nx * Math.max(1, cols - 1);
      const cell = bilinear(summary.field, gx, gy);

      const baseColor = colorForRelief(cell.elevation, cell.land, nx, ny);
      const fieldColor = colorForField(fieldMode, valueForMode(cell, fieldMode), range);
      const blendRatio = fieldMode === 'bathymetry' ? 0.22 : 0.46;
      const rgb = mixColor(baseColor, fieldColor, blendRatio);

      image.data[pixelOffset] = rgb[0];
      image.data[pixelOffset + 1] = rgb[1];
      image.data[pixelOffset + 2] = rgb[2];
      image.data[pixelOffset + 3] = 255;
      pixelOffset += 4;
    }
  }

  context.putImageData(image, margin.left, margin.top);

  context.save();
  context.strokeStyle = 'rgba(255,255,255,0.45)';
  context.lineWidth = 1.05;
  context.strokeRect(margin.left, margin.top, mapWidth, mapHeight);

  context.fillStyle = 'rgba(255,255,255,0.92)';
  context.font = '12px Inter, sans-serif';

  const lonLabels = [subset.lonMin, (subset.lonMin + subset.lonMax) / 2, subset.lonMax];
  const latLabels = [subset.latMin, (subset.latMin + subset.latMax) / 2, subset.latMax];

  context.textAlign = 'center';
  lonLabels.forEach((value, index) => {
    const x = margin.left + (mapWidth * index) / (lonLabels.length - 1);
    context.fillText(`${formatNumber(value)}°`, x, height - 18);
  });

  context.textAlign = 'right';
  latLabels.forEach((value, index) => {
    const y = margin.top + mapHeight - (mapHeight * index) / (latLabels.length - 1);
    context.fillText(`${formatNumber(value)}°`, margin.left - 10, y + (index === latLabels.length - 1 ? -8 : 4));
  });

  context.textAlign = 'center';
  context.fillText('longitude', margin.left + mapWidth / 2, height - 4);

  context.save();
  context.translate(14, margin.top + mapHeight / 2 + 6);
  context.rotate(-Math.PI / 2);
  context.fillText('latitude', 0, 0);
  context.restore();

  context.restore();
  return { margin, mapWidth, mapHeight };
}

function drawVectorOverlay(context, summary, speedRange, controls, mapLayout) {
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

      const scale = Math.min(cellWidth, cellHeight) * 0.9;
      const norm = Math.max(0.08, speedRange.max);
      const dx = (cell.meanU / norm) * scale;
      const dy = (-cell.meanV / norm) * scale;
      const ex = cx + dx;
      const ey = cy + dy;

      context.strokeStyle = 'rgba(7, 20, 30, 0.82)';
      context.lineWidth = 3.5;
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(ex, ey);
      context.stroke();

      context.strokeStyle = 'rgba(236, 247, 255, 0.96)';
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(ex, ey);
      context.stroke();

      const angle = Math.atan2(dy, dx);
      const arrowLength = 6.2;
      context.fillStyle = 'rgba(236, 247, 255, 0.96)';
      context.beginPath();
      context.moveTo(ex, ey);
      context.lineTo(ex - arrowLength * Math.cos(angle - Math.PI / 6), ey - arrowLength * Math.sin(angle - Math.PI / 6));
      context.lineTo(ex - arrowLength * Math.cos(angle + Math.PI / 6), ey - arrowLength * Math.sin(angle + Math.PI / 6));
      context.closePath();
      context.fill();
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
  const fieldRange = scalarRange(summary, fieldMode);
  const speedRange = scalarRange(summary, 'speed');

  const { context, width, height } = setupCanvas(controls.canvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#0a2238';
  context.fillRect(0, 0, width, height);

  const mapLayout = drawOceanMap(context, width, height, summary, fieldMode, fieldRange, subset);
  const vectorCount = showVectors ? drawVectorOverlay(context, summary, speedRange, controls, mapLayout) : 0;
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
    `This demo renders a small static subset (${dataset.metadata.grid_shape[0]} × ${dataset.metadata.grid_shape[1]}, ${dataset.metadata.depth_count} depth levels, ${dataset.time.length} dates) to keep GitHub Pages responsive while preserving real ocean variation.`;

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
