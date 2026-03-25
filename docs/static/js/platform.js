const EXPLORER_DATA_PATH = '../static/data/oneocean_public_currents_subset.json';

const explorerState = {
  dataset: null,
  controls: {},
  map: null,
  scalarLayer: null,
  vectorLayer: null,
};

const FIELD_CONFIG = {
  speed: {
    label: 'current speed',
    legend: 'current-speed tint',
    narrative: 'Current speed is rendered over a map base layer with bathymetry-aware shading.',
    primaryLabel: 'Mean speed',
    secondaryLabel: 'Peak speed',
  },
  temperature: {
    label: 'temperature',
    legend: 'temperature tint',
    narrative: 'Temperature tint highlights thermal variation in the selected depth layer.',
    primaryLabel: 'Mean temperature',
    secondaryLabel: 'Peak temperature',
  },
  salinity: {
    label: 'salinity',
    legend: 'salinity tint',
    narrative: 'Salinity tint highlights spatial water-mass differences.',
    primaryLabel: 'Mean salinity',
    secondaryLabel: 'Peak salinity',
  },
  ssh: {
    label: 'sea surface height',
    legend: 'sea-surface-height tint',
    narrative: 'Sea-surface-height tint highlights sampled surface elevation from the same subset.',
    primaryLabel: 'Mean SSH',
    secondaryLabel: 'Peak |SSH|',
  },
  u: {
    label: 'zonal current (u)',
    legend: 'zonal-current tint',
    narrative: 'Zonal-current tint visualizes east-west current variation.',
    primaryLabel: 'Mean u',
    secondaryLabel: 'Peak |u|',
  },
  v: {
    label: 'meridional current (v)',
    legend: 'meridional-current tint',
    narrative: 'Meridional-current tint visualizes north-south current variation.',
    primaryLabel: 'Mean v',
    secondaryLabel: 'Peak |v|',
  },
  bathymetry: {
    label: 'bathymetry',
    legend: 'bathymetry shading',
    narrative: 'Bathymetry shading visualizes seabed geometry across the selected region.',
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

function toRgba(color, alpha = 1) {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
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

function initializeMap(dataset) {
  const mapElement = document.getElementById('flowFieldMap');
  if (!mapElement || typeof L === 'undefined') {
    return;
  }

  const latMin = dataset.latitude[0];
  const latMax = dataset.latitude[dataset.latitude.length - 1];
  const lonMin = dataset.longitude[0];
  const lonMax = dataset.longitude[dataset.longitude.length - 1];

  const map = L.map(mapElement, {
    zoomControl: true,
    attributionControl: true,
    worldCopyJump: false,
    preferCanvas: true,
  });

  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 13,
    attribution: 'Tiles &copy; Esri &mdash; GEBCO, NOAA, National Geographic, DeLorme, HERE',
  }).addTo(map);

  const bounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax]);
  map.fitBounds(bounds.pad(0.06));

  explorerState.map = map;
  explorerState.scalarLayer = L.layerGroup().addTo(map);
  explorerState.vectorLayer = L.layerGroup().addTo(map);

  window.setTimeout(() => {
    map.invalidateSize();
  }, 20);
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

  window.addEventListener('resize', () => {
    if (explorerState.map) {
      explorerState.map.invalidateSize();
    }
    renderExplorer();
  });

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

function colorForRelief(elevation, land) {
  const depthNorm = clamp(Math.abs(elevation) / 5600, 0, 1);
  if (land) {
    return mixColor([197, 188, 155], [120, 108, 77], clamp((elevation + 15) / 240, 0, 1));
  }
  if (depthNorm < 0.22) return mixColor([78, 153, 190], [41, 112, 160], depthNorm / 0.22);
  if (depthNorm < 0.68) return mixColor([41, 112, 160], [15, 66, 108], (depthNorm - 0.22) / 0.46);
  return mixColor([15, 66, 108], [6, 27, 54], (depthNorm - 0.68) / 0.32);
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

function edgeValues(values) {
  if (values.length === 1) {
    return [values[0] - 0.01, values[0] + 0.01];
  }
  const edges = [];
  edges.push(values[0] - (values[1] - values[0]) / 2);
  for (let i = 0; i < values.length - 1; i += 1) {
    edges.push((values[i] + values[i + 1]) / 2);
  }
  edges.push(values[values.length - 1] + (values[values.length - 1] - values[values.length - 2]) / 2);
  return edges;
}

function bilinearSample(grid, fi, fj) {
  const rows = grid.length;
  const cols = grid[0].length;

  const i0 = clamp(Math.floor(fi), 0, rows - 1);
  const j0 = clamp(Math.floor(fj), 0, cols - 1);
  const i1 = clamp(i0 + 1, 0, rows - 1);
  const j1 = clamp(j0 + 1, 0, cols - 1);

  const ti = clamp(fi - i0, 0, 1);
  const tj = clamp(fj - j0, 0, 1);

  const a = lerp(grid[i0][j0], grid[i0][j1], tj);
  const b = lerp(grid[i1][j0], grid[i1][j1], tj);
  return lerp(a, b, ti);
}

function drawScalarField(dataset, subset, summary, fieldMode, range) {
  explorerState.scalarLayer.clearLayers();
  const rows = summary.field.length;
  const cols = summary.field[0]?.length || 0;
  if (!rows || !cols) return;

  const mapSize = explorerState.map ? explorerState.map.getSize() : { x: 800, y: 520 };
  const width = Math.max(540, Math.min(1100, mapSize.x));
  const height = Math.max(360, Math.min(760, Math.round(mapSize.y * 0.95)));

  const valueGrid = summary.field.map((row) => row.map((cell) => valueForMode(cell, fieldMode)));
  const elevationGrid = summary.field.map((row) => row.map((cell) => cell.elevation));
  const landGrid = summary.field.map((row) => row.map((cell) => (cell.land ? 1 : 0)));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return;

  const image = context.createImageData(width, height);
  const alpha = fieldMode === 'bathymetry' ? 0.22 : 0.52;

  for (let y = 0; y < height; y += 1) {
    const fi = ((height - 1 - y) / Math.max(1, height - 1)) * Math.max(1, rows - 1);
    for (let x = 0; x < width; x += 1) {
      const fj = (x / Math.max(1, width - 1)) * Math.max(1, cols - 1);

      const value = bilinearSample(valueGrid, fi, fj);
      const elevation = bilinearSample(elevationGrid, fi, fj);
      const landProb = bilinearSample(landGrid, fi, fj);
      const land = landProb > 0.5;

      const baseColor = colorForRelief(elevation, land);
      const fieldColor = colorForField(fieldMode, value, range);
      const fill = mixColor(baseColor, fieldColor, alpha);

      const idx = (y * width + x) * 4;
      image.data[idx] = fill[0];
      image.data[idx + 1] = fill[1];
      image.data[idx + 2] = fill[2];
      image.data[idx + 3] = 208;
    }
  }

  context.putImageData(image, 0, 0);

  L.imageOverlay(canvas.toDataURL('image/png'), [[subset.latMin, subset.lonMin], [subset.latMax, subset.lonMax]], {
    opacity: 0.92,
    interactive: false,
    className: 'platform-scalar-overlay',
  }).addTo(explorerState.scalarLayer);
}

function drawVectorField(dataset, subset, summary, controls, speedRange) {
  explorerState.vectorLayer.clearLayers();

  const densityStep = Number(controls.vectorDensity.value);
  const latVals = subset.latIndices.map((index) => dataset.latitude[index]);
  const lonVals = subset.lonIndices.map((index) => dataset.longitude[index]);

  let vectorCount = 0;

  for (let i = 0; i < summary.field.length; i += 1) {
    for (let j = 0; j < summary.field[i].length; j += 1) {
      const cell = summary.field[i][j];
      if (cell.land) continue;
      if (i % densityStep !== 0 || j % densityStep !== 0) continue;

      const lat = latVals[i];
      const lon = lonVals[j];
      const speedNorm = Math.max(0.08, speedRange.max);

      const latScale = 0.07;
      const lonScale = 0.07 / Math.max(0.25, Math.cos((lat * Math.PI) / 180));

      const deltaLat = (cell.meanV / speedNorm) * latScale;
      const deltaLon = (cell.meanU / speedNorm) * lonScale;

      const endLat = lat + deltaLat;
      const endLon = lon + deltaLon;

      L.polyline(
        [
          [lat, lon],
          [endLat, endLon],
        ],
        {
          color: 'rgba(235,247,255,0.95)',
          weight: 1.6,
          opacity: 0.92,
          interactive: false,
        }
      ).addTo(explorerState.vectorLayer);

      L.circleMarker([endLat, endLon], {
        radius: 1.8,
        color: 'rgba(235,247,255,0.95)',
        fillColor: 'rgba(235,247,255,0.95)',
        fillOpacity: 0.95,
        weight: 1,
        interactive: false,
      }).addTo(explorerState.vectorLayer);

      vectorCount += 1;
    }
  }

  return vectorCount;
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
  const { dataset, controls, map } = explorerState;
  if (!dataset || !controls || !map) return;

  const subset = activeSubsetIndices(dataset, controls);
  if (!subset.latIndices.length || !subset.lonIndices.length) return;

  const summary = summarizeSubset(dataset, subset);
  const fieldMode = controls.fieldMode.value;
  const fieldConfig = FIELD_CONFIG[fieldMode];
  const showVectors = controls.vectorOverlay.value === 'on';
  const fieldRange = scalarRange(summary, fieldMode);
  const speedRange = scalarRange(summary, 'speed');

  drawScalarField(dataset, subset, summary, fieldMode, fieldRange);

  const vectorCount = showVectors ? drawVectorField(dataset, subset, summary, controls, speedRange) : 0;
  if (!showVectors) {
    explorerState.vectorLayer.clearLayers();
  }

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
    `This demo renders a small static subset (${dataset.metadata.grid_shape[0]} x ${dataset.metadata.grid_shape[1]}, ${dataset.metadata.depth_count} depth levels, ${dataset.time.length} dates) to keep GitHub Pages responsive while preserving real ocean variation.`;

  controls.meta.innerHTML = [
    `Time: ${formatDateLabel(dataset.time[subset.timeStart])} -> ${formatDateLabel(dataset.time[subset.timeEnd])}`,
    `Depth: ${formatNumber(dataset.depth[subset.depthIndex], 1)} m`,
    `Variable: ${fieldConfig.label}`,
    `Lat: ${formatSigned(subset.latMin)}° -> ${formatSigned(subset.latMax)}°`,
    `Lon: ${formatSigned(subset.lonMin)}° -> ${formatSigned(subset.lonMax)}°`,
  ]
    .map((text) => `<span class="platform-meta-pill">${text}</span>`)
    .join('');

  const fitBounds = L.latLngBounds([subset.latMin, subset.lonMin], [subset.latMax, subset.lonMax]);
  map.fitBounds(fitBounds.pad(0.05), { animate: false });
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
    initializeMap(dataset);
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
