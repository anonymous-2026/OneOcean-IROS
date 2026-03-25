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
    legend: 'vector color: current speed',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes current-speed magnitude.',
    primaryLabel: 'Mean speed',
    secondaryLabel: 'Peak speed',
  },
  temperature: {
    label: 'temperature',
    legend: 'vector color: temperature',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes temperature in the selected depth layer.',
    primaryLabel: 'Mean temperature',
    secondaryLabel: 'Peak temperature',
  },
  salinity: {
    label: 'salinity',
    legend: 'vector color: salinity',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes salinity variation.',
    primaryLabel: 'Mean salinity',
    secondaryLabel: 'Peak salinity',
  },
  ssh: {
    label: 'sea surface height',
    legend: 'vector color: sea-surface height',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes sea-surface-height variation.',
    primaryLabel: 'Mean SSH',
    secondaryLabel: 'Peak |SSH|',
  },
  u: {
    label: 'zonal current (u)',
    legend: 'vector color: zonal current (u)',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes zonal-current variation.',
    primaryLabel: 'Mean u',
    secondaryLabel: 'Peak |u|',
  },
  v: {
    label: 'meridional current (v)',
    legend: 'vector color: meridional current (v)',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes meridional-current variation.',
    primaryLabel: 'Mean v',
    secondaryLabel: 'Peak |v|',
  },
  bathymetry: {
    label: 'bathymetry',
    legend: 'vector color: bathymetry depth',
    narrative: 'Bathymetry is shown as the base map. Arrow color encodes local depth values.',
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
  controls.latMin.value = String(minLat);
  controls.latMax.value = String(maxLat);
  controls.lonMin.value = String(minLon);
  controls.lonMax.value = String(maxLon);
  controls.vectorDensity.value = '1';

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
    controls.latMin.value = String(minLat);
    controls.latMax.value = String(maxLat);
    controls.lonMin.value = String(minLon);
    controls.lonMax.value = String(maxLon);
    controls.vectorDensity.value = '1';
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
  controls.vectorDensityValue.textContent = ['very dense', 'dense', 'balanced', 'light'][Number(controls.vectorDensity.value) - 1];
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

function vectorColorForMode(fieldMode, value, range) {
  const span = Math.max(1e-6, range.max - range.min);
  const ratio = clamp((value - range.min) / span, 0, 1);

  if (fieldMode === 'speed') return mixColor([40, 136, 186], [255, 229, 124], ratio);
  if (fieldMode === 'temperature') return mixColor([33, 112, 190], [246, 121, 55], ratio);
  if (fieldMode === 'salinity') return mixColor([41, 130, 112], [220, 239, 114], ratio);
  if (fieldMode === 'bathymetry') return mixColor([117, 167, 204], [10, 55, 94], ratio);
  if (ratio < 0.5) return mixColor([47, 122, 204], [237, 244, 248], ratio / 0.5);
  return mixColor([237, 244, 248], [230, 100, 58], (ratio - 0.5) / 0.5);
}

function legendGradientForMode(fieldMode) {
  if (fieldMode === 'speed') return 'linear-gradient(90deg, #2888ba, #ffe57c)';
  if (fieldMode === 'temperature') return 'linear-gradient(90deg, #2170be, #f67937)';
  if (fieldMode === 'salinity') return 'linear-gradient(90deg, #298270, #dcef72)';
  if (fieldMode === 'bathymetry') return 'linear-gradient(90deg, #75a7cc, #0a375e)';
  return 'linear-gradient(90deg, #2f7acc, #edf4f8 50%, #e6643a)';
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

function rotatedFieldSample(grid, fi, fj) {
  const rows = grid.length;
  const cols = grid[0].length;
  const rotatedI = (rows - 1) - fi;
  const rotatedJ = (cols - 1) - fj;
  return bilinearSample(grid, rotatedI, rotatedJ);
}

function drawScalarField(dataset, subset, summary, fieldMode, range) {
  explorerState.scalarLayer.clearLayers();
  const rows = summary.field.length;
  const cols = summary.field[0]?.length || 0;
  if (!rows || !cols) return;

  const latVals = subset.latIndices.map((index) => dataset.latitude[index]);
  const lonVals = subset.lonIndices.map((index) => dataset.longitude[index]);
  if (latVals.length < 2 || lonVals.length < 2) return;

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      const sourceI = rows - 1 - i;
      const sourceJ = cols - 1 - j;
      const cell = summary.field[sourceI][sourceJ];
      if (!cell.land) continue;

      const latSouth = i > 0 ? (latVals[i - 1] + latVals[i]) / 2 : latVals[i] - (latVals[1] - latVals[0]) / 2;
      const latNorth = i < rows - 1 ? (latVals[i] + latVals[i + 1]) / 2 : latVals[i] + (latVals[rows - 1] - latVals[rows - 2]) / 2;
      const lonWest = j > 0 ? (lonVals[j - 1] + lonVals[j]) / 2 : lonVals[j] - (lonVals[1] - lonVals[0]) / 2;
      const lonEast = j < cols - 1 ? (lonVals[j] + lonVals[j + 1]) / 2 : lonVals[j] + (lonVals[cols - 1] - lonVals[cols - 2]) / 2;

      L.rectangle(
        [
          [Math.min(latSouth, latNorth), Math.min(lonWest, lonEast)],
          [Math.max(latSouth, latNorth), Math.max(lonWest, lonEast)],
        ],
        {
          stroke: false,
          fillColor: 'rgba(214, 201, 170, 0.66)',
          fillOpacity: 0.66,
          interactive: false,
        }
      ).addTo(explorerState.scalarLayer);
    }
  }
}

function drawVectorField(dataset, subset, summary, controls, speedRange, fieldMode, fieldRange) {
  explorerState.vectorLayer.clearLayers();

  if (!explorerState.map) return 0;

  const densityLevel = Number(controls.vectorDensity.value);
  const pixelStepByDensity = { 1: 10, 2: 14, 3: 20, 4: 28 };
  const targetPixelStep = pixelStepByDensity[densityLevel] || 19;

  const rows = summary.field.length;
  const cols = summary.field[0]?.length || 0;
  if (!rows || !cols) return 0;

  const uGrid = summary.field.map((row) => row.map((cell) => cell.meanU));
  const vGrid = summary.field.map((row) => row.map((cell) => cell.meanV));
  const valueGrid = summary.field.map((row) => row.map((cell) => valueForMode(cell, fieldMode)));
  const landGrid = summary.field.map((row) => row.map((cell) => (cell.land ? 1 : 0)));
  const elevationGrid = summary.field.map((row) => row.map((cell) => cell.elevation));

  const map = explorerState.map;
  const topLeft = map.latLngToLayerPoint([subset.latMax, subset.lonMin]);
  const bottomRight = map.latLngToLayerPoint([subset.latMin, subset.lonMax]);
  const mapWidth = Math.max(1, Math.abs(bottomRight.x - topLeft.x));
  const mapHeight = Math.max(1, Math.abs(bottomRight.y - topLeft.y));
  const nx = clamp(Math.round(mapWidth / targetPixelStep), cols, 120);
  const ny = clamp(Math.round(mapHeight / targetPixelStep), rows, 120);

  const speedNorm = Math.max(0.08, speedRange.max);
  const headAngle = (28 * Math.PI) / 180;

  let vectorCount = 0;

  for (let yi = 0; yi < ny; yi += 1) {
      const fi = (yi / Math.max(1, ny - 1)) * Math.max(1, rows - 1);
      const py = topLeft.y + ((yi + 0.5) / ny) * mapHeight;
    for (let xi = 0; xi < nx; xi += 1) {
      const fj = (xi / Math.max(1, nx - 1)) * Math.max(1, cols - 1);
      const landProb = rotatedFieldSample(landGrid, fi, fj);
      const nearestI = clamp(Math.round((rows - 1) - fi), 0, rows - 1);
      const nearestJ = clamp(Math.round((cols - 1) - fj), 0, cols - 1);
      const nearestCell = summary.field[nearestI][nearestJ];
      const elevation = rotatedFieldSample(elevationGrid, fi, fj);
      if (landProb > 0.35 || nearestCell.land || elevation >= 0) continue;

      const meanU = rotatedFieldSample(uGrid, fi, fj);
      const meanV = rotatedFieldSample(vGrid, fi, fj);
      const speed = Math.sqrt(meanU * meanU + meanV * meanV);
      if (speed < 1e-4) continue;
      const fieldValue = rotatedFieldSample(valueGrid, fi, fj);
      const vectorColor = vectorColorForMode(fieldMode, fieldValue, fieldRange);
      const style = {
        color: toRgba(vectorColor, 0.96),
        weight: 1.8,
        opacity: 0.97,
        interactive: false,
        lineCap: 'round',
        lineJoin: 'round',
      };

      const px = topLeft.x + ((xi + 0.5) / nx) * mapWidth;
      const startPoint = L.point(px, py);
      const unitX = meanU / speed;
      const unitY = -meanV / speed;
      const ratio = clamp(speed / speedNorm, 0, 1);
      const shaftLengthPx = lerp(9, 23, ratio);

      const endPoint = L.point(startPoint.x + unitX * shaftLengthPx, startPoint.y + unitY * shaftLengthPx);
      const startLatLon = map.layerPointToLatLng(startPoint);
      const endLatLon = map.layerPointToLatLng(endPoint);
      L.polyline([startLatLon, endLatLon], style).addTo(explorerState.vectorLayer);

      const headLenPx = Math.max(4.5, shaftLengthPx * 0.35);
      const backX = -unitX;
      const backY = -unitY;
      const leftX = backX * Math.cos(headAngle) - backY * Math.sin(headAngle);
      const leftY = backX * Math.sin(headAngle) + backY * Math.cos(headAngle);
      const rightX = backX * Math.cos(-headAngle) - backY * Math.sin(-headAngle);
      const rightY = backX * Math.sin(-headAngle) + backY * Math.cos(-headAngle);

      const leftPoint = L.point(endPoint.x + leftX * headLenPx, endPoint.y + leftY * headLenPx);
      const rightPoint = L.point(endPoint.x + rightX * headLenPx, endPoint.y + rightY * headLenPx);
      L.polyline([endLatLon, map.layerPointToLatLng(leftPoint)], style).addTo(explorerState.vectorLayer);
      L.polyline([endLatLon, map.layerPointToLatLng(rightPoint)], style).addTo(explorerState.vectorLayer);

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
  controls.surfaceLegendSwatch.className = 'legend-swatch';
  controls.surfaceLegendSwatch.style.background = legendGradientForMode(fieldMode);
  controls.surfaceLegendSwatch.style.border = '1px solid rgba(166, 190, 214, 0.9)';
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

  const vectorCount = showVectors ? drawVectorField(dataset, subset, summary, controls, speedRange, fieldMode, fieldRange) : 0;
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
