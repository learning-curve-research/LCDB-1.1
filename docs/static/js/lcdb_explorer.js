(function () {
  'use strict';

  // ── 28-color palette (assigned dynamically by selection order) ─
  // Ordered for maximum contrast with fewer selections (first = most distinct)
  const PALETTE = [
    '#4E79A7', // blue
    '#F28E2B', // orange
    '#E15759', // red
    '#76B7B2', // teal
    '#59A14F', // green
    '#B07AA1', // mauve
    '#D4A017', // gold
    '#E8735A', // salmon
    '#9C755F', // brown
    '#6B6B6B', // grey
    '#1565C0', // dark blue
    '#E64A19', // dark orange
    '#6A1B9A', // dark purple
    '#00695C', // dark teal
    '#2E7D32', // dark green
    '#EC407A', // pink
    '#C62828', // dark red
    '#F9A825', // amber
    '#37474F', // blue-grey
    '#283593', // indigo
    '#00ACC1', // cyan
    '#7CB342', // lime-green
    '#FF7043', // deep orange
    '#5C6BC0', // slate blue
    '#26A69A', // medium teal
    '#8D6E63', // warm grey-brown
    '#546E7A', // steel blue-grey
    '#D81B60', // deep pink
  ];

  // Learner indices for quick-select buttons
  const IDX_NEW      = [24, 25, 26, 27];
  const IDX_ENSEMBLE = [20, 21, 22, 24, 25, 26, 27];

  // ── State ───────────────────────────────────────────────────
  let meta = null;
  let datasetNames = {};
  let currentData = null;
  let chart = null;
  let selectedSplits = new Set([1]);  // 0=train 1=val 2=test (multi-select)
  const SPLIT_LABELS = ['Train', 'Val', 'Test'];
  const SPLIT_DASH   = [[6, 4], [], []];  // train=dashed, val=solid, test=solid
  let yAxisAuto = true;
  // SVM_Linear=0, Decision Tree=4, KNN=18, ens.RandomForest=21, CatBoost=24
  let selectedLearners = new Set([0, 4, 18, 21, 24]);

  // ── Helpers ─────────────────────────────────────────────────
  function hexToRgba(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // ── Init ────────────────────────────────────────────────────
  async function init() {
    try {
      [meta, datasetNames] = await Promise.all([
        fetch('data/meta.json').then(r => r.json()),
        fetch('data/names.json').then(r => r.json()).catch(() => ({})),
      ]);
    } catch (e) {
      document.getElementById('chart-loading').innerHTML =
        '<i class="fas fa-exclamation-triangle"></i> Could not load data/meta.json';
      document.getElementById('chart-loading').style.display = 'block';
      return;
    }

    buildDatasetSelector();
    buildLearnerCheckboxes();
    buildLearnerDropdown();
    buildSplitControls();
    buildYAxisControls();
    updateDropdownLabel();
    updateColorDots();
    initChart();

    // Load first dataset
    const firstId = meta.dataset_ids[0];
    document.getElementById('dataset-select').value = firstId;
    loadDataset(firstId);
  }

  // ── Dataset selector ────────────────────────────────────────
  function buildDatasetSelector() {
    const select = document.getElementById('dataset-select');
    meta.dataset_ids.forEach(id => {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `#${id}`;
      select.appendChild(opt);
    });
    select.addEventListener('change', e => loadDataset(parseInt(e.target.value)));

    const search = document.getElementById('dataset-search');
    search.addEventListener('input', () => {
      const q = search.value.trim();
      let firstVisible = null;
      [...select.options].forEach(opt => {
        const show = !q || opt.textContent.includes(q);
        opt.style.display = show ? '' : 'none';
        if (show && !firstVisible) firstVisible = opt;
      });
      if (firstVisible) {
        select.value = firstVisible.value;
        loadDataset(parseInt(firstVisible.value));
      }
    });
  }

  // ── Learner checkboxes (flat, in notebook order) ─────────────
  function buildLearnerCheckboxes() {
    const container = document.getElementById('learner-checkboxes');
    const grid = document.createElement('div');
    grid.className = 'learner-flat-grid';

    meta.learners.forEach((name, idx) => {
      const label = document.createElement('label');
      label.className = 'learner-cb-label';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = idx;
      cb.id = `cb-learner-${idx}`;
      cb.checked = selectedLearners.has(idx);
      cb.addEventListener('change', () => {
        if (cb.checked) selectedLearners.add(idx);
        else selectedLearners.delete(idx);
        updateDropdownLabel();
        updateChart();
      });

      const dot = document.createElement('span');
      dot.className = 'color-dot';
      dot.style.backgroundColor = '#ddd';

      label.appendChild(cb);
      label.appendChild(dot);
      label.appendChild(document.createTextNode(name));
      grid.appendChild(label);
    });

    container.appendChild(grid);
  }

  // ── Split checkbox controls (multi-select) ───────────────────
  function buildSplitControls() {
    document.querySelectorAll('input[name="split"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const v = parseInt(cb.value);
        if (cb.checked) selectedSplits.add(v);
        else selectedSplits.delete(v);
        // keep at least one selected
        if (selectedSplits.size === 0) {
          selectedSplits.add(v);
          cb.checked = true;
        }
        updateChart();
      });
    });
  }

  // ── Y-axis radio controls ─────────────────────────────────────
  function buildYAxisControls() {
    document.querySelectorAll('input[name="yaxis"]').forEach(radio => {
      radio.addEventListener('change', () => {
        yAxisAuto = radio.value === 'auto';
        radio.closest('.split-buttons').querySelectorAll('.split-btn').forEach(l => l.classList.remove('active'));
        radio.closest('.split-btn').classList.add('active');
        updateChart();
      });
    });
  }

  // ── Learner dropdown toggle ───────────────────────────────────
  function buildLearnerDropdown() {
    const btn   = document.getElementById('learner-dropdown-btn');
    const panel = document.getElementById('learner-dropdown-panel');

    btn.addEventListener('click', e => {
      e.stopPropagation();
      panel.classList.toggle('open');
    });

    document.addEventListener('click', e => {
      if (!panel.contains(e.target) && e.target !== btn) {
        panel.classList.remove('open');
      }
    });
  }

  function updateDropdownLabel() {
    document.getElementById('learner-btn-label').textContent =
      `${selectedLearners.size} selected`;
  }

  function syncCheckboxes() {
    document.querySelectorAll('[id^="cb-learner-"]').forEach(cb => {
      cb.checked = selectedLearners.has(parseInt(cb.value));
    });
    updateDropdownLabel();
  }

  // ── Chart init ────────────────────────────────────────────────
  function initChart() {
    const ctx = document.getElementById('lc-chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Training Set Size', font: { size: 13 } },
            ticks: {
              callback: v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v,
              maxTicksLimit: 10,
            },
          },
          y: {
            title: { display: true, text: 'Error Rate', font: { size: 13 } },
            min: 0,
            ticks: {
              callback: v => `${(v * 100).toFixed(1)}%`,
            },
          },
        },
        plugins: {
          legend: {
            position: 'right',
            labels: {
              usePointStyle: true,
              pointStyle: 'line',
              // one entry per learner (color = learner; dash shown on chart, not legend)
              generateLabels: (chart) => {
                const seen = new Set();
                return chart.data.datasets
                  .filter(ds => ds.label)
                  .reduce((items, ds, _, arr) => {
                    if (seen.has(ds.label)) return items;
                    seen.add(ds.label);
                    const allIdx = arr
                      .map((d, i) => d.label === ds.label ? i : -1)
                      .filter(i => i >= 0);
                    const hidden = allIdx.every(i => !chart.isDatasetVisible(i));
                    items.push({
                      text: ds.label,
                      fillStyle: ds.borderColor,
                      strokeStyle: ds.borderColor,
                      lineWidth: 2,
                      lineDash: [],
                      hidden,
                      datasetIndex: allIdx[0],
                      _allIdx: allIdx,
                    });
                    return items;
                  }, []);
              },
              boxWidth: 30,
              font: { size: 11 },
            },
            onClick: (e, item, legend) => {
              const chart = legend.chart;
              const allIdx = item._allIdx || [item.datasetIndex];
              const nowHidden = allIdx.every(i => !chart.isDatasetVisible(i));
              allIdx.forEach(i => chart.setDatasetVisibility(i, !nowHidden));
              chart.update();
            },
          },
          tooltip: {
            filter: item => !!item.dataset.label,
            callbacks: {
              label: ctx => {
                const v = ctx.parsed.y;
                return ` ${ctx.dataset.label}: ${(v * 100).toFixed(2)}%`;
              },
            },
          },
        },
        interaction: { mode: 'index', intersect: false },
      },
    });
  }

  // ── Load a dataset JSON ───────────────────────────────────────
  async function loadDataset(datasetId) {
    const loading = document.getElementById('chart-loading');
    loading.style.display = 'block';
    try {
      currentData = await fetch(`data/er/${datasetId}.json`).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
    } catch (e) {
      loading.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Failed to load data for dataset #${datasetId}`;
      return;
    }
    loading.style.display = 'none';

    // show dataset name above chart
    const name = datasetNames[datasetId];
    const titleEl = document.getElementById('dataset-title');
    if (titleEl) titleEl.textContent = name ? `${name}  (OpenML #${datasetId})` : `OpenML #${datasetId}`;

    updateChart();
  }

  // ── Sync dropdown color dots with current color assignment ─────
  function updateColorDots(sortedLearners) {
    const colorMap = new Map((sortedLearners || [...selectedLearners].sort((a,b)=>a-b))
      .map((idx, pos) => [idx, PALETTE[pos % PALETTE.length]]));
    document.querySelectorAll('[id^="cb-learner-"]').forEach(cb => {
      const idx = parseInt(cb.value);
      const dot = cb.parentElement.querySelector('.color-dot');
      if (dot) dot.style.backgroundColor = colorMap.has(idx) ? colorMap.get(idx) : '#ddd';
    });
  }

  // ── Rebuild chart datasets ─────────────────────────────────────
  function updateChart() {
    if (!currentData || !chart) return;

    const anchors = meta.anchors;
    const datasets = [];
    const multiSplit = selectedSplits.size > 1;
    const sortedLearners = [...selectedLearners].sort((a, b) => a - b);

    // update dropdown color dots to reflect current assignment
    updateColorDots(sortedLearners);

    sortedLearners.forEach((idx, position) => {
      const color = PALETTE[position % PALETTE.length];
      const learnerData = currentData[idx]; // [2][137][3]

      [...selectedSplits].sort().forEach(splitIdx => {
        const meanPts = [], upperPts = [], lowerPts = [];

        anchors.forEach((anchor, ai) => {
          const m = learnerData[0][ai]?.[splitIdx];
          const s = learnerData[1][ai]?.[splitIdx];
          if (m === null || m === undefined || isNaN(m)) return;
          meanPts.push({ x: anchor, y: m });
          if (s !== null && s !== undefined && !isNaN(s)) {
            upperPts.push({ x: anchor, y: m + s });
            lowerPts.push({ x: anchor, y: Math.max(0, m - s) });
          } else {
            upperPts.push({ x: anchor, y: m });
            lowerPts.push({ x: anchor, y: m });
          }
        });

        if (meanPts.length === 0) return;

        const dash      = SPLIT_DASH[splitIdx];
        const bandColor = hexToRgba(color, splitIdx === 0 ? 0.08 : 0.12);
        const label     = meta.learners[idx];   // no split suffix; dash style conveys train vs others

        // upper band
        datasets.push({
          label: '',
          data: upperPts,
          borderColor: 'transparent',
          borderWidth: 0,
          backgroundColor: bandColor,
          pointRadius: 0,
          fill: '+1',
          tension: 0.3,
          order: 2,
        });
        // lower band
        datasets.push({
          label: '',
          data: lowerPts,
          borderColor: 'transparent',
          borderWidth: 0,
          backgroundColor: bandColor,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          order: 2,
        });
        // mean line
        datasets.push({
          label,
          data: meanPts,
          borderColor: color,
          backgroundColor: color,
          borderWidth: splitIdx === 0 ? 1.5 : 1.8,
          borderDash: dash,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          order: 1,
        });
      });
    });

    chart.data.datasets = datasets;
    chart.options.scales.y.max = yAxisAuto ? undefined : 1;
    chart.options.scales.y.min = 0;
    chart.update();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
