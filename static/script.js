/* ═══════════════════════════════════════════════════════════════════════════
   F1 RACE PREDICTOR — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API = '';
let perfChartInstance = null;
let teamChartInstance = null;

// ─── Load Dropdowns on Page Load ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [drivers, tracks, teams, years] = await Promise.all([
      fetch(API + '/api/drivers').then(r => r.json()),
      fetch(API + '/api/tracks').then(r => r.json()),
      fetch(API + '/api/teams').then(r => r.json()),
      fetch(API + '/api/years').then(r => r.json()),
    ]);

    populateSelect('driverSelect', drivers, 'Choose a driver');
    populateSelect('trackSelect', tracks, 'Choose a Grand Prix');
    populateSelect('teamSelect', teams, 'Choose a team');
    populateSelect('yearSelect', years.reverse(), 'Choose a year');
  } catch (e) {
    console.error('Failed to load data:', e);
  }
});

function populateSelect(id, items, placeholder) {
  const sel = document.getElementById(id);
  sel.innerHTML = `<option value="">${placeholder}</option>`;
  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = item;
    sel.appendChild(opt);
  });
}

// ─── Analyze Button ─────────────────────────────────────────────────────────
async function analyze() {
  const driver = document.getElementById('driverSelect').value;
  const track = document.getElementById('trackSelect').value;
  const team = document.getElementById('teamSelect').value;
  const year = document.getElementById('yearSelect').value;

  if (!driver || !track || !team) {
    alert('Please select a Driver, Grand Prix, and Team.');
    return;
  }

  const btn = document.getElementById('btnAnalyze');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ driver, track, team, year: year || 2024 })
    });
    const data = await res.json();
    renderDashboard(data, driver, track, team, year);
  } catch (e) {
    console.error('Prediction failed:', e);
    alert('Prediction request failed. Make sure the server is running.');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ─── Render Dashboard ───────────────────────────────────────────────────────
function renderDashboard(data, driver, track, team, year) {
  const dash = document.getElementById('dashboard');
  dash.classList.add('visible');

  document.getElementById('dashTitle').textContent =
    `${driver} — ${track} ${year || ''} (${team})`;

  // ── Prediction Hero ──
  animateGauge(data.win_probability || 0);
  animateNumber('predPosition', data.predicted_position || '—', '', 'P');
  animateNumber('totalWins', data.driver_total_wins || 0);
  animateNumber('trackWins', data.driver_track_wins || 0);

  // ── Driver Profile (year-aware) ──
  const ds = data.driver_stats;
  const driverList = document.getElementById('driverStatsList');
  if (ds) {
    const timeAdv = data.time_advantage;
    const timeAdvStr = timeAdv != null
      ? (timeAdv > 0 ? `<span style="color:var(--accent-green)">+${timeAdv}s faster</span>` : `<span style="color:var(--accent-red)">${Math.abs(timeAdv)}s slower</span>`)
      : 'N/A';
    driverList.innerHTML = `
      <li><span class="stat-key">Career Span</span><span class="stat-val">${ds.career_span}</span></li>
      <li><span class="stat-key">Wins (all-time)</span><span class="stat-val">${ds.total_wins}</span></li>
      <li><span class="stat-key">Wins in ${year || 'Year'}</span><span class="stat-val" style="color:var(--accent-gold);font-weight:700">${data.driver_wins_year || 0}</span></li>
      <li><span class="stat-key">Wins at Track (up to ${year})</span><span class="stat-val" style="color:var(--accent-blue)">${data.driver_track_wins || 0}</span></li>
      <li><span class="stat-key">Racing Style</span><span class="stat-val"><span class="tag red">${ds.racing_style}</span></span></li>
      <li><span class="stat-key">Vs Field Avg (${year})</span><span class="stat-val">${timeAdvStr}</span></li>
      <li><span class="stat-key">Preferred Region</span><span class="stat-val">${ds.preferred_continent}</span></li>
      <li><span class="stat-key">Teams</span><span class="stat-val">${ds.teams.slice(0,3).join(', ')}${ds.teams.length > 3 ? '…' : ''}</span></li>
    `;
  } else {
    driverList.innerHTML = '<li class="no-data">No data available for this driver</li>';
  }

  // ── Performance Chart ──
  renderPerfChart(ds);

  // ── Racing Style ──
  renderStyle(ds, data);

  // ── Team Stats ──
  renderTeamCard(data.team_stats);

  // ── Rivals ──
  renderRivals(data.rivals, data.driver_total_wins);

  // ── Models ──
  renderModels(data);

  // ── Track History ──
  renderTrackHistory(data.track_history, track);

  // Scroll to dashboard
  setTimeout(() => {
    dash.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// ─── Gauge Animation ────────────────────────────────────────────────────────
function animateGauge(percent) {
  const fill = document.getElementById('gaugeFill');
  const valEl = document.getElementById('gaugeValue');
  const circumference = 2 * Math.PI * 70; // r=70
  const offset = circumference - (percent / 100) * circumference;
  fill.style.strokeDasharray = circumference;
  // Reset first
  fill.style.strokeDashoffset = circumference;
  valEl.textContent = '0%';
  requestAnimationFrame(() => {
    setTimeout(() => {
      fill.style.strokeDashoffset = offset;
      countUp(valEl, 0, Math.round(percent), 1200, '%');
    }, 100);
  });
}

// ─── Animated Counter ───────────────────────────────────────────────────────
function animateNumber(elId, target, suffix = '', prefix = '') {
  const el = document.getElementById(elId);
  const num = parseInt(target);
  if (isNaN(num)) { el.textContent = prefix + target; return; }
  countUp(el, 0, num, 1000, suffix, prefix);
}

function countUp(el, from, to, duration, suffix = '', prefix = '') {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
    el.textContent = prefix + Math.round(from + (to - from) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─── Performance Line Chart ─────────────────────────────────────────────────
function renderPerfChart(ds) {
  if (perfChartInstance) perfChartInstance.destroy();
  const canvas = document.getElementById('perfChart');
  if (!ds || !ds.wins_by_year || ds.wins_by_year.length === 0) {
    canvas.parentElement.innerHTML = '<p class="no-data">No win history data</p>';
    return;
  }

  perfChartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: ds.wins_by_year.map(d => d.year),
      datasets: [{
        label: 'Wins',
        data: ds.wins_by_year.map(d => d.wins),
        borderColor: '#e10600',
        backgroundColor: 'rgba(225,6,0,0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#e10600',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: '#555e76', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.03)' } },
        y: { ticks: { color: '#555e76', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.03)' }, beginAtZero: true }
      }
    }
  });
}

// ─── Racing Style & Strengths ───────────────────────────────────────────────
function renderStyle(ds, data) {
  const el = document.getElementById('styleContent');
  if (!ds) { el.innerHTML = '<p class="no-data">No data</p>'; return; }

  let html = `<div style="margin-bottom:16px">
    <span class="tag gold" style="font-size:0.85rem;padding:8px 20px">${ds.racing_style}</span>
  </div>`;

  // Continent wins
  if (ds.wins_by_continent && ds.wins_by_continent.length > 0) {
    html += '<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin-bottom:8px">Wins by Region</p>';
    const maxCont = Math.max(...ds.wins_by_continent.map(c => c.wins));
    ds.wins_by_continent.sort((a, b) => b.wins - a.wins);
    ds.wins_by_continent.forEach(c => {
      const pct = (c.wins / maxCont) * 100;
      html += `<div class="rival-item">
        <div class="rival-name"><span>${c.continent}</span><span style="color:var(--accent-gold)">${c.wins}</span></div>
        <div class="rival-bar-track"><div class="rival-bar-fill" style="width:0" data-width="${pct}%"></div></div>
      </div>`;
    });
  }

  // Strong tracks
  if (ds.strong_tracks && ds.strong_tracks.length > 0) {
    html += '<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin:16px 0 8px">Strongest Tracks</p>';
    ds.strong_tracks.slice(0, 5).forEach(t => {
      html += `<div style="display:flex;justify-content:space-between;padding:6px 0;font-size:0.85rem;border-bottom:1px solid rgba(255,255,255,0.03)">
        <span>${t.track}</span><span style="color:var(--accent-green);font-weight:600">${t.wins} wins</span>
      </div>`;
    });
  }

  el.innerHTML = html;
  setTimeout(() => animateBars(el), 200);
}

// ─── Team Card ──────────────────────────────────────────────────────────────
function renderTeamCard(ts) {
  const list = document.getElementById('teamStatsList');
  if (teamChartInstance) teamChartInstance.destroy();

  if (!ts) {
    list.innerHTML = '<li class="no-data">No data for this team</li>';
    return;
  }

  list.innerHTML = `
    <li><span class="stat-key">Total Wins</span><span class="stat-val">${ts.total_wins}</span></li>
    <li><span class="stat-key">Active Years</span><span class="stat-val">${ts.career_span}</span></li>
    <li><span class="stat-key">Top Driver</span><span class="stat-val">${ts.top_drivers.length > 0 ? ts.top_drivers[0].driver + ' (' + ts.top_drivers[0].wins + ')' : 'N/A'}</span></li>
    <li><span class="stat-key">Winning Drivers</span><span class="stat-val">${ts.drivers.length}</span></li>
  `;

  // Team wins by year chart
  if (ts.wins_by_year && ts.wins_by_year.length > 0) {
    const canvas = document.getElementById('teamChart');
    teamChartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: ts.wins_by_year.map(d => d.year),
        datasets: [{
          label: 'Wins',
          data: ts.wins_by_year.map(d => d.wins),
          backgroundColor: 'rgba(0,144,255,0.5)',
          borderColor: '#0090ff',
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#555e76', maxTicksLimit: 10 }, grid: { display: false } },
          y: { ticks: { color: '#555e76', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.03)' }, beginAtZero: true }
        }
      }
    });
  }
}

// ─── Rivals ─────────────────────────────────────────────────────────────────
function renderRivals(rivals, driverWins) {
  const el = document.getElementById('rivalsContent');
  if (!rivals || rivals.length === 0) {
    el.innerHTML = '<p class="no-data">No overlapping rivals found</p>';
    return;
  }

  const maxWins = Math.max(driverWins || 1, ...rivals.map(r => r.total_wins));
  let html = '';
  rivals.forEach((r, i) => {
    const pct = (r.total_wins / maxWins) * 100;
    const colors = ['#e10600', '#ffd700', '#0090ff', '#00e676', '#b45aff'];
    html += `<div class="rival-item">
      <div class="rival-name">
        <span style="font-weight:600">${r.name}</span>
        <span style="color:${colors[i % colors.length]};font-weight:600">${r.total_wins} wins</span>
      </div>
      <div class="rival-bar-track">
        <div class="rival-bar-fill" style="width:0;background:${colors[i % colors.length]}" data-width="${pct}%"></div>
      </div>
      <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px">${r.overlap_years} overlapping seasons · ${r.overlap_wins} wins in shared era</div>
    </div>`;
  });

  el.innerHTML = html;
  setTimeout(() => animateBars(el), 400);
}

// ─── Models Comparison (with year-aware 5-factor breakdown) ─────────────────
function renderModels(data) {
  const row = document.getElementById('modelsRow');
  row.innerHTML = `
    <div class="model-box">
      <div class="model-name">Random Forest</div>
      <div class="model-val" style="color:var(--accent-red)">${data.predicted_time_rf || 'N/A'}</div>
    </div>
    <div class="model-box">
      <div class="model-name">Gradient Boosting</div>
      <div class="model-val" style="color:var(--accent-gold)">${data.predicted_time_gb || 'N/A'}</div>
    </div>
    <div class="model-box">
      <div class="model-name">Win Chance</div>
      <div class="model-val" style="color:var(--accent-green)">${data.win_probability || 0}%</div>
    </div>
  `;

  const details = document.getElementById('predDetailsList');
  details.innerHTML = `
    <li style="border-bottom:2px solid rgba(225,6,0,0.15);padding-bottom:12px;margin-bottom:4px">
      <span class="stat-key" style="color:var(--text-primary);font-weight:600">📊 5-Factor Breakdown (${data.selected_year || ''})</span>
      <span class="stat-val"></span>
    </li>
    <li><span class="stat-key">Driver Momentum (3yr)</span><span class="stat-val" style="color:var(--accent-green)">${data.momentum_score || 0}%</span></li>
    <li><span class="stat-key">Track Affinity</span><span class="stat-val">${data.track_win_rate || 0}%</span></li>
    <li><span class="stat-key">Team Strength (${data.selected_year || 'yr'})</span><span class="stat-val" style="color:var(--accent-blue)">${data.team_year_strength || 0}%</span></li>
    <li><span class="stat-key">Driver-Team Synergy</span><span class="stat-val">${data.team_synergy_rate || 0}%</span></li>
    <li><span class="stat-key">Year Form (${data.selected_year || 'yr'})</span><span class="stat-val" style="color:var(--accent-gold)">${data.year_form || 0}%</span></li>
    <li style="border-top:1px solid rgba(255,255,255,0.06);margin-top:4px;padding-top:10px">
      <span class="stat-key">Driver Wins in ${data.selected_year || 'Year'}</span>
      <span class="stat-val">${data.driver_wins_year || 0} / ${data.races_in_year || 0} races</span>
    </li>
    <li><span class="stat-key">Team Wins in ${data.selected_year || 'Year'}</span><span class="stat-val">${data.team_wins_year || 0}</span></li>
    <li><span class="stat-key">Recent Form (3yr wins)</span><span class="stat-val">${data.driver_recent_wins || 0}</span></li>
  `;
}

// ─── Track History Table ────────────────────────────────────────────────────
function renderTrackHistory(history, track) {
  const el = document.getElementById('trackHistContent');
  if (!history || history.length === 0) {
    el.innerHTML = `<p class="no-data">No wins recorded at ${track} for this driver</p>`;
    return;
  }

  let html = `<table class="track-table">
    <thead><tr><th>Year</th><th>Team</th><th>Time</th><th>Laps</th></tr></thead><tbody>`;
  history.forEach(h => {
    html += `<tr><td>${h.year}</td><td>${h.team}</td><td>${h.time}</td><td>${h.laps}</td></tr>`;
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

// ─── Animate Bars Utility ───────────────────────────────────────────────────
function animateBars(container) {
  const fills = container.querySelectorAll('.rival-bar-fill');
  fills.forEach((bar, i) => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width;
    }, i * 80);
  });
}
