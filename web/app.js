document.addEventListener('DOMContentLoaded', () => {
  const srcInput = document.getElementById('srcInput');
  const destInput = document.getElementById('destInput');
  const excludeInput = document.getElementById('excludeInput');
  const confidenceRange = document.getElementById('confidenceRange');
  const confValue = document.getElementById('confValue');
  const modelSelect = document.getElementById('modelSelect');
  const batchSizeInput = document.getElementById('batchSizeInput');
  const threadsInput = document.getElementById('threadsInput');
  const executeCheck = document.getElementById('executeCheck');
  const classifyVideosCheck = document.getElementById('classifyVideosCheck');
  const dedupeCheck = document.getElementById('dedupeCheck');
  const preserveFoldersCheck = document.getElementById('preserveFoldersCheck');

  const browseSrcBtn = document.getElementById('browseSrcBtn');
  const browseDestBtn = document.getElementById('browseDestBtn');
  const dryRunBtn = document.getElementById('dryRunBtn');
  const executeBtn = document.getElementById('executeBtn');
  const stopBtn = document.getElementById('stopBtn');

  const statusText = document.getElementById('statusText');
  const progressFill = document.getElementById('progressFill');
  const progressPercent = document.getElementById('progressPercent');
  const consoleBox = document.getElementById('consoleBox');
  const computeDevice = document.getElementById('computeDevice');

  const summaryModal = document.getElementById('summaryModal');
  const modalTitle = document.getElementById('modalTitle');
  const modalStats = document.getElementById('modalStats');
  const proceedExecuteBtn = document.getElementById('proceedExecuteBtn');
  const closeModalBtn = document.getElementById('closeModalBtn');

  // Sync Confidence Range display
  confidenceRange.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Browse Source Folder
  browseSrcBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/browse-src', { method: 'POST' });
      const data = await res.json();
      if (data.path) srcInput.value = data.path;
    } catch (err) {
      appendConsole('error', `[ERROR] Failed to open folder picker: ${err.message}`);
    }
  });

  // Browse Destination Folder
  browseDestBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/browse-dest', { method: 'POST' });
      const data = await res.json();
      if (data.path) destInput.value = data.path;
    } catch (err) {
      appendConsole('error', `[ERROR] Failed to open folder picker: ${err.message}`);
    }
  });

  // Console helper
  function appendConsole(level, message) {
    const line = document.createElement('div');
    line.className = `console-line ${level.toLowerCase()}`;
    line.textContent = message;
    consoleBox.appendChild(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
  }

  // Connect to SSE Stream
  function initSSE() {
    const evtSource = new EventSource('/api/stream');

    evtSource.addEventListener('hardware', (e) => {
      const data = JSON.parse(e.data);
      if (computeDevice && data.device) {
        computeDevice.textContent = data.device;
      }
    });

    evtSource.addEventListener('log', (e) => {
      const data = JSON.parse(e.data);
      appendConsole(data.level || 'info', data.message);
    });

    evtSource.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      if (data.total > 0) {
        const pct = Math.round((data.current / data.total) * 100);
        progressFill.style.width = `${pct}%`;
        progressPercent.textContent = `${pct}%`;
      }
      if (data.status) {
        statusText.textContent = data.status;
      }
    });

    evtSource.addEventListener('done', (e) => {
      const data = JSON.parse(e.data);
      setRunningState(false);
      statusText.textContent = 'Organization Complete';
      showSummary(data.stats, data.dry_run, data.log_file);
    });

    evtSource.onerror = () => {
      statusText.textContent = 'SSE Connection Disconnected... Retrying...';
    };
  }

  initSSE();

  function setRunningState(running) {
    if (dryRunBtn) dryRunBtn.disabled = running;
    if (executeBtn) executeBtn.disabled = running;
    stopBtn.disabled = !running;
    srcInput.disabled = running;
    destInput.disabled = running;
    browseSrcBtn.disabled = running;
    browseDestBtn.disabled = running;
  }

  async function startOrganization(dryRunMode) {
    const actionMode = document.querySelector('input[name="actionMode"]:checked').value;

    const payload = {
      src: srcInput.value,
      dest: destInput.value,
      action: actionMode,
      dry_run: dryRunMode,
      confidence: parseFloat(confidenceRange.value),
      batch_size: parseInt(batchSizeInput.value, 10),
      threads: parseInt(threadsInput.value, 10),
      classify_videos: classifyVideosCheck.checked,
      dedupe: dedupeCheck.checked,
      preserve_folders: preserveFoldersCheck.checked,
      model_name: modelSelect.value,
      exclude_folders: excludeInput.value
    };

    consoleBox.innerHTML = '';
    appendConsole('info', `[INIT] Starting organization (${dryRunMode ? 'DRY RUN PREVIEW' : 'FINAL EXECUTION'})...`);
    progressFill.style.width = '0%';
    progressPercent.textContent = '0%';
    statusText.textContent = dryRunMode ? 'Running Dry Run Preview...' : 'Running Final Execution...';
    setRunningState(true);

    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to start organization');
      }
    } catch (err) {
      appendConsole('error', `[ERROR] ${err.message}`);
      setRunningState(false);
      statusText.textContent = 'Error Starting';
    }
  }

  // Dry Run button handler
  if (dryRunBtn) {
    dryRunBtn.addEventListener('click', () => startOrganization(true));
  }

  // Final Execute button handler
  if (executeBtn) {
    executeBtn.addEventListener('click', () => startOrganization(false));
  }

  // Stop Button handler
  stopBtn.addEventListener('click', async () => {
    try {
      appendConsole('warn', '[WARN] Sending stop request...');
      await fetch('/api/stop', { method: 'POST' });
    } catch (err) {
      appendConsole('error', `[ERROR] Failed to send stop request: ${err.message}`);
    }
  });

  // Modal Summary
  function showSummary(stats, dryRun, logFile) {
    modalTitle.textContent = dryRun ? '🧪 Dry Run Preview Complete' : '🎉 Final Execution Complete';
    let html = `<div><strong>Mode:</strong> ${dryRun ? '<span style="color:#a855f7;">DRY RUN (Preview Only)</span>' : '<span style="color:#00f0ff;">FINAL EXECUTION (' + (stats.action || 'COPY').toUpperCase() + ')</span>'}</div>`;
    html += `<div><strong>Total Candidate Files:</strong> ${stats.total_discovered}</div>`;
    html += `<div><strong>Processed Photos:</strong> ${stats.photos_count}</div>`;
    html += `<div><strong>Processed Videos:</strong> ${stats.videos_count}</div>`;
    html += `<div><strong>Skipped (Done):</strong> ${stats.skipped_count}</div>`;
    html += `<div><strong>Duplicates Identified:</strong> ${stats.duplicates_count}</div>`;
    html += `<div><strong>Errors:</strong> ${stats.error_count}</div><br>`;
    html += `<div><strong>Category Breakdown:</strong></div>`;

    if (stats.categories) {
      for (const [cat, cnt] of Object.entries(stats.categories)) {
        html += `<div>&nbsp;&nbsp;• ${cat}: ${cnt} files</div>`;
      }
    }
    html += `<br><div><strong>Full Log Saved To:</strong><br><small style="word-break: break-all;">${logFile}</small></div>`;

    modalStats.innerHTML = html;

    if (dryRun && proceedExecuteBtn) {
      proceedExecuteBtn.style.display = 'inline-block';
      proceedExecuteBtn.onclick = () => {
        summaryModal.classList.remove('active');
        startOrganization(false);
      };
    } else if (proceedExecuteBtn) {
      proceedExecuteBtn.style.display = 'none';
    }

    summaryModal.classList.add('active');
  }

  closeModalBtn.addEventListener('click', () => {
    summaryModal.classList.remove('active');
  });
});
