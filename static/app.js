let currentRunId = "";
let pollTimer = null;
let latestManifestUrl = "";

const csvFiles = document.getElementById("csvFiles");
const uploadSummary = document.getElementById("uploadSummary");
const runBtn = document.getElementById("runBtn");
const manifestBtn = document.getElementById("manifestBtn");
const zipBtn = document.getElementById("zipBtn");

const statusValue = document.getElementById("statusValue");
const modeValue = document.getElementById("modeValue");
const currentStep = document.getElementById("currentStep");
const progressBar = document.getElementById("progressBar");
const progressMeta = document.getElementById("progressMeta");
const logOutput = document.getElementById("logOutput");

const resultsEmpty = document.getElementById("resultsEmpty");
const resultsArea = document.getElementById("resultsArea");
const resultList = document.getElementById("resultList");
const previewFrame = document.getElementById("previewFrame");

function selectedRunMode() {
  const checked = document.querySelector('input[name="runMode"]:checked');
  return checked ? checked.value : "auto";
}

function updateUploadSummary() {
  const files = Array.from(csvFiles.files || []);
  if (!files.length) {
    uploadSummary.textContent = "No files selected yet.";
    return;
  }

  const lines = [];
  lines.push(`${files.length} file(s) selected`);
  lines.push("");
  files.slice(0, 12).forEach((f, idx) => {
    lines.push(`${idx + 1}. ${f.name}`);
  });
  if (files.length > 12) {
    lines.push(`...and ${files.length - 12} more`);
  }

  const autoMode = files.length === 1 ? "single" : "batch";
  lines.push("");
  lines.push(`Auto mode would run as: ${autoMode}`);

  uploadSummary.textContent = lines.join("\n");
}

function setBusy(isBusy) {
  runBtn.disabled = isBusy;
}

function setStatus(status, cls = "") {
  statusValue.textContent = status;
  statusValue.className = "value status-value";
  if (cls) statusValue.classList.add(cls);
}

function resetResults() {
  resultsEmpty.classList.remove("hidden");
  resultsArea.classList.add("hidden");
  resultList.innerHTML = "";
  previewFrame.src = "";
  manifestBtn.disabled = true;
  latestManifestUrl = "";
  zipBtn.href = "#";
  zipBtn.classList.add("disabled-link");
}

function renderResults(resultPages) {
  if (!resultPages || !resultPages.length) {
    resetResults();
    return;
  }

  resultsEmpty.classList.add("hidden");
  resultsArea.classList.remove("hidden");
  resultList.innerHTML = "";

  resultPages.forEach((row, idx) => {
    const btn = document.createElement("button");
    btn.className = "result-item";
    btn.type = "button";
    btn.innerHTML = `
      <div class="result-name">${row.display_name || row.author_label}</div>
      <div class="result-meta">Scopus ID: ${row.scopus_id || "—"}</div>
    `;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".result-item").forEach(el => el.classList.remove("active"));
      btn.classList.add("active");
      previewFrame.src = row.page_url;
    });
    resultList.appendChild(btn);

    if (idx === 0) {
      btn.classList.add("active");
      previewFrame.src = row.page_url;
    }
  });
}

async function startRun() {
  const files = Array.from(csvFiles.files || []);
  if (!files.length) {
    alert("Please upload at least one Scopus CSV file.");
    return;
  }

  resetResults();
  setBusy(true);
  setStatus("Starting", "running");
  modeValue.textContent = "—";
  currentStep.textContent = "Preparing run…";
  progressBar.style.width = "0%";
  progressMeta.textContent = "0%";
  logOutput.textContent = "Submitting files…";

  const fd = new FormData();
  files.forEach(file => fd.append("files", file));
  fd.append("run_mode", selectedRunMode());

  const resp = await fetch("/api/run", {
    method: "POST",
    body: fd,
  });

  const data = await resp.json();
  if (!resp.ok || !data.ok) {
    setBusy(false);
    setStatus("Error", "error");
    logOutput.textContent = data.error || "Unable to start run.";
    return;
  }

  currentRunId = data.run_id;
  pollStatus();
}

async function pollStatus() {
  if (!currentRunId) return;

  const resp = await fetch(`/api/status/${currentRunId}`);
  const data = await resp.json();

  if (!resp.ok || !data.ok) {
    setBusy(false);
    setStatus("Error", "error");
    logOutput.textContent = data.error || "Unable to fetch status.";
    return;
  }

  modeValue.textContent = data.mode_effective || data.mode_requested || "—";
  currentStep.textContent = data.current_step || "Running";
  progressBar.style.width = `${data.progress_pct || 0}%`;
  progressMeta.textContent = `${data.progress_pct || 0}%`;
  logOutput.textContent = (data.logs || []).join("\n\n") || "No logs yet.";

  if (data.status === "running" || data.status === "queued") {
    setStatus("Running", "running");
    pollTimer = setTimeout(pollStatus, 1500);
    return;
  }

  setBusy(false);

  if (data.status === "success") {
    setStatus("Complete", "success");
    renderResults(data.result_pages || []);
    manifestBtn.disabled = false;

    if (data.zip_download_url) {
      zipBtn.href = data.zip_download_url;
      zipBtn.classList.remove("disabled-link");
    }
  } else {
    setStatus("Error", "error");
    currentStep.textContent = data.error || "Pipeline failed.";
  }
}

async function generateManifest() {
  if (!currentRunId) return;

  manifestBtn.disabled = true;
  const resp = await fetch(`/api/manifest/${currentRunId}`, {
    method: "POST",
  });
  const data = await resp.json();

  if (!resp.ok || !data.ok) {
    manifestBtn.disabled = false;
    alert(data.error || "Manifest generation failed.");
    return;
  }

  latestManifestUrl = data.download_url;
  manifestBtn.disabled = false;
  window.location.href = latestManifestUrl;
}

csvFiles.addEventListener("change", updateUploadSummary);
runBtn.addEventListener("click", startRun);
manifestBtn.addEventListener("click", generateManifest);

updateUploadSummary();
resetResults();
