let currentRunId = "";
let pollTimer = null;
let latestManifestUrl = "";

const csvFiles = document.getElementById("csvFiles");
const uploadSummary = document.getElementById("uploadSummary");
const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const manifestBtn = document.getElementById("manifestBtn");
const zipBtn = document.getElementById("zipBtn");

const firstNameInput = document.getElementById("firstName");
const lastNameInput = document.getElementById("lastName");
const scopusIdInput = document.getElementById("scopusId");

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

const shareMessage = document.getElementById("shareMessage");

function selectedRunMode() {
  const checked = document.querySelector('input[name="runMode"]:checked');
  return checked ? checked.value : "auto";
}

function updateUploadSummary() {
  const files = Array.from(csvFiles.files || []);
  const first = firstNameInput.value.trim();
  const last = lastNameInput.value.trim();
  const scopus = scopusIdInput.value.trim();

  const lines = [];

  if (first || last || scopus) {
    lines.push(`Author: ${first} ${last} (${scopus || "No ID"})`);
    lines.push("");
  }

  if (!files.length) {
    lines.push("No files selected yet.");
    uploadSummary.textContent = lines.join("\n");
    return;
  }

  lines.push(`${files.length} file(s) selected`);
  lines.push("");

  files.slice(0, 10).forEach((f, idx) => {
    lines.push(`${idx + 1}. ${f.name}`);
  });

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
  zipBtn.href = "#";
  zipBtn.classList.add("disabled-link");
  shareMessage.classList.add("hidden");
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
    const wrapper = document.createElement("div");

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

    const shareBtn = document.createElement("button");
    shareBtn.className = "secondary-btn";
    shareBtn.textContent = "Share to Social";

    shareBtn.addEventListener("click", async () => {
      shareMessage.classList.remove("hidden");

      // placeholder (backend PNG route will plug in here)
      const link = document.createElement("a");
      link.textContent = "Download PNG (coming soon)";
      link.href = "#";
      link.className = "primary-btn";

      shareMessage.innerHTML = `
        <p><strong>PNG download is available below for you to share on social media of your choice!</strong></p>
      `;
      shareMessage.appendChild(link);
    });

    wrapper.appendChild(btn);
    wrapper.appendChild(shareBtn);
    resultList.appendChild(wrapper);

    if (idx === 0) {
      btn.classList.add("active");
      previewFrame.src = row.page_url;
    }
  });
}

async function startRun() {
  const files = Array.from(csvFiles.files || []);
  const first = firstNameInput.value.trim();
  const last = lastNameInput.value.trim();
  const scopus = scopusIdInput.value.trim();

  if (!files.length) {
    alert("Please upload a CSV file.");
    return;
  }

  if (files.length === 1 && (!first || !last || !scopus)) {
    alert("First name, last name, and Scopus ID are required for single runs.");
    return;
  }

  resetResults();
  setBusy(true);
  setStatus("Starting", "running");

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  fd.append("run_mode", selectedRunMode());
  fd.append("first_name", first);
  fd.append("last_name", last);
  fd.append("scopus_id", scopus);

  const resp = await fetch("/api/run", {
    method: "POST",
    body: fd,
  });

  const data = await resp.json();

  if (!resp.ok || !data.ok) {
    setBusy(false);
    setStatus("Error", "error");
    logOutput.textContent = data.error || "Failed to start run.";
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
    return;
  }

  modeValue.textContent = data.mode_effective || data.mode_requested || "—";
  currentStep.textContent = data.current_step || "Running";
  progressBar.style.width = `${data.progress_pct || 0}%`;
  progressMeta.textContent = `${data.progress_pct || 0}%`;
  logOutput.textContent = (data.logs || []).join("\n\n");

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
  }
}

async function stopRun() {
  if (!currentRunId) return;

  try {
    await fetch(`/api/stop/${currentRunId}`, { method: "POST" });
    alert("Run stopped.");
    location.reload();
  } catch (err) {
    alert("Stop failed (route may not exist yet).");
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
    alert(data.error || "Manifest failed.");
    return;
  }

  latestManifestUrl = data.download_url;
  manifestBtn.disabled = false;
  window.location.href = latestManifestUrl;
}

csvFiles.addEventListener("change", updateUploadSummary);
firstNameInput.addEventListener("input", updateUploadSummary);
lastNameInput.addEventListener("input", updateUploadSummary);
scopusIdInput.addEventListener("input", updateUploadSummary);

runBtn.addEventListener("click", startRun);
stopBtn.addEventListener("click", stopRun);
manifestBtn.addEventListener("click", generateManifest);

updateUploadSummary();
resetResults();