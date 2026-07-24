// ── CONFIG ──────────────────────────────────────
const API_URL = "https://somalia-displacement-classifier.onrender.com/"; // swap to Render URL after deploy

// ── CONSTANTS ───────────────────────────────────
const MONTHS = [
  "",
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const CAUSE_TO_TYPE = {
  Conflict: "Conflict",
  Drought: "Disaster",
  Flood: "Disaster",
};

const REGION_COORDS = {
  Banaadir: { lat: 2.04, lon: 45.34 },
  Bay: { lat: 3.11, lon: 43.65 },
  Bakool: { lat: 4.2, lon: 44.1 },
  Bari: { lat: 10.45, lon: 50.0 },
  Gedo: { lat: 3.5, lon: 41.8 },
  Hiiraan: { lat: 4.34, lon: 45.3 },
  "Lower Juba": { lat: 0.33, lon: 42.54 },
  "Middle Juba": { lat: 2.08, lon: 42.54 },
  Mudug: { lat: 6.5, lon: 47.5 },
  Sanaag: { lat: 10.4, lon: 47.4 },
  Other: { lat: 5.15, lon: 46.2 },
};

const COLOR_HEX = {
  low: "#22c55e",
  moderate: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

let selectedModel = "xgb";

// ── MODEL SELECTOR ───────────────────────────────
function selectModel(id) {
  selectedModel = id;
  document
    .querySelectorAll(".model-card")
    .forEach((c) => c.classList.remove("active"));
  document.getElementById(`mc-${id}`).classList.add("active");
}

// ── LOAD METRICS FROM API ────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch(`${API_URL}/metrics`);
    if (!res.ok) return;
    const data = await res.json();
    ["lr", "rf", "xgb"].forEach((key) => {
      if (data[key] && data[key].recall) {
        const el = document.getElementById(`recall-${key}`);
        if (el) el.textContent = (data[key].recall * 100).toFixed(0) + "%";
      }
    });
  } catch (_) {
    // API not running yet — dashes stay
  }
}

// ── PREDICT ──────────────────────────────────────
async function predict() {
  const combined_type = document.getElementById("combined_type").value;
  const region = document.getElementById("region").value;
  const month = parseInt(document.getElementById("month").value);
  const duration = parseFloat(document.getElementById("duration_days").value);

  if (!combined_type || !region || !month || isNaN(duration) || duration < 0) {
    alert("Please fill in all fields before running the prediction.");
    return;
  }

  const displacement_type = CAUSE_TO_TYPE[combined_type] || "Disaster";
  const coords = REGION_COORDS[region] || REGION_COORDS["Other"];

  // UI state — loading
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  document.getElementById("loadingRow").classList.add("on");
  document.getElementById("resultSection").classList.remove("on");
  document.getElementById("placeholderCard").style.display = "none";

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        combined_type,
        displacement_type,
        region,
        month,
        duration_days: duration,
        latitude: coords.lat,
        longitude: coords.lon,
        year: new Date().getFullYear(),
        model: selectedModel,
      }),
    });

    if (!res.ok) throw new Error("API returned error " + res.status);

    const data = await res.json();
    renderResult(data, combined_type, region, duration, month);
  } catch (err) {
    alert(
      "Could not reach the API.\n\n" +
        "Make sure the server is running:\n" +
        "  cd backend\n" +
        "  uvicorn app:app --reload",
    );
    document.getElementById("placeholderCard").style.display = "block";
  } finally {
    btn.disabled = false;
    document.getElementById("loadingRow").classList.remove("on");
  }
}

// ── RENDER RESULT ────────────────────────────────
function renderResult(data, cause, region, duration, month) {
  const section = document.getElementById("resultSection");
  section.classList.add("on");

  // Prediction label
  const predEl = document.getElementById("resultPrediction");
  predEl.textContent = data.prediction;
  predEl.className = `result-prediction pred-${data.color}`;

  // Severity badge
  const badgeEl = document.getElementById("severityBadge");
  badgeEl.textContent = data.severity;
  badgeEl.className = `sev-badge sev-${data.color}`;

  // Confidence bar
  const pct = Math.round(data.probability * 100);
  document.getElementById("confValue").textContent = `${pct}%`;
  const bar = document.getElementById("confBar");
  bar.style.width = `${pct}%`;
  bar.style.background = COLOR_HEX[data.color] || "#f97316";

  // Action
  document.getElementById("actionText").textContent = data.action;

  // Details
  document.getElementById("dCause").textContent = cause;
  document.getElementById("dRegion").textContent = region;
  document.getElementById("dDuration").textContent = `${duration} days`;
  document.getElementById("dMonth").textContent = MONTHS[month] || month;

  // Model used
  document.getElementById("modelUsedLabel").innerHTML =
    `Prediction made using <strong>${data.model_name || selectedModel}</strong>`;

  // Scroll into view
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── INIT ─────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
});
