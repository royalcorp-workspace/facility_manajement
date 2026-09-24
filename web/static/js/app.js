/**
 * Facility Management — Web Dashboard Client Application (Fase 5).
 * Zero inline script.
 * Features:
 *   - Real-time Clock UTC / WIB
 *   - Dynamic Camera Switcher & Status Badges (GET /api/cameras)
 *   - Telemetry Polling (KPI Metrics & Camera HUDs)
 *   - Incident Audit Trail Table (GET /api/incidents)
 *   - Interactive Incident Resolve (POST /api/incidents/{id}/resolve)
 *   - High-Res 1080p Evidence Snapshot Modal Viewer
 *   - Automatic API Key injection (X-API-Key header & ?api_key= query param)
 */

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initDynamicCameraSwitcher();
  initTelemetryPolling();
  initIncidentsPolling();
  initEvidenceModal();
});

/* ── Helper: API Key & Authenticated Fetch ─────────────────── */
function getApiKey() {
  return (
    window.FACILITY_API_KEY ||
    document.querySelector('meta[name="api-key"]')?.getAttribute("content") ||
    ""
  );
}

function buildAuthUrl(url) {
  const apiKey = getApiKey();
  if (!apiKey || url.includes("api_key=")) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}api_key=${encodeURIComponent(apiKey)}`;
}

async function apiFetch(url, options = {}) {
  const apiKey = getApiKey();
  const headers = new Headers(options.headers || {});
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  const finalUrl = buildAuthUrl(url);
  return fetch(finalUrl, { ...options, headers });
}

/* ── 1. Live Clock ─────────────────────────────────────────── */
function initClock() {
  const clockEl = document.getElementById("facility-clock");
  if (!clockEl) return;

  function updateTime() {
    const now = new Date();
    clockEl.textContent =
      now.toLocaleTimeString("en-GB", { hour12: false }) + " WIB";
  }
  updateTime();
  setInterval(updateTime, 1000);
}

/* ── 2. Dynamic Camera Switcher & Status Pills ─────────────── */
function initDynamicCameraSwitcher() {
  const switcher = document.getElementById("camera-switcher");
  const pillsContainer = document.getElementById("camera-status-pills");
  const videoGrid = document.getElementById("video-grid");
  const streamFeed = document.getElementById("stream-feed");

  if (!switcher) return;

  switcher.addEventListener("change", (e) => {
    const selected = e.target.value;

    // Single fullscreen kiosk stream feed dengan query param api_key
    if (streamFeed && selected) {
      streamFeed.src = buildAuthUrl(`/video_feed/${selected}`);
      const hudCam = document.getElementById("hud-camera-id");
      if (hudCam) hudCam.textContent = selected.toUpperCase();
      pollActiveStreamStatus(selected);
    }

    // Grid multi-camera view (jika elemen tersedia)
    if (videoGrid) {
      const cards = videoGrid.querySelectorAll(".camera-card");
      cards.forEach((card) => {
        const camId = card.getAttribute("data-camera-id");
        if (selected === "all" || camId === selected) {
          card.classList.remove("hidden");
          if (selected !== "all") {
            card.style.gridColumn = "1 / -1";
          } else {
            card.style.gridColumn = "";
          }
        } else {
          card.classList.add("hidden");
        }
      });
    }
  });

  async function pollActiveStreamStatus(camId) {
    if (!camId || camId === "all") return;
    try {
      const resp = await apiFetch(`/api/status/${camId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const hudStatus = document.getElementById("hud-status");
      const hudFrames = document.getElementById("hud-frames");
      const hudMotion = document.getElementById("hud-motion");

      if (hudStatus && data.status) hudStatus.textContent = data.status.toUpperCase();
      if (hudFrames && data.processed_count !== undefined) hudFrames.textContent = `Frames: ${data.processed_count}`;
      if (hudMotion && data.motion_detected_count !== undefined) hudMotion.textContent = `Motion: ${data.motion_detected_count}`;
    } catch (err) {
      console.debug("Failed to poll active stream status:", err);
    }
  }

  // Polling berkala untuk status kamera aktif
  if (switcher.value) {
    pollActiveStreamStatus(switcher.value);
    setInterval(() => {
      if (switcher.value) pollActiveStreamStatus(switcher.value);
    }, 2000);
  }

  async function refreshCameraPills() {
    if (!pillsContainer) return;
    try {
      const resp = await apiFetch("/api/cameras");
      if (!resp.ok) return;
      const cameras = await resp.json();

      pillsContainer.innerHTML = cameras
        .map(
          (c) => `
          <div class="cam-pill ${c.status === "active" ? "active" : "inactive"}" title="${escapeHtml(c.display_name)} (${c.status})">
            <span class="pill-dot"></span>
            <span>${escapeHtml(c.camera_id)}</span>
          </div>
        `
        )
        .join("");
    } catch (err) {
      console.debug("Failed to refresh camera pills:", err);
    }
  }

  refreshCameraPills();
  setInterval(refreshCameraPills, 5000);
}

/* ── 3. Telemetry Polling ──────────────────────────────────── */
function initTelemetryPolling() {
  const totalFramesEl = document.getElementById("kpi-total-frames");
  const activeCamsEl = document.getElementById("kpi-active-cameras");

  async function pollCameras() {
    const cameraCards = document.querySelectorAll(".camera-card");
    let totalFrames = 0;
    let activeCount = 0;

    for (const card of cameraCards) {
      const camId = card.getAttribute("data-camera-id");
      try {
        const resp = await apiFetch(`/api/status/${camId}`);
        if (!resp.ok) continue;
        const data = await resp.json();

        // Update card HUD elements
        const hudStatus = card.querySelector(".hud-status");
        const hudFrames = card.querySelector(".hud-frames");
        const hudMotion = card.querySelector(".hud-motion");

        if (hudStatus) hudStatus.textContent = data.status.toUpperCase();
        if (hudFrames) hudFrames.textContent = `Frames: ${data.processed_count}`;
        if (hudMotion) hudMotion.textContent = `Motion: ${data.motion_detected_count}`;

        totalFrames += data.processed_count || 0;
        if (data.status === "active") activeCount++;
      } catch (err) {
        console.debug(`Failed to fetch status for ${camId}:`, err);
      }
    }

    if (totalFramesEl) totalFramesEl.textContent = totalFrames.toLocaleString();
    if (activeCamsEl) activeCamsEl.textContent = `${activeCount} / ${cameraCards.length}`;
  }

  pollCameras();
  setInterval(pollCameras, 2000);
}

/* ── 4. Incident Audit Trail & Resolve Action ──────────────── */
let _cachedIncidents = [];

function initIncidentsPolling() {
  const tbody = document.getElementById("incident-table-body");
  const kpiIncidentsEl = document.getElementById("kpi-total-incidents");
  if (!tbody) return;

  async function fetchIncidents() {
    try {
      const resp = await apiFetch("/api/incidents");
      if (!resp.ok) return;
      const incidents = await resp.json();
      _cachedIncidents = incidents;

      if (kpiIncidentsEl) {
        kpiIncidentsEl.textContent = incidents.length;
      }

      if (!incidents || incidents.length === 0) {
        tbody.innerHTML =
          '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Belum ada insiden tercatat</td></tr>';
        return;
      }

      const rows = incidents
        .map((ev) => {
          const eventClass = `badge-${ev.event_type || "enter"}`;
          const timeStr = ev.timestamp
            ? new Date(ev.timestamp).toLocaleTimeString()
            : "-";
          const isResolved = Boolean(ev.is_resolved);
          const statusBadge = isResolved
            ? '<span class="badge-status badge-resolved">RESOLVED</span>'
            : '<span class="badge-status badge-unresolved">UNRESOLVED</span>';

          const hasSnapshot = Boolean(ev.snapshot_path);

          const evidenceBtn = hasSnapshot
            ? `<button class="btn-action btn-evidence" data-incident-id="${ev.id}" title="Lihat Snapshot 1080p">Bukti</button>`
            : '<span style="color: var(--text-muted); font-size: 11px;">-</span>';

          const resolveBtn = isResolved
            ? '<button class="btn-action btn-resolve" disabled title="Telah diselesaikan">Selesai ✓</button>'
            : `<button class="btn-action btn-resolve" data-resolve-id="${ev.id}" title="Tandai telah ditangani">Resolve</button>`;

          return `
            <tr data-row-id="${ev.id}">
              <td>${timeStr}</td>
              <td><strong>${escapeHtml(ev.camera_id)}</strong></td>
              <td>${escapeHtml(ev.zone_id)}</td>
              <td><span class="badge-event ${eventClass}">${escapeHtml(ev.event_type)}</span></td>
              <td>#${ev.track_id ?? "-"} (${escapeHtml(ev.class_label || "objek")})</td>
              <td>${escapeHtml(ev.notes || "-")}</td>
              <td>${statusBadge}</td>
              <td>
                <div class="action-group">
                  ${evidenceBtn}
                  ${resolveBtn}
                </div>
              </td>
            </tr>
          `;
        })
        .join("");

      tbody.innerHTML = rows;
    } catch (err) {
      console.debug("Failed to fetch incidents:", err);
    }
  }

  // Delegated Click Event untuk Tabel (Bukti & Resolve)
  tbody.addEventListener("click", async (e) => {
    const target = e.target.closest("button");
    if (!target) return;

    // Aksi 1: Klik Bukti -> Buka Modal
    if (target.classList.contains("btn-evidence")) {
      const incId = parseInt(target.getAttribute("data-incident-id"), 10);
      const incident = _cachedIncidents.find((i) => i.id === incId);
      if (incident) {
        openEvidenceModal(incident);
      }
      return;
    }

    // Aksi 2: Klik Resolve -> Kirim POST /api/incidents/{id}/resolve
    if (target.classList.contains("btn-resolve")) {
      const incId = target.getAttribute("data-resolve-id");
      if (!incId) return;

      target.disabled = true;
      target.textContent = "Menyimpan...";

      try {
        const resp = await apiFetch(`/api/incidents/${incId}/resolve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });

        if (resp.ok) {
          target.textContent = "Selesai ✓";
          const row = target.closest("tr");
          if (row) {
            const badgeCol = row.querySelector(".badge-status");
            if (badgeCol) {
              badgeCol.className = "badge-status badge-resolved";
              badgeCol.textContent = "RESOLVED";
            }
          }
          fetchIncidents();
        } else {
          target.disabled = false;
          target.textContent = "Gagal (Coba Lagi)";
        }
      } catch (err) {
        console.error("Gagal melakukan resolve insiden:", err);
        target.disabled = false;
        target.textContent = "Error";
      }
    }
  });

  fetchIncidents();
  setInterval(fetchIncidents, 3000);
}

/* ── 5. Evidence Snapshot Modal Viewer ─────────────────────── */
function initEvidenceModal() {
  const modal = document.getElementById("evidence-modal");
  const closeBtn = document.getElementById("modal-close-btn");

  if (!modal) return;

  function closeModal() {
    modal.classList.add("hidden");
    const modalImg = document.getElementById("modal-image");
    if (modalImg) modalImg.src = "";
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", closeModal);
  }

  // Tutup jika klik backdrop di luar card
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  // Tutup jika menekan tombol Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      closeModal();
    }
  });
}

function openEvidenceModal(ev) {
  const modal = document.getElementById("evidence-modal");
  const modalImg = document.getElementById("modal-image");
  const modalTitle = document.getElementById("modal-title");
  const modalLoading = document.getElementById("modal-img-loading");
  const metaGrid = document.getElementById("modal-meta-grid");

  if (!modal || !modalImg || !metaGrid) return;

  const fileName = ev.snapshot_path ? ev.snapshot_path.split(/[\\\\/]/).pop() : "";
  const snapUrl = buildAuthUrl(`/snapshots/${ev.camera_id}/${fileName}`);

  modalTitle.textContent = `Insiden #${ev.id} — ${ev.event_type.toUpperCase()} [${ev.camera_id}]`;

  if (modalLoading) modalLoading.style.display = "block";
  modalImg.style.opacity = "0";

  modalImg.onload = () => {
    if (modalLoading) modalLoading.style.display = "none";
    modalImg.style.opacity = "1";
    modalImg.style.transition = "opacity 0.2s ease";
  };

  modalImg.onerror = () => {
    if (modalLoading) modalLoading.textContent = "Gagal memuat gambar snapshot";
  };

  modalImg.src = snapUrl;

  const isResolved = Boolean(ev.is_resolved);
  const timeFormatted = ev.timestamp ? new Date(ev.timestamp).toLocaleString("id-ID") : "-";
  const resolvedFormatted = ev.resolved_time ? new Date(ev.resolved_time).toLocaleString("id-ID") : "-";

  metaGrid.innerHTML = `
    <div class="meta-card">
      <span class="meta-label">Kamera</span>
      <span class="meta-value">${escapeHtml(ev.camera_id)}</span>
    </div>
    <div class="meta-card">
      <span class="meta-label">Zona ROI</span>
      <span class="meta-value">${escapeHtml(ev.zone_id)}</span>
    </div>
    <div class="meta-card">
      <span class="meta-label">Track ID</span>
      <span class="meta-value">#${ev.track_id ?? "-"} (${escapeHtml(ev.class_label || "objek")})</span>
    </div>
    <div class="meta-card">
      <span class="meta-label">Tipe Event</span>
      <span class="meta-value">${escapeHtml(ev.event_type.toUpperCase())}</span>
    </div>
    <div class="meta-card">
      <span class="meta-label">Waktu Kejadian</span>
      <span class="meta-value">${timeFormatted}</span>
    </div>
    <div class="meta-card">
      <span class="meta-label">Status Penanganan</span>
      <span class="meta-value" style="color: ${isResolved ? "var(--accent-emerald)" : "var(--accent-amber)"};">
        ${isResolved ? "RESOLVED" : "UNRESOLVED"}
      </span>
    </div>
    <div class="meta-card" style="grid-column: 1 / -1;">
      <span class="meta-label">Catatan Insiden & Dwell</span>
      <span class="meta-value">${escapeHtml(ev.notes || "-")}${isResolved ? ` (Selesai pada: ${resolvedFormatted})` : ""}</span>
    </div>
  `;

  modal.classList.remove("hidden");
}

/* ── Helper: Sanitasi HTML ─────────────────────────────────── */
function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
