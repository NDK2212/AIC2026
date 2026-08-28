/**
 * AIC 2026 - Interactive Retrieval SPA Application
 */

// Application State
const state = {
  currentTask: "kis",
  currentQuery: "",
  results: [],
  plan: null,
  trace: null,
  config: null,
  apiKeys: [],
  vqaOverrides: {}, // key: "video_id_frame_id" -> custom answer string
  promotedTop1Key: null,
  activeNeighborTarget: null, // { video_id, frame_id, rank, index }
  selectedNeighborFrame: null, // { video_id, frame_id }
};

// Preset queries for quick testing
const PRESETS = {
  kis: [
    "tìm cảnh người phụ nữ mặc áo dài đỏ cầm hoa sen",
    "xe buýt màu xanh lá cây có chữ Bến Thành",
    "phóng viên nói về dự báo thời tiết tối nay",
    "cảnh sát giao thông đang xử phạt người vi phạm",
  ],
  vqa: [
    "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?",
    "Người đàn ông trong video mặc áo màu gì?",
    "Biển hiệu cửa hàng có chữ gì?",
    "Có bao nhiêu chiếc xe ô tô màu trắng đang đỗ trên đường?",
  ],
  trake: [
    "Chuỗi gồm (1) mở cửa xe (2) bước vào ghế lái (3) khởi động xe",
    "Chuỗi gồm (1) giao bóng (2) đỡ bóng (3) đập bóng qua lưới",
    "Chuỗi gồm (1) chuẩn bị nguyên liệu (2) xào nấu trên bếp (3) bày thức ăn ra đĩa",
  ],
};

// DOM Elements
const el = {
  taskTabs: document.querySelectorAll(".task-tab"),
  currentTaskBadge: document.getElementById("currentTaskBadge"),
  presetQueries: document.getElementById("presetQueries"),
  inputQuery: document.getElementById("inputQuery"),
  btnSearch: document.getElementById("btnSearch"),
  chkUseCache: document.getElementById("chkUseCache"),

  // Pipeline Monitor
  pipelineMonitor: document.getElementById("pipelineMonitor"),
  monitorElapsed: document.getElementById("monitorElapsed"),
  stepDecomp: document.getElementById("step-decomp"),
  stepRetrieval: document.getElementById("step-retrieval"),
  stepRerank: document.getElementById("step-rerank"),
  stepPost: document.getElementById("step-post"),
  stepDetailDecomp: document.getElementById("stepDetailDecomp"),
  stepDetailRetrieval: document.getElementById("stepDetailRetrieval"),
  stepDetailRerank: document.getElementById("stepDetailRerank"),
  stepDetailPost: document.getElementById("stepDetailPost"),
  decompDetailsCard: document.getElementById("decompDetailsCard"),
  decValVisual: document.getElementById("decValVisual"),
  decValOcr: document.getElementById("decValOcr"),
  decValAsr: document.getElementById("decValAsr"),
  decValWeights: document.getElementById("decValWeights"),

  // Results
  resultsToolbar: document.getElementById("resultsToolbar"),
  resultsCount: document.getElementById("resultsCount"),
  pinnedNotice: document.getElementById("pinnedNotice"),
  btnExportCsv: document.getElementById("btnExportCsv"),
  resultsGallery: document.getElementById("resultsGallery"),

  // Settings Drawer
  settingsDrawer: document.getElementById("settingsDrawer"),
  btnToggleSettings: document.getElementById("btnToggleSettings"),
  btnCloseSettings: document.getElementById("btnCloseSettings"),
  btnApplySettings: document.getElementById("btnApplySettings"),
  btnResetDefaults: document.getElementById("btnResetDefaults"),
  dTabs: document.querySelectorAll(".d-tab"),
  dPanes: document.querySelectorAll(".d-tab-pane"),

  // Key Pool
  inputApiKeys: document.getElementById("inputApiKeys"),
  btnSaveKeys: document.getElementById("btnSaveKeys"),
  keySaveStatus: document.getElementById("keySaveStatus"),
  activeKeyChips: document.getElementById("activeKeyChips"),
  keysCountLabel: document.getElementById("keysCountLabel"),

  // Neighbor Modal
  neighborModal: document.getElementById("neighborModal"),
  modalVideoTitle: document.getElementById("modalVideoTitle"),
  neighborFilmstrip: document.getElementById("neighborFilmstrip"),
  metaFrameTitle: document.getElementById("metaFrameTitle"),
  metaFrameBody: document.getElementById("metaFrameBody"),
  btnCloseNeighborModal: document.getElementById("btnCloseNeighborModal"),
  btnCloseNeighborModalFooter: document.getElementById("btnCloseNeighborModalFooter"),
  btnConfirmSelectNeighbor: document.getElementById("btnConfirmSelectNeighbor"),

  // Sliders
  rngWeightOcr: document.getElementById("rngWeightOcr"),
  valWeightOcr: document.getElementById("valWeightOcr"),
  rngWeightAsr: document.getElementById("rngWeightAsr"),
  valWeightAsr: document.getElementById("valWeightAsr"),
  rngWeightDesc: document.getElementById("rngWeightDesc"),
  valWeightDesc: document.getElementById("valWeightDesc"),
  rngWeightVis: document.getElementById("rngWeightVis"),
  valWeightVis: document.getElementById("valWeightVis"),
  rngBlipTopN: document.getElementById("rngBlipTopN"),
  valBlipTopN: document.getElementById("valBlipTopN"),
  rngBlipWeight: document.getElementById("rngBlipWeight"),
  valBlipWeight: document.getElementById("valBlipWeight"),
  rngBgeTopN: document.getElementById("rngBgeTopN"),
  valBgeTopN: document.getElementById("valBgeTopN"),
  rngBgeWeight: document.getElementById("rngBgeWeight"),
  valBgeWeight: document.getElementById("valBgeWeight"),
  rngVlmTopN: document.getElementById("rngVlmTopN"),
  valVlmTopN: document.getElementById("valVlmTopN"),

  // VQA Radios
  lblVqaTextMode: document.getElementById("lblVqaTextMode"),
  lblVqaVisionMode: document.getElementById("lblVqaVisionMode"),
  vqaRadioButtons: document.querySelectorAll("input[name='vqaModeRadio']"),
};

// ==========================================================================
// INITIALIZATION & LIFECYCLE
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadServerStatus();
  loadConfigDefaults();
  renderPresets(state.currentTask);
});

// Setup All UI Event Handlers
function setupEventListeners() {
  // Task switching
  el.taskTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const task = tab.getAttribute("data-task");
      switchTask(task);
    });
  });

  // Query Search
  el.btnSearch.addEventListener("click", executeSearch);
  el.inputQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") executeSearch();
  });

  // Settings Drawer Toggle
  el.btnToggleSettings.addEventListener("click", () => el.settingsDrawer.classList.toggle("open"));
  el.btnCloseSettings.addEventListener("click", () => el.settingsDrawer.classList.remove("open"));
  el.btnApplySettings.addEventListener("click", () => el.settingsDrawer.classList.remove("open"));

  // Drawer Tabs
  el.dTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      el.dTabs.forEach((t) => t.classList.remove("active"));
      el.dPanes.forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const targetPane = document.getElementById(`pane-${tab.getAttribute("data-dtab")}`);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Key Pool Save
  el.btnSaveKeys.addEventListener("click", saveApiKeys);

  // Reset Defaults
  el.btnResetDefaults.addEventListener("click", resetConfigDefaults);

  // Export CSV
  el.btnExportCsv.addEventListener("click", exportSubmissionCsv);

  // Neighbor Modal Controls
  el.btnCloseNeighborModal.addEventListener("click", closeNeighborModal);
  el.btnCloseNeighborModalFooter.addEventListener("click", closeNeighborModal);
  el.btnConfirmSelectNeighbor.addEventListener("click", applySelectedNeighbor);

  // Keyboard navigation for modal filmstrip
  window.addEventListener("keydown", (e) => {
    if (el.neighborModal.style.display !== "none") {
      if (e.key === "Escape") closeNeighborModal();
      if (e.key === "ArrowLeft") navigateNeighborFilmstrip(-1);
      if (e.key === "ArrowRight") navigateNeighborFilmstrip(1);
    }
  });

  // Sync Sliders with badges
  bindSlider("rngWeightOcr", "valWeightOcr");
  bindSlider("rngWeightAsr", "valWeightAsr");
  bindSlider("rngWeightDesc", "valWeightDesc");
  bindSlider("rngWeightVis", "valWeightVis");
  bindSlider("rngBlipTopN", "valBlipTopN");
  bindSlider("rngBlipWeight", "valBlipWeight");
  bindSlider("rngBgeTopN", "valBgeTopN");
  bindSlider("rngBgeWeight", "valBgeWeight");
  bindSlider("rngVlmTopN", "valVlmTopN");

  // VQA Mode Radio Cards
  el.vqaRadioButtons.forEach((rb) => {
    rb.addEventListener("change", () => {
      if (rb.value === "text_only") {
        el.lblVqaTextMode.classList.add("active");
        el.lblVqaVisionMode.classList.remove("active");
      } else {
        el.lblVqaVisionMode.classList.add("active");
        el.lblVqaTextMode.classList.remove("active");
      }
    });
  });
}

function bindSlider(sliderId, badgeId) {
  const s = document.getElementById(sliderId);
  const b = document.getElementById(badgeId);
  if (s && b) {
    s.addEventListener("input", () => {
      b.textContent = s.value;
    });
  }
}

// ==========================================================================
// API CALLS: STATUS & CONFIG
// ==========================================================================
async function loadServerStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.keys_count !== undefined) {
      el.keysCountLabel.textContent = `${data.keys_count} Key(s)`;
    }
    renderKeyChips(data.keys_masked || []);
  } catch (err) {
    console.warn("Could not load status:", err);
  }
}

async function loadConfigDefaults() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    state.config = cfg;
    applyConfigToUI(cfg);
  } catch (err) {
    console.warn("Could not load config:", err);
  }
}

function applyConfigToUI(cfg) {
  if (!cfg) return;
  if (cfg.retrieval_paths) {
    setVal("rngWeightOcr", "valWeightOcr", cfg.retrieval_paths.ocr.weight);
    setVal("rngWeightAsr", "valWeightAsr", cfg.retrieval_paths.asr.weight);
    setVal("rngWeightDesc", "valWeightDesc", cfg.retrieval_paths.description.weight);
    setVal("rngWeightVis", "valWeightVis", cfg.retrieval_paths.visual.weight);
  }
  if (cfg.rerank) {
    setVal("rngBlipTopN", "valBlipTopN", cfg.rerank.blip2.top_n);
    setVal("rngBlipWeight", "valBlipWeight", cfg.rerank.blip2.weight);
    setVal("rngBgeTopN", "valBgeTopN", cfg.rerank.bge.top_n);
    setVal("rngBgeWeight", "valBgeWeight", cfg.rerank.bge.weight);
  }
  if (cfg.vqa) {
    setVal("rngVlmTopN", "valVlmTopN", cfg.vqa.vlm_top_n);
  }
}

function setVal(sliderId, badgeId, val) {
  const s = document.getElementById(sliderId);
  const b = document.getElementById(badgeId);
  if (s && b) {
    s.value = val;
    b.textContent = val;
  }
}

function resetConfigDefaults() {
  if (state.config) {
    applyConfigToUI(state.config);
    alert("Đã khôi phục toàn bộ cấu hình về mặc định!");
  }
}

// ==========================================================================
// API KEY POOL MANAGEMENT
// ==========================================================================
async function saveApiKeys() {
  const text = el.inputApiKeys.value.trim();
  if (!text) {
    alert("Vui lòng nhập ít nhất 1 API key");
    return;
  }
  try {
    el.keySaveStatus.textContent = "Đang lưu...";
    const res = await fetch("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys: text }),
    });
    const data = await res.json();
    if (data.success) {
      el.keySaveStatus.textContent = `✅ Đã lưu ${data.count} keys thành công!`;
      el.keysCountLabel.textContent = `${data.count} Key(s)`;
      loadServerStatus();
      setTimeout(() => {
        el.keySaveStatus.textContent = "";
      }, 3000);
    } else {
      el.keySaveStatus.textContent = `❌ Lỗi: ${data.error}`;
    }
  } catch (err) {
    el.keySaveStatus.textContent = `❌ Lỗi kết nối: ${err}`;
  }
}

function renderKeyChips(maskedList) {
  el.activeKeyChips.innerHTML = "";
  maskedList.forEach((k) => {
    const chip = document.createElement("span");
    chip.className = "key-chip";
    chip.textContent = `🔑 ${k}`;
    el.activeKeyChips.appendChild(chip);
  });
}

// ==========================================================================
// TASK & PRESET SWITCHING
// ==========================================================================
function switchTask(task) {
  state.currentTask = task;
  el.taskTabs.forEach((tab) => {
    if (tab.getAttribute("data-task") === task) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  el.currentTaskBadge.textContent = task.toUpperCase();
  renderPresets(task);

  // Clear previous results & reset view
  state.results = [];
  state.promotedTop1Key = null;
  state.vqaOverrides = {};
  el.pinnedNotice.style.display = "none";
  el.pipelineMonitor.style.display = "none";
  el.resultsToolbar.style.display = "none";

  el.resultsGallery.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">${task === "kis" ? "🎯" : task === "vqa" ? "💬" : "⏱️"}</div>
      <div class="empty-title">Tác vụ ${task.toUpperCase()} sẵn sàng</div>
      <div class="empty-desc">Nhập câu hỏi hoặc chọn mẫu bên trên rồi nhấn <strong>TÌM KIẾM</strong>.</div>
    </div>
  `;
}

function renderPresets(task) {
  el.presetQueries.innerHTML = "";
  const list = PRESETS[task] || [];
  list.forEach((query) => {
    const chip = document.createElement("button");
    chip.className = "preset-chip";
    chip.textContent = query.length > 45 ? query.substring(0, 45) + "..." : query;
    chip.title = query;
    chip.addEventListener("click", () => {
      el.inputQuery.value = query;
      executeSearch();
    });
    el.presetQueries.appendChild(chip);
  });
}

// ==========================================================================
// SEARCH EXECUTION & REAL-TIME PIPELINE MONITOR
// ==========================================================================
async function executeSearch() {
  const query = el.inputQuery.value.trim();
  if (!query) {
    alert("Vui lòng nhập câu truy vấn!");
    return;
  }

  state.currentQuery = query;
  state.promotedTop1Key = null;
  state.vqaOverrides = {};
  el.pinnedNotice.style.display = "none";

  // Gather current parameters from UI
  const params = collectCurrentParams();
  const vqaMode = document.querySelector("input[name='vqaModeRadio']:checked").value;
  const useCache = el.chkUseCache.checked;

  // Show and reset pipeline monitor
  el.pipelineMonitor.style.display = "flex";
  el.resultsToolbar.style.display = "none";
  el.decompDetailsCard.style.display = "none";

  resetPipelineSteps();
  setStepRunning(el.stepDecomp, "Đang phân rã câu hỏi với LLM...");

  const startTime = performance.now();
  const timer = setInterval(() => {
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    el.monitorElapsed.textContent = `Thời gian: ${elapsed}s`;
  }, 100);

  // Disable search button during execution
  el.btnSearch.disabled = true;
  el.btnSearch.style.opacity = "0.7";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: state.currentTask,
        query: query,
        params: params,
        vqa_mode: vqaMode,
        use_cache: useCache,
      }),
    });

    const data = await res.json();
    clearInterval(timer);
    el.btnSearch.disabled = false;
    el.btnSearch.style.opacity = "1";

    if (data.error) {
      alert(`Lỗi thực thi: ${data.error}`);
      return;
    }

    // Complete all monitor steps
    completePipelineSteps(data);

    // Save and render results
    if (state.currentTask === "trake") {
      state.results = data.sequences || [];
      state.plan = data.plan || null;
      renderTrakeResults(state.results, data.plan);
    } else if (state.currentTask === "vqa") {
      state.results = data.candidates || [];
      renderVqaResults(state.results);
    } else {
      state.results = data.candidates || [];
      renderKisResults(state.results);
    }

    el.resultsToolbar.style.display = "flex";
    el.resultsCount.textContent = `${state.results.length} Kết quả (${data.elapsed_total_s}s)`;
  } catch (err) {
    clearInterval(timer);
    el.btnSearch.disabled = false;
    el.btnSearch.style.opacity = "1";
    alert(`Lỗi mạng / máy chủ: ${err}`);
  }
}

function collectCurrentParams() {
  return {
    retrieval_paths: {
      ocr: {
        enabled: document.getElementById("chkPathOcr").checked,
        weight: parseFloat(el.rngWeightOcr.value),
      },
      asr: {
        enabled: document.getElementById("chkPathAsr").checked,
        weight: parseFloat(el.rngWeightAsr.value),
      },
      description: {
        enabled: document.getElementById("chkPathDesc").checked,
        weight: parseFloat(el.rngWeightDesc.value),
      },
      visual: {
        enabled: document.getElementById("chkPathVis").checked,
        weight: parseFloat(el.rngWeightVis.value),
      },
    },
    rerank: {
      blip2: {
        enabled: document.getElementById("chkBlipRerank").checked,
        top_n: parseInt(el.rngBlipTopN.value, 10),
        weight: parseFloat(el.rngBlipWeight.value),
      },
      bge: {
        enabled: document.getElementById("chkBgeRerank").checked,
        top_n: parseInt(el.rngBgeTopN.value, 10),
        weight: parseFloat(el.rngBgeWeight.value),
      },
    },
    vqa: {
      vlm_top_n: parseInt(el.rngVlmTopN.value, 10),
      propagate: document.getElementById("chkVqaPropagate").checked,
    },
    trake: {
      per_step_topk: parseInt(document.getElementById("numTrakeTopk").value, 10),
      coverage_bonus: parseFloat(document.getElementById("numCoverageBonus").value),
      miss_penalty: parseFloat(document.getElementById("numMissPenalty").value),
    },
  };
}

function resetPipelineSteps() {
  [el.stepDecomp, el.stepRetrieval, el.stepRerank, el.stepPost].forEach((step) => {
    step.className = "p-step";
  });
}

function setStepRunning(stepEl, detailText) {
  stepEl.className = "p-step running";
  const d = stepEl.querySelector(".step-detail");
  if (d) d.textContent = detailText;
}

function setStepDone(stepEl, detailText) {
  stepEl.className = "p-step done";
  const d = stepEl.querySelector(".step-detail");
  if (d) d.textContent = detailText;
}

function completePipelineSteps(data) {
  setStepDone(el.stepDecomp, "Hoàn tất phân tích câu hỏi");
  setStepDone(el.stepRetrieval, "Đã truy vấn 4 nhánh song song");
  setStepDone(el.stepRerank, "Hoàn tất BLIP-1 / BGE & Weighted RRF");
  setStepDone(el.stepPost, `Đã xuất ${data.total_results} candidates`);

  if (data.decomposition) {
    el.decompDetailsCard.style.display = "block";
    el.decValVisual.textContent = data.decomposition.image_query || "N/A";
    el.decValOcr.textContent = (data.decomposition.ocr_terms || []).join(", ") || "Không có";
    el.decValAsr.textContent = (data.decomposition.asr_terms || []).join(", ") || "Không có";
    el.decValWeights.textContent = JSON.stringify(data.decomposition.modality_weights || {});
  }
}

// ==========================================================================
// RENDER GALLERY: KIS (KNOWN-ITEM SEARCH)
// ==========================================================================
function renderKisResults(candidates) {
  el.resultsGallery.innerHTML = "";
  if (!candidates || candidates.length === 0) {
    el.resultsGallery.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Không tìm thấy kết quả phù hợp</div></div>`;
    return;
  }

  candidates.forEach((cand, idx) => {
    const isTop1 = idx === 0;
    const card = document.createElement("div");
    card.className = `candidate-card ${isTop1 && state.promotedTop1Key ? "promoted-top1" : ""}`;
    card.id = `cand-card-${cand.video_id}-${cand.frame_id}`;

    card.innerHTML = `
      <div class="card-thumb-wrap">
        <img class="card-thumb" src="${cand.image_url}" loading="lazy" alt="${cand.video_id}_${cand.frame_id}" onerror="this.src='/static/img/placeholder.jpg'">
        <div class="rank-badge">#${idx + 1}</div>
        <div class="score-badge">${cand.score.toFixed(4)}</div>
      </div>
      <div class="card-body">
        <div class="card-meta-row">
          <span class="card-video-id">${cand.video_id}</span>
          <span class="card-frame-id">F: ${cand.frame_id}</span>
          <span class="card-source-tag">${cand.source || "fused"}</span>
        </div>
        <div class="card-actions">
          <button class="btn-card-action btn-pin-top1" onclick="pinCandidateToTop1(${idx})" title="Đưa frame này lên vị trí Top 1">
            ⭐ Đẩy lên #1
          </button>
          <button class="btn-card-action" onclick="openNeighborModal('${cand.video_id}', ${cand.frame_id}, ${idx + 1}, ${idx})" title="Xem 25 frame liên tiếp">
            🎞️ 25 Frame
          </button>
        </div>
      </div>
    `;
    el.resultsGallery.appendChild(card);
  });
}

// ==========================================================================
// RENDER GALLERY: VQA (QUESTION ANSWERING)
// ==========================================================================
function renderVqaResults(candidates) {
  el.resultsGallery.innerHTML = "";
  if (!candidates || candidates.length === 0) {
    el.resultsGallery.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Không tìm thấy kết quả phù hợp</div></div>`;
    return;
  }

  candidates.forEach((cand, idx) => {
    const isTop1 = idx === 0;
    const key = `${cand.video_id}_${cand.frame_id}`;
    const currentAnswer = state.vqaOverrides[key] !== undefined ? state.vqaOverrides[key] : cand.answer || "unknown";

    const card = document.createElement("div");
    card.className = `candidate-card ${isTop1 && state.promotedTop1Key ? "promoted-top1" : ""}`;
    card.id = `cand-card-${cand.video_id}-${cand.frame_id}`;

    card.innerHTML = `
      <div class="card-thumb-wrap">
        <img class="card-thumb" src="${cand.image_url}" loading="lazy" alt="${cand.video_id}_${cand.frame_id}">
        <div class="rank-badge">#${idx + 1}</div>
        ${cand.is_target ? '<span class="score-badge" style="background:rgba(16,185,129,0.2);border-color:#10b981;color:#10b981;">VLM Rep</span>' : ""}
      </div>
      <div class="card-body">
        <div class="card-meta-row">
          <span class="card-video-id">${cand.video_id}</span>
          <span class="card-frame-id">F: ${cand.frame_id}</span>
        </div>

        <div class="vqa-answer-box">
          <label class="vqa-answer-label">Đáp án dự đoán (Có thể chỉnh sửa):</label>
          <input type="text" class="vqa-answer-input" value="${escapeHtml(currentAnswer)}" onchange="updateVqaAnswer('${cand.video_id}', ${cand.frame_id}, this.value)" placeholder="Nhập đáp án...">
        </div>

        <div class="card-actions">
          <button class="btn-card-action btn-pin-top1" onclick="pinCandidateToTop1(${idx})" title="Đưa frame này lên vị trí Top 1">
            ⭐ Đẩy lên #1
          </button>
          <button class="btn-card-action" onclick="openNeighborModal('${cand.video_id}', ${cand.frame_id}, ${idx + 1}, ${idx})" title="Xem 25 frame liên tiếp">
            🎞️ 25 Frame
          </button>
        </div>
      </div>
    `;
    el.resultsGallery.appendChild(card);
  });
}

function updateVqaAnswer(videoId, frameId, newText) {
  const key = `${videoId}_${frameId}`;
  state.vqaOverrides[key] = newText.trim();
}

// ==========================================================================
// RENDER GALLERY: TRAKE (TEMPORAL ACTION SEQUENCE)
// ==========================================================================
function renderTrakeResults(sequences, plan) {
  el.resultsGallery.innerHTML = "";
  if (!sequences || sequences.length === 0) {
    el.resultsGallery.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Không tìm thấy chuỗi sự kiện phù hợp</div></div>`;
    return;
  }

  sequences.forEach((seq, idx) => {
    const isTop1 = idx === 0;
    const card = document.createElement("div");
    card.className = `trake-sequence-card ${isTop1 && state.promotedTop1Key ? "promoted-top1" : ""}`;

    let stepsHtml = "";
    (seq.steps || []).forEach((s, sIdx) => {
      stepsHtml += `
        <div class="trake-step-item" onclick="openNeighborModal('${seq.video_id}', ${s.frame_id}, ${idx + 1}, ${idx})">
          <img class="trake-step-thumb" src="${s.image_url}" loading="lazy" alt="Step ${s.step_index}">
          <div class="trake-step-meta">
            <span style="color:var(--accent-cyan);font-weight:700;">Step ${s.step_index}</span>
            <span>Frame: ${s.frame_id}</span>
          </div>
        </div>
      `;
      if (sIdx < seq.steps.length - 1) {
        stepsHtml += `<div class="trake-arrow">➔</div>`;
      }
    });

    card.innerHTML = `
      <div class="trake-seq-header">
        <div class="trake-seq-title">
          <span class="rank-badge" style="position:static;">#${idx + 1}</span>
          <span class="card-video-id" style="font-size:16px;">Video: ${seq.video_id}</span>
          <span class="score-badge" style="position:static;">Score: ${seq.score.toFixed(4)}</span>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="btn-card-action btn-pin-top1" onclick="pinCandidateToTop1(${idx})">⭐ Đẩy chuỗi lên #1</button>
        </div>
      </div>
      <div class="trake-steps-filmstrip">
        ${stepsHtml}
      </div>
    `;
    el.resultsGallery.appendChild(card);
  });
}

// ==========================================================================
// PIN / PROMOTE CANDIDATE TO TOP 1
// ==========================================================================
window.pinCandidateToTop1 = function (index) {
  if (index === 0) return; // Already Top 1

  const item = state.results[index];
  if (!item) return;

  // Remove from current position and insert at index 0 (shifts others down by 1)
  state.results.splice(index, 1);
  state.results.unshift(item);

  state.promotedTop1Key = `${item.video_id}_${item.frame_id || (item.frames && item.frames[0])}`;
  el.pinnedNotice.style.display = "inline-block";

  // Re-render
  if (state.currentTask === "trake") {
    renderTrakeResults(state.results, state.plan);
  } else if (state.currentTask === "vqa") {
    renderVqaResults(state.results);
  } else {
    renderKisResults(state.results);
  }

  // Smooth scroll to top of gallery
  el.resultsGallery.scrollIntoView({ behavior: "smooth", block: "start" });
};

// ==========================================================================
// 25-NEIGHBOR FILMSTRIP MODAL
// ==========================================================================
window.openNeighborModal = async function (videoId, frameId, rank, index) {
  state.activeNeighborTarget = { videoId, frameId, rank, index };
  state.selectedNeighborFrame = { videoId, frameId };

  el.modalVideoTitle.textContent = `Video: ${videoId} | Target Frame: #${frameId} (Rank #${rank})`;
  el.neighborFilmstrip.innerHTML = `<div style="color:var(--text-muted);padding:20px;">Đang tải 25 frame lân cận từ Elasticsearch & MinIO...</div>`;
  el.neighborModal.style.display = "flex";

  try {
    const res = await fetch(`/api/neighbors/${videoId}/${frameId}`);
    const data = await res.json();
    if (data.error) {
      el.neighborFilmstrip.innerHTML = `<div style="color:var(--accent-rose);padding:20px;">Lỗi: ${data.error}</div>`;
      return;
    }
    renderNeighborFilmstrip(data.neighbors || []);
  } catch (err) {
    el.neighborFilmstrip.innerHTML = `<div style="color:var(--accent-rose);padding:20px;">Lỗi kết nối: ${err}</div>`;
  }
};

function closeNeighborModal() {
  el.neighborModal.style.display = "none";
}

function renderNeighborFilmstrip(neighbors) {
  el.neighborFilmstrip.innerHTML = "";
  if (!neighbors.length) {
    el.neighborFilmstrip.innerHTML = `<div style="color:var(--text-muted);padding:20px;">Không có frame lân cận nào.</div>`;
    return;
  }

  neighbors.forEach((n) => {
    const card = document.createElement("div");
    card.className = `neighbor-card ${n.is_target ? "target selected" : ""}`;
    card.id = `ncard-${n.frame_id}`;

    card.innerHTML = `
      <img class="neighbor-thumb" src="${n.image_url}" loading="lazy" alt="Frame ${n.frame_id}">
      <div class="neighbor-footer">
        <span>F: ${n.frame_id}</span>
        ${n.is_target ? '<span style="color:var(--accent-cyan);font-weight:700;">Target</span>' : ""}
      </div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".neighbor-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.selectedNeighborFrame = { videoId: n.video_id, frame_id: n.frame_id };

      el.metaFrameTitle.textContent = `Frame: #${n.frame_id} ${n.is_target ? "(Target gốc)" : ""}`;
      el.metaFrameBody.innerHTML = `
        <strong>Mô tả cảnh:</strong> ${escapeHtml(n.description || "Không có mô tả")}<br>
        <strong>Chữ OCR:</strong> ${escapeHtml(n.ocr || "Không có chữ")}
      `;
    });

    el.neighborFilmstrip.appendChild(card);
  });
}

function navigateNeighborFilmstrip(direction) {
  const cards = Array.from(document.querySelectorAll(".neighbor-card"));
  const currentIndex = cards.findIndex((c) => c.classList.contains("selected"));
  if (currentIndex === -1) return;

  const nextIndex = Math.max(0, Math.min(cards.length - 1, currentIndex + direction));
  cards[nextIndex].click();
  cards[nextIndex].scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
}

function applySelectedNeighbor() {
  if (!state.activeNeighborTarget || !state.selectedNeighborFrame) {
    closeNeighborModal();
    return;
  }

  const { index } = state.activeNeighborTarget;
  const newFrameId = state.selectedNeighborFrame.frame_id;
  const targetItem = state.results[index];

  if (targetItem) {
    targetItem.frame_id = newFrameId;
    targetItem.image_url = `/api/image/${targetItem.video_id}/${newFrameId}`;

    // Re-render
    if (state.currentTask === "vqa") {
      renderVqaResults(state.results);
    } else if (state.currentTask === "kis") {
      renderKisResults(state.results);
    }
  }

  closeNeighborModal();
}

// ==========================================================================
// EXPORT SUBMISSION CSV
// ==========================================================================
async function exportSubmissionCsv() {
  if (!state.results || state.results.length === 0) {
    alert("Chưa có kết quả để xuất CSV!");
    return;
  }

  const task = state.currentTask;
  const items = [];

  if (task === "kis") {
    state.results.forEach((c) => {
      items.push({ video_id: c.video_id, frame_id: c.frame_id });
    });
  } else if (task === "vqa") {
    state.results.forEach((c) => {
      const key = `${c.video_id}_${c.frame_id}`;
      const answer = state.vqaOverrides[key] !== undefined ? state.vqaOverrides[key] : c.answer || "unknown";
      items.push({ video_id: c.video_id, frame_id: c.frame_id, answer: answer });
    });
  } else if (task === "trake") {
    state.results.forEach((seq) => {
      items.push({ video_id: seq.video_id, frames: seq.frames });
    });
  }

  try {
    const slug = state.currentQuery.toLowerCase().replace(/[^a-z0-9]/g, "_").substring(0, 30);
    const filename = `${task}_${slug || "query"}.csv`;

    const res = await fetch("/api/export_csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, items, filename }),
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    alert(`Lỗi khi xuất file CSV: ${err}`);
  }
}

// Utility: escape HTML
function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// ==========================================================================
// REAL-TIME LIVE BACKEND CONSOLE (SSE)
// ==========================================================================
let logEventSource = null;
let currentLogFilter = "ALL";
const logHistory = [];

function initLiveTerminal() {
  const btnToggle = document.getElementById("btnToggleTerminal");
  const btnClose = document.getElementById("btnCloseTerminal");
  const btnClear = document.getElementById("btnTermClear");
  const drawer = document.getElementById("liveTerminalDrawer");
  const termBody = document.getElementById("terminalBody");
  const badge = document.getElementById("termStatusBadge");
  const autoScroll = document.getElementById("chkTermAutoScroll");
  const filterBtns = document.querySelectorAll(".term-filter");

  if (!btnToggle || !drawer) return;

  // Toggle drawer
  btnToggle.addEventListener("click", () => drawer.classList.toggle("open"));
  btnClose.addEventListener("click", () => drawer.classList.remove("open"));

  // Clear logs
  btnClear.addEventListener("click", () => {
    logHistory.length = 0;
    termBody.innerHTML = "";
  });

  // Filter logs
  filterBtns.forEach((b) => {
    b.addEventListener("click", () => {
      filterBtns.forEach((btn) => btn.classList.remove("active"));
      b.classList.add("active");
      currentLogFilter = b.getAttribute("data-filter");
      renderFilteredLogs();
    });
  });

  // Start SSE connection
  connectLogStream();

  function connectLogStream() {
    if (logEventSource) {
      logEventSource.close();
    }

    logEventSource = new EventSource("/api/logs/stream");

    logEventSource.onopen = () => {
      badge.textContent = "● Live Connected";
      badge.style.color = "var(--accent-emerald)";
      badge.style.borderColor = "rgba(16,185,129,0.3)";
    };

    logEventSource.onmessage = (event) => {
      try {
        const entry = JSON.parse(event.data);
        logHistory.push(entry);
        if (logHistory.length > 500) logHistory.shift();

        if (shouldDisplayLog(entry.level, currentLogFilter)) {
          appendLogLine(entry);
        }
      } catch (err) {
        console.warn("Log parse error:", err);
      }
    };

    logEventSource.onerror = () => {
      badge.textContent = "○ Reconnecting...";
      badge.style.color = "var(--accent-gold)";
      badge.style.borderColor = "rgba(251,191,36,0.3)";
    };
  }

  function shouldDisplayLog(level, filter) {
    if (filter === "ALL") return true;
    if (filter === "INFO") return level === "INFO" || level === "WARNING" || level === "ERROR";
    if (filter === "WARNING") return level === "WARNING" || level === "ERROR";
    if (filter === "ERROR") return level === "ERROR";
    return true;
  }

  function appendLogLine(entry) {
    const div = document.createElement("div");
    const lvl = (entry.level || "INFO").toLowerCase();
    div.className = `term-line term-${lvl}`;

    const tagClass = lvl === "warning" ? "warning" : lvl === "error" ? "error" : lvl === "debug" ? "debug" : "info";

    div.innerHTML = `
      <span class="term-time">${entry.timestamp || ""}</span>
      <span class="term-tag ${tagClass}">[${(entry.level || "INFO").padEnd(5)}]</span>
      <span class="term-logger">${escapeHtml((entry.logger || "").replace("src.", ""))}</span>
      <span class="term-msg">${escapeHtml(entry.message || "")}</span>
    `;

    termBody.appendChild(div);

    if (autoScroll && autoScroll.checked) {
      termBody.scrollTop = termBody.scrollHeight;
    }
  }

  function renderFilteredLogs() {
    termBody.innerHTML = "";
    logHistory.forEach((entry) => {
      if (shouldDisplayLog(entry.level, currentLogFilter)) {
        appendLogLine(entry);
      }
    });
  }
}

// Auto-init terminal on load
document.addEventListener("DOMContentLoaded", () => {
  initLiveTerminal();
});
