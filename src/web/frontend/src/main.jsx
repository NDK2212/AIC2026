import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleDot,
  Clock3,
  Download,
  Eye,
  FileText,
  Film,
  KeyRound,
  ListFilter,
  LoaderCircle,
  Pin,
  Play,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
  X,
} from "lucide-react";
import "./styles.css";

const TASKS = [
  { id: "kis", index: "01", name: "KIS", detail: "Known item", icon: Search },
  { id: "vqa", index: "02", name: "VQA", detail: "Question answer", icon: Eye },
  { id: "trake", index: "03", name: "TRAKE", detail: "Temporal", icon: Clock3 },
];

const PRESETS = {
  kis: [
    "người phụ nữ mặc áo dài đỏ cầm hoa sen",
    "xe buýt màu xanh có chữ Bến Thành",
    "phóng viên nói về dự báo thời tiết",
  ],
  vqa: [
    "Trong video hướng dẫn nấu ăn, người đầu bếp lần lượt cho tiêu xanh, lá chanh và sả vào bụng của tổng cộng 4 con cá. Đây là loài cá gì?",
    "Người đàn ông trong video mặc áo màu gì?",
    "Biển hiệu cửa hàng có chữ gì?",
  ],
  trake: [
    "Chuỗi gồm (1) mở cửa xe (2) bước vào ghế lái (3) khởi động xe",
    "Chuỗi gồm (1) chuẩn bị nguyên liệu (2) xào trên bếp (3) bày ra đĩa",
  ],
};

const FALLBACK_CONFIG = {
  retrieval_paths: {
    ocr: { enabled: true, weight: 1 },
    asr: { enabled: true, weight: 1 },
    description: { enabled: true, weight: 1.2 },
    visual: { enabled: true, weight: 1.5 },
  },
  fusion: { method: "weighted_rrf", rrf_k: 60, adaptive: true },
  rerank: { qwen3_vl: { enabled: true, top_n: 25, weight: 1, device: "cuda" } },
  submission: { top_diverse: 8, head_max_per_video: 3, shot_window: 60, neighbor_expansion: true },
  vqa: { vlm_top_n: 25, vlm_images_per_video: 4, propagate: true, vqa_mode: "vision" },
  trake: { per_step_topk: 150, coverage_bonus: 0.5, miss_penalty: -0.35 },
};

const clone = (value) => JSON.parse(JSON.stringify(value));
const cx = (...items) => items.filter(Boolean).join(" ");
const PROGRESS_ORDER = {
  start: 0,
  split: 1,
  decompose: 1,
  model: 2,
  retrieval: 2,
  rerank: 3,
  evidence: 4,
  answer: 5,
  complete: 6,
};

async function apiJson(url, options) {
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok || data?.error) {
    throw new Error(data?.error || `HTTP ${response.status}`);
  }
  return data;
}

function App() {
  const [task, setTask] = useState("kis");
  const [query, setQuery] = useState("");
  const [config, setConfig] = useState(clone(FALLBACK_CONFIG));
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState([]);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [liveProgress, setLiveProgress] = useState(null);
  const [useCache, setUseCache] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [neighbor, setNeighbor] = useState(null);
  const [pinned, setPinned] = useState(false);

  useEffect(() => {
    Promise.allSettled([apiJson("/api/config"), apiJson("/api/status")]).then(([cfg, health]) => {
      if (cfg.status === "fulfilled") {
        setConfig({ ...clone(FALLBACK_CONFIG), ...cfg.value });
      }
      if (health.status === "fulfilled") setStatus(health.value);
    });
  }, []);

  useEffect(() => {
    const source = new EventSource("/api/logs/stream?tail=0");
    source.onmessage = (event) => {
      try {
        const entry = JSON.parse(event.data);
        if (entry.progress) {
          const incoming = { ...entry.progress, timestamp: entry.timestamp };
          setLiveProgress((current) => {
            if (!current) return incoming;
            const currentOrder = PROGRESS_ORDER[current.phase] ?? 0;
            const incomingOrder = PROGRESS_ORDER[incoming.phase] ?? 0;
            return incomingOrder >= currentOrder ? incoming : current;
          });
        }
      } catch {}
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    if (!loading) return undefined;
    const start = performance.now();
    const timer = window.setInterval(() => setElapsed((performance.now() - start) / 1000), 100);
    return () => window.clearInterval(timer);
  }, [loading]);

  const switchTask = (next) => {
    setTask(next);
    setResults([]);
    setPayload(null);
    setError("");
    setPinned(false);
    setLiveProgress(null);
  };

  const search = async (overrideQuery) => {
    const text = String(overrideQuery ?? query).trim();
    if (!text || loading) return;
    setQuery(text);
    setLoading(true);
    setError("");
    setElapsed(0);
    setPayload(null);
    setResults([]);
    setPinned(false);
    setLiveProgress({
      phase: "start",
      status: "running",
      title: "Đang gửi truy vấn",
      detail: text,
    });
    try {
      const data = await apiJson("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          query: text,
          params: config,
          vqa_mode: config.vqa?.vqa_mode || "vision",
          use_cache: useCache,
        }),
      });
      setPayload(data);
      setResults(task === "trake" ? data.sequences || [] : data.candidates || []);
      setElapsed(Number(data.elapsed_total_s || 0));
    } catch (cause) {
      setError(cause.message || String(cause));
    } finally {
      setLoading(false);
    }
  };

  const promote = (index) => {
    if (index <= 0) return;
    setResults((current) => {
      const next = [...current];
      const [item] = next.splice(index, 1);
      next.unshift(item);
      return next;
    });
    setPinned(true);
  };

  const updateAnswer = (index, answer) => {
    setResults((current) => current.map((item, i) => i === index ? { ...item, answer } : item));
  };

  const openNeighbor = (videoId, frameId, resultIndex, stepIndex = null) => {
    setNeighbor({ videoId, frameId, resultIndex, stepIndex });
  };

  const applyNeighbor = (frameId) => {
    if (!neighbor) return;
    setResults((current) => current.map((item, index) => {
      if (index !== neighbor.resultIndex) return item;
      if (task === "trake" && neighbor.stepIndex !== null) {
        const frames = [...(item.frames || [])];
        frames[neighbor.stepIndex] = frameId;
        const steps = (item.steps || []).map((step, i) => i === neighbor.stepIndex
          ? { ...step, frame_id: frameId, image_url: `/api/image/${item.video_id}/${frameId}` }
          : step);
        return { ...item, frames, steps };
      }
      return { ...item, frame_id: frameId, image_url: `/api/image/${item.video_id}/${frameId}` };
    }));
    setNeighbor(null);
  };

  const exportCsv = async () => {
    if (!results.length) return;
    const items = results.map((item) => task === "trake"
      ? { video_id: item.video_id, frames: item.frames }
      : task === "vqa"
        ? { video_id: item.video_id, frame_id: item.frame_id, answer: item.answer || "unknown" }
        : { video_id: item.video_id, frame_id: item.frame_id });
    const response = await fetch("/api/export_csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, items, filename: `${task}_submission.csv` }),
    });
    if (!response.ok) return setError(`Không thể xuất CSV: HTTP ${response.status}`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${task}_submission.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="shell">
      <Header
        task={task}
        onTask={switchTask}
        status={status}
        onSettings={() => setSettingsOpen(true)}
      />

      <main>
        <QueryDeck
          task={task}
          query={query}
          setQuery={setQuery}
          presets={PRESETS[task]}
          loading={loading}
          onSearch={search}
          useCache={useCache}
          setUseCache={setUseCache}
        />

        {(loading || payload || error) && (
          <PipelineTrace
            task={task}
            loading={loading}
            elapsed={elapsed}
            payload={payload}
            error={error}
            liveProgress={liveProgress}
          />
        )}

        <ResultsHeader
          count={results.length}
          elapsed={elapsed}
          pinned={pinned}
          visible={Boolean(results.length || payload)}
          onExport={exportCsv}
        />

        <ResultGrid
          task={task}
          results={results}
          payload={payload}
          loading={loading}
          error={error}
          onPromote={promote}
          onAnswer={updateAnswer}
          onNeighbor={openNeighbor}
        />
      </main>

      <button className="console-trigger" onClick={() => setConsoleOpen(true)}>
        <TerminalSquare size={15} />
        <span>Live logs</span>
        <i />
      </button>

      <SettingsDrawer
        open={settingsOpen}
        config={config}
        setConfig={setConfig}
        onClose={() => setSettingsOpen(false)}
        onReset={() => setConfig(clone(FALLBACK_CONFIG))}
      />
      <ConsoleDrawer open={consoleOpen} onClose={() => setConsoleOpen(false)} />
      {neighbor && (
        <NeighborModal
          target={neighbor}
          onClose={() => setNeighbor(null)}
          onApply={applyNeighbor}
        />
      )}
    </div>
  );
}

function Header({ task, onTask, status, onSettings }) {
  const services = [
    ["ES", status?.elasticsearch?.connected],
    ["Qdrant", status?.qdrant?.connected],
    ["MinIO", status?.minio?.enabled],
  ];
  return (
    <header className="topbar panel">
      <div className="brand">
        <div className="brand-mark">A26</div>
        <div>
          <strong>AIC 2026</strong>
          <span>Video retrieval workbench</span>
        </div>
      </div>
      <nav className="task-switcher" aria-label="Chọn tác vụ">
        {TASKS.map((item) => {
          const Icon = item.icon;
          return (
            <button key={item.id} className={cx("task-button", task === item.id && "active")} onClick={() => onTask(item.id)}>
              <span className="task-number">{item.index}</span>
              <Icon size={14} />
              <span><b>{item.name}</b><small>{item.detail}</small></span>
            </button>
          );
        })}
      </nav>
      <div className="topbar-right">
        <div className="service-list">
          {services.map(([name, online]) => (
            <span className={cx("service", online === false && "offline")} key={name}><i />{name}</span>
          ))}
          <span className="service"><i />{status?.keys_count ?? 0} keys</span>
        </div>
        <button className="icon-button settings-button" onClick={onSettings} aria-label="Mở cấu hình">
          <Settings2 size={16} />
          <span>Cấu hình</span>
        </button>
      </div>
    </header>
  );
}

function QueryDeck({ task, query, setQuery, presets, loading, onSearch, useCache, setUseCache }) {
  return (
    <section className="query-deck panel">
      <div className="query-kicker">
        <span>{task.toUpperCase()} / QUERY</span>
        <div className="presets">
          {presets.map((preset) => (
            <button key={preset} onClick={() => onSearch(preset)} title={preset}>
              {preset.length > 52 ? `${preset.slice(0, 52)}...` : preset}
            </button>
          ))}
        </div>
      </div>
      <div className="query-row">
        <div className="query-input-wrap">
          <Search size={18} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onSearch()}
            placeholder="Mô tả cảnh, đặt câu hỏi VQA hoặc nhập chuỗi hành động temporal..."
            autoFocus
          />
        </div>
        <label className="cache-control">
          <input type="checkbox" checked={useCache} onChange={(event) => setUseCache(event.target.checked)} />
          Cache
        </label>
        <button className="run-button" disabled={loading || !query.trim()} onClick={() => onSearch()}>
          {loading ? <LoaderCircle className="spin" size={17} /> : <Play size={16} fill="currentColor" />}
          {loading ? "Đang chạy" : "Chạy pipeline"}
        </button>
      </div>
    </section>
  );
}

function PipelineTrace({ task, loading, elapsed, payload, error, liveProgress }) {
  const trace = payload?.trace?.stages || [];
  const steps = task === "vqa"
    ? [
      ["Query split", "Scene, question, evidence query"],
      ["Action retrieval", "Tìm video từ cảnh hành động"],
      ["Fusion + rerank", "Weighted fusion, Qwen3-VL 2B"],
      ["Evidence + answer", "Tìm lần hai trong cùng video"],
    ]
    : task === "trake"
      ? [
        ["Temporal plan", "Tách các bước có thứ tự"],
        ["Step retrieval", "Tìm candidates cho từng bước"],
        ["Alignment", "Anchored bidirectional DP"],
        ["Sequence output", "Kiểm tra thứ tự frame"],
      ]
      : [
        ["Decomposition", "Tách visual, OCR, ASR, description"],
        ["Parallel retrieval", "Chạy các path đang bật"],
        ["Fusion + rerank", "Weighted fusion, Qwen3-VL 2B"],
        ["Diversity", "Shot filtering và neighbor expansion"],
      ];

  const phaseMaps = {
    kis: { start: 0, decompose: 0, model: 1, retrieval: 1, rerank: 2, complete: 4 },
    vqa: { start: 0, split: 0, decompose: 1, model: 1, retrieval: 1, rerank: 2, evidence: 3, answer: 3, complete: 4 },
    trake: { start: 0, plan: 0, decompose: 1, model: 1, retrieval: 1, rerank: 2, complete: 4 },
  };
  const livePhase = loading
    ? (phaseMaps[task]?.[liveProgress?.phase] ?? 0)
    : steps.length;

  const payloadInspector = payload?.split
    ? [
      ["Scene query", payload.split.scene],
      ["Question", payload.split.question],
      ["Evidence query", payload.split.evidence_query],
      ["Answer type", payload.split.type],
    ]
    : payload?.decomposition
      ? [
        ["Visual query", payload.decomposition.image_query],
        ["Description query", payload.decomposition.description_query],
        ["OCR terms", (payload.decomposition.ocr_terms || []).join(", ")],
        ["ASR terms", (payload.decomposition.asr_terms || []).join(", ")],
        ["Weights", JSON.stringify(payload.decomposition.modality_weights || {})],
      ]
      : [];
  const liveInspector = liveProgress?.inspector
    ? Object.entries(liveProgress.inspector)
    : [];
  const inspector = payloadInspector.length ? payloadInspector : liveInspector;

  return (
    <section className={cx("trace-panel panel", error && "has-error")}>
      <div className="section-bar">
        <div><Activity size={14} /><span>PIPELINE TRACE</span></div>
        <code>{elapsed.toFixed(1)} s</code>
      </div>
      {error ? (
        <div className="error-state"><CircleDot size={16} /><span>{error}</span></div>
      ) : (
        <>
          {loading && liveProgress && (
            <div className="live-operation" role="status" aria-live="polite">
              <LoaderCircle className="spin" size={15} />
              <div>
                <strong>{liveProgress.title}</strong>
                <span>{liveProgress.detail || "Backend đang xử lý"}</span>
              </div>
              {liveProgress.timestamp && <time>{liveProgress.timestamp}</time>}
            </div>
          )}
          <div className="trace-steps">
            {steps.map(([name, detail], index) => {
              const done = !loading || index < livePhase;
              const running = loading && index === livePhase;
              const timing = trace[index]?.latency_ms;
              const activeDetail = running && liveProgress?.detail
                ? liveProgress.detail
                : detail;
              return (
                <div className={cx("trace-step", done && "done", running && "running")} key={name}>
                  <div className="trace-index">{done ? <Check size={14} /> : running ? <LoaderCircle className="spin" size={14} /> : index + 1}</div>
                  <div><strong>{name}</strong><span>{timing ? `${activeDetail} / ${timing} ms` : activeDetail}</span></div>
                </div>
              );
            })}
          </div>
          {inspector.length > 0 && (
            <div className="query-inspector">
              {inspector.map(([label, value]) => (
                <div key={label}><span>{label}</span><p>{value || "Không có"}</p></div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function ResultsHeader({ count, elapsed, pinned, visible, onExport }) {
  if (!visible) return null;
  return (
    <section className="results-bar panel">
      <div>
        <ListFilter size={14} />
        <strong>{count} kết quả</strong>
        <span>{elapsed.toFixed(2)} s</span>
        {pinned && <span className="pinned-label"><Pin size={11} /> Đã ghim Top 1</span>}
      </div>
      <button onClick={onExport} disabled={!count}><Download size={14} /> Xuất CSV</button>
    </section>
  );
}

function ResultGrid({ task, results, payload, loading, error, onPromote, onAnswer, onNeighbor }) {
  if (loading) return <ResultSkeleton />;
  if (error) return null;
  if (!results.length) {
    return (
      <section className="empty-stage">
        <div className="empty-mark">{payload ? "0" : "A26"}</div>
        <h2>{payload ? "Không tìm thấy kết quả phù hợp" : "Sẵn sàng truy hồi video"}</h2>
        <p>{payload ? "Thử điều chỉnh retrieval path, fusion weight hoặc truy vấn." : "Chọn tác vụ, nhập truy vấn hoặc dùng một preset để bắt đầu."}</p>
      </section>
    );
  }
  if (task === "trake") {
    return <div className="result-grid temporal-grid">{results.map((item, index) => (
      <TemporalCard key={`${item.video_id}-${index}`} item={item} index={index} onPromote={onPromote} onNeighbor={onNeighbor} />
    ))}</div>;
  }
  return (
    <div className="result-grid">
      {results.map((item, index) => (
        <CandidateCard
          key={`${item.video_id}-${item.frame_id}-${index}`}
          item={item}
          index={index}
          task={task}
          onPromote={onPromote}
          onAnswer={onAnswer}
          onNeighbor={onNeighbor}
        />
      ))}
    </div>
  );
}

function CandidateCard({ item, index, task, onPromote, onAnswer, onNeighbor }) {
  const evidence = item.evidence_frames || [];
  return (
    <article className={cx("candidate-card", evidence.length && "with-evidence")}>
      <div className="action-frame">
        <img src={item.image_url} alt={`${item.video_id} frame ${item.frame_id}`} loading="lazy" />
        <span className="rank">#{index + 1}</span>
        <span className="frame-role">{task === "vqa" ? "ACTION FRAME" : Number(item.score || 0).toFixed(4)}</span>
      </div>
      <div className="candidate-body">
        <div className="candidate-meta"><strong>{item.video_id}</strong><code>F:{item.frame_id}</code>{item.source && <span>{item.source}</span>}</div>
        {task === "vqa" && (
          <label className="answer-field">
            <span>Đáp án dự đoán</span>
            <input value={item.answer || "unknown"} onChange={(event) => onAnswer(index, event.target.value)} />
          </label>
        )}
        {evidence.length > 0 && (
          <div className="evidence-block">
            <div className="evidence-title"><Sparkles size={13} /><span>ANSWER EVIDENCE · CÙNG VIDEO</span></div>
            <div className="evidence-grid">
              {evidence.map((frame) => (
                <button key={`${frame.video_id}-${frame.frame_id}`} onClick={() => onNeighbor(frame.video_id, frame.frame_id, index)}>
                  <img src={frame.image_url} alt={`Evidence frame ${frame.frame_id}`} loading="lazy" />
                  <span>F:{frame.frame_id}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="card-actions">
          <button onClick={() => onPromote(index)} disabled={index === 0}><Pin size={13} /> Ghim Top 1</button>
          <button onClick={() => onNeighbor(item.video_id, item.frame_id, index)}><Film size={13} /> Xem 25 frame</button>
        </div>
      </div>
    </article>
  );
}

function TemporalCard({ item, index, onPromote, onNeighbor }) {
  return (
    <article className="temporal-card">
      <header><div><span>#{index + 1}</span><strong>{item.video_id}</strong><code>{Number(item.score || 0).toFixed(4)}</code></div><button onClick={() => onPromote(index)} disabled={index === 0}><Pin size={13} /> Ghim chuỗi</button></header>
      <div className="temporal-filmstrip">
        {(item.steps || []).map((step, stepIndex) => (
          <React.Fragment key={`${item.video_id}-${step.frame_id}-${stepIndex}`}>
            <button className="temporal-step" onClick={() => onNeighbor(item.video_id, step.frame_id, index, stepIndex)}>
              <img src={step.image_url} alt={`Step ${step.step_index}`} loading="lazy" />
              <span><b>STEP {step.step_index}</b><code>F:{step.frame_id}</code></span>
            </button>
            {stepIndex < item.steps.length - 1 && <ChevronRight className="step-arrow" size={18} />}
          </React.Fragment>
        ))}
      </div>
    </article>
  );
}

function ResultSkeleton() {
  return <div className="result-grid">{Array.from({ length: 6 }, (_, index) => <div className="skeleton-card" key={index}><div /><span /><span /></div>)}</div>;
}

function SettingsDrawer({ open, config, setConfig, onClose, onReset }) {
  const [tab, setTab] = useState("retrieval");
  const [keys, setKeys] = useState("");
  const [keyMessage, setKeyMessage] = useState("");
  const patch = (section, key, value) => setConfig((current) => ({ ...current, [section]: { ...current[section], [key]: value } }));
  const patchDeep = (section, group, key, value) => setConfig((current) => ({
    ...current,
    [section]: { ...current[section], [group]: { ...current[section]?.[group], [key]: value } },
  }));

  useEffect(() => {
    if (!open) return;
    apiJson("/api/keys").then((data) => setKeys((data.keys || []).join("\n"))).catch(() => {});
  }, [open]);

  const saveKeys = async () => {
    try {
      const list = keys.split(/[\n,]+/).map((key) => key.trim()).filter(Boolean);
      const data = await apiJson("/api/keys", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keys: list }) });
      setKeyMessage(`Đã kích hoạt ${data.count} keys`);
    } catch (cause) {
      setKeyMessage(cause.message);
    }
  };

  if (!open) return null;
  const tabs = [
    ["retrieval", "Retrieval", SlidersHorizontal],
    ["reranker", "Qwen reranker", Sparkles],
    ["tasks", "Task", Activity],
    ["submission", "Diversity", Film],
    ["keys", "API pool", KeyRound],
  ];
  return (
    <div className="drawer-layer" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="settings-drawer">
        <header><div><span>RUNTIME SETTINGS</span><h2>Điều khiển pipeline</h2></div><button onClick={onClose} aria-label="Đóng"><X size={18} /></button></header>
        <nav>{tabs.map(([id, label, Icon]) => <button className={tab === id ? "active" : ""} onClick={() => setTab(id)} key={id}><Icon size={13} />{label}</button>)}</nav>
        <div className="settings-content">
          {tab === "retrieval" && <RetrievalSettings config={config} patch={patch} patchDeep={patchDeep} />}
          {tab === "reranker" && <RerankSettings config={config} patchDeep={patchDeep} />}
          {tab === "tasks" && <TaskSettings config={config} patch={patch} />}
          {tab === "submission" && <SubmissionSettings config={config} patch={patch} />}
          {tab === "keys" && <section className="setting-section"><SectionTitle title="NVIDIA NIM API pool" detail="Một key mỗi dòng. Backend tự round-robin và failover." /><textarea rows="8" value={keys} onChange={(event) => setKeys(event.target.value)} placeholder="nvapi-..." /><div className="setting-footer-inline"><button className="primary" onClick={saveKeys}>Lưu và kích hoạt</button><span>{keyMessage}</span></div></section>}
        </div>
        <footer><button onClick={onReset}>Khôi phục mặc định</button><button className="primary" onClick={onClose}><Check size={14} /> Áp dụng</button></footer>
      </aside>
    </div>
  );
}

function SectionTitle({ title, detail }) {
  return <div className="setting-title"><h3>{title}</h3>{detail && <p>{detail}</p>}</div>;
}

function Toggle({ checked, onChange }) {
  return <button type="button" className={cx("toggle", checked && "on")} onClick={() => onChange(!checked)} aria-pressed={checked}><i /></button>;
}

function RangeControl({ label, value, min, max, step, onChange }) {
  return <label className="range-control"><span>{label}<code>{value}</code></span><input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function NumberControl({ label, value, step = 1, onChange }) {
  return <label className="number-control"><span>{label}</span><input type="number" value={value} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function RetrievalSettings({ config, patch, patchDeep }) {
  return <>
    <section className="setting-section"><SectionTitle title="4 retrieval paths" detail="Các path chạy song song. Path tắt sẽ không tạo job và không nhận fusion weight." /><div className="path-grid">{Object.entries(config.retrieval_paths || {}).map(([path, value]) => <div className="path-card" key={path}><div><strong>{path}</strong><Toggle checked={value.enabled} onChange={(next) => patchDeep("retrieval_paths", path, "enabled", next)} /></div><RangeControl label="Fusion weight" value={value.weight} min={0} max={3} step={0.1} onChange={(next) => patchDeep("retrieval_paths", path, "weight", next)} /></div>)}</div></section>
    <section className="setting-section"><SectionTitle title="Fusion" /><div className="control-grid"><label className="number-control"><span>Method</span><select value={config.fusion.method} onChange={(event) => patch("fusion", "method", event.target.value)}><option value="weighted_rrf">Weighted RRF</option><option value="rrf">Standard RRF</option><option value="weighted_norm">Weighted score norm</option></select></label><NumberControl label="RRF K" value={config.fusion.rrf_k} onChange={(value) => patch("fusion", "rrf_k", value)} /><div className="toggle-row"><span><b>Adaptive LLM weights</b><small>Ưu tiên modality theo query</small></span><Toggle checked={config.fusion.adaptive} onChange={(value) => patch("fusion", "adaptive", value)} /></div></div></section>
  </>;
}

function RerankSettings({ config, patchDeep }) {
  const qwen = config.rerank?.qwen3_vl || FALLBACK_CONFIG.rerank.qwen3_vl;
  return <section className="setting-section"><SectionTitle title="Qwen3-VL-Reranker-2B" detail="Rerank một lần sau fusion bằng keyframe, Description, OCR và ASR. GPU chỉ dành cho bước này." /><div className="model-banner"><div><Sparkles size={18} /><span><b>Multimodal post-fusion</b><small>Device: {qwen.device || "cuda"}</small></span></div><Toggle checked={qwen.enabled} onChange={(value) => patchDeep("rerank", "qwen3_vl", "enabled", value)} /></div><RangeControl label="Top N fused candidates" value={qwen.top_n} min={5} max={50} step={5} onChange={(value) => patchDeep("rerank", "qwen3_vl", "top_n", value)} /><RangeControl label="Qwen score weight" value={qwen.weight} min={0} max={2} step={0.1} onChange={(value) => patchDeep("rerank", "qwen3_vl", "weight", value)} /><div className="hardware-note"><CircleDot size={13} /><span>SigLIP, BEiT-3 và Qwen-VL embedding chạy CPU theo config backend.</span></div></section>;
}

function TaskSettings({ config, patch }) {
  return <>
    <section className="setting-section"><SectionTitle title="VQA" detail="Cross-shot evidence luôn tìm trong top video của action retrieval." /><div className="mode-grid"><button className={config.vqa.vqa_mode === "text_only" ? "active" : ""} onClick={() => patch("vqa", "vqa_mode", "text_only")}><FileText size={16} /><b>Text only</b><span>LLM chỉ đọc caption, OCR, ASR</span></button><button className={config.vqa.vqa_mode === "vision" ? "active" : ""} onClick={() => patch("vqa", "vqa_mode", "vision")}><Eye size={16} /><b>Kimi K3 multimodal</b><span>VLM đọc nhiều ảnh + caption + OCR + ASR</span></button></div><RangeControl label="Answer targets" value={config.vqa.vlm_top_n} min={5} max={50} step={5} onChange={(value) => patch("vqa", "vlm_top_n", value)} /><RangeControl label="Images / video" value={config.vqa.vlm_images_per_video || 4} min={1} max={8} step={1} onChange={(value) => patch("vqa", "vlm_images_per_video", value)} /><div className="toggle-row"><span><b>Propagate answer</b><small>Cùng shot, sau đó cùng video</small></span><Toggle checked={config.vqa.propagate} onChange={(value) => patch("vqa", "propagate", value)} /></div></section>
    <section className="setting-section"><SectionTitle title="TRAKE" detail="Temporal planning, per-step retrieval và anchored DP alignment." /><div className="control-grid"><NumberControl label="Top K mỗi step" value={config.trake.per_step_topk} onChange={(value) => patch("trake", "per_step_topk", value)} /><NumberControl label="Coverage bonus" value={config.trake.coverage_bonus} step={0.1} onChange={(value) => patch("trake", "coverage_bonus", value)} /><NumberControl label="Miss penalty" value={config.trake.miss_penalty} step={0.05} onChange={(value) => patch("trake", "miss_penalty", value)} /></div></section>
  </>;
}

function SubmissionSettings({ config, patch }) {
  return <section className="setting-section"><SectionTitle title="Shot diversity và submission" detail="Giảm frame trùng cảnh ở đầu ranking trước khi xuất CSV." /><div className="control-grid"><NumberControl label="Shot window" value={config.submission.shot_window} onChange={(value) => patch("submission", "shot_window", value)} /><NumberControl label="Top diverse" value={config.submission.top_diverse} onChange={(value) => patch("submission", "top_diverse", value)} /><NumberControl label="Max frame / video" value={config.submission.head_max_per_video} onChange={(value) => patch("submission", "head_max_per_video", value)} /><div className="toggle-row"><span><b>Neighbor expansion</b><small>Mở rộng frame lân cận sau head</small></span><Toggle checked={config.submission.neighbor_expansion} onChange={(value) => patch("submission", "neighbor_expansion", value)} /></div></div></section>;
}

function NeighborModal({ target, onClose, onApply }) {
  const [frames, setFrames] = useState([]);
  const [selected, setSelected] = useState(target.frameId);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    apiJson(`/api/neighbors/${encodeURIComponent(target.videoId)}/${target.frameId}`)
      .then((data) => active && setFrames(data.neighbors || []))
      .catch((cause) => active && setError(cause.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [target.videoId, target.frameId]);
  useEffect(() => {
    const handler = (event) => {
      if (event.key === "Escape") onClose();
      const index = frames.findIndex((frame) => frame.frame_id === selected);
      if (event.key === "ArrowLeft" && index > 0) setSelected(frames[index - 1].frame_id);
      if (event.key === "ArrowRight" && index >= 0 && index < frames.length - 1) setSelected(frames[index + 1].frame_id);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [frames, selected, onClose]);
  const selectedFrame = frames.find((frame) => frame.frame_id === selected);
  return (
    <div className="modal-layer" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="neighbor-modal">
        <header><div><span>FILMSTRIP · 25 KEYFRAMES</span><h2>{target.videoId} / target F:{target.frameId}</h2></div><button onClick={onClose}><X size={18} /></button></header>
        <div className="neighbor-content">
          <div className="neighbor-help"><span>Dùng phím mũi tên hoặc cuộn ngang để duyệt.</span><div><ArrowLeft size={14} /><ArrowRight size={14} /></div></div>
          {loading ? <div className="modal-loading"><LoaderCircle className="spin" /> Đang tải filmstrip</div> : error ? <div className="error-state">{error}</div> : <div className="neighbor-strip">{frames.map((frame) => <button className={cx(frame.frame_id === selected && "selected", frame.is_target && "target")} key={frame.frame_id} onClick={() => setSelected(frame.frame_id)}><img src={frame.image_url} alt={`Frame ${frame.frame_id}`} /><span>F:{frame.frame_id}{frame.is_target && <b>TARGET</b>}</span></button>)}</div>}
          <div className="neighbor-meta"><span>FRAME {selected}</span><p><b>Mô tả:</b> {selectedFrame?.description || "Không có mô tả"}</p><p><b>OCR:</b> {selectedFrame?.ocr || "Không có OCR"}</p></div>
        </div>
        <footer><button onClick={onClose}>Đóng</button><button className="primary" onClick={() => onApply(selected)}><Check size={14} /> Dùng frame đang chọn</button></footer>
      </section>
    </div>
  );
}

function ConsoleDrawer({ open, onClose }) {
  const [logs, setLogs] = useState([]);
  const [filter, setFilter] = useState("ALL");
  useEffect(() => {
    if (!open) return undefined;
    const source = new EventSource("/api/logs/stream");
    source.onmessage = (event) => {
      try {
        const entry = JSON.parse(event.data);
        setLogs((current) => [...current.slice(-399), entry]);
      } catch {}
    };
    return () => source.close();
  }, [open]);
  if (!open) return null;
  const visible = logs.filter((entry) => filter === "ALL" || String(entry.level || "INFO").toUpperCase() === filter);
  return (
    <aside className="console-drawer">
      <header><div><TerminalSquare size={15} /><strong>BACKEND CONSOLE</strong><span><i /> LIVE</span></div><div className="console-actions">{["ALL", "INFO", "WARNING", "ERROR"].map((level) => <button className={filter === level ? "active" : ""} onClick={() => setFilter(level)} key={level}>{level === "WARNING" ? "WARN" : level}</button>)}<button onClick={() => setLogs([])}>Xóa</button><button onClick={onClose}><X size={14} /></button></div></header>
      <div className="console-body">{visible.length ? visible.map((entry, index) => <div className={`log-${String(entry.level || "info").toLowerCase()}`} key={index}><time>{entry.time || entry.timestamp || ""}</time><b>[{entry.level || "INFO"}]</b><span>{entry.logger && `${entry.logger}: `}{entry.message || entry.msg || JSON.stringify(entry)}</span></div>) : <p>Đang chờ log từ backend...</p>}</div>
    </aside>
  );
}

createRoot(document.getElementById("root")).render(<App />);
