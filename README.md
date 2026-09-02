# AIC 2026 — Online Search Pipeline

Hệ thống tìm kiếm video cho ba tác vụ vòng sơ tuyển AIC 2026: **Textual KIS**,
**Q&A (VQA)** và **TRAKE**. Dự án giả định dữ liệu đã được index: Qdrant lưu
embedding của keyframe, còn Elasticsearch lưu OCR, ASR và mô tả khung hình.

```
query
  │
  ├─ LLM decompose ──► {ocr_query, asr_query, image_query, modality_weights}
  │
  ├─ OCR    ─► Elasticsearch (ocr_text)              ─► top-K
  ├─ ASR    ─► Elasticsearch (asr_text)              ─► top-K
  └─ VISUAL ─► SigLIP + BEiT-3 + Qwen3-VL text towers ─► Qdrant ─► top-K
                 (embedding models on CPU; vector fusion inside Qdrant)
                            │
                            ▼
              Weighted RRF (Adaptive Score Fusion)
                            │
                            ▼
          Qwen3-VL-Reranker-2B (top-N fused, image + metadata, GPU)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     KIS                  VQA                 TRAKE
  top-100 → CSV     VLM per shot → answer   N-step DP → CSV
```

---

## 1. Cài đặt và cấu hình

### 1.1. Yêu cầu

- Python 3.10 trở lên.
- Node.js 20 trở lên và `npm` để cài hoặc phát triển giao diện React.
- Quyền truy cập Qdrant, Elasticsearch và kho keyframe (thư mục cục bộ hoặc
  MinIO).
- NVIDIA API key cho LLM/VLM. GPU CUDA được khuyến nghị nếu bật
  `rerank.qwen3_vl`; cấu hình mặc định chạy reranker trên CUDA.

Tại thư mục gốc của dự án, chạy:

```bash
make install
```

Lệnh này tạo `.venv`, cài dependency Python và chạy `npm ci` cho frontend. Nếu
chưa có `.env`, lệnh cũng sao chép `.env.example` thành `.env`. Các lệnh `make`
dùng trực tiếp Python trong `.venv`, vì vậy không cần kích hoạt virtualenv.

Nếu chỉ chạy backend bằng bundle React có sẵn, có thể cài riêng phần Python:

```bash
make install-backend
test -f .env || cp .env.example .env
```

### 1.2. Khai báo biến môi trường

Mở `.env` và điền các thông tin kết nối cần thiết. Không commit file `.env`.

| Biến | Mục đích |
|---|---|
| `NVIDIA_API_KEY` | LLM phân rã truy vấn và VLM mặc định |
| `QDRANT_API_KEY` | Xác thực Qdrant; để trống nếu không dùng auth |
| `ES_USER` / `ES_PASSWORD` | Basic authentication của Elasticsearch |
| `ES_API_KEY` | API key Elasticsearch, dùng thay user/password |
| `VLM_API_KEY` | Key riêng cho VLM; mặc định dùng `NVIDIA_API_KEY` |
| `VLM_MODEL` | Model đa phương thức chính, mặc định `moonshotai/kimi-k3` |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | Ghi đè kết nối MinIO trong YAML khi cần |

### 1.3. Trỏ cấu hình đến dữ liệu

Kiểm tra `config/config.yaml`, tối thiểu gồm:

- `qdrant.url`, `qdrant.collection`, `qdrant.vector_names` và
  `qdrant.payload`;
- `elasticsearch.hosts`, `elasticsearch.index` và `elasticsearch.fields`;
- model và `dim` của các encoder đang bật trong `embedding.*` — phải giống hoàn
  toàn model đã dùng khi index;
- `keyframes.root` nếu dùng ảnh cục bộ, hoặc phần `minio` nếu lấy ảnh từ MinIO;
- `rerank.qwen3_vl.device`: chuyển sang `cpu` hoặc tắt `enabled` nếu máy không có
  CUDA.

Kiểm tra kết nối trước, chưa cần tải các model embedding:

```bash
.venv/bin/python -m src.cli inspect --skip-encoders
```

Sau đó chạy kiểm tra đầy đủ để đối chiếu kích thước vector của encoder với
collection Qdrant:

```bash
.venv/bin/python -m src.cli inspect
```

Nếu thiếu checkpoint/reference code của BEiT-3, đặt
`embedding.beit3.enabled: false`; pipeline sẽ tiếp tục bằng các encoder còn bật.

---

## 2. Sử dụng giao diện web

### 2.1. Khởi động

Cách đơn giản nhất là chạy FastAPI cùng bundle React đã build sẵn:

```bash
make backend
```

Mở <http://localhost:7860>. Các đèn trạng thái ở góc trên cho biết kết nối tới
Elasticsearch và Qdrant. API docs nằm tại <http://localhost:7860/api/docs>.

Quy trình sử dụng:

1. Chọn tác vụ **KIS**, **VQA** hoặc **TRAKE** trên thanh đầu trang.
2. Nhập truy vấn rồi nhấn **Chạy pipeline** (hoặc Enter). Với TRAKE, mô tả các
   sự kiện theo đúng thứ tự thời gian.
3. Theo dõi tiến trình và lỗi trong pipeline trace; nút **Live logs** mở log
   backend theo thời gian thực.
4. Kiểm tra kết quả. Có thể ghim một kết quả lên Top 1, mở **Xem 25 frame** để
   chọn keyframe lân cận, sửa câu trả lời VQA, hoặc thay frame của từng bước
   TRAKE.
5. Nhấn **Xuất CSV** sau khi đã chốt thứ tự và nội dung kết quả.

Tên file tải từ web là `kis_submission.csv`, `vqa_submission.csv` hoặc
`trake_submission.csv`. Trước khi nộp, đổi tên file theo đúng tên query của ban
tổ chức, ví dụ `query-p2-1-kis.txt` thành `query-p2-1-kis.csv`.

Nút cấu hình cho phép chỉnh retrieval paths, fusion, reranker, tham số tác vụ,
submission và API-key pool cho lần chạy hiện tại. Các thay đổi này chỉ tồn tại
trong phiên giao diện/request; muốn dùng làm mặc định sau khi khởi động lại, hãy
sửa `config/config.yaml`. API key nhập trên giao diện cũng chỉ được giữ trong bộ
nhớ của backend.

### 2.2. Chế độ phát triển frontend

Chạy backend và Vite hot reload cùng lúc:

```bash
make full
```

Mở <http://localhost:5173> và nhấn `Ctrl+C` để dừng cả hai tiến trình. Có thể đổi
host, port và config mà không sửa Makefile:

```bash
make backend BACKEND_PORT=8000 CONFIG=config/config.yaml
make frontend FRONTEND_PORT=3000 BACKEND_URL=http://127.0.0.1:8000
```

Sau khi sửa frontend, build lại bundle được FastAPI phục vụ:

```bash
make build-frontend
```

---

## 3. Sử dụng CLI và tạo bài nộp

Các ví dụ dưới đây gọi trực tiếp Python trong `.venv`. Nếu đã chạy
`source .venv/bin/activate`, có thể thay `.venv/bin/python` bằng `python`.

### 3.1. Chạy một truy vấn

```bash
# Xem kết quả phân rã truy vấn
.venv/bin/python -m src.cli decompose --query "Tìm video có người mặc áo đỏ"

# Chạy retrieval và in top 20 kèm đóng góp của từng path
.venv/bin/python -m src.cli search --query "Tìm video có người mặc áo đỏ" --topk 20

# Sinh CSV cho từng tác vụ
.venv/bin/python -m src.cli kis queries/query-1-kis.txt
.venv/bin/python -m src.cli qa queries/query-3-qa.txt
.venv/bin/python -m src.cli trake queries/query-4-trake.txt
```

Mặc định CSV được ghi vào `outputs/submission/<tên-query>.csv`. Dùng `--out` để
chọn đường dẫn khác, hoặc `--dry-run` để chạy pipeline và xem trước mà không ghi
CSV.

### 3.2. Chạy hàng loạt

Tên file query phải kết thúc bằng `-kis.txt`, `-qa.txt` hoặc `-trake.txt` để CLI
tự nhận diện tác vụ:

```bash
.venv/bin/python -m src.cli batch queries/ --out-dir outputs/submission/
```

Mỗi file lỗi được ghi nhận riêng; `batch` tiếp tục xử lý các file còn lại và in
bảng tổng kết ở cuối.

### 3.3. Kiểm tra và đóng gói

```bash
.venv/bin/python -m src.cli validate --dir outputs/submission/
.venv/bin/python -m src.cli pack --dir outputs/submission/ --out teamABC1.zip
```

`pack` chỉ tạo ZIP sau khi toàn bộ CSV hợp lệ. Bên trong ZIP luôn có cấu trúc
`submission/<tên-query>.csv` theo yêu cầu cuộc thi.

Các cờ dùng chung: `--config PATH`, `--no-cache`, `-v`/`--verbose` và
`--dry-run`. Dùng `.venv/bin/python -m src.cli --help` hoặc thêm `--help` sau
từng subcommand để xem toàn bộ tùy chọn.

### 3.4. Xử lý lỗi thường gặp

- **ES/Qdrant báo offline:** kiểm tra URL, index/collection, credential trong
  `.env`, sau đó chạy lại `inspect --skip-encoders`.
- **Sai kích thước vector:** model hoặc `dim` trong `embedding.*` không khớp lúc
  index; không nên bỏ qua lỗi này.
- **Không tải được ảnh:** kiểm tra `keyframes.root`; nếu dùng MinIO, kiểm tra
  endpoint, bucket, prefix và credential.
- **CUDA out of memory:** giảm `rerank.qwen3_vl.top_n`/`max_pixels`, chuyển
  reranker sang CPU hoặc tắt reranker.
- **NVIDIA API lỗi:** kiểm tra API key. Fallback mặc định cần Ollama đang chạy và
  có model `qwen3-vl:2b-instruct`; nếu không dùng fallback, đặt
  `llm.fallback.enabled: false` và `vlm.fallback.enabled: false`.
- **Kết quả tìm kiếm chưa tốt:** xem `outputs/runs/<timestamp>_<query>.json` để
  kiểm tra decomposition, kết quả từng path, fusion weights và top fused.
  Log tổng nằm ở `outputs/runs/run.log`.

---

## 4. Why the ranking looks the way it does

Scoring is `R@k = max` over the first `k` rows, averaged over
`k ∈ {1, 5, 20, 50, 100}`. Three consequences are built into the code:

1. **Always submit 100 rows.** A deep row can only ever help; it is a free lottery
   ticket. `submission.max_rows` is only a cap, never a target to fall short of.
2. **The head is expensive.** `R@1` and `R@5` are two of the five terms, so
   `submission/builder.py` forces the first `top_diverse` rows onto different
   shots *and* caps how many may come from one video — five frames of a single
   wrong shot would throw away four fifths of a term.
3. **The tail is cheap.** The ground-truth window `[s, e]` is narrow (under 10
   frames for TRAKE) while keyframes are sparse, so from `start_rank` onward the
   builder splices in neighbouring keyframes of strong candidates.

---

## 5. TRAKE: the alignment algorithm

1. An LLM reads the query and emits `N` chronologically ordered key moments —
   `N` is taken from the query itself when it states one.
2. Every step is retrieved independently through the normal four-path pipeline
   (steps run in parallel).
3. Step scores are min-max normalised **per step**, so no step dominates the
   objective and `miss_penalty` means the same thing everywhere.
4. Videos are ranked by summed best-per-step score plus a coverage bonus; the
   top `max_videos` go to alignment. The DP always runs *inside* one video, which
   is what guarantees the single-video output the rules require.
5. Inside a video, an **anchored bidirectional DP** picks one frame per step:

   * `prefix[j][i]` — best score covering steps `0..j` with step `j` at frame `i`
   * `suffix[j][i]` — best score covering steps `j..N-1` with step `j` at frame `i`
   * the best sequence through anchor candidate `i` is
     `prefix[a][i] + suffix[a][i] − score[a][i]`

   Transitions use a sliding-window maximum (a monotonic deque), so the temporal
   constraints `min_gap` / `max_gap` cost nothing extra. Complexity is
   `O(N² · K log K)` per video — never `O(K^N)`. Enumerating anchor candidates
   also yields genuinely distinct runner-up paths for free.
6. A step with no candidate is **skipped, not fatal**: it costs `miss_penalty`,
   and its frame is interpolated between its neighbours (extrapolated at the
   ends) then snapped onto a real keyframe. TRAKE scores a *fraction* of matched
   moments — 3 of 4 in the right video is 0.75, while dropping the right video is 0.
7. Sequences are ranked globally and the first five rows are spread across at
   least `head_min_videos` videos, because a wrong video scores exactly zero.

`refine_frame(video_id, frame_id, step_description)` in `tasks/trake.py` is an
identity hook left in place for a future dense local search around a candidate.

Works for any `N ≥ 1`; tested at `N = 2, 3, 4, 6`. `anchor_index` is clamped
into range automatically.

---

## 6. Configuration reference

### `qdrant`
| key | meaning |
|---|---|
| `url`, `api_key`, `timeout` | connection (`api_key` comes from the env) |
| `collection` | collection holding one point per keyframe |
| `vector_names.siglip` / `.beit3` | named vectors inside that collection |
| `payload.video_id` / `.frame_id` | payload fields carrying the identity |
| `prefetch_limit` | candidates each vector branch retrieves before fusing |
| `search_limit` | results returned after fusion |
| `fusion` | `rrf` \| `dbsf` \| `manual` (manual = fuse client-side) |

### `elasticsearch`
| key | meaning |
|---|---|
| `hosts`, `index`, `timeout`, `verify_certs` | connection |
| `fields.ocr` / `.asr` / `.video_id` / `.frame_id` | field names in your index |
| `size` | hits per text path |
| `min_score` | drop weak BM25 hits (0 = keep all) |
| `boosts.match` / `.phrase` / `.terms` | clause weights in the bool query |
| `boosts.exact_phrase` | phrase boost when the query wants literal on-screen text |
| `phrase_slop` | `match_phrase` slop |

### `embedding.siglip` / `embedding.beit3`
| key | meaning |
|---|---|
| `enabled` | turn the encoder — and its Qdrant branch — off |
| `backend` | `transformers` \| `open_clip` (SigLIP), `torchscale` \| `transformers` (BEiT-3) |
| `model_id` | **must match the model used at indexing time** |
| `dim` | asserted against the collection's vector width at start-up |
| `device` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `checkpoint_path`, `tokenizer_path` | BEiT-3 weights and `beit3.spm` |
| `max_length`, `batch_size` | tokenisation and batching |

### `llm` / `vlm`
| key | meaning |
|---|---|
| `model` | model id (`vlm.model` must be a vision model) |
| `temperature`, `top_p`, `max_tokens` | sampling |
| `enable_thinking` | `<think>` blocks; the JSON parser strips them |
| `max_retries`, `retry_backoff` | transport retries on timeout, 429 and 5xx |
| `llm.json_retries` | retries when a completion is not valid JSON |
| `timeout` | HTTP timeout in seconds for OpenAI-compatible providers |
| `fallback.*` | secondary provider/model/base URL used after the primary fails |
| `vlm.max_workers` | frames asked in parallel |
| `vlm.image_max_side` | downscale before base64 to save tokens |

VQA defaults to multimodal mode. For each candidate video, Kimi K3 receives up
to `vqa.vlm_images_per_video` action/evidence frames in one request. Each image
is interleaved with its Description, OCR and ASR metadata, so the model can use
both the pixels and the indexed text when answering. The primary model comes
from `VLM_MODEL` in `.env` (the existing `VLM_model` spelling is also accepted).

The default configuration makes one NVIDIA/Kimi attempt and then falls back to
the local Ollama OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1`,
using `qwen3-vl:2b-instruct`. The same multi-image and metadata payload is sent
to the fallback. After a primary timeout, that client stays on the local
fallback until the backend is restarted, avoiding another remote timeout for
every candidate video.

### `fusion`
| key | meaning |
|---|---|
| `method` | `rrf` \| `weighted_rrf` \| `weighted_norm` |
| `rrf_k` | the `k` in `w / (k + rank)` |
| `weights` | static per-path weights, used when `adaptive: false` (ratios, not capped) |
| `adaptive` | use the LLM's `modality_weights` instead |
| `adaptive_floor` | minimum weight per active path, so the LLM cannot kill one |

### `submission`
| key | meaning |
|---|---|
| `max_rows` | hard cap (100) |
| `top_diverse` | rows in the protected diversity head |
| `head_max_per_video` | max head rows from one video |
| `shot_window` | frames closer than this count as the same shot |
| `neighbor_expansion.enabled` | splice neighbouring keyframes into the tail |
| `neighbor_expansion.start_rank` | first rank expansion may occupy |
| `neighbor_expansion.offsets` | frame offsets to try, nearest first |

### `vqa`
| key | meaning |
|---|---|
| `mode` | `vision` (default: images + Description/OCR/ASR) or `text_only` |
| `vlm_top_n` | how many representative action-scene frames reach the answerer |
| `vlm_images_per_video` | maximum action/evidence images combined in one VLM request per video |
| `cross_shot_evidence` | search answer-bearing shots elsewhere in the same candidate videos |
| `evidence_video_top_n` | number of top scene-matched videos searched for secondary evidence |
| `evidence_top_n` | maximum secondary evidence frames sent to the answerer |
| `propagate` | reuse answers within the same shot/video before global fallback |
| `answer_max_chars` | hard limit (100 per the rules) |
| `fallback_answer` | used when everything is UNKNOWN — never left empty |

### `trake`
| key | meaning |
|---|---|
| `anchor_index` | 0-based anchor step (clamped into range) |
| `per_step_topk` | candidates retrieved per step |
| `max_videos` | videos handed to the DP |
| `paths_per_video` | distinct sequences kept per video |
| `coverage_bonus` | reward per step a video actually covers |
| `miss_penalty` | cost of interpolating a step |
| `min_gap` / `max_gap` | temporal constraints between consecutive steps |
| `allow_fill` | permit skipped-and-interpolated steps at all |
| `head_min_videos` / `head_window` | video spread required in the head |

### `rerank`, `cache`, `runs`
`rerank.qwen3_vl` applies `Qwen/Qwen3-VL-Reranker-2B` once to the fused top-N.
It scores the query against each keyframe image plus available Description/OCR/ASR
metadata. The embedding models use CPU while this reranker uses CUDA. Legacy
`rerank.blip2` and `rerank.bge` remain parse-compatible but are disabled. `cache.dir` holds the LLM,
VLM, embedding and keyframe-index caches — reruns cost no quota. `runs.dir`
holds the per-query JSON traces and `run.log`.

---

## 7. Submission format

Enforced by `submission/writer.py` and re-checked by `submission/validator.py`:

* UTF-8 **without** BOM, `,` delimiter, `\n` line endings, **no header row**
* at most 100 rows; `video_id` carries **no** `.mp4`; `frame_id` is an integer
* Q&A answers ≤ 100 characters, never empty; quoted only when they contain
  `,` `"` or a newline, with inner `"` escaped as `""` (`csv.QUOTE_MINIMAL`)
* TRAKE rows all carry exactly `num_events` frames in increasing order —
  the writer raises rather than emit a row that would fail to parse
* the zip contains a `submission/` folder; bare CSVs at the root are rejected

`pack` runs the validator first and refuses to build an invalid archive — a
malformed file still burns one of your three submissions.

---

## 8. Tests

```bash
pytest -q
```

169 tests, no network access required — the LLM, VLM, Qdrant and Elasticsearch
are all faked.

| file | covers |
|---|---|
| `test_fusion.py` | RRF / weighted RRF formulas, single-path candidates, determinism, weight normalisation |
| `test_trake_dp.py` | the DP at `N = 2, 3, 4, 6`; `min_gap`/`max_gap` never violated; step filling and its penalty; anchors; head diversity |
| `test_builder.py` | shot and video diversity in the head, neighbour expansion, row caps |
| `test_writer.py` | CSV escaping, no header, ≤ 100 rows, TRAKE width assertions |
| `test_validator.py` | every rejection class, plus zip layout |
| `test_utils.py` | JSON extraction from `<think>`/fenced/chatty output, answer cleanup, keyframe mapping |
| `test_pipeline.py` | path dispatch, failure isolation, adaptive vs static weights, run traces |
| `test_end_to_end.py` | all three tasks from query to a CSV the validator accepts |
| `test_cli.py` | global flags on either side of the subcommand, `validate`/`pack` exit codes |

---

## 9. Layout

```
src/
├── cli.py                    entrypoint, one subcommand per operation
├── config.py                 YAML + env → validated dataclasses
├── schemas.py                Candidate, DecomposeResult, TrakePlan, ...
├── clients/                  qdrant, elastic, llm (JSON-hardened), vlm
├── embedding/                SigLIP / BEiT-3 text towers, lazy + cached
├── retrieval/                decompose, search_text, search_visual, fusion,
│                             rerank, pipeline
├── tasks/                    kis, vqa, trake
├── submission/               builder, writer, validator, packer
└── utils/                    keyframe_index, cache, text_norm
```
