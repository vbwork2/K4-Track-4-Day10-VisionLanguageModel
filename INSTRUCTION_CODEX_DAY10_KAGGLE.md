# CODEX INSTRUCTION — Day 10 VLM · Kaggle QLoRA · Evidence · Resume

Repository: https://github.com/vbwork2/K4-Track-4-Day10-VisionLanguageModel  
Nguồn yêu cầu bắt buộc: `README.md`, `GUIDE.md`, `RUBRIC.md`, `REPORT.md`, `configs/e0.yaml`…`configs/e5.yaml` (đọc lại trên branch hiện hành trước khi sửa).

## Vai trò / nguyên tắc không thương lượng

Bạn là kỹ sư ML triển khai bài lab **cá nhân**. Hoàn thiện phần mã còn placeholder, sao cho tác giả có thể tự chạy và tự giải thích được kết quả. **Không tự bịa số liệu, kết quả GPU, log hoặc câu trả lời báo cáo.** Khi không có GPU/Kaggle credentials, chỉ xác nhận unit test, smoke test CPU và để kết quả train là `NOT_RUN`; không tự tạo `report/evidence/*.json` giả.

**Không sửa hoặc ghi đè** bất kỳ file nào bên trong `scripts/` và `tests/` do đề cung cấp. Giữ cấu trúc / format `REPORT.md` (bảng và 10 câu). Không đổi split, seed, dataset_version, metric, ground-truth; không sửa `GUIDE.md` / `RUBRIC.md` để “qua bài”. Không commit ảnh dataset, model weight, adapter, optimizer, token, logs lớn, `data/`, `work/`, `.env` hoặc checkpoint. Chỉ thêm mã nguồn, notebook, cấu hình bổ sung, tài liệu ngắn, test của người làm vào vị trí **riêng** ngoài `tests/` gốc (ví dụ `student_tests/`).

### Việc đầu tiên
1. `git status -sb; git log -5 --oneline; find . -maxdepth 3 -type f | sort`; đọc hết README, GUIDE, RUBRIC, REPORT, configs và kiểm tra lại `lab_day10.ipynb`.
2. Phát hiện nhánh hiện hành; dùng feature branch `feat/day10-kaggle-qlora` nếu chưa ở feature branch; **không force push, không xóa history, không reset thay đổi người dùng**.
3. Chạy `python -m pytest -q` *trước* khi sửa, ghi baseline; nếu thiếu dependencies, cài và ghi trạng thái.
4. Tạo bảng checklist đối chiếu từng yêu cầu với từng file/mã kiểm tra; giữ bảng này ở cuối `docs/IMPLEMENTATION_STATUS.md`. Đừng đánh dấu GPU train hoàn thành khi chưa chạy thật.
5. Tôn trọng notebook Kaggle kèm theo `kaggle_day10_qlora_resume.ipynb`: có thể tái sử dụng/refactor và đồng bộ module vào repo, nhưng tránh có 2 pipeline khác nhau sinh kết quả khác nhau.

## Kiến trúc đề xuất (có thể chỉnh nếu có lý do kỹ thuật rõ ràng)

```
lab_day10.ipynb                   # hoàn thiện 9 mục của notebook gốc, không bỏ mục
kaggle_day10_qlora_resume.ipynb   # entrypoint Kaggle, chứa/nhúng workflow train và resume
student/labels.py                # YOLO -> bbox_2d, order ổn định, thang 0–1000
student/dataset.py               # image+messages; loss chỉ trên assistant
student/model.py                 # model 4-bit, whitelist/skip ViT và lm_head
student/training.py              # E1/E2/E3, GradScaler, AdamW, cosine
student/checkpointing.py         # full-state save/load + remote backup
student/inference.py             # E0/E1/E2/E3/E4, JSONL incremental
student/experiments.py           # orchestrator, setup E0–E4, config fingerprint
student/reward.py                # E5 tùy chọn: reward IoU có unit tests
student_tests/                   # tests thêm của sinh viên, không chạm tests/ gốc
docs/KAGGLE_RUNBOOK.md           # hướng dẫn Kaggle + secrets + backup + recovery
docs/IMPLEMENTATION_STATUS.md    # checkpoint, lệnh test, bằng chứng, giới hạn
```

Đường dẫn có thể khác để khớp repo, nhưng các trách nhiệm phải rõ. Giữ `scripts/score.py`, `scripts/prepare_data.py`, `scripts/check_submission.py` **nguyên byte**. Không thay `configs/e1.yaml`–`e3.yaml` để lách khác biệt setup; nếu cần batch micro nhỏ hơn cho GPU Kaggle, ghi override vào run manifest và báo cáo, giữ **effective batch = batch_size × grad_accum** khi có thể.

## Checkpoint triển khai: 1 checkpoint công việc = ít nhất 1 commit Git

**CP00 — Audit, plan, môi trường (`chore(day10): audit lab requirements and kaggle workflow`)**
- Liệt kê yêu cầu từ rubric, kiểm tra package pinned ở `requirements-colab.txt`, GPU runtime, dataset mount Kaggle.
- Check `python -m pytest -q`, lưu kết quả và log môi trường tại `docs/IMPLEMENTATION_STATUS.md` (không ghi token).
- Commit các file tài liệu/plan ngay sau khi test.

**CP01 — Dữ liệu chuẩn, chống leakage (`feat(day10): prepare grouped splits and validate labels`)**
- Gọi nguyên bản `scripts/prepare_data.py --source <KAGGLE_INPUT_FOLDER>` nếu dataset đã Add Input, hoặc `--kaggle` nếu có Kaggle credentials; **không tự chia train/val/test**.
- Đọc `data/splits/{train,val,test}.jsonl`, `meta.json`; assert session không giao nhau; so sánh số mẫu và thống kê classes với README.
- Vẽ ≥ 6 ảnh overlay nhãn để kiểm tra trực quan. Kiểm tra và log `dataset_version`.
- Unit tests YOLO → JSON 0–1000 → vẽ pixel, clipping 0–1000, `[]`, thứ tự `(y1, x1)`.
- Commit riêng.

**CP02 — Encoding/masking (`feat(day10): implement vlm chat encoding and masked collator`)**
- Viết system/user prompt trả **DUY NHẤT** JSON array `[ {"label":"...", "bbox_2d":[x1,y1,x2,y2]} ]` với classes `pothole/crack/manhole`, toạ độ **nguyên 0–1000**.
- Xây user message chứa ảnh PIL và prompt, assistant message chứa đáp án.
- `processor.apply_chat_template` cho prompt-only và full; **assert prompt input_ids là tiền tố của full input_ids**.
- Mask `labels[:prompt_len] = -100`, pads `-100`; không tính loss trên prompt/vision.
- Collator pad text nhưng **concat** `pixel_values` và `image_grid_thw` đúng kích thước, không pad vision; hỗ trợ `mm_token_type_ids` nếu API trả về.
- Log và kiểm tra số image token thực tế (đề kỳ vọng **220/ảnh 640×360**) và decode các token labels; nếu processor version khác thì fail rõ + điều tra, không sửa metric.
- CPU tests và commit.

**CP03 — Model/LoRA setup (`feat(day10): configure quantized qwen3 vl lora variants`)**
- Model ID: `Qwen/Qwen3-VL-4B-Instruct`. NF4/double quant, fp16, một GPU; bỏ qua lượng tử hoá **toàn bộ `visual` và `lm_head`** theo GUIDE.
- In và **assert** module names thật trước khi set target. E1/E2: 252 LoRA linear modules thuộc LLM (36×7). E3: 348 (=252+24×4) gồm ViT. Nếu upstream API thay đổi, điều tra và báo minh bạch, không giả số.
- E1: ViT+projector frozen, LoRA LLM. E2: E1 + full-train `visual.merger` và `visual.deepstack_merger_list` (~109M). E3: E2 + LoRA ViT attention+MLP.
- Huấn luyện merger ở fp32; đảm bảo dtype của đầu vào module phù hợp, gradient chảy qua ViT khi E3, và **lưu độc lập toàn bộ state của projector**. LoRA adapter đơn thuần không chứa projector.
- Unit/smoke (nếu GPU có), số trainable params, dtype, trainable names; commit.

**CP04 — E0 zero-shot (`feat(day10): implement E0 zero shot inference and scoring`)**
- Suy luận greedy, `max_new_tokens: 256`, không nhìn nhãn test khi viết prompt.
- Mỗi image_id đúng một dòng JSONL `{"image_id":"...","raw":"..."}`, giữ raw text để debug; hỗ trợ resume dự đoán dang dở bằng cách bỏ qua ID đã ghi, flush sau mỗi ảnh.
- Chỉ khi có **full 311 test predictions** chạy **nguyên bản** `python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json`.
- In 5 raw outputs, parse_valid_rate, ví dụ lỗi; không điền số giả.
- Commit mã; tách commit evidence khi đã chạy GPU thật.

**CP05 — Training, auto-logs, durable resume (`feat(day10): add resumable fp16 qlora training and remote checkpoints`)**
- Implement AdamW, LR=2e-4, warmup 5%, cosine, max_steps=300, grad accumulation×4, batch=2 theo configs; nếu trên Kaggle OOM thử batch=1/accum=8, ghi rõ override.
- GradScaler fp16, finite-loss assertion, clip grad norm=1.0, gradient checkpointing (use_reentrant=False), disable `use_cache` khi train.
- **Sau mỗi optimizer step:** append+flush `metrics.jsonl` (step, loss, LR, grad_norm, scaler, elapsed, max_gpu_GB, throughput nếu đo được). Log text `train.log` / TensorBoard optional, JSON run manifest ghi thư viện/version/git SHA/dataset_version/seed/setup/model/LoRA/optimizer/start/end time/exception.
- **Mỗi N optimizer steps (khuyến nghị 25; cấu hình được):** checkpoint **đầy đủ** gồm LoRA adapter, projector state E2/E3, optimizer, scheduler, scaler, torch/cuda/python/numpy RNG states, global_step, config fingerprint, dataset_version, trainable_params. Save **atomic**, đánh dấu `complete` sau cùng, đọc lại verify trước khi backup.
- Nếu session Kaggle chết, checkpoint tại `/kaggle/working` chỉ là bản **best-effort**. Phải hỗ trợ upload lên **private Hugging Face model repo** bằng Kaggle Secrets (`HF_TOKEN`, `HF_REPO_ID`) sau mỗi checkpoint và tự tải bản mới nhất còn hợp lệ ở lần chạy sau. Không đưa secret lên stdout/notebook/git. **Không được tuyên bố restart từ checkpoint được sau hard reset nếu chưa test remote upload/download.** Thiếu HF credential thì fail trước training khi yêu cầu durable resume.
- Resume: ưu tiên checkpoint local hợp lệ, so với remote step và dùng checkpoint mới hơn; đối chiếu config hash/dataset_version, restore mọi state rồi train từ `global_step+1`, không lặp bước hoặc xoá log. Nếu remote sync lỗi: retry có backoff, cảnh báo **FAIL_BACKUP**, giữ checkpoint local; cấu hình fail-closed nếu yêu cầu đảm bảo bền vững.
- Tests cho: corrupt/incomplete checkpoint, mismatch dataset/setup, restore optimizer+step+RNG, interrupted→resumed step monotonic, thiếu remote token.
- Commit riêng ngay sau khi test.

**CP06 — E1/E2/E3 evaluation (`feat(day10): evaluate all supervised variants with canonical scorer`)**
- Chạy từng setup trên Kaggle, lấy **log/số thật**; không cố chạy 3 setup trong 1 session nếu thiếu GPU.
- Sau train, load **base + đúng LoRA + đúng projector** (nếu E2/E3); chạy val nhỏ để debug, test đầy đủ **chỉ khi finalized**.
- Tạo `work/e1.jsonl`, `work/e2.jsonl`, `work/e3.jsonl` rồi gọi `scripts/score.py --split test` kèm đủ `--meta trainable_params=… --meta peak_vram_gb=… --meta train_minutes=…` (ba số **dương**, chính xác).
- Evidence `report/evidence/e1.json`, `e2.json`, `e3.json` do scorer sinh; kiểm tra consistency `dataset_version`. `parse_valid_rate(E1) >= .95` và `F1(E1)-F1(E0) >= .10` là **mục tiêu rubric**, không phải kết quả mặc định; nếu không đạt, debug training/mask/scale.
- Tạo ảnh dự đoán đúng/sai với GT/Pred ≥6 ảnh và chú thích vào `report/evidence` (tệp ảnh nhỏ mới được commit).
- Commit mã trước, evidence và báo cáo chỉ khi đã có kết quả thật.

**CP07 — E4 ablation (`feat(day10): add single variable ablation and comparison`)**
- Chọn **MỘT** trong `rank`, `data_size`, `coord_scale`; dễ triển khai Kaggle nhất: **rank 4 và 64**, alpha=2r, giữ toàn bộ yếu tố khác so với baseline E1/E2/E3 đã chọn. Nếu rank 16 baseline thì **không** được lấy làm đối chứng cho rank 4/64; chạy đủ 2 giá trị thật.
- Không đổi data split, prompt, max_steps, optimizer, seed, eval protocol giữa hai lần run. Nếu chọn `coord_scale=pixel`, đổi đồng bộ nhãn+prompt+`--coord pixel` (không đụng GT/scorer).
- Tạo hai file evidence dạng `report/evidence/e4_rank_4.json` và `...64.json` (hoặc tên đúng biến đã chọn); so CI95 và chi phí.
- Commit.

**CP08 — E5 bonus (tùy thời gian/GPU) (`feat(day10): implement optional iou reward and grpo`)**
- Tạo reward Python thuần dựa trên `scripts.score.parse_output` và `iou_xyxy`: JSON hỏng→0; GT/pred đều rỗng→1; match greedy cùng lớp IoU≥0.5; không rỗng→0.2 + 0.8 × (tổng IoU matched)/max(n_gt,n_pred,1).
- Tests 5 tình huống: perfect, malformed, wrong class, hallucinated box, both empty; rồi mới thử `trl==1.9.2` GRPO 4 generations, max completion 128, ~60 steps, `use_vllm=False`.
- Nếu không đủ VRAM/thời gian, hoàn thành reward tests và đánh dấu `NOT_RUN` cho GRPO. Không bịa log.
- Commit.

**CP09 — Report/QA (`docs(day10): complete evidence backed report and submission checks`)**
- Điền **đúng 10 câu** của `REPORT.md` (tiếng Việt ngắn, giải thích bằng hiểu biết thực tế và số đo đúng), giữ nguyên heading và bảng.
- Bảng E0–E3: F1, CI95, FP/image, per-class F1, VRAM, minutes; trích JSON evidence đã sinh; E4 so hai biến thể.
- Test `python -m pytest -q`, `python scripts/check_submission.py`. In status PASS/FAIL; nếu thiếu GPU/evidence, giữ FAIL đúng bản chất thay vì fabricate.
- Xem `git status`, `git ls-files | grep` forbidden; ≥5 commits có ý nghĩa; push branch & tạo PR nếu quyền cho phép, không merge vào main khi chưa được duyệt.

## Acceptance criteria bắt buộc
- README/GUIDE/RUBRIC/tools/tests nguyên bản; `python -m pytest -q` PASS.
- Code thực sự implement E0/E1/E2/E3/E4 và có thể chạy Kaggle, không còn `YOUR CODE HERE` trong notebook cần nộp.
- Split session không leak, nhãn norm1000 và loss mask đúng; model target/dtype/trainable assertions đúng.
- Đủ log metrics theo step, checkpoint atomic với **resume sau ngắt phiên trên Kaggle qua HF private repo** đã test bằng train ngắn → kill → reboot → resume từ step > 0.
- Projector riêng phải xuất hiện và được khôi phục trong E2/E3.
- Evidence chỉ sinh bằng `scripts/score.py` từ output thật; báo cáo bằng tiếng Việt, ngắn gọn và có số liệu.
- Từng CP có commit tách biệt; mỗi commit ghi `git show --stat --oneline HEAD` + kết quả test/status vào `docs/IMPLEMENTATION_STATUS.md` (tránh commit đệ quy do update status; có thể cập nhật status ở commit kế tiếp).

## Định dạng báo kết thúc của Codex

```text
Checkpoint | Commit SHA | Files | Test | GPU/CPU | Status
CP00 ... CP09

Kaggle notebook: <repo path>
Runbook Kaggle: <repo path>
Evidence thật: <liệt kê file, hoặc NOT_RUN>
Kaggle resume test: LOCAL_PASS / REMOTE_PASS / NOT_RUN (lý do)
Known blockers: ...
```

**Bắt đầu ngay theo thứ tự, không hỏi lại khi dữ liệu đã có trong repo. Sau mỗi checkpoint hoàn tất test thì COMMIT ngay.**
