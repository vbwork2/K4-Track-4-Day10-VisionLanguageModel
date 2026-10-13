# Chạy Lab Day 10 trên Kaggle

Notebook chính là kaggle_day10_qlora_resume.ipynb. Notebook clone đúng branch triển khai, cài package theo requirements-colab.txt, chạy CPU tests, xác nhận dữ liệu và sau đó gọi duy nhất pipeline trong student/.

## 1. Tạo Kaggle Notebook

1. Mở kaggle_day10_qlora_resume.ipynb từ branch feat/day10-kaggle-qlora trên GitHub bằng Import Notebook.
2. Trong Notebook settings, chọn GPU T4 và bật Internet.
3. Chọn Add Input và thêm dataset Road Damage có cấu trúc chứa images/ và labels-YOLO/.
4. Trong Add-ons → Secrets, tạo:
   - HF_TOKEN: Hugging Face token có quyền tạo/cập nhật model repo riêng tư.
   - HF_REPO_ID: tên repo dạng username/day10-qwen3vl-checkpoints. Tạo repo private hoặc để notebook tạo và xác nhận repo private.
5. Chạy các cell theo thứ tự. Không ghi secret vào cell output, notebook source, hoặc repository.

E0 không cần Hugging Face secret. E1–E4 dừng trước khi tải model nếu thiếu secret hoặc repo không private.

## 2. Dữ liệu và preflight

Notebook tìm dataset trong /kaggle/input, rồi gọi nguyên bản scripts/prepare_data.py --source ... --out /kaggle/working/day10_data. Split giữ nguyên logic và seed của đề. Trước khi tải model, notebook kiểm tra dataset_version, thống kê README, session không giao nhau, đường dẫn ảnh và sáu overlay ground-truth.

Notebook chạy cả tests/ gốc và student_tests/. PyTorch processor smoke test xác nhận prompt-prefix, mask assistant, collation, và kỳ vọng 220 image tokens cho ảnh 640×360. Nếu phiên bản processor trả số khác, dừng để điều tra thay vì đổi chuẩn hoặc scorer.

## 3. Chạy E0–E3

Chỉnh EXPERIMENT ở cell cấu hình thành một giá trị cho mỗi lần chạy:

- E0: zero-shot; không huấn luyện và không cần backup HF.
- E1: LoRA trên LLM; ViT/projector đóng băng.
- E2: E1 cộng projector đầy đủ ở FP32.
- E3: E2 cộng LoRA trên attention/MLP của ViT.

Mặc định notebook dùng MICRO_BATCH_SIZE=1, GRAD_ACCUM=8, effective batch 8, 300 optimizer steps, checkpoint mỗi 25 bước. Đây là override Kaggle để giảm peak VRAM; các file configs/e1.yaml–e3.yaml vẫn giữ nguyên. Mọi override được đưa vào fingerprint và run manifest.

Run artifacts nằm dưới /kaggle/working/day10_runs/<run_id>/. Mỗi run lưu JSONL metrics sau từng optimizer step, train.log, run.json, và checkpoint tại step 0, mỗi 25 bước, cùng bước cuối. Checkpoint gồm adapter, projector riêng cho E2/E3, optimizer, scheduler, GradScaler, RNG, metrics, fingerprint, dataset version và SHA-256 cho file.

Khi mở session mới, chạy lại đúng branch, experiment và cấu hình. Runner so checkpoint local với private HF repo, chọn step mới nhất còn hợp lệ, xác nhận fingerprint/dataset/setup rồi khôi phục trạng thái. Nếu upload checkpoint thất bại, run dừng với FAIL_BACKUP và giữ bản local; không tiếp tục train khi remote backup bắt buộc chưa xác nhận.

**Giới hạn kiểm chứng:** mã có luồng tải lên/tải xuống và khôi phục remote, nhưng lượt triển khai này không có Kaggle credentials hoặc Kaggle runtime nên kịch bản hard reset → remote download → resume từ step > 0 chưa được chạy thực tế. Hãy coi tính bền vững qua hard reset là chưa xác minh cho đến khi thực hiện một run ngắn có chủ đích trên Kaggle.

## 4. Suy luận và evidence

Runner chỉ gọi scripts/score.py khi JSONL chứa đúng một prediction cho mọi ảnh thuộc split yêu cầu. Evidence test được ghi ở report/evidence/; E1–E3 và E4 kèm trainable_params, peak_vram_gb, train_minutes. Sáu ảnh GT/Pred được lưu dưới report/evidence/figures/.

EVAL_SPLIT=val dành cho debug; chỉ test mới sinh evidence dùng để nộp. Không chỉnh prompt hoặc siêu tham số dựa trên test.

## 5. E4 và E5

EXPERIMENT=E4 gọi run_rank_ablation() một lượt, chạy rank 4 rồi rank 64 với baseline E1, seed 42, effective batch 8, 300 bước và cùng scoring protocol. Nếu Kaggle ngắt giữa hai run, chạy lại notebook cùng cấu hình; fingerprint riêng cho mỗi rank sẽ khôi phục đúng run. Evidence có tên e4_rank_4_base_e1.json và e4_rank_64_base_e1.json; comparator kiểm tra cùng dataset version, test split, prediction đầy đủ, CI95 và metadata chi phí.

E5 reward thuần Python có unit tests. GRPO là tùy chọn; notebook này không tự chạy GRPO.

## 6. Tải artifacts

Cell cuối tạo ZIP gồm logs, JSONL predictions, scorer evidence và hình overlays, không gồm checkpoint hoặc trọng số. Tải file từ output Kaggle. Chỉ đưa evidence thật và hình minh họa cần nộp vào GitHub; không đưa data/, work/, token, adapter, optimizer hay model weights.

REPORT.md hiện đánh dấu phần cần số liệu Kaggle là NOT_RUN. Điền báo cáo từ evidence thật sau khi chạy. python scripts/check_submission.py sẽ báo thiếu evidence cho đến khi E0–E4 thật sự hoàn tất.
