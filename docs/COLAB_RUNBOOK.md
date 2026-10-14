# Chạy lại bài Day 10 trên Colab

Notebook chạy thực tế: `colab_day10_qlora_resume.ipynb`; notebook theo bố cục đề: `lab_day10.ipynb`. Cả hai gọi pipeline trong `student/`.

1. Mở notebook từ nhánh `main`, chọn runtime T4 GPU.
2. Chạy các cell theo thứ tự. Mặc định `EXPERIMENTS=["E0", "E1", "E2", "E3", "E4"]`; có thể chọn riêng một thí nghiệm. Dataset công khai được tải bằng `kagglehub`, chia bằng script gốc. Có thể dùng `DATASET_ZIP` hoặc `DATA_SOURCE_DIR`.
3. Với E1–E4, thêm Colab Secrets `HF_TOKEN` có quyền ghi và `HF_REPO_ID` trỏ tới repo private. Bật Notebook access; không ghi token vào code/output.
4. Huấn luyện 300 bước, micro-batch 1, tích luỹ 8, checkpoint mỗi 25 bước. E4 chạy rank 4/64, alpha=2r, baseline E1. Kết quả nộp cho thấy E3 tốt nhất; E4 hiện chưa đáp ứng yêu cầu ablation trên setup tốt nhất.
5. Sau gián đoạn, chạy lại cùng cấu hình và Secrets. Runner kiểm tra checkpoint local/HF và fingerprint trước khi khôi phục. ZIP được cung cấp chưa chứng minh khôi phục sau hard reset.
6. Tải `e0_e4_logs_evidence.zip` ở cell cuối. ZIP gồm dự đoán, bằng chứng, log và ảnh; trọng số/checkpoint lưu trên HF. Sau khi runtime mất, output local có thể cần sinh lại.

Notebook giữ `CODE_REVISION=19525bd179bc2c3942dbd066fd1690c04e132572` để tương thích fingerprint của lần chạy đã nộp; cell vá đồng bộ cấu hình lượng tử hoá và vòng train với mã đã chạy. Không đổi revision/cấu hình giữa chừng nếu muốn resume checkpoint cũ.

Các sửa lỗi được giữ trong mã nộp: bỏ backend `torchao` không tương thích, loại `model.visual` và `lm_head` khỏi lượng tử hoá, giảm loss scale rồi thử lại khi gradient FP16 tràn số. Scheduler và bước log chỉ tăng sau cập nhật thành công.

Bằng chứng thật E0–E4 nằm trong `report/evidence/`, nhật ký tài nguyên ở `report/logs/`; số liệu và giới hạn ở `REPORT.md`. Chạy kiểm tra nộp bằng `python scripts/check_submission.py`.
