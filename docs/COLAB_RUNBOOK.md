# Chạy Day 10 trên Google Colab

Notebook chính: `colab_day10_qlora_resume.ipynb`. Notebook lab vẫn có đúng 22 cell; cả Colab và Kaggle gọi cùng pipeline `student/`.

1. Mở notebook từ nhánh `feat/day10-kaggle-qlora` trong Colab bằng tài khoản Google của bạn. Có thể lưu bản sao vào Drive để giữ cấu hình và outputs.
2. Chọn Runtime → Change runtime type → T4 GPU. Nếu tài khoản không được cấp GPU, dừng và thử lại khi có tài nguyên; notebook kiểm tra CUDA trước khi chạy model.
3. Mặc định `EXPERIMENT="E0"`. Chạy cell cấu hình, cài package và preflight trước. Dataset công khai được tải bằng `kagglehub`, rồi chia bằng script gốc. Không cần Kaggle CLI hoặc Kaggle account cho đường tải công khai; nếu dịch vụ yêu cầu đăng nhập/chặn tải, dùng `DATASET_ZIP` hoặc `DATA_SOURCE_DIR`.
4. Để chạy E1–E4, thêm Colab Secrets `HF_TOKEN` (quyền ghi) và `HF_REPO_ID` (dạng `username/day10-qwen3vl-checkpoints`), bật Notebook access. Repo backup phải private. Không dán token vào code hoặc output.
5. Chạy các cell theo thứ tự. Mỗi lượt chọn một experiment: E0, E1, E2, E3, rồi E4. E4 tự chạy rank 4 và 64, baseline E1. Giữ 300 bước, micro batch 1, accumulation 8, checkpoint mỗi 25 bước.
6. Sau sự cố, mở lại cùng notebook, cùng cấu hình và Secrets, chạy lại từ đầu. Runner tìm checkpoint hợp lệ mới nhất local/HF, kiểm tra fingerprint rồi khôi phục. Không đổi branch/code revision hoặc tham số khi muốn resume. Khi remote backup thất bại, run dừng và giữ checkpoint local.
7. Cell cuối tải ZIP `e0_logs_evidence.zip` (hoặc E1/E2/E3/E4 tương ứng). ZIP chứa evidence thật, predictions, logs, hình minh họa; checkpoint và trọng số lưu trên HF. Tải ZIP sau mỗi experiment vì file trong `/content` có thể mất khi runtime bị xóa.

Google account chỉ xác định notebook và runtime Colab; HF Secrets dùng tài khoản Hugging Face của bạn. Notebook không chứa email hoặc token. Không cần mount Drive để pipeline chạy.

Lượt triển khai mới chỉ xác minh CPU/static. Chưa chạy GPU hoặc thử mất runtime → tải remote checkpoint → resume trên Colab. REPORT giữ NOT_RUN cho đến khi có evidence thật. Yêu cầu chạy GPU trên Kaggle trước đây được thay bằng Colab theo chỉ dẫn mới của người dùng.
