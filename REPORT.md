Họ tên: <điền họ tên>
`dataset_version`: d10-21e4070301 (đã xác nhận bằng script chia dữ liệu gốc trên bản dataset hiện có)

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E1 | LoRA LLM | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E2 | + projector | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E3 | + LoRA ViT | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E4 | rank 4/64, baseline E1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Ước lượng trong GUIDE: tổng mô hình ≈4,44B; LLM ≈4,0B; ViT ≈415M. Chưa nạp model trên Kaggle để xác nhận revision. E1 cập nhật LoRA LLM; E2 mở merger/DeepStack projector; E3 thêm LoRA attention/MLP của ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
NOT_RUN: chưa sinh đủ 311 output E0 trên Kaggle nên chưa có parse_valid_rate hoặc ví dụ raw thật.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
NOT_RUN: chưa chấm F1 với output pixel. Khi khác scale, box dự đoán và ground truth không cùng hệ quy chiếu nên IoU/F1 không còn ý nghĩa; cần chạy scorer nguyên bản với đúng `--coord` để đo.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Loss chỉ áp dụng lên token assistant để model học JSON đáp án. Nếu tính trên prompt và token ảnh, loss sẽ khuyến khích tái tạo đầu vào thay vì dự đoán box.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Thiết kế giữ toàn bộ visual không lượng tử hóa và mở projector ở FP32; checkpoint lưu riêng trạng thái projector. Chưa chạy forward/backward trên Qwen thật để xác nhận dtype và gradient trong Kaggle.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Đã kiểm tra trên dataset cục bộ bằng script gốc: train 1.422 ảnh (pothole 1.074, crack 1.750, manhole 691 box); val 276 (66, 470, 103); test 311 (121, 299, 163). Có 18 phiên, mỗi phiên nằm trong đúng một split. Chia ngẫu nhiên có thể đưa frame liền kề vào cả train và test, làm F1 lạc quan do rò rỉ.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
NOT_RUN: chưa có train/eval E1–E3 hoặc VRAM/thời gian đo thực trên Kaggle.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
NOT_RUN: chưa có CI95 từ 311 dự đoán test thật; chưa thể kết luận chênh lệch setup có ý nghĩa.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
NOT_RUN: mới kiểm tra overlay ground-truth dữ liệu; chưa có ảnh dự đoán E0–E4 để phân tích lớp lỗi.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
NOT_RUN: runner đã chuẩn bị E4 rank 4 và 64 cùng baseline E1, seed, effective batch và số bước; chưa chạy hai thí nghiệm nên không có kết quả hoặc CI95.

## Nhìn lại

Chưa kiểm tra trên Kaggle: processor cần xác nhận 220 image tokens; luồng upload checkpoint lên HF private repo và resume sau hard reset cũng chưa được chạy thực tế. Chưa có bằng chứng model để đối chiếu lỗi ngoài những gì scorer bắt được.
