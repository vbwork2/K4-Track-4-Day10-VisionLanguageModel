Họ tên: Bùi Quang Vinh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | N/A | N/A | 0.0969 | 0.0721–0.1258 | 1.0932 | 0.1477 | 0.0224 | 0.2162 |
| E1 | LoRA LLM | 33030144 | 9.057 | 62.65 | 0.2631 | 0.2271–0.3036 | 1.1608 | 0.1463 | 0.1884 | 0.5126 |
| E2 | + projector | 142135296 | 6.772 | 13.75 | 0.3364 | 0.2956–0.3820 | 1.0193 | 0.2180 | 0.2526 | 0.5965 |
| E3 | + LoRA ViT | 148426752 | 10.177 | 17.00 | 0.4244 | 0.3811–0.4670 | 0.7878 | 0.2828 | 0.3249 | 0.7023 |
| E4 RANK 4 | rank 4, baseline E1 | 8257536 | 8.197 | 10.57 | 0.2670 | 0.2325–0.3068 | 1.1929 | 0.1895 | 0.1902 | 0.5174 |
| E4 RANK 64 | rank 64, baseline E1 | 132120576 | 10.072 | 13.60 | 0.2662 | 0.2287–0.3035 | 1.1415 | 0.2043 | 0.1809 | 0.4982 |

Nguồn: ZIP log do người làm cung cấp; chấm lại bằng `scripts/score.py` nguyên bản. Mỗi setup có đủ 311 dự đoán, không thiếu ID; CI95 bootstrap 1.000 lần. E0 không huấn luyện nên tài nguyên là N/A. Log E1–E4 nằm ở `report/logs/`, ánh xạ tên và SHA-256 ở `report/IMPORT_MANIFEST.json`.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Theo GUIDE, tổng ≈4,44B; LLM ≈4,0B, visual ≈415M (bao gồm projector). Log xác nhận projector có 109.105.152 tham số qua chênh lệch E2−E1. E1 cập nhật LoRA LLM (33.030.144); E2 thêm projector (142.135.296); E3 thêm 6.291.456 tham số LoRA ViT (148.426.752). ZIP không có phép đếm riêng từng khối model gốc.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 hợp lệ 290/311 = 93,25%; có 21 output không parse được. Lỗi quan sát nổi bật là box crack quá rộng và bỏ sót đối tượng: crack có 230 FP, 293 FN, F1=0,0224. Ví dụ thật `vlcsnap-00201`: `[{"label":"crack","bbox_2d":[0,457,998,998]}]`, phủ gần hết chiều ngang thay vì vết nứt nhỏ trong ảnh.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Đã chấm lại cùng 311 output E3: đúng `norm1000` cho F1=0,4244, cố tình dùng `--coord pixel` cho F1=0,0000 (0 TP). Với ảnh 640×360, scorer hiểu nhầm toạ độ rồi phóng box theo 1000/640 và 1000/360; box mất giao với GT. Thang 0–1000 biểu diễn vị trí tương đối, không phụ thuộc kích thước ảnh. Kết quả đối chứng ở `report/analysis/e3_wrong_coord_pixel.json`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Chỉ token assistant mang nhãn cần học; prompt, token ảnh và padding được mask −100. Tính loss cả đầu vào khiến model học tái tạo prompt/token đặc biệt, làm loãng tín hiệu định vị. Test CPU kiểm tra prefix, mask và collator; ZIP không có log xác nhận 220 image tokens từ processor thật.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Projector lượng tử hoá 4-bit không còn trọng số float thông thường để AdamW cập nhật đầy đủ. Mã loại `model.visual` và `lm_head` khỏi lượng tử hoá, đưa projector được mở sang FP32 và lưu/nạp riêng ngoài adapter. FP32 giữ độ chính xác cho cập nhật và giảm rủi ro tràn số FP16. Chênh lệch E2−E1 xác nhận số tham số được mở; ZIP không chứa checkpoint để kiểm tra trực tiếp dtype/state sau reload.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Chia theo phiên tránh frame liền kề xuất hiện ở cả train/test và làm F1 cao giả. Chạy lại script gốc cho 18 phiên không giao nhau: train 1.422 ảnh, box pothole/crack/manhole = 1.074/1.750/691; val 276 ảnh, 66/470/103; test 311 ảnh, 121/299/163. Dataset version khớp cả sáu bằng chứng.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E2 hơn E1 0,0733 F1; E3 hơn E2 0,0879 và giảm FP/ảnh từ 1,0193 xuống 0,7878. E3 tốt nhất về độ chính xác; CI E2/E3 vẫn giao rất ít nên chưa khẳng định ý nghĩa thống kê chỉ từ CI riêng. E1 đạt JSON 100% và tăng 0,1662 F1 so E0. VRAM/phút lấy nguyên từ meta; E1 chậm hơn nhiều và E2 dùng VRAM thấp bất thường, nên chưa kết luận hiệu suất phần cứng khi thiếu thông tin GPU/phiên chạy.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 có độ rộng E0/E1/E2/E3 lần lượt 0,0537/0,0765/0,0865/0,0859. Với 311 ảnh cùng các phiên quay, bootstrap theo ảnh có thể lạc quan vì ảnh không hoàn toàn độc lập. Không nên xem chênh lệch nhỏ là chắc chắn; cần kiểm định ghép cặp hoặc thêm phiên/test trước khi khái quát.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Trong E3, pothole kém nhất: F1=0,2828, recall=0,2314, bỏ sót 93/121 box; crack F1=0,3249, manhole=0,7023. Giả thuyết: ổ gà nhỏ/xa và giống mảng tối gây nhầm; crack mảnh làm IoU nhạy với lệch box. Đây là suy luận từ lỗi/ảnh, chưa có thống kê kích thước để xác nhận. Sáu ví dụ GT/Pred được chú thích bên dưới.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Đổi rank 4→64, alpha=2r; hai manifest giữ cùng baseline E1, seed 42, prompt, LR, 300 bước, micro-batch 1 × tích luỹ 8. F1 0,2670→0,2662 (giảm 0,0009), CI giao rộng; rank 4 ít tham số và VRAM hơn (8,197 so 10,072 GB). Chưa thấy lợi ích tăng rank. Hạn chế: E4 chạy trên E1, chưa đúng yêu cầu baseline tốt nhất là E3; không đổi nhãn để che sai lệch này.

## Ảnh minh hoạ E3

Box dày 3 px là GT, box mảnh 2 px là Pred; nhãn ghi trực tiếp trên ảnh. TP tính cùng lớp, IoU ≥ 0,5.

1. `vlcsnap-00200`: Đúng crack và manhole (2 TP).

![E3 example 1](report/evidence/figures/e3/example_01_vlcsnap-00200.jpg)

2. `vlcsnap-00201`: Crack lệch box (1 FP, 1 FN), bỏ sót manhole (1 FN).

![E3 example 2](report/evidence/figures/e3/example_02_vlcsnap-00201.jpg)

3. `vlcsnap-00202`: Đúng manhole; bịa thêm pothole (1 FP).

![E3 example 3](report/evidence/figures/e3/example_03_vlcsnap-00202.jpg)

4. `vlcsnap-00203`: Hai crack sai box (2 FP), bỏ sót crack GT (1 FN).

![E3 example 4](report/evidence/figures/e3/example_04_vlcsnap-00203.jpg)

5. `vlcsnap-00204`: Đúng manhole (1 TP).

![E3 example 5](report/evidence/figures/e3/example_05_vlcsnap-00204.jpg)

6. `vlcsnap-00205`: Đúng hai crack, thêm một crack sai (1 FP).

![E3 example 6](report/evidence/figures/e3/example_06_vlcsnap-00205.jpg)

## Nhìn lại

Loss đầu→cuối E1/E2/E3: 1,1267→0,5396 / 1,1275→0,5294 / 1,1275→0,5123; tất cả có 300 bước hữu hạn. Thời gian E1/E2 trong manifest có timestamp đầu/cuối không khớp log train; bảng giữ `train_minutes` đo bằng elapsed, không lấy chênh timestamp. E3/E4 có thêm thời gian lưu checkpoint cuối. ZIP chưa chứng minh hard-reset → remote resume; E5 GRPO chưa chạy. Test sau cập nhật: 118 passed, 4 skipped, 1 failed do test mẫu yêu cầu REPORT còn placeholder tên trong khi checker yêu cầu xoá placeholder; giữ nguyên test gốc. Bỏ riêng test mẫu mâu thuẫn: 118 passed, 4 skipped, 1 deselected; kiểm tra nộp bài: 0 FAIL, 0 WARN.
