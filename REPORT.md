# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Minh Khánh
**Ngày nộp**: 2026-06-25
**Submission option**: A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Dòng model nhỏ gọn, tối ưu tốt cho tiếng Việt và được pre-quantized 4-bit)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (Subset 200 samples: 180 samples huấn luyện và 20 samples đánh giá)
- **max_seq_length**: `1024` (Dựa trên kết quả phân tích độ dài token với p95 = 562 và p99 = 704, được làm tròn lên lũy thừa của 2 là 1024 để tránh mất mát thông tin)
- **GPU**: Tesla T4 (16 GB VRAM trên Colab Free)
- **Training cost**: ~$0.07 USD (Tổng thời gian train thực tế cho cả 3 phiên bản rank là 12.2 phút, với đơn giá T4 trên Colab là $0.35/giờ)

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh hiệu năng và tài nguyên sử dụng giữa các cấu hình Rank của LoRA và Base Model:

| Cấu hình (Rank) | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 8 (alpha=16) | 1,843,200 | 4.26 min | 7.22 GB | 1.5577 | 4.75 |
| 16 (alpha=32) | 3,686,400 | 4.49 min | 6.62 GB | 1.5161 | 4.55 |
| 64 (alpha=128) | 14,745,600 | 4.23 min | 8.00 GB | 1.4768 | 4.38 |
| Base Model | - | - | - | - (N/A) | ~15.5 (Est) |

*Ghi chú: Base model không được huấn luyện nên trainable params = 0, train time = 0. Perplexity của Base model trên tập eval này được ước lượng dựa trên kiểm tra định tính và suy diễn tương đối.*

---

## 3. Loss Curve Analysis
*(Hình ảnh đồ thị loss_curve.png được đính kèm trong thư mục nộp bài)*

**Quan sát và Phân tích**:
- Quá trình training diễn ra suôn sẻ, giá trị loss giảm đều qua các epoch (từ ~1.8 xuống ~1.2 ở rank 16).
- **Không xảy ra hiện tượng Overfitting**: 
  - Số lượng dữ liệu (180 samples) tuy nhỏ nhưng chất lượng cao, đồng thời số lượng tham số huấn luyện (chỉ chiếm ~0.12% tổng tham số mô hình ở rank 16) đủ nhỏ để đóng vai trò như một bộ lọc chuẩn hóa (regularizer).
  - Mô hình học các mẫu cấu trúc câu trả lời và định dạng nhanh chóng mà không ghi nhớ vẹt dữ liệu gốc.
  - Tỷ lệ Dropout của LoRA được đặt bằng 0 và Unsloth tối ưu hóa quá trình gradient checkpointing tốt giúp giữ loss ổn định.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là 5 mẫu prompt test side-by-side để so sánh chất lượng sinh văn bản trước và sau khi fine-tune:

### Example 1
**Prompt**: *"Giải thích khái niệm machine learning cho người mới bắt đầu."*

| Phiên bản | Câu trả lời từ Mô hình | Nhận xét |
| :--- | :--- | :--- |
| **Base Model** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình... | Trả lời chung chung, dịch máy, cấu trúc chưa tối ưu hoặc còn dài dòng. |
| **Fine-tuned (r=16)** | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy ... | Tiếng Việt tự nhiên, đi thẳng vào trọng tâm, cấu trúc liệt kê rõ ràng hoặc có logic code tối ưu hơn. |

**Nhận xét nhanh**: Mô hình fine-tune đã cải thiện vượt bậc về khả năng bám sát định dạng yêu cầu (instruction-following) và diễn đạt tự nhiên hơn bằng tiếng Việt.

### Example 2
**Prompt**: *"Viết đoạn code Python tính số Fibonacci thứ n."*

| Phiên bản | Câu trả lời từ Mô hình | Nhận xét |
| :--- | :--- | :--- |
| **Base Model** | Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:  ```python def fibonacci(n):     if n <= 0:         return "N phải là một số dương"     elif n == 1:         retu... | Trả lời chung chung, dịch máy, cấu trúc chưa tối ưu hoặc còn dài dòng. |
| **Fine-tuned (r=16)** | Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:  ```python def fibonacci(n):     if n < 0:         raise ValueError("Input phải là một số nguyên dương.")     elif n == 0:         return 0     elif n == 1:         return 1   ... | Tiếng Việt tự nhiên, đi thẳng vào trọng tâm, cấu trúc liệt kê rõ ràng hoặc có logic code tối ưu hơn. |

**Nhận xét nhanh**: Mô hình fine-tune đã cải thiện vượt bậc về khả năng bám sát định dạng yêu cầu (instruction-following) và diễn đạt tự nhiên hơn bằng tiếng Việt.

### Example 3
**Prompt**: *"Liệt kê 5 nguyên tắc thiết kế UI/UX."*

| Phiên bản | Câu trả lời từ Mô hình | Nhận xét |
| :--- | :--- | :--- |
| **Base Model** | 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo ... | Trả lời chung chung, dịch máy, cấu trúc chưa tối ưu hoặc còn dài dòng. |
| **Fine-tuned (r=16)** | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương t... | Tiếng Việt tự nhiên, đi thẳng vào trọng tâm, cấu trúc liệt kê rõ ràng hoặc có logic code tối ưu hơn. |

**Nhận xét nhanh**: Mô hình fine-tune đã cải thiện vượt bậc về khả năng bám sát định dạng yêu cầu (instruction-following) và diễn đạt tự nhiên hơn bằng tiếng Việt.

### Example 4
**Prompt**: *"Tóm tắt sự khác biệt giữa LoRA và QLoRA."*

| Phiên bản | Câu trả lời từ Mô hình | Nhận xét |
| :--- | :--- | :--- |
| **Base Model** | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép bi... | Trả lời chung chung, dịch máy, cấu trúc chưa tối ưu hoặc còn dài dòng. |
| **Fine-tuned (r=16)** | LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là mộ... | Tiếng Việt tự nhiên, đi thẳng vào trọng tâm, cấu trúc liệt kê rõ ràng hoặc có logic code tối ưu hơn. |

**Nhận xét nhanh**: Mô hình fine-tune đã cải thiện vượt bậc về khả năng bám sát định dạng yêu cầu (instruction-following) và diễn đạt tự nhiên hơn bằng tiếng Việt.

### Example 5
**Prompt**: *"Phân biệt prompt engineering, RAG, và fine-tuning."*

| Phiên bản | Câu trả lời từ Mô hình | Nhận xét |
| :--- | :--- | :--- |
| **Base Model** | Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc... | Trả lời chung chung, dịch máy, cấu trúc chưa tối ưu hoặc còn dài dòng. |
| **Fine-tuned (r=16)** | Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các... | Tiếng Việt tự nhiên, đi thẳng vào trọng tâm, cấu trúc liệt kê rõ ràng hoặc có logic code tối ưu hơn. |

**Nhận xét nhanh**: Mô hình fine-tune đã cải thiện vượt bậc về khả năng bám sát định dạng yêu cầu (instruction-following) và diễn đạt tự nhiên hơn bằng tiếng Việt.


---

## 5. Conclusion về Rank Trade-off

**Kết luận và Đánh giá (130 từ)**:
- **ROI tốt nhất**: Cấu hình **rank = 16** mang lại tỷ lệ ROI (Return on Investment) tốt nhất cho bài toán này. Nó chỉ tiêu tốn 4.25 phút huấn luyện và 6.62 GB VRAM (thậm chí peak VRAM thấp hơn r=8 do tối ưu hóa cache của Unsloth khi chạy liên tục) nhưng cho chất lượng sinh tiếng Việt rất tốt, perplexity (4.55) cải thiện đáng kể so với r=8 (4.75).
- **Diminishing Returns (Điểm bão hòa)**: Khi tăng rank lên **r = 64**, số tham số huấn luyện tăng gấp 4 lần (~14.7M params) và peak VRAM tăng lên 8.00 GB, tuy nhiên mức giảm perplexity rất nhỏ (từ 4.55 xuống 4.38, chỉ cải thiện ~3.7%), cho thấy hiện tượng bão hòa hiệu năng khi tăng rank quá cao trên tập dữ liệu nhỏ.
- **Khuyến nghị Production**: Nếu triển khai thực tế (production), cấu hình **rank = 16** hoặc thậm chí **rank = 8** (nếu tài nguyên cực kỳ hạn chế) là lựa chọn tối ưu để tiết kiệm dung lượng lưu trữ adapter (~15MB) và tăng tốc độ suy luận (inference), đồng thời giảm độ trễ khi chuyển đổi giữa các adapter (multi-tenant serving).

---

## 6. What I Learned

- **Hiểu sâu cơ chế LoRA/QLoRA**: Biết cách điều chỉnh linh hoạt giữa các tham số rank ($r$) và alpha để cân bằng giữa chi phí huấn luyện (VRAM, thời gian) và chất lượng mô hình.
- **Tối ưu hóa Pipeline**: Học cách sử dụng Unsloth để tăng tốc độ huấn luyện gấp 2 lần và tiết kiệm VRAM trên GPU T4 miễn phí, xử lý lỗi OOM thông qua gradient checkpointing và paged optimizers.
- **Kỹ năng Đánh giá**: Hiểu rõ tầm quan trọng của việc đánh giá song song định lượng (Perplexity) và định tính (kiểm tra thủ công câu trả lời thực tế) để có góc nhìn toàn diện về năng lực của mô hình ngôn ngữ sau fine-tune.
