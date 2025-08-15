
# Báo cáo Mô hình PhoBERT: Phân loại cảm xúc phản hồi sinh viên

## 1. Mục tiêu & Tổng quan
Hệ thống được xây dựng nhằm phân loại phản hồi sinh viên thành ba nhãn cảm xúc:
- **Negative** (Tiêu cực)
- **Neutral** (Trung tính)
- **Positive** (Tích cực)

PhoBERT – một mô hình Transformer tối ưu cho tiếng Việt – được sử dụng để tận dụng khả năng hiểu ngữ nghĩa và ngữ cảnh sâu hơn so với các phương pháp dựa trên TF-IDF + LDA.

---

## 2. Pipeline & Thành phần chính

### 2.1. Chuẩn bị dữ liệu
- Dữ liệu đầu vào đã được **làm sạch** và có cột văn bản (`sentence_clean`) cùng cột nhãn (`sentiment`).
- Ba tập tách riêng: **train**, **validation**, **test**.
- Yêu cầu: Nhãn là số nguyên liên tục từ 0 đến 2.

### 2.2. Chuyển đổi sang `DatasetDict`
Hàm `_frames_to_hfds`:
- Đổi tên cột nhãn thành `'labels'` theo chuẩn HuggingFace.
- Trả về `DatasetDict` gồm 3 split: train, validation, test.

### 2.3. Tokenization
- Dùng `AutoTokenizer` của **vinai/phobert-base**.
- Cắt ngữ cảnh ở độ dài tối đa `max_len=256`.
- Padding động bằng `DataCollatorWithPadding`.

### 2.4. Huấn luyện
- **Model:** `AutoModelForSequenceClassification` (PhoBERT) với `num_labels=3`.
- **Loss:** Cross-Entropy.
- **Optimizer:** AdamW (qua `TrainingArguments`).
- **Batch size:** 16 (train) / 32 (eval).
- **Learning rate:** 2e-5.
- **Epoch:** 3.
- **Weight decay:** 0.01.
- **Seed fix:** Đảm bảo kết quả reproducible trên CPU/GPU.

### 2.5. Đánh giá
- **Metrics:** Accuracy & F1-Macro.
- **`compute_metrics`**:
  - Lấy logits → argmax → so sánh với label thật.
  - Tính `accuracy` và `f1_macro`.

---

## 3. Kết quả thực nghiệm

Sau khi huấn luyện PhoBERT trong 3 epoch trên tập dữ liệu phản hồi sinh viên, mô hình đạt được kết quả như sau:

### 3.1. Thông số huấn luyện
- **Số epoch:** 3
- **Batch size:** 16 (train) / 32 (eval)
- **Learning rate:** 2e-5
- **Max length:** 256
- **Weight decay:** 0.01
- **Optimizer:** AdamW
- **Loss function:** CrossEntropyLoss

### 3.2. Kết quả trên tập Validation
| Epoch | Train Loss | Validation Accuracy | Validation Loss |
|-------|------------|---------------------|-----------------|
| 1     | 0.3087     | 0.9381              | 0.0835          |
| 2     | 0.1860     | 0.9431              | 0.0481          |
| 3     | 0.1478     | 0.9458              | 0.0489          |

### 3.3. Kết quả trên tập Test
- **Accuracy:** 0.9318 (~93.18%)
- **Macro F1:** 0.825
- **Weighted F1:** 0.932

**Bảng chi tiết Precision / Recall / F1-score:**

| Lớp      | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.92      | 0.96   | 0.94     | 1409    |
| Neutral  | 0.65      | 0.42   | 0.51     | 167     |
| Positive | 0.95      | 0.97   | 0.96     | 1590    |
| **Accuracy** |           |        | **0.93** | **3166**|
| **Macro Avg**| 0.83      | 0.78   | 0.82     | 3166    |
| **Weighted Avg**| 0.93      | 0.93   | 0.93     | 3166    |

### 3.4. Ma trận nhầm lẫn (Confusion Matrix)
```
[[1354   20   35]
 [  55   69   43]
 [  55   21 1514]]
```
- **Nhận xét:**
    - Lớp **Negative** và **Positive** được phân loại chính xác rất cao (Recall > 0.95).
    - Lớp **Neutral** vẫn còn yếu, Recall chỉ 0.42, cho thấy mô hình vẫn khó phân biệt câu trung tính với hai thái cực còn lại.
    - Sai lệch chủ yếu xảy ra khi câu Neutral bị nhầm thành Negative hoặc Positive.

**So với LDA:**
- Accuracy tăng từ ~84% lên ~93% (+9% absolute).
- F1-Macro tăng từ ~0.64–0.67 lên ~0.82, cải thiện rõ rệt khả năng phân loại lớp thiểu số.

---

## 4. Đề xuất cải tiến
1. **Data Augmentation cho lớp Neutral**
   - Back-translation (Vi → En → Vi)
   - Synonym replacement
   - Paraphrasing

2. **Fine-tuning sâu hơn**
   - Thử `vinai/phobert-large`.
   - Tăng epoch (4–5) kèm early stopping.

3. **Kỹ thuật regularization**
   - Sử dụng dropout cao hơn hoặc weight decay mạnh hơn để giảm overfit.

4. **Kết hợp với embedding khác**
   - So sánh PhoBERT với XLM-RoBERTa, BERT multilingual.

5. **Tích hợp interpretability**
   - Dùng LIME/SHAP để giải thích dự đoán, giúp hiểu lý do mô hình chọn nhãn.
