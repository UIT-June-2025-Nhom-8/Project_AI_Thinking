# 🤖 Hệ Thống Phân Tích Phản Hồi Sinh Viên Bằng AI

**Dự án AI Thinking - Nhóm 8 | Trường Đại học Công nghệ Thông tin (UIT)**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 👥 Thành Viên Nhóm 8

| MSSV     | Họ và Tên         | Vai Trò                            |
| -------- | ----------------- | ---------------------------------- |
| 25410056 | Lã Xuân Hồng      | Team Lead, Documentation           |
| 25410034 | Lê Quang Hoài Đức | Vietnamese Text Preprocessing      |
| 25410150 | Nguyễn Minh Trọng | LDA & PhoBERT Implementation       |
| 25410088 | Trần Thanh Long   | Comparison Analysis, Visualization |
| 25410104 | Nguyễn Minh Nhật  | Data Analysis, Model Evaluation    |

## 📋 Tổng Quan Dự Án

Hệ thống **phân tích cảm xúc và phân loại chủ đề** trong phản hồi sinh viên tiếng Việt, sử dụng hai phương pháp tiếp cận chính:

### 🎯 Mục Tiêu Chính

- **Phân loại cảm xúc** thành 3 nhóm: TIÊU CỰC (46%), TRUNG LẬP (4.3%), TÍCH CỰC (49.7%)
- **Phân tích 4 chủ đề**: Giảng viên, Chương trình đào tạo, Cơ sở vật chất, Khác
- **So sánh hiệu suất** giữa phương pháp truyền thống (LDA) và hiện đại (PhoBERT)

### 📊 Kết Quả Chính

| Phương Pháp      | Độ Chính Xác | F1-Macro Score | Ưu Điểm                      |
| ---------------- | ------------ | -------------- | ---------------------------- |
| **LDA + TF-IDF** | 84%          | 0.64           | Nhanh, ít tài nguyên         |
| **PhoBERT**      | **93%**      | **0.82**       | Hiệu suất cao, hiểu ngữ cảnh |

### 🔧 Công Nghệ Sử Dụng

- **Dataset**: Vietnamese Students Feedback (UIT-NLP) - 16,175 mẫu
- **Preprocessing**: Underthesea tokenizer, custom Vietnamese text cleaner
- **Traditional ML**: Linear Discriminant Analysis + TF-IDF
- **Deep Learning**: PhoBERT (Vietnamese BERT) với fine-tuning

## � Cài Đặt và Sử Dụng

### Yêu Cầu Hệ Thống

```bash
Python 3.11+
PyTorch 2.0+
Transformers 4.20+
```

### Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/UIT-June-2025-Nhom-8/Project_AI_Thinking.git
cd Project_AI_Thinking

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc venv\Scripts\activate  # Windows

# Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
pip install underthesea  # Vietnamese NLP toolkit
pip install jupyter notebook
```

### Chạy Dự Án

#### 1. Chạy Pipeline Chính (So sánh LDA vs PhoBERT)

```bash
cd src
python main.py
```

#### 2. Chạy Jupyter Notebooks

```bash
# Phân tích chất lượng dataset
jupyter notebook src/notebooks/Review_dataset_quality.ipynb

# Demo tương tác
jupyter notebook src/notebooks/main.ipynb
```

#### 3. Chạy Từng Thành Phần

```bash
# Chỉ LDA Classifier
python src/LDA_classifier.py

# Chỉ PhoBERT Classifier
python src/phoBERT_classifier.py
```

## 📁 Cấu Trúc Thư Mục

```
Project_AI_Thinking/
├── 📂 src/                                    # Mã nguồn chính
│   ├── main.py                               # Pipeline chính - So sánh LDA vs PhoBERT
│   ├── VN_preprocessor.py                    # Tiền xử lý văn bản tiếng Việt
│   ├── LDA_classifier.py                     # Bộ phân loại LDA
│   ├── TFIDF_vectorlizer.py                 # Vector hóa TF-IDF
│   ├── phoBERT_classifier.py                # Mô hình PhoBERT cho sentiment analysis
│   └── 📂 notebooks/                         # Jupyter notebooks
│       ├── Review_dataset_quality.ipynb     # Phân tích chất lượng dataset
│       └── main.ipynb                       # Demo tương tác
├── 📂 reports/                               # Báo cáo và tài liệu
│   ├── Project.md                           # Mô tả chi tiết dự án
│   ├── Overall_Report.md                    # Báo cáo tổng hợp toàn diện
│   ├── LDA_Classifier_Report.md             # Phân tích chi tiết LDA
│   ├── PhoBERT_Final_Report.md              # Báo cáo mô hình PhoBERT
│   ├── LDA_PhoBERT_Comparison_Report.md     # So sánh LDA vs PhoBERT
│   ├── Review_dataset_quality_report.md     # Báo cáo chất lượng dataset
│   └── 📂 images/                            # Biểu đồ và visualization
│       ├── Accuracy_Comparison.png          # So sánh độ chính xác
│       ├── F1Macro_Comparison.png           # So sánh F1-Macro score
│       ├── ConfusionMatrix_RawData.png      # Ma trận nhầm lẫn (Raw data)
│       ├── ConfusionMatrix_UTData.png       # Ma trận nhầm lẫn (Underthesea)
│       └── PhoBERT_Simple_Confusion_Matrix.png # Ma trận nhầm lẫn PhoBERT
├── 📂 docs/                                  # Tài liệu lý thuyết
│   └── 📂 Phan_Tich_Tong_Quat/              # Phân tích tổng quát bài toán
├── 📂 .github/                               # GitHub configuration
├── 📂 .vscode/                               # VS Code settings
├── .env.example                             # Template cho environment variables
├── .gitignore                               # Git ignore file
└── README.md                                # File này
```

## 🧠 Kiến Trúc Hệ Thống

### 1. Phương Pháp LDA (Traditional ML)

```
Raw Text → VN Preprocessor → TF-IDF → LDA → Sentiment Classification
```

### 2. Phương Pháp PhoBERT (Deep Learning)

```
Raw Text → PhoBERT Tokenizer → PhoBERT Model → Fine-tuning → Sentiment Classification
```

## 📊 Dataset

- **Nguồn**: [Vietnamese Students Feedback (UIT-NLP)](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Tổng số mẫu**: 16,175 phản hồi sinh viên
- **Phân chia**: Train/Validation/Test
- **Ngôn ngữ**: Tiếng Việt
- **Nhãn cảm xúc**: NEGATIVE (0), NEUTRAL (1), POSITIVE (2)
- **Chủ đề**: lecturer, training_program, facility, others

## 🔍 Thành Phần Chính

### 📝 Vietnamese Preprocessor (`VN_preprocessor.py`)

- Làm sạch văn bản tiếng Việt (Unicode normalization)
- Tách từ với Underthesea tokenizer
- Loại bỏ stop words và ký tự đặc biệt
- Chuẩn hóa văn bản

### 🎯 LDA Classifier (`LDA_classifier.py`)

- Linear Discriminant Analysis với SVD solver
- Tích hợp TF-IDF vectorizer
- Cross-validation để tối ưu hyperparameters
- Báo cáo chi tiết performance metrics

### 🤖 PhoBERT Classifier (`phoBERT_classifier.py`)

- Fine-tuning PhoBERT cho sentiment analysis
- Custom dataset class cho Vietnamese text
- Training loop với validation
- Advanced evaluation metrics

## 📈 Kết Quả Thực Nghiệm

### So Sánh Hiệu Suất

| Metric        | LDA (Raw) | LDA (Underthesea) | PhoBERT  |
| ------------- | --------- | ----------------- | -------- |
| **Accuracy**  | 81%       | 84%               | **93%**  |
| **F1-Macro**  | 0.55      | 0.64              | **0.82** |
| **Precision** | 0.81      | 0.84              | **0.93** |
| **Recall**    | 0.81      | 0.84              | **0.93** |

### Phân Tích Chi Tiết

- **PhoBERT** vượt trội với cải thiện 9% accuracy và 18% F1-macro
- **Underthesea tokenizer** cải thiện 3% cho phương pháp LDA
- Cả hai mô hình đều có xu hướng phân loại nhầm NEUTRAL thành POSITIVE

## 🎯 Ứng Dụng Thực Tế

- **Phản hồi sinh viên tự động**: Phân tích ý kiến về giảng viên và chương trình học
- **Hệ thống đánh giá chất lượng giáo dục**: Theo dõi sentiment theo thời gian
- **Công cụ hỗ trợ ra quyết định**: Cải thiện chất lượng giảng dạy dựa trên feedback

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ

**Nhóm 8 - AI Thinking Project**

- Email: [25410056@ms.uit.edu.vn](mailto:25410056@ms.uit.edu.vn)
- Repository: [https://github.com/UIT-June-2025-Nhom-8/Project_AI_Thinking](https://github.com/UIT-June-2025-Nhom-8/Project_AI_Thinking)

## 🙏 Acknowledgments

- [UIT-NLP Lab](https://nlp.uit.edu.vn/) cho Vietnamese Students Feedback dataset
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) team cho pre-trained Vietnamese BERT
- [Underthesea](https://github.com/undertheseanlp/underthesea) cho Vietnamese NLP toolkit

---

<div align="center">

**🎓 Dự án được thực hiện bởi Nhóm 8 - Môn Tư Duy Trí Tuệ Nhân Tạo**  
**Đại học Quốc Gia Thành phố Hồ Chí Minh - Trường Đại học Công nghệ Thông tin (UIT) - 2025**

</div>
