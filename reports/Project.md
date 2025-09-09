# Dự Án Phân Tích Phản Hồi Sinh Viên Bằng AI - Nhóm 8

## 📋 Tổng Quan Dự Án

Đây là dự án **AI Thinking của Nhóm 8 - Trường Đại học Công nghệ Thông tin (UIT)** nhằm phát triển một ứng dụng AI để **phân tích cảm xúc và phân loại chủ đề trong phản hồi của sinh viên** bằng tiếng Việt.

## 🎯 Mục Tiêu Chính

Xây dựng hệ thống phân tích phản hồi sinh viên Việt Nam với khả năng:

1. **Phân loại cảm xúc** thành 3 nhóm:

   - **TIÊU CỰC** (46.0% dữ liệu)
   - **TRUNG LẬP** (4.3% dữ liệu)
   - **TÍCH CỰC** (49.7% dữ liệu)

2. **Phân tích 4 chủ đề chính**:
   - Giảng viên (lecturer)
   - Chương trình đào tạo (training_program)
   - Cơ sở vật chất (facility)
   - Khác (others)

## 📊 Dữ Liệu Sử Dụng

- **Dataset**: Vietnamese Students Feedback (UIT-NLP)
- **Nguồn**: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
- **Tổng số mẫu**: 16,175 phản hồi sinh viên
- **Ngôn ngữ**: Tiếng Việt
- **Đặc điểm**: Phản hồi về các học phần tại trường đại học

## 🔧 Phương Pháp Tiếp Cận

### Phương Pháp 1: LDA (Linear Discriminant Analysis)

- **Tiền xử lý văn bản tiếng Việt** với tùy chọn tokenization bằng Underthesea
- **Vector hóa TF-IDF** để trích xuất đặc trưng
- **Phân loại LDA** cho sentiment analysis
- **Kết quả**: Đạt độ chính xác ~81-84%

### Phương Pháp 2: PhoBERT

- **Mô hình Transformer** được tối ưu cho tiếng Việt
- **Kết quả vượt trội**: Đạt **93% độ chính xác** và **0.82 F1-macro score**
- Hiệu suất cao hơn đáng kể so với phương pháp LDA

## 🛠️ Các Thành Phần Chính

### 1. Mô-đun Tiền Xử Lý (`VNPreprocessor`)

- Làm sạch văn bản tiếng Việt
- Tách từ (tokenization)
- Loại bỏ stop words
- Chuẩn hóa Unicode cho ký tự có dấu

### 2. Vector Hóa TF-IDF (`TFIDFVectorizer`)

- Chuyển đổi văn bản thành vector số
- Sử dụng n-gram (1,2)
- Tối đa 5,000 đặc trưng

### 3. Bộ Phân Loại LDA (`LDAClassifier`)

- Thuật toán Linear Discriminant Analysis
- Hỗ trợ validation và đánh giá mô hình
- Tích hợp các chỉ số đánh giá chi tiết

### 4. Pipeline Chính (`main.py`)

- Quy trình hoàn chỉnh từ dữ liệu thô đến kết quả
- So sánh hiệu suất giữa dữ liệu raw và được xử lý bằng Underthesea
- Tạo báo cáo và biểu đồ đánh giá

## 📈 Kết Quả Đạt Được

### So Sánh Hiệu Suất

| Mô hình               | Accuracy | F1-Macro | Đặc điểm                                        |
| --------------------- | -------- | -------- | ----------------------------------------------- |
| **LDA (Raw)**         | 0.84     | 0.64     | Khá tốt, nhưng yếu ở lớp Neutral                |
| **LDA (Underthesea)** | 0.81     | 0.67     | Không cải thiện đáng kể so với Raw              |
| **PhoBERT**           | 0.93     | 0.82     | Cải thiện rõ rệt cả độ chính xác và cân bằng F1 |

### Phân Tích Chất Lượng Dữ Liệu

- **Mất cân bằng lớp**: Tỷ lệ 11.52 giữa lớp đa số và thiểu số
- **Phát hiện nhãn sai**: 461 mẫu có khả năng bị gán nhãn không chính xác
- **Đề xuất cải thiện**: Cần cân bằng dữ liệu và kiểm tra lại nhãn

## 🎓 Ứng Dụng Thực Tiễn

Hệ thống này có thể giúp các trường đại học:

- **Tự động phân tích phản hồi sinh viên** về các khía cạnh khác nhau của trải nghiệm học tập
- **Đánh giá cảm xúc** đối với giảng viên, chương trình đào tạo, cơ sở vật chất
- **Đưa ra quyết định dựa trên dữ liệu** để cải thiện chất lượng giáo dục
- **Phát hiện sớm các vấn đề** trong quá trình giảng dạy và học tập

## 👥 Thành Viên Nhóm 8

| MSSV     | Họ và Tên         |
| -------- | ----------------- |
| 25410056 | Lã Xuân Hồng      |
| 25410034 | Lê Quang Hoài Đức |
| 25410150 | Nguyễn Minh Trọng |
| 25410088 | Trần Thanh Long   |
| 25410104 | Nguyễn Minh Nhật  |

## 🔧 Yêu Cầu Kỹ Thuật

- **Python 3.11** (khuyến nghị)
- **Thư viện chính**: underthesea, scikit-learn, pandas, numpy
- **Thư viện phụ trợ**: matplotlib, seaborn (cho visualization)
- **Dataset**: Vietnamese Students Feedback từ Hugging Face

## 📁 Cấu Trúc Thư Mục

```
Project_AI_Thinking/
├── .github/                                    # GitHub configuration
│   └── pull_request_template.md              # PR template
├── .vscode/                                   # VS Code settings
│   └── settings.json                         # Editor configuration
├── src/                                       # Mã nguồn chính
│   ├── main.py                               # Pipeline chính - So sánh LDA với/không Underthesea
│   ├── vn_preprocessor.py                    # Tiền xử lý tiếng Việt
│   ├── LDA_classifier.py                     # Bộ phân loại LDA
│   ├── TFIDF_vectorlizer.py                 # Vector hóa TF-IDF
│   ├── 1.review_dataset_quality.ipynb       # Phân tích chất lượng dữ liệu
│   └── demo.ipynb                           # Demo và ví dụ
├── reports/                                  # Báo cáo và tài liệu chính
│   ├── Project.md                           # Mô tả dự án tổng quan (file hiện tại)
│   ├── Overall_Report.md                    # Báo cáo tổng hợp toàn diện
│   ├── Review_dataset_quality_report.md     # Báo cáo chất lượng dataset
│   ├── LDA_Classifier_Report.md             # Báo cáo chi tiết LDA
│   ├── PhoBERT_Final_Report.md              # Báo cáo mô hình PhoBERT
│   ├── LDA_PhoBERT_Comparison_Report.md     # So sánh LDA vs PhoBERT
│   └── images/                              # Biểu đồ và hình ảnh
│       ├── Accuracy_Comparison.png          # So sánh độ chính xác
│       ├── F1Macro_Comparison.png           # So sánh F1-Macro score
│       ├── ConfusionMatrix_RawData.png      # Ma trận nhầm lẫn - Raw data
│       ├── ConfusionMatrix_UTData.png       # Ma trận nhầm lẫn - Underthesea
│       └── PhoBERT_Simple_Confusion_Matrix.png # Ma trận nhầm lẫn PhoBERT
├── docs/                                     # Tài liệu phân tích lý thuyết
│   └── Phan_Tich_Tong_Quat/                 # Phân tích bài toán tổng quát
│       ├── PhanTichBaiToan_AIThinking_ChuanHoa.MD
│       ├── PhanTichBaiToan_AIThinking_Co-so-ly-thuyet-va-giai-phap.md
│       └── *.pdf                            # Các file PDF tương ứng
├── create_confusion_matrix_viz.py           # Script tạo visualization cho LDA
├── generate_phobert_confusion_matrix.py     # Script tạo ma trận nhầm lẫn PhoBERT
├── .env.example                             # Template cho environment variables
├── .gitignore                               # Git ignore file
└── README.md                                # Hướng dẫn chung dự án
```

### 📝 Mô Tả Chi Tiết Các Thành Phần

#### 🔧 Source Code (`src/`)

- **`main.py`**: Pipeline chính thực hiện so sánh LDA với và không có Underthesea
- **`vn_preprocessor.py`**: Xử lý đặc thù tiếng Việt (Unicode, tokenization, stop words)
- **`LDA_classifier.py`**: Implementation LDA với SVD solver
- **`TFIDF_vectorlizer.py`**: TF-IDF vectorization với n-gram support
- **`*.ipynb`**: Jupyter notebooks cho phân tích và demo

#### 📊 Reports (`reports/`)

- **`Project.md`**: Mô tả tổng quan dự án (file hiện tại)
- **`Overall_Report.md`**: Báo cáo tổng hợp toàn diện với phân tích chi tiết
- **Individual Reports**: Báo cáo chi tiết từng thành phần và so sánh
- **`images/`**: Tất cả visualization và confusion matrices

#### 📚 Documentation (`docs/`)

### 📝 Mô Tả Chi Tiết Các Thành Phần

#### 🔧 Source Code (`src/`)

- **`main.py`**: Pipeline chính thực hiện so sánh LDA với và không có Underthesea
- **`vn_preprocessor.py`**: Xử lý đặc thù tiếng Việt (Unicode, tokenization, stop words)
- **`LDA_classifier.py`**: Implementation LDA với SVD solver
- **`TFIDF_vectorlizer.py`**: TF-IDF vectorization với n-gram support
- **`*.ipynb`**: Jupyter notebooks cho phân tích và demo

#### 📊 Documentation (`docs/`)

- **Reports**: Báo cáo chi tiết từng thành phần và tổng hợp
- **Visualizations**: Confusion matrices, performance comparisons
- **Theoretical Analysis**: Phân tích lý thuyết và giải pháp

#### 🎨 Visualization Scripts

- **`create_confusion_matrix_viz.py`**: Tạo confusion matrix cho LDA models
- **`generate_phobert_confusion_matrix.py`**: Tạo confusion matrix cho PhoBERT

## 🚀 Hướng Phát Triển

1. **Cải thiện mô hình**: Thử nghiệm các kiến trúc deep learning khác
2. **Mở rộng dữ liệu**: Thu thập thêm phản hồi từ nhiều trường đại học
3. **Tích hợp thời gian thực**: Phát triển API để phân tích phản hồi trực tiếp
4. **Giao diện người dùng**: Xây dựng web interface cho việc sử dụng dễ dàng
5. **Đa ngôn ngữ**: Mở rộng hỗ trợ cho các ngôn ngữ khác trong khu vực

## 📊 Đóng Góp Khoa Học

Dự án này đóng góp vào lĩnh vực xử lý ngôn ngữ tự nhiên tiếng Việt bằng cách:

- So sánh hiệu quả giữa phương pháp truyền thống (LDA) và deep learning (PhoBERT)
- Cung cấp pipeline hoàn chỉnh cho sentiment analysis tiếng Việt
- Phân tích chi tiết chất lượng dataset và đề xuất cải thiện
- Tạo framework có thể tái sử dụng cho các bài toán tương tự

---

## 🔬 Phân Tích Chi Tiết Kỹ Thuật

### 📊 Dataset và Phân Tích Chất Lượng Dữ Liệu

#### Thống Kê Dataset Vietnamese Students Feedback

- **Tổng số mẫu**: 16,175 phản hồi sinh viên
- **Phân bố train/validation/test**: Đã được chia sẵn theo chuẩn
- **Đặc điểm văn bản**:
  - Chiều dài trung bình: 58.8 ký tự (14.2 từ)
  - Phạm vi: 4-718 ký tự
  - Văn bản rất ngắn (<10 ký tự): 104 mẫu
  - Không có văn bản rỗng hoặc quá dài

#### Phân Tích Mất Cân Bằng Dữ Liệu

- **NEGATIVE**: 7,439 mẫu (46.0%)
- **POSITIVE**: 8,038 mẫu (49.7%)
- **NEUTRAL**: 698 mẫu (4.3%)
- **Tỷ lệ mất cân bằng**: 11.52 (giữa lớp đa số và thiểu số)

#### Phân Tích Từ Vựng Chi Tiết

- **Tổng số từ**: 203,815
- **Từ duy nhất**: 2,845
- **Độ phong phú từ vựng**: 0.0140
- **Từ xuất hiện một lần**: 40.0%
- **Từ phổ biến**: viên, giảng, dạy, thầy, sinh, học, bài, tình, không, và

#### Phân Tích Theo Từng Nhãn Cảm Xúc

| Nhãn         | Tổng từ    | Từ duy nhất | Độ phong phú       | Từ đặc trưng                         |
| ------------ | ---------- | ----------- | ------------------ | ------------------------------------ |
| **NEGATIVE** | Nhiều nhất | 2,070       | 0.0206             | không, nên, nhiều, bài, mờ, hư, tệ   |
| **NEUTRAL**  | Ít nhất    | 743         | 0.1364 (cao nhất)  | Đa dạng chủ đề                       |
| **POSITIVE** | Trung bình | 1,832       | 0.0187 (thấp nhất) | tình, nhiệt, rất, dễ, hiểu, cởi, hoà |

#### Phân Tích Tương Quan Sentiment-Topic

- **Topic 0 (lecturer)**: Cân bằng NEGATIVE (35%) và POSITIVE (62%)
- **Topic 1 (training_program)**: Chủ yếu NEGATIVE (77%)
- **Topic 2 (facility)**: Gần như hoàn toàn NEGATIVE (96%)
- **Topic 3 (others)**: Phân bố đều các nhãn

#### Phát Hiện Nhãn Sai Tiềm Năng

- **461 mẫu** có khả năng bị gán nhãn sai (dựa trên độ tin cậy model baseline > 0.7)
- **Ví dụ tiêu biểu**:
  - "nên cho thực hành nhiều hơn" (nhãn POSITIVE) → Model dự đoán NEGATIVE (99.9%)
  - "cô dạy rất nhiệt tình, tận tâm và chu đáo" (nhãn NEGATIVE) → Model dự đoán POSITIVE (99.9%)

### 🛠️ Phân Tích Kỹ Thuật Chi Tiết

#### Mô-đun VNPreprocessor

**Chức năng chính**:

- Chuẩn hóa Unicode cho ký tự tiếng Việt có dấu
- Loại bỏ URL, email, số, dấu câu không cần thiết
- Tách từ với/không sử dụng Underthesea
- Loại bỏ stop words tiếng Việt (24 từ cốt lõi)

**Thách thức đặc thù tiếng Việt**:

- Tính đơn lập và không có dấu phân cách từ rõ ràng
- 6 thanh điệu với dấu phụ ảnh hưởng nghĩa
- Biểu đạt mỉa mai, châm biếm phức tạp
- Cấu trúc phủ định đa dạng

**Kết quả so sánh**:

- **Raw data** (không Underthesea): 84% accuracy
- **Underthesea processed**: 81% accuracy
- Kết luận: Underthesea có thể loại bỏ thông tin quan trọng trong ngữ cảnh này

#### Mô-đun TFIDFVectorizer

**Tham số tối ưu**:

- `max_features=5000`: Giữ lại 5000 từ/n-gram quan trọng nhất
- `ngram_range=(1,2)`: Kết hợp unigram và bigram
- `min_df=2`: Loại bỏ từ xuất hiện <2 document
- `max_df=0.8`: Loại bỏ từ xuất hiện >80% document

**Công thức TF-IDF**:

```
TF-IDF(t,d) = TF(t,d) × IDF(t)
TF(t,d) = số lần từ t xuất hiện trong document d / tổng số từ trong d
IDF(t) = log(N / df(t))
```

#### Mô-đun LDAClassifier

**Đặc điểm kỹ thuật**:

- Sử dụng `solver='svd'` cho ổn định số học
- Validation tự động với cross-validation
- Xử lý lỗi cho dữ liệu không hợp lệ
- Tích hợp các metrics đánh giá chi tiết

### 📈 Kết Quả Thực Nghiệm Chi Tiết

#### So Sánh Hiệu Suất LDA vs PhoBERT

| Metric                 | LDA (Raw) | LDA (Underthesea) | PhoBERT | Cải thiện |
| ---------------------- | --------- | ----------------- | ------- | --------- |
| **Accuracy**           | 0.84      | 0.81              | 0.93    | +9%       |
| **F1-Macro**           | 0.64      | 0.67              | 0.82    | +18%      |
| **Precision NEGATIVE** | 0.86      | 0.84              | 0.92    | +6%       |
| **Recall NEGATIVE**    | 0.86      | 0.82              | 0.96    | +10%      |
| **Precision NEUTRAL**  | 0.29      | 0.22              | 0.65    | +36%      |
| **Recall NEUTRAL**     | 0.25      | 0.26              | 0.42    | +17%      |
| **Precision POSITIVE** | 0.87      | 0.87              | 0.95    | +8%       |
| **Recall POSITIVE**    | 0.89      | 0.87              | 0.97    | +8%       |

#### Phân Tích Ma Trận Nhầm Lẫn PhoBERT

```
Predicted:    NEG  NEU  POS
Actual: NEG  [1354  20   35]  ← 96.3% chính xác
        NEU  [  55  69   43]  ← 41.3% chính xác
        POS  [  55  21 1514]  ← 95.2% chính xác
```

**Nhận xét**:

- Lớp NEGATIVE và POSITIVE: Hiệu suất rất cao (>95%)
- Lớp NEUTRAL: Vẫn là thách thức lớn nhất (41.3% recall)
- Xu hướng nhầm lẫn: NEUTRAL → NEGATIVE/POSITIVE

#### Đánh Giá Model Baseline

| Mô hình                 | Accuracy | Macro F1 | Nhận xét                                |
| ----------------------- | -------- | -------- | --------------------------------------- |
| **Naive Bayes**         | 0.844    | 0.575    | Yếu ở lớp NEUTRAL                       |
| **Logistic Regression** | 0.897    | 0.668    | Accuracy cao nhất trong ML truyền thống |
| **Logistic (Balanced)** | 0.842    | 0.702    | Macro F1 cao nhất nhờ cân bằng lớp      |

### 🔍 Phân Tích Sâu Về Thách Thức

#### Thách Thức Lớp NEUTRAL

**Nguyên nhân**:

1. **Mất cân bằng dữ liệu nghiêm trọng**: Chỉ 4.3% tổng số mẫu
2. **Tính mơ hồ ngữ nghĩa**: Khó phân biệt với positive/negative nhẹ
3. **Ngữ cảnh văn hóa**: Sinh viên Việt Nam ít diễn đạt trung lập

**Đề xuất giải pháp**:

- Áp dụng SMOTE cho data augmentation
- Sử dụng focal loss để xử lý mất cân bằng
- Thu thập thêm dữ liệu NEUTRAL chất lượng cao
- Xem xét gộp với lớp có cảm xúc yếu

#### Hiệu Suất Underthesea vs Raw

**Kết quả bất ngờ**: Raw data > Underthesea processed

**Phân tích nguyên nhân**:

1. **Over-segmentation**: Underthesea có thể tách quá mức từ ghép quan trọng
2. **Stop words aggressive**: Loại bỏ từ có thể mang thông tin cảm xúc
3. **Domain mismatch**: Underthesea được train trên tổng quát, không phù hợp hoàn toàn với domain giáo dục

### 📋 Phương Pháp Luận Khoa Học

#### Thiết Kế Thực Nghiệm

1. **Kiểm định chéo 5-fold**: Đảm bảo tính ổn định kết quả
2. **Stratified sampling**: Duy trì tỷ lệ nhãn trong mỗi fold
3. **Seed cố định**: Reproducible results (seed=42)
4. **Metrics đa chiều**: Accuracy, F1-macro, F1-weighted, confusion matrix

#### Pipeline Đánh Giá

```
Dataset → Tiền xử lý → Feature Engineering →
Cross Validation → Model Training →
Evaluation → Statistical Testing → Report
```

#### Cơ Sở Lý Thuyết

**Linear Discriminant Analysis (LDA)**:

- Tìm hyperplane tối ưu phân tách các lớp
- Tối đa hóa tỷ lệ between-class vs within-class variance
- Giả định: Dữ liệu tuân theo phân phối Gaussian
- Hiệu quả với dữ liệu có ít nhiễu và đặc trưng tuyến tính

**PhoBERT Architecture**:

- Dựa trên BERT-base với 12 layers, 768 hidden units
- Pre-trained trên 20GB văn bản tiếng Việt
- Fine-tuning với learning rate 2e-5, batch size 16
- Dropout 0.1, weight decay 0.01

### 🚀 Đề Xuất Cải Tiến Kỹ Thuật

#### Cải Thiện Data Quality

1. **Active Learning**: Chọn mẫu thông tin cao để labeling bổ sung
2. **Data Augmentation**: Back-translation, synonym replacement
3. **Pseudo-labeling**: Sử dụng high-confidence predictions

#### Nâng Cấp Model Architecture

1. **Ensemble Methods**: Kết hợp LDA + PhoBERT + XGBoost
2. **Multi-task Learning**: Học đồng thời sentiment + topic
3. **Hierarchical Models**: Phân cấp topic → aspect → sentiment

#### Tối Ưu Hyperparameters

1. **Automated Search**: Optuna, Ray Tune
2. **Multi-objective Optimization**: Pareto optimal points
3. **Early Stopping**: Ngăn overfitting với patience monitoring

### 📊 Tác Động và Ứng Dụng Mở Rộng

#### Ứng Dụng Trong Giáo Dục

1. **Dashboard thời gian thực**: Monitoring feedback liên tục
2. **Cảnh báo sớm**: Phát hiện vấn đề chất lượng giảng dạy
3. **Phân tích xu hướng**: Theo dõi thay đổi cảm xúc theo thời gian
4. **Personalized recommendations**: Đề xuất cải thiện cho từng giảng viên

#### Mở Rộng Cho Các Domain Khác

1. **E-commerce**: Phân tích review sản phẩm
2. **Social Media**: Monitoring brand sentiment
3. **Healthcare**: Phân tích phản hồi bệnh nhân
4. **Government**: Phân tích ý kiến công chúng

### 🎯 Đóng Góp Khoa Học Cụ Thể

#### Contribution 1: Comprehensive Vietnamese Education Sentiment Analysis

- First systematic ABSA framework for Vietnamese education domain
- Specialized preprocessing pipeline for educational terminology
- Domain-specific performance benchmarks

#### Contribution 2: Comparative Analysis of Traditional vs Modern Approaches

- Empirical comparison: TF-IDF+LDA vs PhoBERT
- Analysis of trade-offs: accuracy vs interpretability vs computational cost
- Guidelines for model selection based on resource constraints

#### Contribution 3: Data Quality Analysis Framework

- Systematic approach to detecting mislabeled samples
- Comprehensive vocabulary analysis by sentiment classes
- Statistical significance testing for model comparisons

#### Contribution 4: Practical Deployment Considerations

- End-to-end pipeline from raw text to actionable insights
- Production-ready code with error handling and validation
- Scalable architecture for real-world deployment

---

_Dự án được thực hiện trong khuôn khổ môn học AI Thinking - UIT 2025_
