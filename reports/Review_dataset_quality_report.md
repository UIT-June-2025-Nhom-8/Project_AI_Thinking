
# Báo cáo Đánh giá Chất lượng Dataset: Vietnamese Students Feedback

**Dataset**: Vietnamese Students Feedback (UIT-NLP)
- **Source**: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
- **Tổng số mẫu**: 16,175
- **Nguồn**: Phản hồi sinh viên về các học phần tại trường đại học
- **Ngôn ngữ**: Tiếng Việt
- **Labels**: 
  - Sentiment: 3 classes (Negative, Neutral, Positive)
  - Topic: 4 classes (chủ đề khác nhau)

## Tổng quan

Báo cáo này trình bày kết quả đánh giá chất lượng của dataset "Vietnamese Students Feedback" được sử dụng để phân tích cảm xúc (Sentiment Analysis) và phân loại chủ đề (Topic Classification) trong các phản hồi của sinh viên.

*   **Kích thước Dataset:** Dataset bao gồm 16,175 mẫu.
*   **Các Cột:** Dataset có 3 cột chính: `sentence` (văn bản phản hồi), `sentiment` (nhãn cảm xúc: 0-negative, 1-neutral, 2-positive), và `topic` (0-lecturer, 1-training_program, 2-facility và 3-others).

## 1. Thống kê Cơ bản

*   **Số lượng mẫu:** 16,175
*   **Số lượng đặc trưng (cột):** 3
*   **Giá trị thiếu:** Không có giá trị thiếu nào được tìm thấy trong dataset.
*   **Thống kê Văn bản:**
    *   Chiều dài văn bản trung bình: 58.8 ký tự
    *   Số từ trung bình: 14.2 từ
    *   Chiều dài tối thiểu: 4 ký tự
    *   Chiều dài tối đa: 718 ký tự

## 2. Phân bố Nhãn (Sentiment)

Dataset cho phân tích cảm xúc có sự mất cân bằng đáng kể giữa các lớp:

*   **NEGATIVE (0):** 7,439 mẫu (46.0%)
*   **NEUTRAL (1):** 698 mẫu (4.3%)
*   **POSITIVE (2):** 8,038 mẫu (49.7%)

Tỷ lệ mất cân bằng lớp là **11.52**, cho thấy sự chênh lệch lớn giữa lớp chiếm đa số (POSITIVE) và lớp chiếm thiểu số (NEUTRAL). **Điều này cần được xử lý bằng các kỹ thuật cân bằng dữ liệu khi xây dựng mô hình.**

## 3. Phân tích Chất lượng Văn bản

*   **Văn bản rỗng:** 0
*   **Văn bản rất ngắn (<10 ký tự):** 104 mẫu
*   **Văn bản rất dài (>1000 ký tự):** 0
*   **Văn bản chứa ký tự đặc biệt:** 16,175 mẫu (tất cả các mẫu đều chứa ít nhất một ký tự đặc biệt hoặc dấu câu).
*   **Văn bản chứa URL/Email:** 0

Đa số các văn bản có độ dài và số từ trong phạm vi hợp lý. Có một số lượng nhỏ các văn bản rất ngắn, cần kiểm tra xem chúng có cung cấp đủ thông tin hữu ích hay không. Việc tất cả các văn bản chứa ký tự đặc biệt là điều bình thường đối với dữ liệu tiếng Việt có dấu và các ký tự đặc biệt thông thường.

## 4. Phân tích Từ vựng

### Tổng thể

*   Tổng số từ: 203,815
*   Số từ duy nhất: 2,845
*   Độ phong phú từ vựng: 0.0140
*   Các từ phổ biến nhất: viên, giảng, dạy, thầy, sinh, học, bài, tình, không, và.

Dataset có số lượng từ duy nhất tương đối thấp so với tổng số từ, dẫn đến độ phong phú từ vựng thấp. Điều này có thể do tính chất chuyên ngành của dữ liệu (liên quan đến trường học, giảng dạy). 40.0% số từ chỉ xuất hiện duy nhất một lần, cho thấy sự đa dạng ở mức thấp đối với các từ hiếm.

### Theo Nhãn

Phân tích từ vựng theo từng nhãn cho thấy:

*   **NEGATIVE:** Tổng từ nhiều nhất, độ phong phú từ vựng 0.0206. Các từ phổ biến liên quan đến phản hồi tiêu cực (không, nên, nhiều, bài). Các từ độc nhất hàng đầu (mờ, hư, sài, hỏng, tệ) phản ánh các vấn đề cụ thể.
*   **NEUTRAL:** Tổng từ ít nhất, nhưng độ phong phú từ vựng cao nhất (0.1364). Điều này có thể là do số lượng mẫu ít hơn và các văn bản trung lập có thể đề cập đến nhiều chủ đề khác nhau một cách ngắn gọn.
*   **POSITIVE:** Số lượng từ và từ duy nhất ở mức trung bình, độ phong phú từ vựng thấp nhất (0.0187). Các từ phổ biến và độc nhất hàng đầu (tình, nhiệt, rất, dễ, hiểu, cởi, hoà, hăng) liên quan rõ ràng đến các phản hồi tích cực về giảng viên và việc học.

## 5. Hiệu suất Model Baseline

Việc đánh giá bằng các mô hình baseline (Naive Bayes, Logistic Regression) cung cấp cái nhìn ban đầu về tính phân tách của dữ liệu:

| Mô hình                 | Độ chính xác (Accuracy) | Macro F1 | Nhận xét                                                                 |
| :---------------------- | :---------------------- | :------- | :------------------------------------------------------------------------ |
| Naive Bayes             | 0.844                   | 0.575    | Hiệu suất thấp trên lớp NEUTRAL do mất cân bằng.                         |
| Logistic Regression     | **0.897**               | 0.668    | Độ chính xác cao nhất, nhưng Macro F1 vẫn bị ảnh hưởng bởi lớp NEUTRAL. |
| Logistic (Balanced) | 0.842                   | **0.702**| Macro F1 cao nhất, cho thấy khả năng xử lý lớp thiểu số tốt hơn nhờ cân bằng lớp. |

Mô hình Logistic Regression (Balanced) cho thấy hiệu suất cân bằng tốt nhất giữa các lớp dựa trên chỉ số Macro F1, mặc dù độ chính xác tổng thể thấp hơn. Điều này nhấn mạnh tầm quan trọng của việc xử lý mất cân bằng lớp.

## 6. Phát hiện Nhãn tiềm năng bị Sai

Đã phát hiện **461** mẫu có khả năng bị gán nhãn sai dựa trên độ tin cậy cao của mô hình baseline. Việc xem xét thủ công các mẫu này (ví dụ: các mẫu có độ tin cậy > 0.7) là rất quan trọng để cải thiện chất lượng nhãn.

Ví dụ các mẫu tiềm năng bị sai nhãn:

1.  **Index 15378:** "nên cho thực hành nhiều hơn ." (POSITIVE) -> Model predicts NEGATIVE (0.999)
2.  **Index 3074:** "thầy nên dạy nhiều hơn , nên dạy kỹ lý thuyết hơn ." (POSITIVE) -> Model predicts NEGATIVE (0.999)
3.  **Index 12365:** "cô dạy rất nhiệt tình , tận tâm và chu đáo ." (NEGATIVE) -> Model predicts POSITIVE (0.999)

Các ví dụ này cho thấy sự mơ hồ hoặc nhầm lẫn trong việc gán nhãn ban đầu, đặc biệt là giữa các phản hồi có chứa cả yếu tố tích cực và tiêu cực hoặc các đề xuất mang tính xây dựng.

## 7. Phân tích Tương quan Sentiment - Topic

Biểu đồ heatmap về phân bố cảm xúc theo chủ đề (chuẩn hóa theo cột) cho thấy:

*   **Topic 0-lecturer:** Có sự phân bố khá đều giữa NEGATIVE (0.35) và POSITIVE (0.62).
*   **Topic 1-training_program:** Chủ yếu là NEGATIVE (0.77), với một phần nhỏ POSITIVE (0.18).
*   **Topic 2-facility:** Gần như hoàn toàn là NEGATIVE (0.96).
*   **Topic 3-others:** Có sự phân bố giữa NEGATIVE (0.40), NEUTRAL (0.28) và POSITIVE (0.32).

Kết quả này gợi ý rằng một số chủ đề (như Topic 1-training_program và Topic 2-facility) có xu hướng nhận được phản hồi tiêu cực nhiều hơn, trong khi các chủ đề khác (như Topic 0-lecturer và Topic 3-others) có sự pha trộn hoặc nghiêng về tích cực/trung lập. Điều này có thể hữu ích trong việc hiểu rõ hơn về các khía cạnh nào của trải nghiệm sinh viên đang nhận được phản hồi tiêu cực.

## 8. Đề xuất

*   **Cân bằng dữ liệu:** Do sự mất cân bằng đáng kể trong phân bố nhãn cảm xúc, nên áp dụng các kỹ thuật như SMOTE, sử dụng `class_weight` trong mô hình, hoặc lấy mẫu lại để cải thiện hiệu suất cho các lớp thiểu số.
*   **Kiểm tra nhãn sai:** Tiến hành xem xét thủ công các mẫu được xác định là có khả năng bị gán nhãn sai để sửa chữa, từ đó nâng cao độ tin cậy của dataset.
*   **Tiền xử lý nâng cao:** Khám phá các kỹ thuật tiền xử lý văn bản nâng cao hơn (ví dụ: loại bỏ stop words chuyên ngành, chuẩn hóa từ, xử lý các ký hiệu cảm xúc/teencode nếu có) để cải thiện chất lượng dữ liệu đầu vào cho mô hình.
*   **Phân tích sâu hơn các lớp thiểu số:** Do lớp NEUTRAL có số lượng mẫu rất ít, cần phân tích sâu hơn các mẫu này để hiểu rõ đặc điểm của chúng và có thể cân nhắc việc gộp với lớp khác nếu phù hợp ngữ cảnh.

## Chất lượng Tổng thể

Dataset có kích thước tốt và không có giá trị thiếu, nhưng cần cải thiện về sự cân bằng nhãn và có một số lượng mẫu tiềm năng bị gán nhãn sai cần xem xét lại.

 __Báo cáo được tổng hợp dựa trên kết quả của src/1.review_dataset_quality.ipynb__

