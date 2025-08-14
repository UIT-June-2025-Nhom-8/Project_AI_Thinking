import pandas as pd
import numpy as np
from vn_preprocessor import VNPreprocessor
from datasets import load_dataset
from LDA_classifier import LDAClassifier
from TFIDF_vectorlizer import TFIDFVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

# {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

########################################
#
#
# PLEASE USE PYTHON 3.11 TO RUN THIS CODE AND INSTALL THE REQUIRED LIBRARIES (ESPECIALLY UNDERTHESEA)
#
#
########################################

ds = load_dataset(
    "uitnlp/vietnamese_students_feedback", revision="refs/convert/parquet"
)

df_train = pd.DataFrame(ds["train"][:])
df_val = pd.DataFrame(ds["validation"][:])
df_test = pd.DataFrame(ds["test"][:])

prep = VNPreprocessor(
    text_col="sentence",
    analyzer="word",
    use_underthesea=True,
    add_clean=True,
    add_no_stop=True,
    add_tokens=False,
    add_lengths=True,
)

prep_raw = VNPreprocessor(
    text_col="sentence",
    analyzer="word",
    use_underthesea=False,
    add_clean=True,
    add_no_stop=True,
    add_tokens=False,
    add_lengths=True,
)

### Nếu không muốn gộp các tập dữ liệu lại, có thể sử dụng:

# transform với data raw
df_raw_train = prep_raw.transform(df_train)
df_raw_val = prep_raw.transform(df_val)
df_raw_test = prep_raw.transform(df_test)

# transform với Underthesea
df_underthesea_train = prep.transform(df_train)
df_underthesea_val = prep.transform(df_val)
df_underthesea_test = prep.transform(df_test)

# print(df_train.head())
# print(df_val.head())
# print(df_test.head())

### Nếu muốn gộp các tập dữ liệu lại và random split:
# Gộp các tập dữ liệu lại
# df = pd.concat([df_train, df_val, df_test], ignore_index=True)

# splits = prep.split(
#     df, stratify_col="sentiment", test_size=0.2, val_size=0.2, random_state=42
# )

# splits_without_underthesea = prep_without_underthesea.split(
#     df, stratify_col="sentiment", test_size=0.2, val_size=0.2, random_state=42
# )


### Vectorlizer
vectorlizer = TFIDFVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)

## vectorlizer cho tập train
df_underthesea_train = vectorlizer.fit_transform(
    df_underthesea_train, text_column="sentence_nostop", new_column="vector"
)
df_raw_train = vectorlizer.transform(
    df_raw_train, text_column="sentence_nostop", new_column="vector"
)

# Ghép thành DataFrame
df_display = pd.DataFrame(
    {
        "Raw sentence": df_raw_train["sentence_nostop"],
        "Underthesea sentence": df_underthesea_train["sentence_nostop"],
    }
)

df_vectorized_display = pd.DataFrame(
    {
        "Raw vector": df_raw_train["vector"],
        "Underthesea vector": df_underthesea_train["vector"],
    }
)

print("DataFrame hiển thị:")
print(df_display.head())
print("DataFrame vectorized hiển thị:")
print(df_vectorized_display.head())

## vectorlizer cho tập test
df_raw_test = vectorlizer.transform(
    df_raw_test, text_column="sentence_nostop", new_column="vector"
)
df_underthesea_test = vectorlizer.transform(
    df_underthesea_test, text_column="sentence_nostop", new_column="vector"
)

### LDA Classifier
raw_lda_classifier = LDAClassifier(solver="svd")
underthesea_lda_classifier = LDAClassifier(solver="svd")

# # Train model
raw_lda_classifier.fit(
    np.array(df_raw_train["vector"].tolist()),
    np.array(df_raw_train["sentiment"].tolist()),
)
underthesea_lda_classifier.fit(
    np.array(df_underthesea_train["vector"].tolist()),
    np.array(df_underthesea_train["sentiment"].tolist()),
)
print("Model đã được train thành công.")

### Dự đoán và đánh giá mô hình
## RAW
raw_actual = df_raw_test["sentiment"]
raw_accuracy = raw_lda_classifier.score(
    np.array(df_raw_test["vector"].tolist()), raw_actual
)

## UNDERTHESEA
underthesea_actual = df_underthesea_test["sentiment"]
underthesea_accuracy = underthesea_lda_classifier.score(
    np.array(df_underthesea_test["vector"].tolist()), underthesea_actual
)

print(f"Độ chính xác của mô hình trên tập test không underthesea: {raw_accuracy:.2f}")
print(f"Độ chính xác của mô hình trên tập test có underthesea: {underthesea_accuracy :.2f}")

# Add analysis/plots after LDA training
print("\n--- Analysis after LDA Training ---")

sentiment_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# 4. Classification Report
print("\nClassification Report (Raw Data):")
raw_pred = raw_lda_classifier.predict(np.array(df_raw_test["vector"].tolist()))
print(classification_report(raw_actual, raw_pred, target_names=sentiment_labels))

print("\nClassification Report (Underthesea Data):")
underthesea_pred = underthesea_lda_classifier.predict(np.array(df_underthesea_test["vector"].tolist()))
print(classification_report(underthesea_actual, underthesea_pred, target_names=sentiment_labels))

# 5. Confusion Matrix
print("\nConfusion Matrix (Raw Data):")
cm_raw = confusion_matrix(raw_actual, raw_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_labels, yticklabels=sentiment_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Raw Data)')
plt.show()

print("\nConfusion Matrix (Underthesea Data):")
cm_underthesea = confusion_matrix(underthesea_actual, underthesea_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_underthesea, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_labels, yticklabels=sentiment_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Underthesea Data)')
plt.show()

