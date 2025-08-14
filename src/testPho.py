import pandas as pd
from vn_preprocessor import VNPreprocessor
from datasets import load_dataset
from phobert_sentiment import train_phobert_from_frames
from phobert_cl import train_phobert
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

### Nếu không muốn gộp các tập dữ liệu lại, có thể sử dụng:
df_train = prep.transform(df_train)
df_val = prep.transform(df_val)
df_test = prep.transform(df_test)

# print(df_train.head())
# print(df_val.head())
# print(df_test.head())

### Nếu muốn gộp các tập dữ liệu lại và random split:
# Gộp các tập dữ liệu lại
# df = pd.concat([df_train, df_val, df_test], ignore_index=True)

# splits = prep.split(
#     df, stratify_col="sentiment", test_size=0.2, val_size=0.2, random_state=42
# )

#metrics = train_phobert_from_frames(df_train, df_val, df_test)
metrics = train_phobert(df_train, df_val, df_test)
print("\nReturned metrics dict:", metrics)