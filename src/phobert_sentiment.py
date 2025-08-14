# Define the corrected train_phobert_from_frames function directly in this cell
from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch
import random

TaskType = Literal["sentiment", "topic"]

def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def _frames_to_hfds(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str) -> DatasetDict:
    # Ensure label column is named 'labels' for Trainer
    train_df = train_df.rename(columns={label_col: 'labels'})
    val_df = val_df.rename(columns={label_col: 'labels'})
    test_df = test_df.rename(columns={label_col: 'labels'})

    return DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

def train_phobert(
    # === dữ liệu đã làm sạch (bắt buộc) ===
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,

    # === cấu hình nhãn & cột văn bản ===
    task: TaskType = "sentiment",              # "sentiment"(3 lớp) hoặc "topic"(4 lớp)
    text_col: str = "sentence_clean",          # ví dụ cột sau VNPreprocessor: sentence_clean / sentence_nostop / ...
    label_col: Optional[str] = None,           # mặc định auto theo task; có thể override

    # === mô hình & hparams ===
    model_name: str = "vinai/phobert-base",
    max_len: int = 256,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    weight_decay: float = 0.01,
    seed: int = 42,

    # === tiện ích ===
    print_demo_inference: bool = False,        # True để in demo suy luận cuối cùng
) -> Dict[str, float]:
    """
    Train PhoBERT chỉ từ các DataFrame đã làm sạch.
    - KHÔNG lưu checkpoint/model (nhẹ máy).
    - In metrics ra màn hình và trả về dict metrics trên tập test.

    Yêu cầu:
      - `text_col` tồn tại trong cả 3 DataFrame.
      - `label_col` tồn tại (mặc định: 'sentiment' khi task='sentiment' hoặc 'topic' khi task='topic').
      - Nhãn là số nguyên liên tục từ 0..(num_labels-1).
    """
    _set_seed(seed)

    # --- xác định cột nhãn & số lớp ---
    if label_col is None:
        label_col = "sentiment" if task == "sentiment" else "topic"
    if task == "sentiment":
        num_labels = 3
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    else:
        num_labels = 4
        id2label = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

    # --- kiểm tra cột bắt buộc ---
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if text_col not in df.columns:
            raise ValueError(f"[{split_name}] thiếu cột văn bản '{text_col}'.")
        if label_col not in df.columns:
             raise ValueError(f"[{split_name}] thiếu cột nhãn '{label_col}'.")
        if not all(df[label_col].apply(lambda x: isinstance(x, int) and 0 <= x < num_labels)):
             raise ValueError(f"[{split_name}] nhãn phải là số nguyên liên tục từ 0 đến {num_labels-1}.")


    # --- chuyển dataframe sang huggingface dataset ---
    hf_dataset = _frames_to_hfds(train_df, val_df, test_df, label_col)

    # --- tokenization ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize_function(examples):
        # Ensure 'labels' are kept during tokenization
        return tokenizer(examples[text_col], truncation=True, max_length=max_len)

    tokenized_dataset = hf_dataset.map(_tokenize_function, batched=True)

    # --- chuẩn bị dữ liệu cho training ---
    # DataCollatorWithPadding will automatically handle padding and include labels if present
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(sorted(train_df['sentiment'].unique()))  # hoặc label_col bạn dùng
    print(sorted(val_df['sentiment'].unique()))
    print(sorted(test_df['sentiment'].unique()))
    # --- xác định metric ---
    # Load both accuracy and f1 metrics
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):

      # eval_pred có thể là tuple (predictions, labels) hoặc EvalPrediction.
      # - predictions: logits shape [N, num_labels] hoặc tuple(logits, ...)
      # - labels:      shape [N]

      # Hỗ trợ cả 2 kiểu đầu vào
      try:
          logits, labels = eval_pred
      except Exception:
          logits, labels = eval_pred.predictions, eval_pred.label_ids

      # Nếu transformers trả tuple(logits, ...), lấy phần đầu
      if isinstance(logits, (tuple, list)):
          logits = logits[0]

      # Chuyển sang numpy int
      preds = np.argmax(logits, axis=-1)
      preds = preds.astype(int) if hasattr(preds, "astype") else np.asarray(preds, dtype=int)
      labels = labels.astype(int) if hasattr(labels, "astype") else np.asarray(labels, dtype=int)

      # TÍNH 2 CHỈ SỐ: accuracy và F1 macro (đa lớp)
      out = {
          "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
          "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
      }
      return out


    # --- load model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label
    )

    # --- training args: không lưu checkpoint ---
    args = TrainingArguments(
        output_dir="phobert_tmp_out",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="phobert_tmp_logs",
        logging_steps=10,
        report_to="none",
        seed=seed,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # --- trainer ---
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer, # Keep tokenizer here for DataCollatorWithPadding
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- train ---
    trainer.train()

    # --- evaluate trên tập test ---
    metrics = trainer.evaluate(tokenized_dataset["test"])
    # print(f"Metrics trên tập test: {metrics}")

    # --- demo inference (tùy chọn) ---
    if print_demo_inference:
        print("\n--- Demo Inference ---")
        pipe = trainer.get_inference_pipeline()
        for i in range(min(5, len(test_df))):
            text = test_df[text_col].iloc[i]
            true_label = test_df[label_col].iloc[i]
            pred = pipe(text)[0]
            pred_label = int(pred['label'].split('_')[-1])
            print(f"Văn bản: {text}")
            print(f"Nhãn thật: {id2label[true_label]} ({true_label})")
            print(f"Nhãn dự đoán: {pred['label']} ({pred_label})")
            print("-" * 20)


    return metrics

# Add checks for PyTorch version and CUDA availability
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))