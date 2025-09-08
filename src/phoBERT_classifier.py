import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union, Tuple, Optional


class PhoBERTDataset(Dataset):
    """
    Dataset class cho dữ liệu phân loại cảm xúc tiếng Việt sử dụng PhoBERT.

    Attributes:
        sentences (List[str]): Danh sách câu tiếng Việt.
        labels (List[int]): Danh sách nhãn cảm xúc (0=Negative, 1=Neutral, 2=Positive).
        tokenizer (AutoTokenizer): Tokenizer của PhoBERT.
        max_len (int): Độ dài tối đa của chuỗi token.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
        """
        Khởi tạo dataset từ DataFrame.

        Args:
            df: DataFrame chứa cột 'sentence' và 'sentiment'.
            tokenizer: Tokenizer để mã hóa câu.
            max_len: Độ dài tối đa cho câu (mặc định 128).
        """
        self.sentences = df["sentence"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Lấy một mẫu dữ liệu đã được token hóa.

        Args:
            idx: Chỉ số mẫu.

        Returns:
            dict: {"input_ids", "attention_mask", "labels"} dạng torch.Tensor.
        """
        text = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class PhoBERTSentimentClassifier:
    """
    Huấn luyện, đánh giá, dự đoán và benchmark mô hình phân loại cảm xúc tiếng Việt dùng PhoBERT.

    Hỗ trợ 3 lớp cảm xúc: 0=Negative, 1=Neutral, 2=Positive.

    Attributes:
        device: Thiết bị huấn luyện (CPU/GPU).
        tokenizer: Tokenizer PhoBERT.
        model: Mô hình PhoBERT đã thêm classification head.
        max_len: Độ dài tối đa câu.
        batch_size: Kích thước batch khi huấn luyện.
        lr: Learning rate.
        train_loader: DataLoader tập huấn luyện.
        val_loader: DataLoader tập validation.
        test_loader: DataLoader tập test.

    Example:
        >>> clf = PhoBERTSentimentClassifier(batch_size=16, lr=2e-5)
        >>> clf.prepare_dataloader(train_df, val_df, test_df)
        >>> clf.train(epochs=3)
        >>> report = clf.evaluate_on_test(return_report=True)
        >>> print(report["text_report"])  # classification report chuẩn sklearn
        >>> clf.save_model("./model_phobert")
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        max_len: int = 128,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> None:
        """
        Khởi tạo PhoBERTSentimentClassifier.

        Args:
            model_name: Tên model PhoBERT trên HuggingFace (mặc định "vinai/phobert-base").
            num_labels: Số lượng nhãn phân loại (mặc định 3).
            max_len: Độ dài tối đa câu (mặc định 128).
            batch_size: Kích thước batch (mặc định 16).
            lr: Learning rate (mặc định 2e-5).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)

        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr

        # Placeholders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def prepare_dataloader(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """
        Tạo DataLoader cho train, validation và test.

        Args:
            train_df: DataFrame tập huấn luyện.
            val_df: DataFrame tập validation.
            test_df: DataFrame tập test.
        """
        train_dataset = PhoBERTDataset(train_df, self.tokenizer, self.max_len)
        val_dataset = PhoBERTDataset(val_df, self.tokenizer, self.max_len)
        test_dataset = PhoBERTDataset(test_df, self.tokenizer, self.max_len)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self, epochs: int = 3) -> None:
        """
        Huấn luyện mô hình trên `self.train_loader` và đánh giá tạm thời trên `self.val_loader`.

        Args:
            epochs: Số epoch huấn luyện (mặc định 3).
        """
        assert self.train_loader is not None and self.val_loader is not None, (
            "Bạn cần gọi prepare_dataloader trước khi train."
        )

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_training_steps = epochs * len(self.train_loader)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self.model.train()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            loop = tqdm(self.train_loader, leave=True)
            for batch in loop:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                loop.set_description(f"Epoch {epoch + 1}")
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / max(1, len(self.train_loader))
            val_acc = self.evaluate_accuracy(self.val_loader)
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

    def _predict_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán logits cho một batch tensor.

        Args:
            input_ids: Tensor input ids (B, L).
            attention_mask: Tensor attention mask (B, L).

        Returns:
            Tensor: Logits (B, num_labels).
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    def predict(
        self, texts: Union[str, List[str]], return_proba: bool = False, batch_size: Optional[int] = None
    ) -> Union[int, List[int], Tuple[List[int], np.ndarray]]:
        """
        Dự đoán nhãn cho 1 câu hoặc danh sách câu.

        Args:
            texts: Câu hoặc danh sách câu tiếng Việt.
            return_proba: Nếu True, trả thêm xác suất softmax cho từng lớp.
            batch_size: Kích thước batch khi dự đoán danh sách (mặc định dùng self.batch_size).

        Returns:
            - Nếu input là str và return_proba=False: int (nhãn dự đoán)
            - Nếu input là List[str] và return_proba=False: List[int]
            - Nếu return_proba=True: (List[int], np.ndarray[shape=(N, num_labels)])
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        bs = batch_size or self.batch_size
        preds: List[int] = []
        probas: List[np.ndarray] = []

        self.model.eval()
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            logits = self._predict_batch(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            if return_proba:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probas.extend(probs)

        if single_input and not return_proba:
            return preds[0]
        if return_proba:
            return preds, np.vstack(probas) if len(probas) else np.empty((0,))
        return preds

    def predict_on_loader(self, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        Dự đoán toàn bộ nhãn trên một DataLoader và trả về y_true, y_pred.

        Args:
            dataloader: DataLoader cần dự đoán.

        Returns:
            (y_true, y_pred): Danh sách nhãn thật và nhãn dự đoán (cùng thứ tự).
        """
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy().tolist()

                logits = self._predict_batch(input_ids, attention_mask)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()

                y_true.extend(labels)
                y_pred.extend(pred)
        return y_true, y_pred

    def evaluate_accuracy(self, dataloader: DataLoader) -> float:
        """
        Tính accuracy trên một DataLoader.

        Args:
            dataloader: DataLoader để đánh giá.

        Returns:
            float: Accuracy.
        """
        y_true, y_pred = self.predict_on_loader(dataloader)
        return float(accuracy_score(y_true, y_pred))

    def evaluate_detailed(self, dataloader: DataLoader) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Tính các chỉ số đánh giá chi tiết: accuracy, precision/recall/F1 (macro/micro/weighted),
        classification report dạng text, và confusion matrix.

        Args:
            dataloader: DataLoader để đánh giá.

        Returns:
            dict: {
                'accuracy': float,
                'precision_macro': float,
                'recall_macro': float,
                'f1_macro': float,
                'precision_micro': float,
                'recall_micro': float,
                'f1_micro': float,
                'precision_weighted': float,
                'recall_weighted': float,
                'f1_weighted': float,
                'confusion_matrix': np.ndarray (shape [num_labels, num_labels]),
                'text_report': str
            }
        """
        y_true, y_pred = self.predict_on_loader(dataloader)
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        p_w, r_w, f_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], zero_division=0
        )
        return {
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
            "precision_micro": float(p_micro),
            "recall_micro": float(r_micro),
            "f1_micro": float(f_micro),
            "precision_weighted": float(p_w),
            "recall_weighted": float(r_w),
            "f1_weighted": float(f_w),
            "confusion_matrix": cm,
            "text_report": report,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def evaluate_on_test(self, return_report: bool = False) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Đánh giá mô hình trên tập test và (tuỳ chọn) trả về báo cáo chi tiết.

        Args:
            return_report: Nếu True, trả về dict báo cáo chi tiết thay vì chỉ in ra.

        Returns:
            dict: Giống như `evaluate_detailed`, chỉ khi return_report=True.
        """
        assert self.test_loader is not None, "Chưa có test_loader, hãy gọi prepare_dataloader trước."
        metrics = self.evaluate_detailed(self.test_loader)
        if return_report:
            return metrics
        print(metrics["text_report"])  # Giữ cách dùng cũ
        return metrics

    @staticmethod
    def benchmark_predictions(
        y_true: List[int], predictions: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        So sánh nhiều mô hình bằng cách cung cấp cùng một `y_true` và các `y_pred` của từng mô hình.

        Args:
            y_true: Danh sách nhãn thật.
            predictions: Dict mapping từ tên mô hình -> danh sách nhãn dự đoán (cùng thứ tự với y_true).

        Returns:
            pd.DataFrame: Bảng tổng hợp các chỉ số (accuracy, precision/recall/F1 macro & weighted).

        Example:
            >>> # y_true lấy từ clf.evaluate_on_test(return_report=True)["y_true"]
            >>> table = PhoBERTSentimentClassifier.benchmark_predictions(
            ...     y_true,
            ...     {
            ...         "PhoBERT": y_pred_phobert,
            ...         "SVM": y_pred_svm,
            ...         "LogReg": y_pred_lr,
            ...     },
            ... )
            >>> print(table)
        """
        rows = []
        for name, y_pred in predictions.items():
            acc = accuracy_score(y_true, y_pred)
            p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            p_w, r_w, f_w, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            rows.append(
                {
                    "model": name,
                    "accuracy": acc,
                    "precision_macro": p_macro,
                    "recall_macro": r_macro,
                    "f1_macro": f_macro,
                    "precision_weighted": p_w,
                    "recall_weighted": r_w,
                    "f1_weighted": f_w,
                }
            )
        df = pd.DataFrame(rows).sort_values(by=["accuracy", "f1_macro", "f1_weighted"], ascending=False)
        return df

    def test(self) -> None:
        """
        (Giữ cách cũ) Đánh giá nhanh trên test set bằng cách in `classification_report` ra màn hình.
        """
        assert self.test_loader is not None, "Chưa có test_loader, hãy gọi prepare_dataloader trước."
        y_true, y_pred = self.predict_on_loader(self.test_loader)
        print(
            classification_report(
                y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], zero_division=0
            )
        )

    def save_model(self, path: str = "./phobert_sentiment") -> None:
        """
        Lưu model và tokenizer xuống thư mục.

        Args:
            path: Đường dẫn thư mục lưu (mặc định "./phobert_sentiment").

        Example:
            >>> clf.save_model("./my_phobert")
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str = "./phobert_sentiment") -> None:
        """
        Tải lại model và tokenizer từ thư mục.

        Args:
            path: Đường dẫn thư mục chứa model/tokenizer (mặc định "./phobert_sentiment").

        Example:
            >>> clf.load_model("./my_phobert")
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
