import re, unicodedata
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import train_test_split

try:
    from underthesea import word_tokenize as uts_tok

    _HAS_UTS = True
except Exception:
    _HAS_UTS = False

# --- regex & stopwords ---
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MAIL = re.compile(r"[\w\.-]+@[\w\.-]+", re.I)
_NUM = re.compile(r"(?<!\w)([+-]?\d[\d,\.]*)")
_PUNC = re.compile(r"[^\w\sÀ-ỹ]")
_WS = re.compile(r"\s+")
VN_STOPWORDS = {
    "và", "với", "của", "cho", "trên", "dưới", "trong", "ngoài", "từ", "như",
    "thì", "là", "mà", "bởi", "vì", "nên", "hay", "hoặc", "các", "những",
    "một", "mọi", "vài", "nhiều", "ít", "đây", "đó", "này", "kia", "ấy",
    "ta", "tôi", "bạn", "họ"
}

def _clean_basic(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = _URL.sub("<url>", s)
    s = _MAIL.sub("<email>", s)
    s = _NUM.sub("<num>", s)
    s = _PUNC.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


@dataclass
class VNPreprocessor:
    text_col: str = "sentence"
    analyzer: str = "word"  # "word" | "char"
    use_underthesea: Optional[bool] = None  # None=auto (dùng nếu có)
    add_clean: bool = True
    add_no_stop: bool = True  # chỉ tác dụng khi analyzer="word"
    add_tokens: bool = False
    add_lengths: bool = True
    custom_stopwords: Optional[set] = None

    def _tokenize_series(self, s: pd.Series) -> pd.Series:
        """
        Tách token cho từng phần tử trong Series dựa vào analyzer và underthesea (nếu có).

        Args:
            s (pd.Series): Series chứa văn bản đã làm sạch.

        Returns:
            pd.Series: Series chứa list các token (theo từ hoặc ký tự).
        """
        if self.analyzer == "char":
            return s.apply(list)
        # analyzer = word:
        want_uts = (
            _HAS_UTS if self.use_underthesea is None else bool(self.use_underthesea)
        )
        if want_uts and _HAS_UTS:
            # underthesea trả chuỗi có token nối _; tách theo khoảng trắng để được list token
            return s.apply(lambda x: uts_tok(x, format="text").split())
        return s.str.split()

    def _remove_stopwords(self, toks: pd.Series) -> pd.Series:
        """
        Loại bỏ các stopword khỏi list token trong Series.

        Args:
            toks (pd.Series): Series chứa list token.

        Returns:
            pd.Series: Series chứa list token đã loại bỏ stopword.
        """
        if self.analyzer != "word":
            return toks
        sw = (
            self.custom_stopwords if self.custom_stopwords is not None else VN_STOPWORDS
        )
        return toks.apply(lambda ts: [t for t in ts if (t not in sw and len(t) > 1)])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý DataFrame: làm sạch, tách từ, loại bỏ stopword, sinh các trường đặc trưng.

        Args:
            df (pd.DataFrame): DataFrame đầu vào chứa cột văn bản.

        Returns:
            pd.DataFrame: DataFrame mới với các trường đã tiền xử lý.
        """
        assert self.text_col in df.columns, f"Thiếu cột {self.text_col}"
        out = df.copy()

        # 1) clean
        if self.add_clean or self.add_tokens or self.add_no_stop or self.add_lengths:
            clean_col = f"{self.text_col}_clean"
            out[clean_col] = out[self.text_col].astype(str).apply(_clean_basic)

        # 2) tokenize (+ optional stopword)
        if self.add_tokens or self.add_no_stop or self.add_lengths:
            toks = self._tokenize_series(out[clean_col])
            toks2 = self._remove_stopwords(toks) if self.add_no_stop else toks

            if self.add_tokens:
                out[f"{self.text_col}_tokens"] = toks.apply(lambda x: " ".join(x))
            if self.add_no_stop:
                out[f"{self.text_col}_nostop"] = toks2.apply(lambda x: " ".join(x))
            if self.add_lengths:
                out[f"{self.text_col}_len_chars"] = out[clean_col].str.len()
                out[f"{self.text_col}_len_words"] = (
                    (toks2 if self.add_no_stop else toks).apply(len)
                    if self.analyzer == "word"
                    else None
                )
        return out

    def _safe_split(
        self,
        df_proc: pd.DataFrame,
        test_size: float,
        stratify_col: Optional[str],
        random_state: int,
    ):
        """
        Chia DataFrame thành hai phần với train_test_split, có stratify nếu đủ điều kiện.

        Args:
            df_proc (pd.DataFrame): DataFrame đã tiền xử lý.
            test_size (float): Tỉ lệ dữ liệu cho tập test.
            stratify_col (Optional[str]): Tên cột để stratify.
            random_state (int): Seed ngẫu nhiên.

        Returns:
            tuple: (train_df, test_df)
        """
        y = (
            df_proc[stratify_col]
            if stratify_col and stratify_col in df_proc.columns
            else None
        )
        try:
            return train_test_split(
                df_proc, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # rơi về không stratify nếu lớp quá ít
            return train_test_split(
                df_proc, test_size=test_size, random_state=random_state, stratify=None
            )

    def split(
        self,
        df: pd.DataFrame,
        stratify_col: Optional[str] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        """
        Chia DataFrame thành 3 phần: train, validation, test.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            stratify_col (Optional[str]): Cột dùng để stratify.
            test_size (float): Tỉ lệ tập test.
            val_size (float): Tỉ lệ tập validation.
            random_state (int): Seed ngẫu nhiên.

        Returns:
            dict: {'train': train_df, 'val': val_df, 'test': test_df}
        """
        assert 0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1
        df_proc = self.transform(df)

        # train vs (val+test)
        holdout = test_size + val_size
        train_df, tmp_df = self._safe_split(
            df_proc,
            test_size=holdout,
            stratify_col=stratify_col,
            random_state=random_state,
        )

        # val vs test
        rel_test = test_size / holdout
        y_tmp = (
            tmp_df[stratify_col]
            if stratify_col and stratify_col in tmp_df.columns
            else None
        )
        try:
            val_df, test_df = train_test_split(
                tmp_df, test_size=rel_test, random_state=random_state, stratify=y_tmp
            )
        except ValueError:
            val_df, test_df = train_test_split(
                tmp_df, test_size=rel_test, random_state=random_state, stratify=None
            )

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }
