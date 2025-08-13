from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class TFIDFVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8):
        """
        max_features: số lượng từ tối đa giữ lại
        ngram_range: phạm vi n-gram (1,1) = unigram, (1,2) = unigram + bigram
        min_df: bỏ các từ xuất hiện ít hơn min_df documents
        max_df: bỏ các từ xuất hiện trong > max_df documents
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.feature_names = None
        self.is_fitted = False

    def fit(self, data, text_column=None):
        """
        Huấn luyện TF-IDF (không trả về vector)
        """
        if isinstance(data, (list, pd.Series)):
            self.vectorizer.fit(data)
        elif isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValueError("Phải chỉ định 'text_column' nếu input là DataFrame")
            self.vectorizer.fit(data[text_column])
        else:
            raise TypeError("Dữ liệu phải là list/Series hoặc DataFrame")

        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return self

    def fit_transform(self, data, text_column=None, new_column="vector"):
        """
        Huấn luyện TF-IDF và trả về vector
        """
        if isinstance(data, (list, pd.Series)):
            tfidf_matrix = self.vectorizer.fit_transform(data)
        elif isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValueError("Phải chỉ định 'text_column' nếu input là DataFrame")
            tfidf_matrix = self.vectorizer.fit_transform(data[text_column])
            data[new_column] = list(tfidf_matrix.toarray())
        else:
            raise TypeError("Dữ liệu phải là list/Series hoặc DataFrame")

        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return data if isinstance(data, pd.DataFrame) else tfidf_matrix

    def transform(self, data, text_column=None, new_column="vector"):
        """
        Biến đổi dữ liệu mới sang TF-IDF (phải fit trước)
        """
        if not self.is_fitted:
            raise ValueError("❌ Vectorizer chưa được fit. Hãy gọi .fit() hoặc .fit_transform() trước.")

        if isinstance(data, (list, pd.Series)):
            return self.vectorizer.transform(data)
        elif isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValueError("Phải chỉ định 'text_column' nếu input là DataFrame")
            tfidf_matrix = self.vectorizer.transform(data[text_column])
            data[new_column] = list(tfidf_matrix.toarray())
            return data
        else:
            raise TypeError("Dữ liệu phải là list/Series hoặc DataFrame")

    def get_feature_names(self):
        """Trả về danh sách các từ khóa (features)"""
        return self.feature_names
