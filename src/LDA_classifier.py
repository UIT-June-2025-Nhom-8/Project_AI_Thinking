from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np

class LDAClassifier:
    def __init__(self, solver='svd'):
        self.solver = solver
        self.model = LinearDiscriminantAnalysis(solver=self.solver)
        self.vectorizer = None  # Chỉ dùng nếu X là text
        self.is_fitted = False

    def _validate_data(self, X, y=None, is_train=False):
        # Check None
        if X is None:
            raise ValueError("❌ Dữ liệu X không được để trống.")
        if is_train and y is None:
            raise ValueError("❌ Khi train, y không được để trống.")

        # Check length
        if is_train:
            if len(X) != len(y):
                raise ValueError(f"❌ Số mẫu X ({len(X)}) và y ({len(y)}) không khớp.")

        # Check type: Nếu X là text → throw error nếu chưa train vectorizer
        if isinstance(X[0], str):
            if is_train:
                raise ValueError("❌ Dữ liệu X là text, vui lòng sử dụng vectorizer để biến đổi dữ liệu.")
            else:
                if self.vectorizer is None:
                    raise ValueError("❌ Model chưa được train với dữ liệu text.")
                X = self.vectorizer.transform(X).toarray()
        else:
            # Nếu là numpy / list số
            X = np.array(X)
        return X, y

    def fit(self, X, y):
        X, y = self._validate_data(X, y, is_train=True)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("❌ Model chưa được train.")
        X, _ = self._validate_data(X, is_train=False)
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("❌ Model chưa được train.")
        X, _ = self._validate_data(X, is_train=False)
        return self.model.predict_proba(X)

    def score(self, X, y):
        if not self.is_fitted:
            raise ValueError("❌ Model chưa được train.")
        X, y = self._validate_data(X, y, is_train=False)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
