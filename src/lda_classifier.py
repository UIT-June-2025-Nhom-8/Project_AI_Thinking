"""
Linear Discriminant Analysis (LDA) Classifier Pipeline
TF-IDF → LDA (Dimensionality Reduction) → Sentiment Classification

Giải thích pipeline:
1. Text preprocessing → TF-IDF vectorization
2. Linear Discriminant Analysis → Dimensionality reduction + feature extraction
3. Reduced features → Sentiment classification
4. Compare performance với BERT

Mục tiêu: So sánh LDA-based approach với BERT cho sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# LDA (Linear Discriminant Analysis) và libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Additional classifiers để so sánh
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Visualization
import warnings
warnings.filterwarnings('ignore')

class LDAClassifier:
    """
    Linear Discriminant Analysis-based Sentiment Classifier
    
    Workflow:
    1. Text → TF-IDF → High-dimensional vectors
    2. TF-IDF → LDA → Low-dimensional discriminant features
    3. LDA features → Direct classification (LDA có built-in classifier)
    4. So sánh với additional classifiers trên LDA features
    """
    
    def __init__(self, text_col: str = "sentence_clean", label_col: str = "sentiment"):
        """
        Initialize Linear Discriminant Analysis Classifier
        
        Args:
            text_col: Tên column chứa text đã preprocessing
            label_col: Tên column chứa sentiment labels (0,1,2)
        """
        self.text_col = text_col
        self.label_col = label_col
        
        # Main components
        self.tfidf_vectorizer = None    # TF-IDF vectorizer
        self.lda_model = None           # Linear Discriminant Analysis model
        
        # Additional classifiers để compare
        self.additional_classifiers = {}
        self.best_classifier = None
        
        # Results storage
        self.lda_features = None        # LDA-transformed features
        self.results = {}
        
    def fit(self, train_df: pd.DataFrame, 
            val_df: pd.DataFrame = None,
            max_features: int = 1000,
            random_state: int = 42) -> Dict[str, Any]:
        """
        Train Linear Discriminant Analysis classifier pipeline
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional, để validate performance)
            max_features: Số features tối đa cho TF-IDF
            random_state: Random seed cho reproducibility
            
        Returns:
            Dictionary chứa thông tin training results
        """
        print("Bắt đầu training Linear Discriminant Analysis Pipeline...")
        print("Pipeline: Text → TF-IDF → LDA (Dimensionality Reduction) → Classification")
        
        # Step 1: TF-IDF Vectorization
        print("Bước 1: TF-IDF Vectorization...")
        print("- Chuyển text thành numerical vectors")
        print("- TF-IDF weight các từ quan trọng")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),        # Unigrams + bigrams
            min_df=2,                  # Ignore rare words
            max_df=0.8,                # Ignore too common words
            stop_words=None
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(train_df[self.text_col])
        print(f"Kết quả TF-IDF shape: {tfidf_matrix.shape} (documents, features)")
        
        # Step 2: Linear Discriminant Analysis
        print("Bước 2: Linear Discriminant Analysis...")
        print("- LDA tìm linear combinations của features để separate classes tốt nhất")
        print("- Reduce dimensions từ high-dim TF-IDF về low-dim discriminant space")
        print("- Built-in classifier: LDA vừa reduce dimension vừa classify luôn")
        
        # Get labels
        y_train = train_df[self.label_col]
        n_classes = len(np.unique(y_train))
        max_components = min(tfidf_matrix.shape[1], n_classes - 1)
        
        print(f"- Số classes: {n_classes}")
        print(f"- LDA sẽ tạo tối đa {max_components} discriminant functions")
        
        # Initialize LDA
        self.lda_model = LinearDiscriminantAnalysis(
            solver='svd',  # Suitable cho high-dimensional data
            store_covariance=True
        )
        
        # Fit LDA (nó sẽ vừa learn discriminant functions vừa fit classifier)
        print("Đang training LDA...")
        self.lda_model.fit(tfidf_matrix.toarray(), y_train)  # LDA cần dense array
        
        # Transform training data để lấy LDA features
        self.lda_features = self.lda_model.transform(tfidf_matrix.toarray())
        print(f"Kết quả LDA features shape: {self.lda_features.shape} (documents, discriminant_features)")
        
        print(f"- LDA đã reduce từ {tfidf_matrix.shape[1]} features về {self.lda_features.shape[1]} discriminant features")
        print("- Mỗi discriminant feature là linear combination tối ưu để separate classes")
        
        # Step 3: Chỉ sử dụng LDA built-in classifier
        print("Bước 3: Sử dụng LDA Built-in Classifier...")
        print("- LDA có classifier tích hợp sẵn, không cần train thêm")
        print("- LDA classifier sử dụng discriminant functions để classify")
        
        # Compile results
        results = {
            'max_features': max_features,
            'tfidf_features': tfidf_matrix.shape[1],
            'lda_components': self.lda_features.shape[1],
            'n_classes': n_classes
        }
        
        print("Hoàn thành Linear Discriminant Analysis training!")
        return results
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict sentiment using LDA built-in classifier
        
        Args:
            test_df: Test dataframe
            
        Returns:
            Array chứa predictions
        """
        # Transform test data
        tfidf_test = self.tfidf_vectorizer.transform(test_df[self.text_col])
        
        # Sử dụng LDA built-in classifier
        predictions = self.lda_model.predict(tfidf_test.toarray())
        
        return predictions
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluate LDA built-in classifier trên test set
        """
        print("Đang evaluate Linear Discriminant Analysis...")
        
        # Transform test data
        tfidf_test = self.tfidf_vectorizer.transform(test_df[self.text_col])
        y_test = test_df[self.label_col]
        
        # Evaluate LDA built-in classifier
        print(f"\nĐang evaluate LDA BUILT-IN CLASSIFIER...")
        y_pred_lda = self.lda_model.predict(tfidf_test.toarray())
        
        accuracy_lda = accuracy_score(y_test, y_pred_lda)
        f1_lda = f1_score(y_test, y_pred_lda, average='weighted')
        
        print(f"   Accuracy: {accuracy_lda:.4f}")
        print(f"   F1-Score: {f1_lda:.4f}")
        
        results = {
            'lda': {
                'accuracy': accuracy_lda,
                'f1_score': f1_lda,
                'predictions': y_pred_lda,
                'classification_report': classification_report(y_test, y_pred_lda, output_dict=True)
            }
        }
        
        self.results = results
        self.best_classifier = 'lda'
        
        print(f"\nSử dụng LDA Built-in Classifier (Accuracy: {accuracy_lda:.4f})")
        
        return results
    
    def _validate_classifiers(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate tất cả classifiers trên validation set
        """
        # Transform validation data
        tfidf_val = self.tfidf_vectorizer.transform(val_df[self.text_col])
        lda_features_val = self.lda_model.transform(tfidf_val.toarray())
        y_val = val_df[self.label_col]
        
        val_results = {}
        
        # Validate LDA built-in
        y_pred_lda = self.lda_model.predict(tfidf_val.toarray())
        accuracy_lda = accuracy_score(y_val, y_pred_lda)
        f1_lda = f1_score(y_val, y_pred_lda, average='weighted')
        
        val_results['lda'] = {
            'accuracy': accuracy_lda,
            'f1_score': f1_lda
        }
        print(f"   LDA: Acc={accuracy_lda:.4f}, F1={f1_lda:.4f}")
        
        # Validate additional classifiers
        for name, clf in self.additional_classifiers.items():
            y_pred = clf.predict(lda_features_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            val_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            print(f"   {name}: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        return val_results
    
    def analyze_discriminant_functions(self):
        """
        Analyze các discriminant functions mà LDA đã học được
        """
        print("\nPHÂN TÍCH DISCRIMINANT FUNCTIONS:")
        print("=" * 50)
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"Số discriminant functions: {self.lda_model.coef_.shape[0]}")
        print(f"Mỗi function là combination của {self.lda_model.coef_.shape[1]} TF-IDF features")
        
        # Analyze từng discriminant function
        for i, coef in enumerate(self.lda_model.coef_):
            print(f"\nDiscriminant Function {i}:")
            
            # Top positive weights (words supporting this class)
            top_pos_idx = coef.argsort()[-5:][::-1]
            print("   Top positive words:")
            for idx in top_pos_idx:
                if coef[idx] > 0:
                    word = feature_names[idx]
                    # Ensure proper UTF-8 display
                    print(f"     {word}: {coef[idx]:.4f}")
            
            # Top negative weights
            top_neg_idx = coef.argsort()[:5]
            print("   Top negative words:")
            for idx in top_neg_idx:
                if coef[idx] < 0:
                    word = feature_names[idx]
                    # Ensure proper UTF-8 display
                    print(f"     {word}: {coef[idx]:.4f}")
    
    def visualize(self, save_plots: bool = True):
        """
        Create visualizations cho LDA results
        """
        print("Đang tạo các biểu đồ...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: LDA features distribution
        plt.subplot(2, 3, 1)
        if self.lda_features.shape[1] >= 2:
            plt.scatter(self.lda_features[:, 0], self.lda_features[:, 1], 
                       c=np.random.choice(3, len(self.lda_features)), alpha=0.6)
            plt.xlabel('Discriminant Function 1')
            plt.ylabel('Discriminant Function 2')
            plt.title('LDA Feature Space')
        
        # Plot 2: Classifier performance comparison
        if self.results:
            plt.subplot(2, 3, 2)
            accuracies = [result['accuracy'] for result in self.results.values()]
            f1_scores = [result['f1_score'] for result in self.results.values()]
            names = list(self.results.keys())
            
            x = np.arange(len(names))
            width = 0.35
            
            plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            plt.xlabel('Classifier')
            plt.ylabel('Score')
            plt.title('Classifier Performance')
            plt.xticks(x, names, rotation=45)
            plt.legend()
        
        # Plot 3: Feature importance heatmap
        plt.subplot(2, 3, 3)
        if hasattr(self.lda_model, 'coef_'):
            coef_matrix = self.lda_model.coef_
            sns.heatmap(coef_matrix, cmap='RdBu', center=0, cbar=True)
            plt.title('Discriminant Function Coefficients')
            plt.xlabel('TF-IDF Features (sample)')
            plt.ylabel('Discriminant Functions')
        
        # Plot 4: Dimensionality reduction visualization
        plt.subplot(2, 3, 4)
        original_dim = len(self.tfidf_vectorizer.get_feature_names_out())
        reduced_dim = self.lda_features.shape[1]
        
        plt.bar(['Original TF-IDF', 'LDA Reduced'], [original_dim, reduced_dim])
        plt.title('Dimensionality Reduction')
        plt.ylabel('Number of Features')
        plt.yscale('log')
        
        # Plot 5: Class separation visualization  
        plt.subplot(2, 3, 5)
        # Simplified class separation plot
        if self.lda_features.shape[1] >= 1:
            plt.hist(self.lda_features[:, 0], bins=20, alpha=0.7)
            plt.xlabel('First Discriminant Function')
            plt.ylabel('Frequency')
            plt.title('Distribution along DF1')
        
        # Plot 6: Model info
        plt.subplot(2, 3, 6)
        info_text = f"""
        LDA Model Info:
        
        Original Features: {original_dim}
        LDA Features: {reduced_dim}
        Classes: {len(np.unique(getattr(self, '_y_train', [0,1,2])))}
        
        Reduction Ratio: 
        {original_dim/reduced_dim:.1f}x smaller
        """
        plt.text(0.1, 0.5, info_text, fontsize=10, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Model Summary')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('linear_lda_classifier_results.png', dpi=300, bbox_inches='tight')
            print("Đã lưu biểu đồ vào 'linear_lda_classifier_results.png'")
        
        plt.show()
    
    def get_summary(self) -> str:
        """
        Generate comprehensive summary report
        """
        lines = []
        lines.append("=" * 50)
        lines.append("TÓM TẮT KẾT QUẢ LINEAR DISCRIMINANT ANALYSIS")
        lines.append("=" * 50)
        
        if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer:
            original_features = len(self.tfidf_vectorizer.get_feature_names_out())
            lda_features = self.lda_features.shape[1] if self.lda_features is not None else 0
            
            lines.append(f"Thông tin model:")
            lines.append(f"   - TF-IDF features: {original_features}")
            lines.append(f"   - LDA features: {lda_features}")
            lines.append(f"   - Reduction ratio: {original_features/max(lda_features,1):.1f}x")
        
        if self.results:
            lines.append(f"\nKết quả classification:")
            for name, result in self.results.items():
                lines.append(f"   - {name.title()}: Acc={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
            lines.append(f"   - Tốt nhất: {self.best_classifier}")
        
        return "\n".join(lines)


def run_linear_lda_classifier(train_df: pd.DataFrame, 
                             val_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             text_col: str = "sentence_clean",
                             label_col: str = "sentiment",
                             max_features: int = 1000) -> Tuple[LDAClassifier, Dict]:
    """
    Run complete Linear Discriminant Analysis pipeline
    """
    print("BẮT ĐẦU LINEAR DISCRIMINANT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize classifier
    lda_clf = LDAClassifier(text_col=text_col, label_col=label_col)
    
    # Train
    train_results = lda_clf.fit(
        train_df, 
        val_df=val_df,
        max_features=max_features
    )
    
    # Analyze discriminant functions
    lda_clf.analyze_discriminant_functions()
    
    # Evaluate trên test set
    eval_results = lda_clf.evaluate(test_df)
    
    # Create visualizations
    lda_clf.visualize()
    
    # Print summary
    print("\n" + lda_clf.get_summary())
    
    return lda_clf, eval_results
