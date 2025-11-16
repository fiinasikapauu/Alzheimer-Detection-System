"""
Alzheimer ML Engine - Machine Learning Logic
Berisi semua fungsi untuk preprocessing, training, dan evaluasi model SVM
Disesuaikan dengan pipeline notebook asli
"""

import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go


class AlzheimerMLEngine:
    """Engine untuk semua operasi Machine Learning"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []
        self.target_col = None
        self.numeric_cols = []
        self.cat_cols = []
        self.preprocessor = None
        self.label_encoder = None
        self.labels = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None
        self.is_trained = False
        
    def load_dataset(self, uploaded_file):
        """Load dataset dari uploaded file"""
        try:
            self.df = pd.read_csv(uploaded_file)
            return {
                'success': True,
                'message': f'Dataset berhasil dimuat: {self.df.shape[0]} baris x {self.df.shape[1]} kolom',
                'data': self.df
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membaca CSV: {e}',
                'error': str(e)
            }
    
    def select_features(self, target_col, selected_features):
        """Select fitur tertentu dari dataset"""
        if self.df is None:
            return {'success': False, 'message': 'Dataset belum dimuat'}
        
        try:
            self.target_col = target_col
            self.selected_features = selected_features
            
            # df_selected = df[selected_features + [target_col]]
            self.X = self.df[selected_features].copy()
            self.y = self.df[target_col].copy()
            
            return {
                'success': True,
                'message': f'Fitur terpilih: {len(selected_features)} fitur, Target: {target_col}',
                'X_shape': self.X.shape,
                'y_shape': self.y.shape
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal memilih fitur: {e}',
                'error': str(e)
            }
    
    def get_descriptive_stats(self):
        """Statistik awal (describe, missing values)"""
        if self.X is None or self.y is None:
            return {'success': False, 'message': 'Fitur belum dipilih'}
        
        try:
            stats = {
                'describe': self.X.describe(),
                'missing_values': self.X.isnull().sum(),
                'duplicates': self.df.duplicated().sum(),
                'target_distribution': self.y.value_counts(),
                'data_types': self.X.dtypes
            }
            
            return {
                'success': True,
                'message': 'Statistik berhasil dihitung',
                'stats': stats
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal menghitung statistik: {e}',
                'error': str(e)
            }
    
    def identify_data_types(self):
        """Identifikasi tipe data (numerical vs categorical) - sesuai kode asli"""
        if self.X is None:
            return {'success': False, 'message': 'Fitur belum dipilih'}
        
        try:
            # Sesuai dengan kode asli: int64, float64 untuk numerik
            self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.cat_cols = self.X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
            
            return {
                'success': True,
                'message': 'Tipe data berhasil diidentifikasi',
                'numeric_cols': self.numeric_cols,
                'categorical_cols': self.cat_cols
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal mengidentifikasi tipe data: {e}',
                'error': str(e)
            }
    
    def build_preprocessing_pipeline(self, numeric_impute='median', encode_categoricals=False):
        """Build preprocessing pipeline - sesuai kode asli (hanya numerik)"""
        if not self.numeric_cols and not self.cat_cols:
            self.identify_data_types()
        
        try:
            # Sesuai kode asli: hanya numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=numeric_impute)),
                ('scaler', StandardScaler())
            ])
            
            # Hanya untuk fitur numerik (sesuai kode asli)
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, self.numeric_cols)
            ])
            
            return {
                'success': True,
                'message': 'Preprocessing pipeline berhasil dibuat (numerik only)',
                'numeric_features': len(self.numeric_cols),
                'categorical_features': 0
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membuat preprocessing pipeline: {e}',
                'error': str(e)
            }
    
    def encode_target_labels(self):
        """Encode target labels - sesuai kode asli (langsung gunakan y)"""
        if self.y is None:
            return {'success': False, 'message': 'Target belum dipilih'}
        
        try:
            # Sesuai kode asli: langsung gunakan y tanpa encoding khusus
            # Karena target sudah dalam format yang benar (0/1 atau string)
            self.y_encoded = self.y.copy()
            self.labels = sorted(self.y.unique())
            self.label_encoder = None  # Tidak perlu encoder
            
            return {
                'success': True,
                'message': 'Target labels siap digunakan',
                'labels': self.labels,
                'encoded': False
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal memproses target: {e}',
                'error': str(e)
            }
    
    def train_test_split_data(self, test_size=0.2, random_state=42):
        """Train-test split"""
        if self.X is None or not hasattr(self, 'y_encoded'):
            return {'success': False, 'message': 'Data belum siap untuk split'}
        
        try:
            # Stratified split jika memungkinkan
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y_encoded, 
                    test_size=test_size, 
                    random_state=random_state, 
                    stratify=self.y_encoded
                )
            except Exception:
                # Fallback ke split biasa
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y_encoded, 
                    test_size=test_size, 
                    random_state=random_state
                )
            
            return {
                'success': True,
                'message': f'Data berhasil di-split: {len(self.X_train)} train, {len(self.X_test)} test',
                'train_size': len(self.X_train),
                'test_size': len(self.X_test)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal split data: {e}',
                'error': str(e)
            }
    
    def check_class_distribution(self):
        """Class distribution check"""
        if self.y_train is None:
            return {'success': False, 'message': 'Data training belum tersedia'}
        
        try:
            train_dist = pd.Series(self.y_train).value_counts()
            test_dist = pd.Series(self.y_test).value_counts()
            
            return {
                'success': True,
                'message': 'Distribusi kelas berhasil dihitung',
                'train_distribution': train_dist.to_dict(),
                'test_distribution': test_dist.to_dict(),
                'train_counts': train_dist,
                'test_counts': test_dist
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal menghitung distribusi kelas: {e}',
                'error': str(e)
            }
    
    def create_svm_pipeline(self):
        """SVM Pipeline (preprocessor + SVC(probability=True))"""
        if self.preprocessor is None:
            return {'success': False, 'message': 'Preprocessor belum dibuat'}
        
        try:
            self.svm_pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', SVC(probability=True, random_state=42))
            ])
            
            return {
                'success': True,
                'message': 'SVM Pipeline berhasil dibuat',
                'pipeline_steps': list(self.svm_pipeline.named_steps.keys())
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membuat SVM pipeline: {e}',
                'error': str(e)
            }
    
    def train_model_with_gridsearch(self, use_gridsearch=True, cv=5):
        """GridSearchCV sesuai kode asli dengan parameter lengkap"""
        if not hasattr(self, 'svm_pipeline'):
            self.create_svm_pipeline()
        
        if self.X_train is None or self.y_train is None:
            return {'success': False, 'message': 'Data training belum tersedia'}
        
        try:
            if use_gridsearch:
                # Parameter grid sesuai kode asli
                param_grid = {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': [0.1, 0.01, 'scale'],
                    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'classifier__class_weight': [None, 'balanced']
                }
                
                # GridSearch dengan recall_weighted scoring (sesuai kode asli)
                grid = GridSearchCV(
                    estimator=self.svm_pipeline,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='recall_weighted',
                    verbose=1,
                    n_jobs=-1
                )
                
                print("Memulai pelatihan model dan tuning hyperparameter...")
                grid.fit(self.X_train, self.y_train)
                print("Grid Search selesai.")
                
                self.model = grid.best_estimator_
                self.best_params = grid.best_params_
                self.grid_results = grid
                
                result_msg = f"Model terbaik: {self.best_params}"
            else:
                # Training biasa
                self.model = self.svm_pipeline
                self.model.fit(self.X_train, self.y_train)
                self.best_params = None
                
                result_msg = "Model SVM berhasil dilatih"
            
            self.is_trained = True
            
            return {
                'success': True,
                'message': result_msg,
                'best_params': self.best_params,
                'model_trained': True
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal melatih model: {e}',
                'error': str(e)
            }
    
    def predict_and_evaluate(self):
        """Predict + Evaluation"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'message': 'Model belum dilatih'}
        
        try:
            # Predictions
            y_pred = self.model.predict(self.X_test)
            
            # Probabilities (untuk ROC curve)
            try:
                y_prob = self.model.predict_proba(self.X_test)
                if y_prob.shape[1] >= 2:
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = None
            except Exception:
                y_prob_positive = None
            
            # Metrics sesuai kode asli
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred, 
                target_names=[str(l) for l in self.labels],
                output_dict=True
            )
            
            # ROC Curve menggunakan auc dari sklearn.metrics (sesuai kode asli)
            roc_data = None
            if y_prob_positive is not None and len(np.unique(self.y_test)) == 2:
                try:
                    fpr, tpr, thresholds = roc_curve(self.y_test, y_prob_positive)
                    roc_auc = auc(fpr, tpr)  # Menggunakan auc() bukan roc_auc_score()
                    roc_data = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc,
                        'thresholds': thresholds
                    }
                except Exception:
                    pass
            
            return {
                'success': True,
                'message': f'Evaluasi selesai - Akurasi: {accuracy:.4f}',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_prob_positive,
                'roc_data': roc_data,
                'labels': self.labels
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal melakukan evaluasi: {e}',
                'error': str(e)
            }
    
    def calculate_feature_importance(self, n_repeats=10):
        """Feature importance menggunakan permutation importance"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'message': 'Model belum dilatih'}
        
        try:
            # Permutation importance
            r = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=n_repeats, random_state=42
            )
            importances = r.importances_mean
            
            # Recover feature names
            feature_names = []
            if len(self.numeric_cols) > 0:
                feature_names.extend(self.numeric_cols)
            
            if len(self.cat_cols) > 0:
                try:
                    # Get categories from onehot encoder
                    ohe = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                    cat_names = ohe.get_feature_names_out(self.cat_cols).tolist()
                    feature_names.extend(cat_names)
                except Exception:
                    # Fallback
                    feature_names.extend([f'cat_{i}' for i in range(len(importances) - len(self.numeric_cols))])
            
            if len(feature_names) != len(importances):
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create DataFrame
            imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return {
                'success': True,
                'message': 'Feature importance berhasil dihitung',
                'importance_df': imp_df,
                'feature_names': feature_names
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal menghitung feature importance: {e}',
                'error': str(e)
            }
    
    def create_prediction_results(self):
        """Preview sample prediction untuk download"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'message': 'Model belum dilatih'}
        
        try:
            # Get predictions
            eval_result = self.predict_and_evaluate()
            if not eval_result['success']:
                return eval_result
            
            # Create results DataFrame
            results_df = self.X_test.copy()
            results_df['y_true'] = self.y_test
            results_df['y_pred'] = eval_result['predictions']
            
            if eval_result['probabilities'] is not None:
                results_df['probability'] = eval_result['probabilities']
            
            # Decode labels jika ada label encoder
            if self.label_encoder is not None:
                results_df['y_true_label'] = self.label_encoder.inverse_transform(results_df['y_true'])
                results_df['y_pred_label'] = self.label_encoder.inverse_transform(results_df['y_pred'])
            
            return {
                'success': True,
                'message': f'Hasil prediksi siap: {len(results_df)} sampel',
                'results_df': results_df,
                'sample_preview': results_df.head(20)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membuat hasil prediksi: {e}',
                'error': str(e)
            }
    
    def predict_single_sample(self, input_data):
        """Prediksi untuk satu sampel - sesuai kode asli"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'message': 'Model belum dilatih'}
        
        try:
            # Buat DataFrame dari input (sesuai kode asli)
            if isinstance(input_data, dict):
                user_data = pd.DataFrame([input_data])
            else:
                user_data = pd.DataFrame([input_data], columns=self.selected_features)
            
            # Prediksi
            prediction = self.model.predict(user_data)
            proba = self.model.predict_proba(user_data)[0]
            
            # Hasil sesuai format kode asli
            diagnosis = "Positif Alzheimer" if prediction[0] == 1 else "Negatif Alzheimer"
            confidence = max(proba) * 100
            
            return {
                'success': True,
                'message': f'Prediksi berhasil: {diagnosis}',
                'prediction': prediction[0],
                'prediction_label': diagnosis,  # Tambahkan key ini untuk konsistensi
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probabilities': {
                    'Negatif Alzheimer': proba[0] * 100,
                    'Positif Alzheimer': proba[1] * 100
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal melakukan prediksi: {e}',
                'error': str(e)
            }
    
    def get_sample_predictions_summary(self, n_samples=10):
        """Contoh hasil prediksi vs aktual - sesuai kode asli"""
        if not self.is_trained or self.model is None:
            return {'success': False, 'message': 'Model belum dilatih'}
        
        try:
            y_pred = self.model.predict(self.X_test)
            
            # Contoh hasil prediksi (sesuai kode asli)
            results = []
            benar = 0
            
            for i in range(min(n_samples, len(self.X_test))):
                actual = self.y_test.iloc[i]
                pred = y_pred[i]
                status = " Benar" if actual == pred else " Salah"
                if actual == pred:
                    benar += 1
                
                results.append({
                    'sampel': i + 1,
                    'aktual': actual,
                    'prediksi': pred,
                    'status': status
                })
            
            # Ringkasan evaluasi (sesuai kode asli)
            total = len(self.y_test)
            jumlah_benar = (self.y_test == y_pred).sum()
            jumlah_salah = total - jumlah_benar
            
            eval_result = self.predict_and_evaluate()
            
            summary = {
                'total_data_uji': total,
                'prediksi_benar': jumlah_benar,
                'prediksi_salah': jumlah_salah,
                'akurasi': eval_result['accuracy'],
                'precision': eval_result['precision'],
                'recall': eval_result['recall'],
                'f1_score': eval_result['f1_score']
            }
            
            return {
                'success': True,
                'message': f'Summary prediksi {n_samples} sampel berhasil dibuat',
                'sample_results': results,
                'summary': summary
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membuat summary: {e}',
                'error': str(e)
            }
    
    def get_correlation_matrix(self):
        """Generate correlation matrix untuk visualisasi"""
        if self.X is None:
            return {'success': False, 'message': 'Fitur belum dipilih'}
        
        try:
            numeric_features = self.X.select_dtypes(include=[np.number])
            if len(numeric_features.columns) < 2:
                return {'success': False, 'message': 'Tidak cukup fitur numerik untuk korelasi'}
            
            corr_matrix = numeric_features.corr()
            
            return {
                'success': True,
                'message': 'Matriks korelasi berhasil dibuat',
                'correlation_matrix': corr_matrix
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gagal membuat matriks korelasi: {e}',
                'error': str(e)
            }


# Helper functions
def download_csv(df, filename):
    """Helper function untuk download CSV"""
    return df.to_csv(index=False).encode('utf-8')


def create_confusion_matrix_plot(cm, labels):
    """Create confusion matrix plot"""
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title_text='Confusion Matrix', xaxis_side='top')
    return fig


def create_roc_curve_plot(fpr, tpr, auc_score):
    """Create ROC curve plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.update_layout(
        title=f'ROC Curve (AUC = {auc_score:.4f})', 
        xaxis_title='False Positive Rate', 
        yaxis_title='True Positive Rate'
    )
    return fig


def create_feature_importance_plot(imp_df, top_n=20):
    """Create feature importance plot"""
    top_features = imp_df.head(top_n)
    fig = px.bar(
        top_features, 
        x='importance', 
        y='feature', 
        orientation='h', 
        title=f'Top {len(top_features)} Feature Importance'
    )
    return fig
