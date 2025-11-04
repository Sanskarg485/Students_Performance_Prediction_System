# ============================================
# model.py - Handles ML Model Loading & Prediction
# ============================================

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class StudentPerformanceModel:
    """Handles loading, preprocessing, and prediction for student performance model."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.expected_features = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and encoder from disk."""
        try:
            with open("best_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open("label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            # Save expected feature names from the scaler (if available)
            if hasattr(self.scaler, "feature_names_in_"):
                self.expected_features = list(self.scaler.feature_names_in_)
            print("✅ Model, Scaler, and Label Encoder loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model artifacts: {e}")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for consistency."""
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess uploaded CSV to match training schema."""
        df = df.copy()
        df = self._normalize_columns(df)

        # Mapping of common alternate column names to training names
        column_mapping = {
            'hours_studied': 'Hours Studied',
            'study_hours': 'Hours Studied',
            'previous_scores': 'Previous Scores',
            'past_scores': 'Previous Scores',
            'attendance_percent': 'Attendance',
            'attendance_percentage': 'Attendance',
            'exam_score': 'Exam Score',
            'test_score': 'Exam Score',
            'sleep_hours': 'Sleep Hours',
            'extracurricular_activities': 'Extracurricular Activities',
            'performance_index': 'Performance Index',
            'performance_score': 'Performance Index',
        }

        # Rename columns if matches exist
        for col in df.columns:
            if col in column_mapping:
                df.rename(columns={col: column_mapping[col]}, inplace=True)

        # Handle missing expected features
        if self.expected_features is not None:
            missing_cols = [c for c in self.expected_features if c not in df.columns]
            for col in missing_cols:
                df[col] = 0  # Add missing features with default 0

            # Keep only the expected columns (in correct order)
            df = df[self.expected_features]

        # Handle NaN values
        df = df.fillna(df.median(numeric_only=True))

        # Scale numeric columns
        X_scaled = self.scaler.transform(df)
        return X_scaled

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict student performance and append results."""
        X_scaled = self.preprocess(df)
        preds = self.model.predict(X_scaled)
        labels = self.label_encoder.inverse_transform(preds)
        df["Predicted_Performance"] = labels
        return df

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature importance if supported by model."""
        try:
            importances = self.model.feature_importances_
            features = (
                self.expected_features
                if self.expected_features
                else df.select_dtypes(include=np.number).columns
            )
            return pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
        except AttributeError:
            return pd.DataFrame(columns=["Feature", "Importance"])
