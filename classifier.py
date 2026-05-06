"""
classifier.py
Loads pre-trained Random Forest model and classifies network flow records.
Uses CIC-DDoS2019 dataset features and labels.
FIXED: Handles feature name validation for scikit-learn version compatibility
"""

import logging
import pickle
import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

# Your exact 30 features from CIC-DDoS2019
FEATURE_COLS = [
    "Packet Length Min",
    "Fwd Packet Length Min",
    "Avg Fwd Segment Size",
    "Packet Length Max",
    "Fwd Packets Length Total",
    "Avg Packet Size",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Max",
    "Packet Length Mean",
    "Flow Bytes/s",
    "Subflow Fwd Bytes",
    "Bwd Packets/s",
    "Fwd Act Data Packets",
    "Flow Packets/s",
    "Flow IAT Std",
    "Fwd Packets/s",
    "Flow IAT Max",
    "ACK Flag Count",
    "Fwd IAT Max",
    "Flow IAT Mean",
    "Flow Duration",
    "Fwd IAT Std",
    "Init Fwd Win Bytes",
    "Fwd IAT Mean",
    "Total Fwd Packets",
    "Idle Std",
    "Bwd IAT Max",
    "Fwd IAT Total",
    "Packet Length Variance",
    "Idle Max",
]

# Your exact 18 attack classes from CIC-DDoS2019
ATTACK_LABELS = [
    "Benign",
    "DrDoS_DNS",
    "DrDoS_LDAP",
    "DrDoS_MSSQL",
    "DrDoS_NTP",
    "DrDoS_NetBIOS",
    "DrDoS_SNMP",
    "DrDoS_UDP",
    "LDAP",
    "MSSQL",
    "NetBIOS",
    "Portmap",
    "Syn",
    "TFTP",
    "UDP",
    "UDP-lag",
    "UDPLag",
    "WebDDoS",
]

# Google Drive direct download link for your model
MODEL_DRIVE_URL = (
    "https://drive.google.com/uc?export=download&id=1ye3pcl-tIZGPksxqI60lRa_HEQaPi34P"
)
MODEL_PATH = Path(__file__).parent / "rf_model_imbalanced.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"
FEATURE_NAMES_PATH = Path(__file__).parent / "feature_names.pkl"


class NetworkFlowClassifier:
    """Random Forest classifier for CIC-DDoS2019 network flow records."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = FEATURE_COLS
        self.classes_ = ATTACK_LABELS
        self.shap_explainer = None
        self._load()
        self._init_shap()

    def _download_model(self):
        """Download model from Google Drive if not present locally."""
        if not MODEL_PATH.exists():
            logger.info("Downloading model from Google Drive...")
            print("Downloading model from Google Drive, please wait...")
            gdown.download(MODEL_DRIVE_URL, str(MODEL_PATH), quiet=False)
            print("Model downloaded successfully.")

    def _load(self):
        """Load model, scaler and label encoder."""
        # Download model if needed
        self._download_model()

        # Load model
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully.")

        # Load scaler
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded successfully.")
        else:
            logger.warning("Scaler not found - proceeding without scaling")

        # Load label encoder
        if LABEL_ENCODER_PATH.exists():
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully.")
        else:
            logger.warning("Label encoder not found - using model classes directly")

        # Load feature names if available
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, "rb") as f:
                saved_features = pickle.load(f)
                # Use saved features if they match our expected list
                if len(saved_features) == len(FEATURE_COLS):
                    self.feature_names = saved_features
                    logger.info(
                        f"Loaded {len(self.feature_names)} feature names from file"
                    )

        # Use model classes if available
        if hasattr(self.model, "classes_"):
            self.classes_ = list(self.model.classes_)

    def _init_shap(self):
        """Initialize SHAP explainer for feature importance."""
        try:
            import shap

            # Use TreeExplainer for Random Forest
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialised successfully.")
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            self.shap_explainer = None
        except Exception as e:
            logger.warning(f"Could not initialize SHAP: {e}")
            self.shap_explainer = None

    def _prepare_features(self, df: pd.DataFrame):
        """
        Extract and prepare features for model input.
        Returns: (original_features_df, scaled_features_array)
        """
        # Check for required features
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        # Extract features in correct order
        X_raw = df[self.feature_names].fillna(0).astype(float)

        # Convert to numpy array to avoid sklearn feature name validation issues
        X_array = X_raw.values

        # Scale if scaler is available AND feature count matches
        if self.scaler is not None:
            # Check if scaler expects the same number of features
            expected_features = getattr(self.scaler, "n_features_in_", None)

            if expected_features is not None and expected_features != X_array.shape[1]:
                logger.warning(
                    f"Scaler expects {expected_features} features but got {X_array.shape[1]}. "
                    f"Skipping scaling - Random Forest works well without it."
                )
                X_scaled = X_array
            else:
                # Transform without feature name checking
                X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        return X_raw, X_scaled

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify network flow records.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the 30 CIC-DDoS2019 feature columns.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns:
            - attack_type  : predicted class label
            - confidence   : probability of predicted class (0-1)
            - is_attack    : True when label != Benign
            - class_probs  : dict mapping each class to its probability
            - shap_values  : feature importance (if SHAP enabled)
        """
        # Prepare features
        X_raw, X_model = self._prepare_features(df)

        # Predict
        predictions = self.model.predict(X_model)
        probabilities = self.model.predict_proba(X_model)

        # Decode labels if encoder available
        if self.label_encoder is not None:
            attack_types = self.label_encoder.inverse_transform(predictions)
        else:
            attack_types = predictions

        # Build result
        result = df.copy()
        result["attack_type"] = attack_types
        result["confidence"] = probabilities.max(axis=1).round(4)
        result["is_attack"] = result["attack_type"] != "Benign"
        result["class_probs"] = [
            {cls: round(float(prob), 4) for cls, prob in zip(self.classes_, row)}
            for row in probabilities
        ]

        # Add SHAP feature importance if available
        if self.shap_explainer is not None:
            try:
                # Calculate SHAP values for this batch
                shap_values = self.shap_explainer.shap_values(X_model)

                # For multiclass, shap_values is a list of arrays (one per class)
                # We'll store the SHAP values for the predicted class
                if isinstance(shap_values, list):
                    # Get SHAP values for predicted class
                    result["shap_values"] = [
                        {
                            "feature_importance": {
                                feat: float(shap_values[pred_class][i, j])
                                for j, feat in enumerate(self.feature_names)
                            },
                            "top_features": self._get_top_features(
                                shap_values[pred_class][i]
                            ),
                        }
                        for i, pred_class in enumerate(predictions)
                    ]
                else:
                    # Binary classification case
                    result["shap_values"] = [
                        {
                            "feature_importance": {
                                feat: float(shap_values[i, j])
                                for j, feat in enumerate(self.feature_names)
                            },
                            "top_features": self._get_top_features(shap_values[i]),
                        }
                        for i in range(len(shap_values))
                    ]

                logger.debug(f"SHAP values calculated for {len(result)} records")
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
                result["shap_values"] = [None] * len(result)
        else:
            result["shap_values"] = [None] * len(result)

        logger.info(
            "Classified %d records: %d attacks detected",
            len(result),
            result["is_attack"].sum(),
        )
        return result

    def _get_top_features(self, shap_row, top_n=10):
        """Extract top N most important features from SHAP values."""
        feature_importance = {
            self.feature_names[i]: abs(float(shap_row[i])) for i in range(len(shap_row))
        }
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_n]

    def classify_batch(self, df: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
        """
        Classify in batches to manage memory for large datasets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the 30 CIC-DDoS2019 feature columns.
        batch_size : int
            Number of rows to process per batch.

        Returns
        -------
        pd.DataFrame
            Classified DataFrame with all rows.
        """
        if len(df) <= batch_size:
            return self.classify(df)

        logger.info("Classifying %d records in batches of %d", len(df), batch_size)
        results = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            batch_result = self.classify(batch)
            results.append(batch_result)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    "Processed batch %d/%d",
                    (i // batch_size) + 1,
                    (len(df) + batch_size - 1) // batch_size,
                )

        final_result = pd.concat(results, ignore_index=False)
        logger.info(
            "Batch classification complete: %d total records", len(final_result)
        )
        return final_result

    def classify_single(self, row: dict) -> dict:
        """Classify a single flow record passed as a dictionary."""
        df = pd.DataFrame([row])
        result = self.classify(df)
        return {
            "attack_type": result["attack_type"].iloc[0],
            "confidence": result["confidence"].iloc[0],
            "is_attack": result["is_attack"].iloc[0],
            "class_probs": result["class_probs"].iloc[0],
            "shap_values": result["shap_values"].iloc[0],
        }
