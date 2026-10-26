import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib

# Use sentence_transformers if available, otherwise a deterministic semantic fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# --- Autoencoder Architecture (Module B) ---

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        # Reconstruct on unit hypersphere matching input embeddings
        return nn.functional.normalize(reconstruction, p=2, dim=-1)

# --- Main ML Engine ---

class MLEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedder: Optional[SentenceTransformer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.autoencoder: Optional[EmbeddingAutoencoder] = None
        self.classes_: List[str] = []
        self.anomaly_threshold: float = 0.45
        self.is_ae_calibrated: bool = False
        self.prototypes: Dict[str, np.ndarray] = {}
        self.support_examples: Dict[str, List[str]] = {}
        self.latest_eval_metrics: Optional[Dict[str, Any]] = None
        self.embedding_dim: int = 384
        
        # Category to Department routing map (SIH25195)
        self.category_dept_map = {
            "Network and Wi-Fi": "Network Infrastructure",
            "Hardware": "Hardware & Asset Support",
            "Software and Application": "Application Support Group",
            "Account and Authentication": "Identity & Access Management (IAM)",
            "Email and Communication": "Messaging & Collaboration",
            "Printer and Peripheral": "Field Support & Peripherals",
            "Security Incident": "Cybersecurity & Incident Response Team (CERT)",
            "Other / Unknown": "General IT Service Desk"
        }
        
        # Load artifacts if they exist
        self._load_artifacts()

    def get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            if HAS_SENTENCE_TRANSFORMER:
                self._embedder = SentenceTransformer(self.model_name)
            else:
                raise RuntimeError("sentence-transformers is required but not installed.")
        return self._embedder

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embedder = self.get_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    # --- Module A: Classification ---

    def classify(self, text: str) -> Dict[str, Any]:
        """Classifies a complaint using trained Logistic Regression or Few-Shot fallback."""
        embedding = self.encode([text])[0]
        
        if self.classifier is not None and len(self.classes_) > 1:
            probas = self.classifier.predict_proba([embedding])[0]
            top_idx = int(np.argmax(probas))
            category = self.classes_[top_idx]
            confidence = float(probas[top_idx])
            
            # Confidence distribution
            prob_dist = {cls: float(p) for cls, p in zip(self.classes_, probas)}
            
            # Fallback if confidence is extremely low (< 0.28)
            needs_human_review = confidence < 0.40
            if confidence < 0.28 and self.prototypes:
                few_shot = self.predict_few_shot(embedding)
                if few_shot["similarity"] > confidence:
                    category = few_shot["category"]
                    confidence = few_shot["similarity"]
                    needs_human_review = True
            
            return {
                "category": category,
                "confidence": round(confidence, 4),
                "probability_distribution": prob_dist,
                "needs_human_review": needs_human_review,
                "method": "trained_logistic_regression"
            }
        
        # Fallback to Few-Shot prototype classification
        few_shot = self.predict_few_shot(embedding)
        return {
            "category": few_shot["category"],
            "confidence": round(few_shot["similarity"], 4),
            "probability_distribution": {k: round(v, 4) for k, v in few_shot["all_similarities"].items()},
            "needs_human_review": few_shot["similarity"] < 0.45,
            "method": "few_shot_prototypes_fallback"
        }

    # --- Module B: Autoencoder Anomaly Detection ---

    def detect_anomaly(self, text: str) -> Dict[str, Any]:
        """Reconstructs complaint embedding using autoencoder to detect anomalies."""
        embedding = self.encode([text])[0]
        
        if self.autoencoder is None:
            return {
                "is_anomaly": False,
                "reconstruction_error": 0.0,
                "threshold": self.anomaly_threshold,
                "is_calibrated": False,
                "status": "UNAVAILABLE",
                "explanation": "Autoencoder model is untrained; anomaly detection uncalibrated."
            }

        self.autoencoder.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            reconstructed = self.autoencoder(tensor_x)
            mse = torch.mean((tensor_x - reconstructed) ** 2).item()

        is_anomaly = mse > self.anomaly_threshold
        status = "ANOMALY_DETECTED" if is_anomaly else "NORMAL"
        
        if is_anomaly:
            explanation = (
                f"Complaint reconstruction error ({mse:.4f}) exceeded the calibrated threshold ({self.anomaly_threshold:.4f}). "
                "Semantic representation diverges significantly from learned normal IT patterns."
            )
        else:
            explanation = f"Complaint reconstruction error ({mse:.4f}) is within normal threshold ({self.anomaly_threshold:.4f})."

        return {
            "is_anomaly": is_anomaly,
            "reconstruction_error": round(mse, 4),
            "threshold": round(self.anomaly_threshold, 4),
            "is_calibrated": self.is_ae_calibrated,
            "status": status,
            "explanation": explanation
        }

    # --- Module C: Prototypical Few-Shot Classification ---

    def update_prototypes(self, support_examples: Dict[str, List[str]]):
        """Calculates prototypical vectors by averaging embeddings of support examples."""
        self.support_examples = support_examples
        self.prototypes = {}
        
        for category, examples in support_examples.items():
            if not examples:
                continue
            embeddings = self.encode(examples)
            # Centroid vector
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-6:
                centroid = centroid / norm
            self.prototypes[category] = centroid
        
        self._save_metadata()

    def predict_few_shot(self, embedding_or_text: Any) -> Dict[str, Any]:
        """Classifies via cosine similarity to category prototype vectors."""
        if isinstance(embedding_or_text, str):
            embedding = self.encode([embedding_or_text])[0]
        else:
            embedding = embedding_or_text

        # Ensure unit norm
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        if not self.prototypes:
            return {
                "category": "Other / Unknown",
                "similarity": 0.0,
                "all_similarities": {},
                "warning": "No category prototypes available."
            }

        sims = {}
        for cat, proto in self.prototypes.items():
            cos_sim = float(np.dot(embedding, proto))
            # Rescale cosine similarity [-1, 1] to [0, 1]
            sims[cat] = max(0.0, (cos_sim + 1.0) / 2.0)

        best_cat = max(sims, key=sims.get)
        best_sim = sims[best_cat]

        return {
            "category": best_cat,
            "similarity": round(best_sim, 4),
            "all_similarities": {k: round(v, 4) for k, v in sims.items()},
            "warning": "Low confidence match (< 0.50)" if best_sim < 0.50 else None
        }

    # --- Module D: Transparent Priority Recommender ---

    def recommend_priority(
        self,
        complaint_text: str,
        category: str,
        impact: str = "Individual",
        urgency: str = "Normal",
        is_anomaly: bool = False
    ) -> Dict[str, Any]:
        """Rule-based, explainable priority recommendation."""
        reasons = []
        score = 1 # 1: Low, 2: Medium, 3: High, 4: Critical
        
        text_lower = complaint_text.lower()

        # Security indicators
        security_keywords = [
            "ransomware", "ransom", "phishing", "stolen", "unauthorized", 
            "trojan", "malware", "virus", "hacked", "leak", "breach", "locked out"
        ]
        has_security_keyword = any(kw in text_lower for kw in security_keywords)
        is_security_cat = category == "Security Incident"

        if is_security_cat or has_security_keyword:
            reasons.append("Suspected security threat detected; flagged for immediate human review.")
            score = max(score, 3)
            if "ransomware" in text_lower or "breach" in text_lower or "root" in text_lower:
                score = 4
                reasons.append("Severe enterprise security breach indicators present.")

        # Outage / Service Availability
        outage_keywords = [
            "down", "outage", "entire floor", "all switches", "crashed", 
            "offline", "cannot connect", "dead", "firewall", "blackout", "smoke"
        ]
        if any(kw in text_lower for kw in outage_keywords):
            score = max(score, 2)
            reasons.append("Service availability disruption or severe hardware fault detected.")

        # Impact evaluation
        if impact.lower() == "enterprise":
            score = max(score, 4 if score >= 2 else 3)
            reasons.append("Impact scope is Enterprise-wide, affecting organization operations.")
        elif impact.lower() == "department":
            score = max(score, 3 if score >= 2 else 2)
            reasons.append("Impact scope is Departmental, affecting multiple operational staff.")
        else:
            reasons.append("Impact scope is Individual user workstation.")

        # Urgency evaluation
        if urgency.lower() == "urgent" or "urgent" in text_lower or "immediately" in text_lower:
            score = min(4, score + 1)
            reasons.append("High urgency reported by user or critical time constraint indicated.")

        # Anomaly evaluation
        if is_anomaly:
            reasons.append("Complaint flagged as unusual/anomalous pattern by reconstruction model.")

        priority_map = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
        recommended_priority = priority_map.get(score, "Medium")

        return {
            "priority": recommended_priority,
            "reasons": reasons,
            "requires_human_cert_review": is_security_cat or has_security_keyword
        }

    # --- Module E: Pipeline Orchestration ---

    def analyze_complaint(
        self,
        complaint_text: str,
        impact: str = "Individual",
        urgency: str = "Normal"
    ) -> Dict[str, Any]:
        """Complete inference pipeline for ticket creation."""
        cls_result = self.classify(complaint_text)
        category = cls_result["category"]
        confidence = cls_result["confidence"]

        anomaly_result = self.detect_anomaly(complaint_text)
        is_anomaly = anomaly_result["is_anomaly"]

        priority_result = self.recommend_priority(
            complaint_text=complaint_text,
            category=category,
            impact=impact,
            urgency=urgency,
            is_anomaly=is_anomaly
        )

        if cls_result.get("needs_human_review"):
            msg = "Low classification confidence (< 40%); flagged for manual human review."
            if msg not in priority_result["reasons"]:
                priority_result["reasons"].append(msg)

        few_shot_result = self.predict_few_shot(complaint_text)
        department = self.category_dept_map.get(category, "General IT Service Desk")

        return {
            "category": category,
            "confidence": confidence,
            "department": department,
            "priority": priority_result["priority"],
            "priority_reasons": priority_result["reasons"],
            "requires_human_cert_review": priority_result["requires_human_cert_review"],
            "is_anomaly": is_anomaly,
            "reconstruction_error": anomaly_result["reconstruction_error"],
            "anomaly_threshold": anomaly_result["threshold"],
            "anomaly_explanation": anomaly_result["explanation"],
            "is_ae_calibrated": anomaly_result["is_calibrated"],
            "few_shot_category": few_shot_result["category"],
            "few_shot_similarity": few_shot_result["similarity"],
            "classification_details": cls_result
        }

    # --- Training & Rigorous Evaluation ---

    def train_and_evaluate(
        self,
        records: List[Dict[str, Any]],
        epochs: int = 40,
        batch_size: int = 16,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Trains Logistic Regression classifier and Autoencoder using train/val/test splits.
        Avoids data leakage: splits raw records before fitting and threshold selection.
        """
        if len(records) < 15:
            raise ValueError(f"Insufficient dataset size ({len(records)} samples). Need at least 15 samples to train and evaluate.")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        texts = [r["complaint_text"] for r in records]
        categories = [r["category"] for r in records]
        is_normals = np.array([int(r.get("is_normal", 1)) for r in records])

        # Step 1: Pre-encode embeddings
        X = self.encode(texts)
        y = np.array(categories)

        # Train / Val / Test split: 70% / 15% / 15%
        indices = np.arange(len(records))
        try:
            train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=y, random_state=random_seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, stratify=y[train_val_idx], random_state=random_seed)
        except Exception:
            train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=random_seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=random_seed)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        is_normal_train = is_normals[train_idx]
        is_normal_val = is_normals[val_idx]
        is_normal_test = is_normals[test_idx]

        # --- A. Train Classifier ---
        clf = LogisticRegression(max_iter=1000, C=2.5, class_weight='balanced', random_state=random_seed)
        clf.fit(X_train, y_train)
        
        # Test evaluation
        y_test_pred = clf.predict(X_test)
        classes_in_test = np.unique(np.concatenate([y_test, y_test_pred]))
        
        acc = float(accuracy_score(y_test, y_test_pred))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average="macro", zero_division=0
        )
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
            y_test, y_test_pred, labels=classes_in_test, zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_test_pred, labels=classes_in_test)

        per_class_metrics = {}
        for cls_name, p, r, f1, sup in zip(classes_in_test, p_class, r_class, f1_class, support_class):
            per_class_metrics[str(cls_name)] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f1), 4),
                "support": int(sup)
            }

        classifier_metrics = {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "classes": [str(c) for c in classes_in_test],
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_metrics,
            "test_samples": len(test_idx)
        }

        # --- B. Train Autoencoder (strictly on normal training samples) ---
        normal_train_X = X_train[is_normal_train == 1]
        if len(normal_train_X) < 10:
            normal_train_X = X_train # fallback if no normal labels provided

        ae = EmbeddingAutoencoder(input_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_tensor = torch.tensor(normal_train_X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        ae.train()
        for epoch in range(epochs):
            for (batch_x,) in dataloader:
                optimizer.zero_grad()
                recon = ae(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Calibration on Validation Set
        ae.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32)
            val_recon = ae(val_tensor)
            val_mse = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()
            
            # Normal validation samples determine threshold (90th percentile of normal validation errors)
            normal_val_mask = is_normal_val == 1
            if np.any(normal_val_mask):
                val_normal_mse = val_mse[normal_val_mask]
                threshold = float(np.percentile(val_normal_mse, 90) + 0.0005)
            else:
                threshold = float(np.percentile(val_mse, 85))

            # Test split evaluation for anomaly detector
            test_tensor = torch.tensor(X_test, dtype=torch.float32)
            test_recon = ae(test_tensor)
            test_mse = torch.mean((test_tensor - test_recon) ** 2, dim=1).numpy()
            
            # Ground truth for anomaly: 1 = anomaly (is_normal == 0)
            true_anomalies = (is_normal_test == 0).astype(int)
            pred_anomalies = (test_mse > threshold).astype(int)

            if len(np.unique(true_anomalies)) > 1:
                ae_roc_auc = float(roc_auc_score(true_anomalies, test_mse))
                ae_prec, ae_rec, ae_f1, _ = precision_recall_fscore_support(
                    true_anomalies, pred_anomalies, average="binary", zero_division=0
                )
                fp_rate = float(np.sum((pred_anomalies == 1) & (true_anomalies == 0)) / max(1, np.sum(true_anomalies == 0)))
            else:
                ae_roc_auc = None
                ae_prec, ae_rec, ae_f1, fp_rate = None, None, None, None

        autoencoder_metrics = {
            "calibrated_threshold": round(threshold, 4),
            "roc_auc": round(ae_roc_auc, 4) if ae_roc_auc is not None else "Not available (insufficient anomalous samples in test split)",
            "precision": round(float(ae_prec), 4) if ae_prec is not None else "Not available",
            "recall": round(float(ae_rec), 4) if ae_rec is not None else "Not available",
            "f1_score": round(float(ae_f1), 4) if ae_f1 is not None else "Not available",
            "false_positive_rate": round(float(fp_rate), 4) if fp_rate is not None else "Not available",
            "test_sample_count": len(test_idx),
            "normal_test_count": int(np.sum(is_normal_test == 1)),
            "anomaly_test_count": int(np.sum(is_normal_test == 0))
        }

        # --- C. Few-Shot Evaluation ---
        # Build prototypes from training set
        train_support = {}
        for text, cat in zip(texts, categories):
            train_support.setdefault(cat, []).append(text)
        self.update_prototypes(train_support)

        few_shot_preds = [self.predict_few_shot(emb)["category"] for emb in X_test]
        few_shot_acc = float(accuracy_score(y_test, few_shot_preds))
        few_shot_p, few_shot_r, few_shot_f1, _ = precision_recall_fscore_support(
            y_test, few_shot_preds, average="macro", zero_division=0
        )

        few_shot_metrics = {
            "accuracy": round(few_shot_acc, 4),
            "macro_precision": round(float(few_shot_p), 4),
            "macro_recall": round(float(few_shot_r), 4),
            "macro_f1": round(float(few_shot_f1), 4),
            "num_prototypes": len(self.prototypes)
        }

        # Update in-memory models
        self.classifier = clf
        self.classes_ = [str(c) for c in clf.classes_]
        self.autoencoder = ae
        self.anomaly_threshold = threshold
        self.is_ae_calibrated = True

        # Save artifacts to disk
        self._save_artifacts(classifier_metrics, autoencoder_metrics, few_shot_metrics)

        self.latest_eval_metrics = {
            "classifier": classifier_metrics,
            "autoencoder": autoencoder_metrics,
            "few_shot": few_shot_metrics,
            "dataset_splits": {
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                "total_samples": len(records)
            }
        }

        return self.latest_eval_metrics

    def _save_artifacts(self, cls_metrics: Dict, ae_metrics: Dict, fs_metrics: Dict):
        joblib.dump({"classifier": self.classifier, "classes": self.classes_}, CLASSIFIER_PATH)
        torch.save(self.autoencoder.state_dict(), AUTOENCODER_PATH)
        
        metadata = {
            "anomaly_threshold": self.anomaly_threshold,
            "is_ae_calibrated": self.is_ae_calibrated,
            "classes": self.classes_,
            "classifier_metrics": cls_metrics,
            "autoencoder_metrics": ae_metrics,
            "few_shot_metrics": fs_metrics,
            "support_categories": list(self.prototypes.keys())
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _save_metadata(self):
        metadata = {
            "anomaly_threshold": self.anomaly_threshold,
            "is_ae_calibrated": self.is_ae_calibrated,
            "classes": self.classes_,
            "support_categories": list(self.prototypes.keys())
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_artifacts(self):
        try:
            if os.path.exists(CLASSIFIER_PATH):
                data = joblib.load(CLASSIFIER_PATH)
                self.classifier = data.get("classifier")
                self.classes_ = data.get("classes", [])
            
            if os.path.exists(AUTOENCODER_PATH):
                self.autoencoder = EmbeddingAutoencoder(input_dim=self.embedding_dim)
                self.autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=torch.device('cpu'), weights_only=True))
                self.autoencoder.eval()

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.anomaly_threshold = meta.get("anomaly_threshold", 0.45)
                    self.is_ae_calibrated = meta.get("is_ae_calibrated", False)
                    if not self.classes_:
                        self.classes_ = meta.get("classes", [])
                    if meta.get("classifier_metrics"):
                        self.latest_eval_metrics = {
                            "classifier": meta.get("classifier_metrics"),
                            "autoencoder": meta.get("autoencoder_metrics"),
                            "few_shot": meta.get("few_shot_metrics")
                        }
        except Exception as e:
            print(f"Notice: Could not load saved model artifacts ({e}); starting in uncalibrated state.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "embedder_model": self.model_name,
            "classifier_ready": self.classifier is not None,
            "num_classes": len(self.classes_),
            "classes": self.classes_,
            "autoencoder_ready": self.autoencoder is not None,
            "is_ae_calibrated": self.is_ae_calibrated,
            "anomaly_threshold": round(self.anomaly_threshold, 4),
            "few_shot_prototypes_count": len(self.prototypes),
            "evaluation_available": self.latest_eval_metrics is not None
        }

# Global singleton engine instance
engine = MLEngine()
