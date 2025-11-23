"""
Automated Machine Learning Pipeline.

Automatically selects models, engineers features, tunes hyperparameters,
and builds production-ready ML pipelines.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    Automated ML pipeline for classification and regression tasks.

    Automatically:
    - Detects problem type (classification/regression)
    - Engineers features
    - Selects best model
    - Tunes hyperparameters
    - Trains final model
    - Generates deployment artifacts
    """

    def __init__(
        self,
        task: str = "auto",
        metric: str = "auto",
        time_budget: int = 300,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize AutoML pipeline.

        Args:
            task: Task type ('auto', 'classification', 'regression')
            metric: Metric to optimize ('auto' or specific metric)
            time_budget: Time budget in seconds
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.task = task
        self.metric = metric
        self.time_budget = time_budget
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.feature_importances = None
        self.pipeline_history = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Fit AutoML pipeline.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Self
        """
        logger.info("Starting AutoML pipeline")
        start_time = datetime.utcnow()

        # Step 1: Detect task type
        if self.task == "auto":
            self.task = self._detect_task_type(y_train)
            logger.info(f"Detected task type: {self.task}")

        # Step 2: Select metric
        if self.metric == "auto":
            self.metric = self._select_metric(self.task)
            logger.info(f"Selected metric: {self.metric}")

        # Step 3: Feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_val_processed = self._engineer_features(
            X_train, X_val
        )

        # Step 4: Model selection
        logger.info("Selecting best model...")
        model_candidates = self._get_model_candidates(self.task)

        best_model_name = None
        best_model_score = float('-inf')

        for model_name, model_class, param_space in model_candidates:
            logger.info(f"Evaluating {model_name}...")

            # Quick evaluation
            score = self._quick_evaluate(
                model_class, X_train_processed, y_train, self.metric
            )

            self.pipeline_history.append({
                'model': model_name,
                'score': score,
                'timestamp': datetime.utcnow().isoformat()
            })

            if score > best_model_score:
                best_model_score = score
                best_model_name = model_name
                self.best_model_class = model_class
                self.best_param_space = param_space

        logger.info(f"Best model: {best_model_name} (score: {best_model_score:.4f})")

        # Step 5: Hyperparameter tuning
        logger.info("Tuning hyperparameters...")

        from src.automl.hyperparameter_tuner import HyperparameterTuner, TuningMethod

        tuner = HyperparameterTuner(
            method=TuningMethod.BAYESIAN,
            metric=self.metric,
            n_trials=min(50, self.time_budget // 10),
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        tuning_result = tuner.tune(
            self.best_model_class,
            self.best_param_space,
            X_train_processed,
            y_train,
            X_val_processed,
            y_val
        )

        self.best_params = tuning_result['best_params']
        self.best_score = tuning_result['best_score']

        # Step 6: Train final model
        logger.info("Training final model with best parameters...")
        self.best_model = self.best_model_class(**self.best_params)
        self.best_model.fit(X_train_processed, y_train)

        # Step 7: Feature importance
        self._compute_feature_importances(X_train_processed)

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"AutoML pipeline complete in {duration:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_processed, _ = self._engineer_features(X, None)
        return self.best_model.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only).

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")

        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_processed, _ = self._engineer_features(X, None)
        return self.best_model.predict_proba(X_processed)

    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect whether task is classification or regression."""
        unique_values = len(np.unique(y))
        n_samples = len(y)

        # If small number of unique values compared to samples, likely classification
        if unique_values / n_samples < 0.05:
            return "classification"
        else:
            return "regression"

    def _select_metric(self, task: str) -> str:
        """Select appropriate metric based on task."""
        if task == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"

    def _engineer_features(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Engineer features automatically.

        Args:
            X: Input features
            X_val: Validation features

        Returns:
            Processed features
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Handle missing values
        if not hasattr(self, 'imputer_'):
            self.imputer_ = SimpleImputer(strategy='mean')
            X_imputed = self.imputer_.fit_transform(X)
        else:
            X_imputed = self.imputer_.transform(X)

        # Scale features
        if not hasattr(self, 'scaler_'):
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler_.transform(X_imputed)

        # Process validation set
        if X_val is not None:
            X_val_imputed = self.imputer_.transform(X_val)
            X_val_scaled = self.scaler_.transform(X_val_imputed)
        else:
            X_val_scaled = None

        return X_scaled, X_val_scaled

    def _get_model_candidates(self, task: str) -> List[Tuple]:
        """Get model candidates for given task."""
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC

            return [
                (
                    "RandomForest",
                    RandomForestClassifier,
                    {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
                    }
                ),
                (
                    "GradientBoosting",
                    GradientBoostingClassifier,
                    {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 10}
                    }
                ),
                (
                    "LogisticRegression",
                    LogisticRegression,
                    {
                        'C': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True},
                        'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']},
                        'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']}
                    }
                )
            ]
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.svm import SVR

            return [
                (
                    "RandomForest",
                    RandomForestRegressor,
                    {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
                    }
                ),
                (
                    "GradientBoosting",
                    GradientBoostingRegressor,
                    {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 10}
                    }
                ),
                (
                    "Ridge",
                    Ridge,
                    {
                        'alpha': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True}
                    }
                )
            ]

    def _quick_evaluate(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        metric: str
    ) -> float:
        """Quick model evaluation using cross-validation."""
        from sklearn.model_selection import cross_val_score

        model = model_class(random_state=self.random_state)

        scores = cross_val_score(
            model, X, y,
            cv=3,
            scoring=metric,
            n_jobs=self.n_jobs
        )

        return scores.mean()

    def _compute_feature_importances(self, X: np.ndarray):
        """Compute feature importances if model supports it."""
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances = self.best_model.feature_importances_

            # Log top features
            if self.feature_importances is not None:
                n_features = min(10, len(self.feature_importances))
                top_indices = np.argsort(self.feature_importances)[-n_features:][::-1]

                logger.info("Top feature importances:")
                for i, idx in enumerate(top_indices, 1):
                    logger.info(f"  {i}. Feature {idx}: {self.feature_importances[idx]:.4f}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the best model."""
        return {
            'model_class': self.best_model_class.__name__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'metric': self.metric,
            'task': self.task,
            'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None
        }

    def save_model(self, path: str):
        """Save trained model."""
        import joblib

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler_,
            'imputer': self.imputer_,
            'task': self.task,
            'metric': self.metric,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'feature_importances': self.feature_importances
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str):
        """Load trained model."""
        import joblib

        model_data = joblib.load(path)

        pipeline = cls(task=model_data['task'], metric=model_data['metric'])
        pipeline.best_model = model_data['model']
        pipeline.scaler_ = model_data['scaler']
        pipeline.imputer_ = model_data['imputer']
        pipeline.best_params = model_data['best_params']
        pipeline.best_score = model_data['best_score']
        pipeline.feature_importances = model_data['feature_importances']

        logger.info(f"Model loaded from {path}")
        return pipeline

    def generate_deployment_config(self) -> Dict[str, Any]:
        """Generate configuration for model deployment."""
        return {
            'model_type': 'sklearn',
            'task': self.task,
            'preprocessing': {
                'scaling': 'standard',
                'imputation': 'mean'
            },
            'model_params': self.best_params,
            'metrics': {
                'training_score': float(self.best_score),
                'metric_name': self.metric
            },
            'deployment_recommendations': self._get_deployment_recommendations()
        }

    def _get_deployment_recommendations(self) -> Dict[str, Any]:
        """Get deployment platform recommendations based on model characteristics."""
        # Estimate model size
        import sys

        model_size_mb = sys.getsizeof(self.best_model) / (1024 * 1024)

        recommendations = {
            'model_size_mb': float(model_size_mb),
            'recommended_platforms': []
        }

        # Platform recommendations based on size and latency
        if model_size_mb < 50:
            recommendations['recommended_platforms'].append({
                'platform': 'AWS Lambda',
                'reason': 'Small model size, cost-effective for variable workloads'
            })
            recommendations['recommended_platforms'].append({
                'platform': 'Vercel',
                'reason': 'Edge deployment for global low latency'
            })

        if model_size_mb < 250:
            recommendations['recommended_platforms'].append({
                'platform': 'GCP Cloud Run',
                'reason': 'Containerized deployment with auto-scaling'
            })

        recommendations['recommended_platforms'].append({
            'platform': 'AWS SageMaker',
            'reason': 'Managed ML infrastructure with monitoring'
        })

        recommendations['recommended_platforms'].append({
            'platform': 'Kubernetes',
            'reason': 'Maximum control and scalability'
        })

        return recommendations
