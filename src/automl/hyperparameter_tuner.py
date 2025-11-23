"""
Hyperparameter tuning for ML models.

Supports multiple optimization strategies including grid search,
random search, Bayesian optimization, and evolutionary algorithms.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class TuningMethod(Enum):
    """Hyperparameter tuning methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"


class HyperparameterTuner:
    """
    Hyperparameter tuner for ML models.

    Finds optimal hyperparameters using various search strategies.
    """

    def __init__(
        self,
        method: TuningMethod = TuningMethod.RANDOM_SEARCH,
        metric: str = "accuracy",
        maximize: bool = True,
        n_trials: int = 50,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            method: Tuning method to use
            metric: Metric to optimize
            maximize: Whether to maximize metric (vs minimize)
            n_trials: Number of trials for random/bayesian search
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed
        """
        self.method = method
        self.metric = metric
        self.maximize = maximize
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.best_params = None
        self.best_score = None
        self.trials_history = []

    def tune(
        self,
        model_class: Callable,
        param_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters.

        Args:
            model_class: Model class or creation function
            param_space: Hyperparameter search space
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cv: Number of cross-validation folds

        Returns:
            Best hyperparameters and score
        """
        logger.info(f"Starting hyperparameter tuning with {self.method.value}")
        logger.info(f"Search space: {param_space}")

        start_time = datetime.utcnow()

        if self.method == TuningMethod.GRID_SEARCH:
            result = self._grid_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )
        elif self.method == TuningMethod.RANDOM_SEARCH:
            result = self._random_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )
        elif self.method == TuningMethod.BAYESIAN:
            result = self._bayesian_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )
        elif self.method == TuningMethod.EVOLUTIONARY:
            result = self._evolutionary_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )
        elif self.method == TuningMethod.HYPERBAND:
            result = self._hyperband_search(
                model_class, param_space, X_train, y_train, X_val, y_val
            )
        else:
            raise ValueError(f"Unknown tuning method: {self.method}")

        duration = (datetime.utcnow() - start_time).total_seconds()

        result['tuning_time_seconds'] = duration
        result['n_trials'] = len(self.trials_history)

        logger.info(f"Tuning complete in {duration:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return result

    def _grid_search(
        self,
        model_class,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        cv
    ):
        """Grid search over all parameter combinations."""
        from sklearn.model_selection import GridSearchCV

        # Convert param space to sklearn format
        sklearn_params = self._convert_param_space(param_space)

        # Create base estimator
        base_estimator = model_class()

        # Grid search
        grid_search = GridSearchCV(
            base_estimator,
            sklearn_params,
            scoring=self.metric,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }

    def _random_search(
        self,
        model_class,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        cv
    ):
        """Random search over parameter space."""
        from sklearn.model_selection import RandomizedSearchCV

        sklearn_params = self._convert_param_space(param_space)
        base_estimator = model_class()

        random_search = RandomizedSearchCV(
            base_estimator,
            sklearn_params,
            n_iter=self.n_trials,
            scoring=self.metric,
            cv=cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_
        }

    def _bayesian_search(
        self,
        model_class,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        cv
    ):
        """Bayesian optimization using Optuna."""
        try:
            import optuna

            def objective(trial):
                # Sample hyperparameters
                params = {}
                for param_name, param_config in param_space.items():
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_config['type'] == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high'],
                                log=param_config.get('log', False)
                            )
                        elif param_config['type'] == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name,
                                param_config['choices']
                            )

                # Train model with cross-validation
                from sklearn.model_selection import cross_val_score

                model = model_class(**params)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv,
                    scoring=self.metric,
                    n_jobs=1
                )

                return scores.mean()

            # Create study
            direction = "maximize" if self.maximize else "minimize"
            study = optuna.create_study(direction=direction)

            # Optimize
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )

            self.best_params = study.best_params
            self.best_score = study.best_value

            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'study': study
            }

        except ImportError:
            logger.warning("Optuna not installed, falling back to random search")
            return self._random_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )

    def _evolutionary_search(
        self,
        model_class,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        cv
    ):
        """Evolutionary algorithm for hyperparameter search."""
        try:
            from sklearn.model_selection import cross_val_score
            import random

            population_size = 20
            generations = self.n_trials // population_size

            # Initialize population
            population = []
            for _ in range(population_size):
                params = self._sample_params(param_space)
                population.append(params)

            best_score_global = float('-inf') if self.maximize else float('inf')
            best_params_global = None

            for generation in range(generations):
                # Evaluate population
                scores = []
                for params in population:
                    model = model_class(**params)
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv,
                        scoring=self.metric,
                        n_jobs=1
                    )
                    score = cv_scores.mean()
                    scores.append(score)

                    # Track best
                    if self.maximize:
                        if score > best_score_global:
                            best_score_global = score
                            best_params_global = params.copy()
                    else:
                        if score < best_score_global:
                            best_score_global = score
                            best_params_global = params.copy()

                # Selection (tournament)
                selected = []
                for _ in range(population_size):
                    i1, i2 = random.sample(range(population_size), 2)
                    if self.maximize:
                        winner = i1 if scores[i1] > scores[i2] else i2
                    else:
                        winner = i1 if scores[i1] < scores[i2] else i2
                    selected.append(population[winner].copy())

                # Crossover and mutation
                new_population = []
                for i in range(0, population_size, 2):
                    parent1 = selected[i]
                    parent2 = selected[i + 1] if i + 1 < population_size else selected[0]

                    # Crossover
                    child1, child2 = self._crossover(parent1, parent2)

                    # Mutation
                    child1 = self._mutate(child1, param_space)
                    child2 = self._mutate(child2, param_space)

                    new_population.extend([child1, child2])

                population = new_population[:population_size]

                logger.info(f"Generation {generation + 1}/{generations}, Best: {best_score_global:.4f}")

            self.best_params = best_params_global
            self.best_score = best_score_global

            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'generations': generations
            }

        except Exception as e:
            logger.error(f"Evolutionary search failed: {e}")
            return self._random_search(
                model_class, param_space, X_train, y_train, X_val, y_val, cv
            )

    def _hyperband_search(
        self,
        model_class,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val
    ):
        """Hyperband algorithm for efficient hyperparameter search."""
        # Simplified Hyperband implementation
        logger.info("Using simplified Hyperband (falling back to random search)")
        return self._random_search(
            model_class, param_space, X_train, y_train, X_val, y_val, cv=3
        )

    def _convert_param_space(self, param_space):
        """Convert param space to sklearn format."""
        sklearn_params = {}
        for param_name, param_config in param_space.items():
            if isinstance(param_config, list):
                sklearn_params[param_name] = param_config
            elif isinstance(param_config, dict):
                if param_config['type'] == 'int':
                    sklearn_params[param_name] = list(range(
                        param_config['low'],
                        param_config['high'] + 1
                    ))
                elif param_config['type'] == 'categorical':
                    sklearn_params[param_name] = param_config['choices']
        return sklearn_params

    def _sample_params(self, param_space):
        """Sample parameters from param space."""
        import random

        params = {}
        for param_name, param_config in param_space.items():
            if isinstance(param_config, list):
                params[param_name] = random.choice(param_config)
            elif isinstance(param_config, dict):
                if param_config['type'] == 'int':
                    params[param_name] = random.randint(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    if param_config.get('log', False):
                        log_low = np.log(param_config['low'])
                        log_high = np.log(param_config['high'])
                        params[param_name] = np.exp(
                            random.uniform(log_low, log_high)
                        )
                    else:
                        params[param_name] = random.uniform(
                            param_config['low'],
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    params[param_name] = random.choice(param_config['choices'])
        return params

    def _crossover(self, parent1, parent2):
        """Crossover two parameter sets."""
        import random

        child1 = {}
        child2 = {}

        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _mutate(self, params, param_space, mutation_rate=0.1):
        """Mutate parameter set."""
        import random

        mutated = params.copy()

        for param_name in mutated.keys():
            if random.random() < mutation_rate:
                # Sample new value
                param_config = param_space[param_name]
                if isinstance(param_config, dict):
                    if param_config['type'] == 'int':
                        mutated[param_name] = random.randint(
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        mutated[param_name] = random.uniform(
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        mutated[param_name] = random.choice(param_config['choices'])

        return mutated

    def get_tuning_history(self):
        """Get complete tuning history."""
        return self.trials_history

    def plot_tuning_history(self, save_path: str = None):
        """Plot tuning history."""
        try:
            import matplotlib.pyplot as plt

            if not self.trials_history:
                logger.warning("No tuning history available")
                return

            trials = list(range(len(self.trials_history)))
            scores = [t['score'] for t in self.trials_history]

            plt.figure(figsize=(10, 6))
            plt.plot(trials, scores, 'b-', alpha=0.5)
            plt.plot(trials, np.maximum.accumulate(scores) if self.maximize else np.minimum.accumulate(scores), 'r-', linewidth=2)
            plt.xlabel('Trial')
            plt.ylabel(f'{self.metric}')
            plt.title('Hyperparameter Tuning Progress')
            plt.grid(True, alpha=0.3)
            plt.legend(['Trial Score', 'Best Score'])

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved tuning plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not installed, cannot plot history")
