#!/usr/bin/env python
# Created by "Thieu" at 15:44, 03/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graforvfl.network.gfo_rvfl_cv import GfoRvflCV


class GfoRvflComparator:
    """
    A class to compare different optimizers for the GfoRvflCV model.

    Attributes:
        problem_type (str): Type of problem, either 'regression' or 'classification'.
        bounds (dict): Bounds for the hyperparameters.
        optim_list (list): List of optimizers to compare.
        optim_params_list (list): List of parameters for each optimizer.
        scoring (str): Scoring metric to evaluate the model.
        cv (int): Number of cross-validation folds.
        seed (int): Random seed for reproducibility.
        verbose (bool): Verbosity mode.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(self, problem_type="regression", bounds=None,
                 optim_list=None, optim_params_list=None,
                 scoring="MSE", cv=None,
                 seed=None, verbose=True, **kwargs):
        """
        Initializes the GfoRvflComparator with the given parameters.

        Args:
            problem_type (str): Type of problem, either 'regression' or 'classification'.
            bounds (list): Bounds for the hyperparameters.
            optim_list (list): List of optimizers to compare.
            optim_params_list (list): List of parameters for each optimizer.
            scoring (str): Scoring metric to evaluate the model.
            cv (int): Number of cross-validation folds.
            seed (int): Random seed for reproducibility.
            verbose (bool): Verbosity mode.
            **kwargs: Additional keyword arguments.
        """
        if len(optim_list) != len(optim_params_list):
            raise ValueError("Length of optim_list and optim_params_list must be the same.")

        self.problem_type = problem_type
        self.bounds = bounds
        self.optim_list = optim_list
        self.optim_params_list = optim_params_list
        self.scoring = scoring
        self.cv = cv
        self.seed = seed
        self.verbose = verbose
        self.kwargs = kwargs
        self.generator = np.random.default_rng(seed)
        self.optimizer_names = []

    def run(self, X_train, y_train, X_test, y_test, n_trials=3,
            list_metrics=("MSE", "NSE", "KGE", "R", "MAE"),
            save_results=True, save_models=False, path_save="history"):
        """
        Run comparison across all optimizers.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            X_test (array-like): Testing data features.
            y_test (array-like): Testing data labels.
            n_trials (int): Number of trials to run for each optimizer.
            list_metrics (tuple): List of metrics to evaluate.
            save_results (bool): Whether to save the results to files.
            save_models (bool): Whether to save the trained models.
            path_save (str): Path to save the results and models.

        Returns:
            tuple: List of trained models, list of training losses, and DataFrame of metric results.
        """
        Path(path_save).mkdir(parents=True, exist_ok=True)
        if n_trials < 1:
            n_trials = 1
        seed_list = self.generator.choice(range(1, 100), n_trials, replace=False)

        metric_results = []  # Store all trial results for this optimizer
        metric_unfold = []
        list_trained_models = []
        list_loss_train = []
        for optim, params in zip(self.optim_list, self.optim_params_list):
            print(f"Running optimizer: {optim} with params: {params}")

            trial_models = []
            trial_loss_train = {}

            for idx_trial, seed in enumerate(seed_list):
                print(f"\tTrial {idx_trial+1}/{n_trials} with seed: {seed}...")

                # Ghi lại thời gian chạy
                start_time = time.perf_counter()

                # Tạo model với optimizer hiện tại
                model = GfoRvflCV(problem_type=self.problem_type, bounds=self.bounds,
                                  optim=optim, optim_params=params,
                                  scoring=self.scoring, cv=self.cv,
                                  seed=seed, verbose=self.verbose, **self.kwargs)
                # Train model
                model.fit(X_train, y_train)
                # Ghi lại thời gian chạy
                elapsed_time = time.perf_counter() - start_time
                scores = model.best_estimator.scores(X_test, y_test, list_metrics=list_metrics)

                # Lưu kết quả
                metric_results.append({
                    "optimizer": model.optim.name,
                    "trial": idx_trial,
                    "best_scores": scores,
                    "best_params": model.best_params,
                    "optimizer_params": params,
                    "time_seconds": elapsed_time,
                })
                trial_models.append(model)
                trial_loss_train[f"trial_{idx_trial}"] = model.loss_train

                # Handle metric unfold to save in csv file
                res = {"optimizer": model.optim.name, "trial": idx_trial, "time_seconds": elapsed_time, **scores}
                metric_unfold.append(res)

                # Save optimizer name for later use
                if idx_trial == 0:
                    self.optimizer_names.append(model.optim.name)

            list_trained_models.append(trial_models)
            list_loss_train.append(trial_loss_train)

            if save_results:
                pd.DataFrame(trial_loss_train).to_csv(f"{path_save}/{model.optim.name}-loss_train.csv", index=False)
            if save_models:
                for idx, model in enumerate(trial_models):
                    with open(f"{path_save}/{model.optim.name}-trial_{idx}-model.pkl", "wb") as ff:
                        pickle.dump(model, ff)

        if save_results:
            met = list(list_metrics) + ["time_seconds"]
            pd.DataFrame(metric_results).to_csv(f"{path_save}/metric_results_full.csv", index=False)
            df = pd.DataFrame(metric_unfold)
            df_mean = df.groupby("optimizer")[met].mean().reset_index()
            df_std = df.groupby("optimizer")[met].std().reset_index()

            df_mean.to_csv(f"{path_save}/metric_results_mean.csv", index=False)
            df_std.to_csv(f"{path_save}/metric_results_std.csv", index=False)
            df.to_csv(f"{path_save}/metric_results_unfold.csv", index=False)

        return list_trained_models, list_loss_train, pd.DataFrame(metric_results)

    def plot_loss_train_per_trial(self, path_read="history", path_save="history",
                                  fig_size=(7, 5), exts=(".png", ".pdf"), verbose=False):
        """
        Plot comparison of loss_train for each trial.

        Args:
            path_read (str): Path where the loss_train files are saved.
            path_save (str): Path where to save the figures.
            fig_size (tuple): Size of the figure.
            exts (tuple): File extensions for saving the figures.
            verbose (bool): Whether to print additional information.
        """
        y_label = "Accuracy" if self.problem_type == "classification" else "Loss"
        Path(path_read).mkdir(parents=True, exist_ok=True)
        Path(path_save).mkdir(parents=True, exist_ok=True)
        dfs = []
        for optim_name in self.optimizer_names:
            df = pd.read_csv(f"{path_read}/{optim_name}-loss_train.csv")
            dfs.append(df)

        # Số lượng trials (giả sử tất cả models có số trial như nhau)
        n_trials = dfs[0].shape[1]
        fig_size = fig_size if isinstance(fig_size, tuple) else (7, 5)
        for idx in range(n_trials):
            plt.figure(figsize=fig_size)
            for df, name in zip(dfs, self.optimizer_names):
                plt.plot(df[f"trial_{idx}"], label=name)

            plt.title(f"Trial {idx}")
            plt.xlabel("Iteration")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            for ext in exts:
                plt.savefig(f"{path_save}/loss_train_trial_{idx}{ext}")
            if verbose:
                plt.show()

    def plot_loss_train_average(self, path_read="history", path_save="history",
                                fig_size=(7, 5), exts=(".png", ".pdf"), verbose=False):
        """
        Plot average loss_train across trials for each model.

        Args:
            path_read (str): Path where the loss_train files are saved.
            path_save (str): Path where to save the figures.
            fig_size (tuple): Size of the figure.
            exts (tuple): File extensions for saving the figures.
            verbose (bool): Whether to print additional information.
        """
        Path(path_read).mkdir(parents=True, exist_ok=True)
        Path(path_save).mkdir(parents=True, exist_ok=True)

        y_label = "Average Accuracy" if self.problem_type == "classification" else "Average Loss"

        fig_size = fig_size if isinstance(fig_size, tuple) else (7, 5)
        plt.figure(figsize=fig_size)
        for optim_name in self.optimizer_names:
            df = pd.read_csv(f"{path_read}/{optim_name}-loss_train.csv")
            mean_loss = df.mean(axis=1)
            plt.plot(mean_loss, label=optim_name)

        plt.title(f"{y_label} Of Training Set Over Trials")
        plt.xlabel("Iteration")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        for ext in exts:
            plt.savefig(f"{path_save}/loss_train_average{ext}")
        if verbose:
            plt.show()

    def plot_metric_boxplot(self, path_read="history", path_save="history",
                            fig_size=(7, 5), exts=(".png", ".pdf"), verbose=False):
        """
        Plot boxplot for each metric.

        Args:
            path_read (str): Path where the loss_train files are saved.
            path_save (str): Path where to save the figures.
            fig_size (tuple): Size of the figure.
            exts (tuple): File extensions for saving the figures.
            verbose (bool): Whether to print additional information.
        """
        Path(path_read).mkdir(parents=True, exist_ok=True)
        Path(path_save).mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(f"{path_read}/metric_results_unfold.csv")
        metrics = df.columns.difference(["optimizer", "trial", "time_seconds"])

        fig_size = fig_size if isinstance(fig_size, tuple) else (7, 5)
        for metric in metrics:
            plt.figure(figsize=fig_size)
            df.boxplot(column=metric, by="optimizer")
            plt.title(f"Boxplot for {metric}")
            plt.suptitle("")
            plt.xlabel("Optimizer")
            plt.ylabel(metric)
            plt.tight_layout()
            for ext in exts:
                plt.savefig(f"{path_save}/metric_boxplot_{metric}{ext}")
            if verbose:
                plt.show()

    def plot_average_runtime(self, path_read="history", path_save="history",
                             fig_size=(7, 5), exts=(".png", ".pdf"), verbose=False):
        """
        Plot average runtime for each model.

        Args:
            path_read (str): Path where the loss_train files are saved.
            path_save (str): Path where to save the figures.
            fig_size (tuple): Size of the figure.
            exts (tuple): File extensions for saving the figures.
            verbose (bool): Whether to print additional information.
        """
        Path(path_read).mkdir(parents=True, exist_ok=True)
        Path(path_save).mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(f"{path_read}/metric_results_unfold.csv")
        avg_runtime = df.groupby("optimizer")["time_seconds"].mean()

        fig_size = fig_size if isinstance(fig_size, tuple) else (7, 5)
        plt.figure(figsize=fig_size)
        avg_runtime.plot(kind="bar")
        plt.xlabel("Optimizer")
        plt.ylabel("Average Runtime (seconds)")
        plt.title("Average Runtime Comparison")
        plt.tight_layout()
        for ext in exts:
            plt.savefig(f"{path_save}/average_runtime_chart{ext}")
        if verbose:
            plt.show()
