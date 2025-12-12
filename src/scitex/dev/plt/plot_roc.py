#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_roc.py - ROC curve and AUC

"""
ROC curve (Receiver Operating Characteristic) for binary classification.
"""

import numpy as np
import scitex as stx


def plot_roc(plt, rng, ax=None):
    """ROC curve with multiple classifiers and AUC.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes
        The axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    from sklearn.metrics import roc_curve, auc

    # Generate synthetic classification data
    n_samples = 500

    # True labels
    y_true = rng.integers(0, 2, n_samples)

    # Simulate different classifiers with varying performance
    classifiers = [
        ("Good classifier", 0.85),
        ("Average classifier", 0.70),
        ("Poor classifier", 0.55),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (name, auc_target), color in zip(classifiers, colors):
        # Generate scores that would produce approximately the target AUC
        # Better classifiers have more separation between classes
        separation = (auc_target - 0.5) * 4
        y_score = rng.normal(y_true * separation, 1, n_samples)
        y_score = 1 / (1 + np.exp(-y_score))  # Sigmoid to get probabilities

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

        # Add confidence band (simulated)
        ax.fill_between(fpr, tpr * 0.95, np.minimum(tpr * 1.05, 1.0), alpha=0.1, color=color)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    # Mark optimal threshold point for best classifier
    ax.scatter([0.15], [0.85], s=100, c="red", marker="*", zorder=10, label="Optimal threshold")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves - Classifier Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


def plot_precision_recall(plt, rng, ax=None):
    """Precision-Recall curve with multiple classifiers.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes
        The axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    from sklearn.metrics import precision_recall_curve, average_precision_score

    # Generate synthetic classification data (imbalanced)
    n_samples = 500
    n_positive = int(n_samples * 0.3)  # 30% positive class

    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_samples - n_positive)])
    rng.shuffle(y_true)

    classifiers = [
        ("Good classifier", 0.85),
        ("Average classifier", 0.65),
        ("Poor classifier", 0.45),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (name, ap_target), color in zip(classifiers, colors):
        separation = (ap_target - 0.3) * 3
        y_score = rng.normal(y_true * separation, 1, n_samples)
        y_score = 1 / (1 + np.exp(-y_score))

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        ax.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AP = {ap:.3f})")

    # Plot baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline (AP = {baseline:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig, ax


