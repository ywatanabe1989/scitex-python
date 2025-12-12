#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_3d.py - 3D plot examples

"""
3D plotting examples demonstrating various 3D visualization types.

Includes:
- 3D line plots
- 3D scatter plots
- 3D surface plots
- 3D wireframe plots
- 3D bar plots
- 3D contour plots
"""

import numpy as np
import scitex as stx


def plot_3d_line(plt, rng, ax=None):
    """3D line plot (parametric curve).

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    # Parametric helix
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)

    ax.plot(x, y, z, linewidth=2, label="Helix")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Line (Helix)")
    ax.legend()

    return fig, ax


def plot_3d_scatter(plt, rng, ax=None):
    """3D scatter plot.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    n = 100
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)
    colors = rng.random(n)
    sizes = 50 * rng.random(n) + 20

    scatter = ax.scatter(x, y, z, c=colors, s=sizes, cmap="viridis", alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter")

    return fig, ax


def plot_3d_surface(plt, rng, ax=None):
    """3D surface plot.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    # Create mesh
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Surface")

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    return fig, ax


def plot_3d_wireframe(plt, rng, ax=None):
    """3D wireframe plot.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    # Create mesh
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y)

    ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Wireframe")

    return fig, ax


def plot_3d_bar(plt, rng, ax=None):
    """3D bar plot.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    # Bar positions
    x_pos = np.arange(4)
    y_pos = np.arange(3)
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    # Bar dimensions
    dx = dy = 0.5
    dz = rng.uniform(1, 5, len(x_pos))

    colors = plt.cm.viridis(dz / dz.max()) if hasattr(plt, 'cm') else None
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("3D Bar")

    return fig, ax


def plot_3d_contour(plt, rng, ax=None):
    """3D contour plot (contour on surface).

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes3D
        The 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure

    # Create mesh
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2)

    # Surface with contour
    ax.contour3D(X, Y, Z, 30, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Contour")

    return fig, ax


def plot_3d_gallery(plt, rng):
    """Gallery of all 3D plot types.

    Creates a 2x3 grid showing different 3D visualization types.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    fig : Figure
        The figure object
    axes : list
        List of 3D axes
    """
    fig = plt.figure(figsize=(15, 10))

    # Create 2x3 grid of 3D subplots
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")

    # Plot each type
    plot_3d_line(plt, rng, ax=ax1)
    plot_3d_scatter(plt, rng, ax=ax2)
    plot_3d_surface(plt, rng, ax=ax3)
    plot_3d_wireframe(plt, rng, ax=ax4)
    plot_3d_bar(plt, rng, ax=ax5)
    plot_3d_contour(plt, rng, ax=ax6)

    fig.tight_layout()
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]


