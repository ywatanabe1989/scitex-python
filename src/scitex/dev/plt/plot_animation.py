#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_animation.py - Animation plot examples

"""
Animation examples demonstrating matplotlib animation capabilities.

Includes:
- Line animation (wave propagation)
- Scatter animation (particle motion)
- Bar animation (growing bars)
- Surface animation (rotating 3D surface)

Note: These functions return animation objects, not static figures.
Use fig.savefig() for static frames, or anim.save() for video output.
"""

import numpy as np
import scitex as stx


def create_line_animation(plt, rng, frames=100, interval=50):
    """Create animated line plot (wave propagation).

    Parameters
    ----------
    plt : module
        Plotting module (must be matplotlib.pyplot)
    rng : numpy.random.Generator
        Random number generator
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    fig : Figure
        The figure object
    anim : FuncAnimation
        The animation object

    Example
    -------
    >>> fig, anim = create_line_animation(plt, rng)
    >>> anim.save("wave.gif", writer="pillow", fps=20)
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 4 * np.pi, 200)
    line, = ax.plot([], [], "-", linewidth=2)
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Wave Propagation Animation")
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        phase = frame * 0.1
        y = np.sin(x - phase) * np.exp(-0.1 * np.abs(x - phase))
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=interval, blit=True)
    return fig, anim


def create_scatter_animation(plt, rng, frames=100, interval=50):
    """Create animated scatter plot (particle motion).

    Parameters
    ----------
    plt : module
        Plotting module (must be matplotlib.pyplot)
    rng : numpy.random.Generator
        Random number generator
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    fig : Figure
        The figure object
    anim : FuncAnimation
        The animation object

    Example
    -------
    >>> fig, anim = create_scatter_animation(plt, rng)
    >>> anim.save("particles.gif", writer="pillow", fps=20)
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))

    n_particles = 50
    # Initial positions
    x = rng.uniform(-5, 5, n_particles)
    y = rng.uniform(-5, 5, n_particles)
    # Velocities
    vx = rng.uniform(-0.1, 0.1, n_particles)
    vy = rng.uniform(-0.1, 0.1, n_particles)
    colors = rng.random(n_particles)
    sizes = 50 * rng.random(n_particles) + 20

    scatter = ax.scatter(x, y, c=colors, s=sizes, cmap="viridis", alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Particle Motion Animation")
    ax.set_aspect("equal")

    def update(frame):
        nonlocal x, y
        # Update positions
        x += vx
        y += vy
        # Bounce off walls
        mask_x = np.abs(x) > 5
        mask_y = np.abs(y) > 5
        vx[mask_x] *= -1
        vy[mask_y] *= -1
        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)
        scatter.set_offsets(np.c_[x, y])
        return scatter,

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    return fig, anim


def create_bar_animation(plt, rng, frames=50, interval=100):
    """Create animated bar chart (growing bars).

    Parameters
    ----------
    plt : module
        Plotting module (must be matplotlib.pyplot)
    rng : numpy.random.Generator
        Random number generator
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    fig : Figure
        The figure object
    anim : FuncAnimation
        The animation object

    Example
    -------
    >>> fig, anim = create_bar_animation(plt, rng)
    >>> anim.save("bars.gif", writer="pillow", fps=10)
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["A", "B", "C", "D", "E"]
    target_values = rng.uniform(20, 100, len(categories))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    bars = ax.bar(categories, np.zeros(len(categories)), color=colors)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_title("Bar Chart Animation")

    def update(frame):
        progress = min(1.0, frame / (frames * 0.8))
        for bar, target in zip(bars, target_values):
            bar.set_height(target * progress)
        return bars

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    return fig, anim


def create_heatmap_animation(plt, rng, frames=50, interval=100):
    """Create animated heatmap (diffusion simulation).

    Parameters
    ----------
    plt : module
        Plotting module (must be matplotlib.pyplot)
    rng : numpy.random.Generator
        Random number generator
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    fig : Figure
        The figure object
    anim : FuncAnimation
        The animation object

    Example
    -------
    >>> fig, anim = create_heatmap_animation(plt, rng)
    >>> anim.save("diffusion.gif", writer="pillow", fps=10)
    """
    from matplotlib.animation import FuncAnimation
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial state - hot spot in center
    size = 50
    data = np.zeros((size, size))
    data[size//2, size//2] = 100

    im = ax.imshow(data, cmap="hot", vmin=0, vmax=50, interpolation="bilinear")
    ax.set_title("Heat Diffusion Animation")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(frame):
        nonlocal data
        # Simulate diffusion
        data = gaussian_filter(data, sigma=0.5)
        # Add small random noise
        data += rng.uniform(0, 0.1, data.shape)
        im.set_array(data)
        return im,

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    return fig, anim


def create_3d_rotation_animation(plt, rng, frames=100, interval=50):
    """Create animated 3D surface (rotation).

    Parameters
    ----------
    plt : module
        Plotting module (must be matplotlib.pyplot)
    rng : numpy.random.Generator
        Random number generator
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    fig : Figure
        The figure object
    anim : FuncAnimation
        The animation object

    Example
    -------
    >>> fig, anim = create_3d_rotation_animation(plt, rng)
    >>> anim.save("rotation.gif", writer="pillow", fps=20)
    """
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create surface
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Surface Rotation")

    def update(frame):
        ax.view_init(elev=30, azim=frame * 3.6)
        return surf,

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    return fig, anim


# Registry of animation creators
ANIMATIONS = {
    "line": create_line_animation,
    "scatter": create_scatter_animation,
    "bar": create_bar_animation,
    "heatmap": create_heatmap_animation,
    "3d_rotation": create_3d_rotation_animation,
}


def list_animations():
    """List available animation types."""
    return list(ANIMATIONS.keys())


def get_animation(name):
    """Get animation creator function by name."""
    if name in ANIMATIONS:
        return ANIMATIONS[name]
    raise KeyError(f"Unknown animation: {name}. Available: {list(ANIMATIONS.keys())}")


