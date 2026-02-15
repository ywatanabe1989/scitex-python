#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/_image/__init__.py

"""Image kind - Embedded image elements.

An image bundle contains an embedded image file.
Used for logos, photographs, diagrams in figures.

Structure:
- payload/image.*: The image file (png, jpg, svg, etc.)
- canonical/node.json: Position and sizing information
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage


def render_image(
    ax: "Axes",
    image: Union[str, Path, "np.ndarray"],
    x: float = 0,
    y: float = 0,
    width: Optional[float] = None,
    height: Optional[float] = None,
    alpha: float = 1.0,
    interpolation: str = "antialiased",
    **kwargs: Any,
) -> Optional["AxesImage"]:
    """Render embedded image on axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render on
    image : str, Path, or ndarray
        Image file path or numpy array
    x, y : float
        Position of bottom-left corner in axes coordinates
    width, height : float, optional
        Size in axes coordinates (None = auto from image aspect)
    alpha : float
        Opacity (0-1)
    interpolation : str
        Interpolation method (antialiased, nearest, bilinear, etc.)
    **kwargs : Any
        Additional kwargs passed to ax.imshow()

    Returns
    -------
    AxesImage or None
        The created image object, or None on error
    """
    import numpy as np

    # Load image if path provided
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image

            img_data = np.array(Image.open(image))
        except ImportError:
            import matplotlib.pyplot as plt

            img_data = plt.imread(str(image))
    else:
        img_data = image

    # Calculate extent based on position and size
    img_h, img_w = img_data.shape[:2]
    aspect = img_w / img_h

    if width is not None and height is not None:
        w, h = width, height
    elif width is not None:
        w = width
        h = width / aspect
    elif height is not None:
        h = height
        w = height * aspect
    else:
        # Default to small size in axes coords
        w, h = 0.2, 0.2 / aspect

    extent = [x, x + w, y, y + h]

    # Render image
    img_obj = ax.imshow(
        img_data,
        extent=extent,
        aspect="auto",
        alpha=alpha,
        interpolation=interpolation,
        transform=ax.transAxes,
        **kwargs,
    )

    return img_obj


def load_image(path: Union[str, Path]) -> "np.ndarray":
    """Load image from file.

    Parameters
    ----------
    path : str or Path
        Path to image file

    Returns
    -------
    ndarray
        Image data as numpy array
    """
    import numpy as np

    try:
        from PIL import Image

        return np.array(Image.open(path))
    except ImportError:
        import matplotlib.pyplot as plt

        return plt.imread(str(path))


__all__ = ["render_image", "load_image"]

# EOF
