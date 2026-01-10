# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cv/_filters.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-08
# # File: src/scitex/cv/_filters.py
# """Image filtering utilities using cv2."""
# 
# from __future__ import annotations
# 
# import cv2
# import numpy as np
# 
# 
# def blur(
#     img: np.ndarray,
#     ksize: int = 5,
#     method: str = "gaussian",
# ) -> np.ndarray:
#     """Apply blur to an image.
# 
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image.
#     ksize : int
#         Kernel size (must be odd).
#     method : str
#         Blur method: 'gaussian', 'median', 'box', 'bilateral'.
# 
#     Returns
#     -------
#     np.ndarray
#         Blurred image.
#     """
#     if ksize % 2 == 0:
#         ksize += 1
# 
#     if method == "gaussian":
#         return cv2.GaussianBlur(img, (ksize, ksize), 0)
#     elif method == "median":
#         return cv2.medianBlur(img, ksize)
#     elif method == "box":
#         return cv2.blur(img, (ksize, ksize))
#     elif method == "bilateral":
#         return cv2.bilateralFilter(img, ksize, 75, 75)
#     else:
#         raise ValueError(f"Unknown blur method: {method}")
# 
# 
# def sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
#     """Sharpen an image.
# 
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image.
#     strength : float
#         Sharpening strength.
# 
#     Returns
#     -------
#     np.ndarray
#         Sharpened image.
#     """
#     kernel = np.array(
#         [
#             [0, -1, 0],
#             [-1, 5, -1],
#             [0, -1, 0],
#         ],
#         dtype=np.float32,
#     )
# 
#     if strength != 1.0:
#         kernel = np.eye(3) + strength * (kernel - np.eye(3))
# 
#     return cv2.filter2D(img, -1, kernel)
# 
# 
# def edge_detect(
#     img: np.ndarray,
#     method: str = "canny",
#     low: int = 50,
#     high: int = 150,
# ) -> np.ndarray:
#     """Detect edges in an image.
# 
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image.
#     method : str
#         Edge detection method: 'canny', 'sobel', 'laplacian'.
#     low, high : int
#         Thresholds for Canny detector.
# 
#     Returns
#     -------
#     np.ndarray
#         Edge image.
#     """
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img
# 
#     if method == "canny":
#         return cv2.Canny(gray, low, high)
#     elif method == "sobel":
#         sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#         return np.uint8(np.sqrt(sobel_x**2 + sobel_y**2))
#     elif method == "laplacian":
#         return np.uint8(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))
#     else:
#         raise ValueError(f"Unknown edge method: {method}")
# 
# 
# def threshold(
#     img: np.ndarray,
#     thresh: int = 127,
#     maxval: int = 255,
#     method: str = "binary",
# ) -> np.ndarray:
#     """Apply thresholding to an image.
# 
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image.
#     thresh : int
#         Threshold value.
#     maxval : int
#         Maximum value.
#     method : str
#         Threshold method: 'binary', 'binary_inv', 'trunc', 'tozero',
#         'tozero_inv', 'otsu', 'adaptive_mean', 'adaptive_gaussian'.
# 
#     Returns
#     -------
#     np.ndarray
#         Thresholded image.
#     """
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img
# 
#     method_map = {
#         "binary": cv2.THRESH_BINARY,
#         "binary_inv": cv2.THRESH_BINARY_INV,
#         "trunc": cv2.THRESH_TRUNC,
#         "tozero": cv2.THRESH_TOZERO,
#         "tozero_inv": cv2.THRESH_TOZERO_INV,
#     }
# 
#     if method == "otsu":
#         _, result = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     elif method == "adaptive_mean":
#         result = cv2.adaptiveThreshold(
#             gray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
#         )
#     elif method == "adaptive_gaussian":
#         result = cv2.adaptiveThreshold(
#             gray,
#             maxval,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             11,
#             2,
#         )
#     else:
#         thresh_type = method_map.get(method, cv2.THRESH_BINARY)
#         _, result = cv2.threshold(gray, thresh, maxval, thresh_type)
# 
#     return result
# 
# 
# def denoise(
#     img: np.ndarray,
#     strength: int = 10,
#     method: str = "fastNl",
# ) -> np.ndarray:
#     """Denoise an image.
# 
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image.
#     strength : int
#         Denoising strength.
#     method : str
#         Denoising method: 'fastNl', 'bilateral'.
# 
#     Returns
#     -------
#     np.ndarray
#         Denoised image.
#     """
#     if method == "fastNl":
#         if len(img.shape) == 3:
#             return cv2.fastNlMeansDenoisingColored(img, None, strength, strength)
#         else:
#             return cv2.fastNlMeansDenoising(img, None, strength)
#     elif method == "bilateral":
#         return cv2.bilateralFilter(img, 9, strength * 7.5, strength * 7.5)
#     else:
#         raise ValueError(f"Unknown denoise method: {method}")
# 
# 
# __all__ = [
#     "blur",
#     "sharpen",
#     "edge_detect",
#     "threshold",
#     "denoise",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cv/_filters.py
# --------------------------------------------------------------------------------
