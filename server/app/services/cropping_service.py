import cv2
import numpy as np
from typing import Tuple

class CroppingService:
    def process(
        self,
        image: np.ndarray,
        top_left: Tuple[float, float],
        top_right: Tuple[float, float],
        bottom_right: Tuple[float, float],
        bottom_left: Tuple[float, float],
    ) -> np.ndarray:
        """
        Crops the given image using the specified corner points.

        Parameters:
            image (np.ndarray): The input image to be cropped.
            top_left (Tuple[float, float]): Coordinates of the top-left corner (x, y).
            top_right (Tuple[float, float]): Coordinates of the top-right corner (x, y).
            bottom_right (Tuple[float, float]): Coordinates of the bottom-right corner (x, y).
            bottom_left (Tuple[float, float]): Coordinates of the bottom-left corner (x, y).

        Returns:
            np.ndarray: The cropped image.
        """
        # Convert corner points to NumPy arrays
        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # Check if the quadrilateral is an axis-aligned rectangle
        if self._is_axis_aligned_rectangle(pts):
            # Use simple array slicing for cropping
            x_min, y_min = np.min(pts, axis=0).astype(int)
            x_max, y_max = np.max(pts, axis=0).astype(int)

            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            # Perform the cropping
            cropped_image = image[y_min:y_max, x_min:x_max]
        else:
            # Use perspective transformation for asymmetrical cropping
            cropped_image = self._perspective_crop(image, pts)

        return cropped_image

    def _is_axis_aligned_rectangle(self, pts: np.ndarray, tol: float = 1e-5) -> bool:
        """
        Checks if the given four points form an axis-aligned rectangle.

        Parameters:
            pts (np.ndarray): An array of four points with shape (4, 2).
            tol (float): Tolerance for floating-point comparisons.

        Returns:
            bool: True if the points form an axis-aligned rectangle, False otherwise.
        """
        # Extract x and y coordinates
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]

        # Sort points based on x and y coordinates
        x_sorted = np.sort(x_coords)
        y_sorted = np.sort(y_coords)

        # Check if opposite sides are equal and aligned
        width1 = abs(x_sorted[0] - x_sorted[1])
        width2 = abs(x_sorted[2] - x_sorted[3])
        height1 = abs(y_sorted[0] - y_sorted[1])
        height2 = abs(y_sorted[2] - y_sorted[3])

        # Check if widths and heights are equal within tolerance
        is_width_equal = abs(width1 - width2) < tol
        is_height_equal = abs(height1 - height2) < tol

        # Check if sides are aligned with axes
        is_axis_aligned = (
            all(abs(pts[:, 0] - x_coords.astype(int)) < tol) or
            all(abs(pts[:, 1] - y_coords.astype(int)) < tol)
        )

        return is_width_equal and is_height_equal and is_axis_aligned

    def _perspective_crop(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Performs perspective cropping on the image using the given points.

        Parameters:
            image (np.ndarray): The input image.
            pts (np.ndarray): An array of four points with shape (4, 2).

        Returns:
            np.ndarray: The perspective-cropped image.
        """
        # Order points in consistent order: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(pts)

        # Compute the dimensions of the new image
        (tl, tr, br, bl) = rect

        # Compute the width of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        # Compute the height of the new image
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype='float32')

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders points in the following order: top-left, top-right, bottom-right, bottom-left.

        Parameters:
            pts (np.ndarray): An array of four points with shape (4, 2).

        Returns:
            np.ndarray: The ordered points.
        """
        rect = np.zeros((4, 2), dtype='float32')

        # Sum and difference of points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Top-left point has the smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point has the largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right point has the smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left has the largest difference
        rect[3] = pts[np.argmax(diff)]

        return rect
