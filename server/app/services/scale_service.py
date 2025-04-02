import cv2
import numpy as np

class ScaleService:
    def process(
        self, 
        image: np.ndarray, 
        scale_factor_x: float, 
        scale_factor_y: float
    ) -> np.ndarray:
        """
        Scales the given image by the specified scale factors along the x and y axes.

        Parameters:
            image (np.ndarray): The input image to be scaled.
            scale_factor_x (float): The scaling factor along the x-axis (width).
            scale_factor_y (float): The scaling factor along the y-axis (height).

        Returns:
            np.ndarray: The scaled image.

        Raises:
            ValueError: If scale factors are not positive numbers.
        """
        if scale_factor_x <= 0 or scale_factor_y <= 0:
            raise ValueError("Scale factors must be positive numbers.")

        # Calculate new dimensions
        original_height, original_width = image.shape[:2]
        new_width = int(original_width * scale_factor_x)
        new_height = int(original_height * scale_factor_y)

        # Ensure dimensions are at least 1 pixel
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Perform the scaling
        scaled_image = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_LINEAR
        )

        return scaled_image
