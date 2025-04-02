import cv2
import numpy as np
from typing import Tuple
from pydantic import BaseModel

class PixelationDetectionResponse(BaseModel):
    is_pixelated: bool
    pixelation_score: float

class PixelationDetectionService:
    def __init__(self):
        pass
    
    def is_pixelated(self, image: np.ndarray, threshold: float = 0.3, tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Detects pixelation in the image.

        Args:
            image (np.ndarray): Image in OpenCV format.
            threshold (float): Pixelation detection threshold (default is 0.3).
            tolerance (float): Tolerance for pixelation detection (default is 0.05).

        Returns:
            Tuple[bool, float]: A tuple containing two values:
                - bool: True if the image is pixelated, False otherwise.
                - float: The pixelation ratio of the image.

        Raises:
            Exception: If there's an error during pixelation detection.
        """
        try:
            # Call detect_pixelation method
            pixelated, pixelation_ratio = self.detect_pixelation(image, sensitivity=tolerance, threshold=threshold)
            
            return pixelated, pixelation_ratio
        
        except Exception as e:
            raise Exception(f"Error detecting pixelation: {str(e)}")
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances the image by increasing resolution and applying edge enhancement.

        Parameters:
            image (np.ndarray): Input image in OpenCV format.

        Returns:
            np.ndarray: Enhanced image.
        """
        # Increase resolution by resizing
        scale_factor = 2  # Adjustable scale factor
        enhanced_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Apply edge enhancement using a kernel
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

        return enhanced_image

    def detect_pixelation(self, image: np.ndarray, sensitivity=0.3, threshold=0.05) -> Tuple[bool, float]:
        """
        Detects pixelation in the image using the Laplacian method.

        Parameters:
            image (np.ndarray): Input image in OpenCV format.
            sensitivity (float): Sensitivity for pixelation detection.
            threshold (float): Pixelation detection threshold.

        Returns:
            Tuple[bool, float]: A tuple containing two values:
                - bool: True if the image is pixelated, False otherwise.
                - float: The pixelation ratio of the image.
        """
        try:
            enhanced_image = self.enhance_image(image)
            
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            laplacian_abs = np.abs(laplacian)
            _, binary_image = cv2.threshold(laplacian_abs, sensitivity * np.max(laplacian_abs), 255, cv2.THRESH_BINARY)
            binary_image = binary_image.astype(np.uint8)
            
            pixelated_area = np.sum(binary_image > 0)
            total_area = binary_image.size
            pixelation_ratio = pixelated_area / total_area
            
            pixelated = pixelation_ratio > threshold
            
            return pixelated, pixelation_ratio
        
        except Exception as e:
            raise Exception(f"Error detecting pixelation: {str(e)}")

def get_pixelation_detection_service():
    return PixelationDetectionService()
