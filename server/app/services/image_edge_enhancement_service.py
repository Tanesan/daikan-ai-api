# image_edge_enhancement_service.py

import cv2
import numpy as np
from typing import Optional
from PIL import Image
import io
from pydantic import BaseModel
from enum import Enum

class EnhancementMethod(str, Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    UNSHARP_MASK = "unsharp_mask"
    HIGH_PASS_FILTER = "high_pass_filter"
    SHARPEN = "sharpen"
    LAPLACIAN_FILTER = "laplacian_filter"
    EDGE_ENHANCE = "edge_enhance"
    SOBEL_FILTER = "sobel_filter"

class ImageProcessingParams(BaseModel):
    kernel_size: Optional[int] = 5
    alpha: Optional[float] = 1.5
    method: EnhancementMethod = EnhancementMethod.GAUSSIAN_BLUR

class ImageEdgeEnhancementService:
    
    def __init__(self):
        pass
    
    def process(self, image: bytes, kernel_size: int = 5, alpha: float = 1.5, method: str = "gaussian_blur") -> Image:
        """
        Process an image to enhance its edges using the specified algorithm.

        Args:
            image_data (bytes): Raw image data
            kernel_size (int): Kernel size for edge enhancement algorithms. Default is 5.
            alpha (float): Alpha parameter for unsharp mask algorithm. Default is 1.5.
            method (str): Edge enhancement method. Options:
                - gaussian_blur
                - unsharp_mask
                - high_pass_filter
                - sharpen
                - laplacian_filter
                - edge_enhance
                - sobel_filter
              Default is gaussian_blur.

        Returns:
            Image: PIL Image object representing the processed image

        Raises:
            ValueError: If the method parameter is invalid
        """
        try:            
            # Process image
            params = ImageProcessingParams(kernel_size=kernel_size, alpha=alpha, method=method)
            processed_image = self.process_image(image, params)
            
            return processed_image
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def process_image(self, image: Image, params: ImageProcessingParams):
        """
        Process an image to enhance its edges using the specified algorithm.

        Args:
            image (Image): PIL Image object
            params (ImageProcessingParams): Parameters for edge enhancement

        Returns:
            Image: PIL Image object representing the processed image

        Raises:
            ValueError: If the method parameter is invalid
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply Gaussian Blur first
        blurred_image = self.apply_gaussian_blur(image, kernel_size=params.kernel_size)
        
        # Apply the chosen method
        if params.method == EnhancementMethod.GAUSSIAN_BLUR:
            result_image = blurred_image
        elif params.method == EnhancementMethod.UNSHARP_MASK:
            result_image = self.apply_unsharp_mask(blurred_image, kernel_size=params.kernel_size, alpha=params.alpha)
        elif params.method == EnhancementMethod.HIGH_PASS_FILTER:
            result_image = self.apply_high_pass_filter(blurred_image)
        elif params.method == EnhancementMethod.SHARPEN:
            result_image = self.apply_sharpen(blurred_image)
        elif params.method == EnhancementMethod.LAPLACIAN_FILTER:
            result_image = self.apply_laplacian_filter(blurred_image)
        elif params.method == EnhancementMethod.EDGE_ENHANCE:
            result_image = self.apply_edge_enhance(blurred_image)
        elif params.method == EnhancementMethod.SOBEL_FILTER:
            result_image = self.apply_sobel_filter(blurred_image)
        else:
            raise ValueError("Unsupported method")

        # Convert the processed image to PIL for return
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image

    def apply_gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian Blur to the image."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_unsharp_mask(self, image, kernel_size=5, alpha=1.5):
        """Apply Unsharp Mask to the image."""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
        return sharpened

    def apply_high_pass_filter(self, image):
        """Apply High Pass Filter to the image."""
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        high_pass = cv2.filter2D(image, -1, kernel)
        return cv2.add(image, high_pass)

    def apply_sharpen(self, image):
        """Apply Sharpen filter to the image."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def apply_laplacian_filter(self, image):
        """Apply Laplacian filter to the image."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        return cv2.add(image, laplacian_abs)

    def apply_edge_enhance(self, image):
        """Enhance edges in the image."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def apply_sobel_filter(self, image):
        """Apply Sobel filter to the image."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = cv2.convertScaleAbs(sobel)
        return cv2.add(image, sobel)

def get_image_edge_enhancement_service():
    return ImageEdgeEnhancementService()