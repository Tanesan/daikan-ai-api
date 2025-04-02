import cv2
import numpy as np
from typing import Tuple
from pydantic import BaseModel

class PixelationResponse(BaseModel):
    is_pixelated: bool
    pixelation_score: float

class PixelationService:
    def __init__(self):
        pass
    
    def is_pixelated(self, image_data: bytes, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Detect if an image is pixelated within a certain threshold.

        Args:
            image_data (bytes): The image data.
            threshold (float): The pixelation threshold. 
                Scores above this value are considered pixelated. 
                Defaults to 0.5.

        Returns:
            Tuple[bool, float]: A tuple containing:
                - bool: True if the image is considered pixelated, False otherwise.
                - float: The pixelation score, where higher values indicate more pixelation.

        Raises:
            ValueError: If there's an error processing the image.
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate the magnitude of gradients
            magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Normalize the magnitude
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Calculate the average magnitude
            avg_magnitude = np.mean(magnitude)

            # Calculate color transitions
            diff_horiz = np.abs(img[:, 1:] - img[:, :-1]).mean()
            diff_vert = np.abs(img[1:, :] - img[:-1, :]).mean()
            color_transition = (diff_horiz + diff_vert) / 2

            # Normalize color transition
            color_transition = color_transition / 255

            # Combine edge sharpness and color transition
            pixelation_score = (1 - avg_magnitude/255 + color_transition) / 2

            # Check if the score is above the threshold
            is_pixelated = pixelation_score > threshold

            return is_pixelated, pixelation_score

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

def get_pixelation_service():
    return PixelationService()