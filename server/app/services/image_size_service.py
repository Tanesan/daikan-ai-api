from PIL import Image
import io
from typing import Tuple
from pydantic import BaseModel

class ImageSizeResponse(BaseModel):
    too_large: bool
    message: str

class ImageSizeService:
    def __init__(self):
        pass
    
    def is_image_too_large(self, image_data: bytes, threshold: int) -> Tuple[bool, str]:
        """
        Check if an image is too large based on a given threshold.

        Args:
            image_data (bytes): The image data.
            threshold (int): The maximum allowed dimension (width or height).

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if the image is too large, False otherwise.
                - str: A message if the image is too large, empty string otherwise.

        Raises:
            ValueError: If there's an error processing the image.
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                width, height = img.size
                if width > threshold or height > threshold:
                    return True, f"The size of the image you have provided is too large. Maximum allowed dimension is {threshold}px, but image is {width}x{height}px."
                return False, ""
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

def get_image_size_service():
    return ImageSizeService()