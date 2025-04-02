from pydantic import BaseModel
from PIL import Image


class ImageRequest(BaseModel):
    """
    This model defines the structure of the request payload, which includes
    a Base64 encoded image.

    Attributes:
        image_base64 (str): The Base64 encoded image data as a string.
    """
    image_base64: str

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resizes the image proportionally such that the larger dimension is set to max_size.

    This function maintains the aspect ratio of the image to prevent distortion during resizing.

    Args:
        image (Image.Image): The PIL.Image object to be resized.
        max_size (int): The maximum pixel value for the larger dimension of the image.

    Returns:
        Image.Image: The resized PIL.Image object.
    """
    width, height = image.size

    # Calculate new dimensions to ensure the larger dimension is max_size
    if width > height:
        new_width = max_size
        new_height = int((new_width / width) * height)
    else:
        new_height = max_size
        new_width = int((new_height / height) * width)

    return image.resize((new_width, new_height), Image.LANCZOS)

