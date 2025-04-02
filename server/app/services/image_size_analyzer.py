from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Tuple

app = FastAPI(
    title="Image Size Analyzer",
    description="API to analyze if an image is too large",
    version="1.0.0"
)

def image_too_large(image_data: bytes, threshold: int) -> Tuple[bool, str]:
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

@app.post("/analyze_image_size", response_model=dict)
async def analyze_image_size(
    image: UploadFile = File(..., description="The image file to analyze"),
    threshold: int = Query(1000, description="Maximum allowed dimension in pixels")
):
    """
    Analyze if an uploaded image is too large based on a given threshold.

    Parameters:
    - **image**: The image file to be analyzed. Must be PNG, JPG, JPEG, GIF, BMP, or WebP.
    - **threshold**: Maximum allowed dimension (width or height) in pixels. Default is 1000.

    Returns:
    A JSON object containing:
    - **too_large** (bool): Indicates if the image is too large.
    - **message** (str): A message if the image is too large, empty string otherwise.

    Raises:
    - **HTTPException 400**: If the file type is invalid or there's an error processing the image.

    Example:
    ```
    {
        "too_large": true,
        "message": "The size of the image you have provided is too large. Maximum allowed dimension is 1000px, but image is 1200x800px."
    }
    ```
    """
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported formats are PNG, JPG, JPEG, GIF, BMP, and WebP.")

    contents = await image.read()
    try:
        too_large, message = image_too_large(contents, threshold)
        return JSONResponse({
            "too_large": too_large,
            "message": message
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)