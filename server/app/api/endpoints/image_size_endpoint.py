import logging
from fastapi import FastAPI, HTTPException, UploadFile, Depends, File, Query, APIRouter
from app.services.image_size_service import ImageSizeService, ImageSizeResponse, get_image_size_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze_image_size/", response_model=ImageSizeResponse, summary="Analyze image size", description="Upload an image and check if it's too large based on a given threshold.")
async def analyze_image_size(
    image: UploadFile = File(..., description="The image file to analyze"),
    threshold: int = Query(1000, description="Maximum allowed dimension in pixels"),
    image_size_service: ImageSizeService = Depends(get_image_size_service)
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
    try:
        contents = await image.read()
        too_large, message = image_size_service.is_image_too_large(contents, threshold)
        return {
            "too_large": too_large,
            "message": message
        }
    except ValueError as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)