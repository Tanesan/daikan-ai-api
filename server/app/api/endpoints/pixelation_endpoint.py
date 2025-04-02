import logging
from fastapi import FastAPI, HTTPException, UploadFile, Depends, File, Form, APIRouter
from app.services.pixelation_service import PixelationService, PixelationResponse, get_pixelation_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze_pixelation/", response_model=PixelationResponse, summary="Analyze image pixelation", description="Upload an image and check if it's pixelated based on a given threshold.")
async def analyze_pixelation(
    image: UploadFile = File(..., description="The image file to analyze"),
    threshold: float = Form(0.5, description="Pixelation threshold (0-1)"),
    pixelation_service: PixelationService = Depends(get_pixelation_service)
):
    """
    Analyze if an uploaded image is pixelated based on a given threshold.

    Parameters:
    - **image**: The image file to be analyzed. Must be PNG, JPG, JPEG, GIF, BMP, or WebP.
    - **threshold**: Pixelation threshold. Scores above this value are considered pixelated. Default is 0.1.

    Returns:
    A JSON object containing:
    - **is_pixelated** (bool): Indicates if the image is considered pixelated.
    - **pixelation_score** (float): The pixelation score, where higher values indicate more pixelation.

    Raises:
    - **HTTPException 400**: If the file type is invalid or there's an error processing the image.

    Example:
    ```
    {
        "is_pixelated": true,
        "pixelation_score": 0.15
    }
    ```
    """
    try:
        contents = await image.read()
        is_pixelated, pixelation_score = pixelation_service.is_pixelated(contents, threshold)
        return {
            "is_pixelated": is_pixelated,
            "pixelation_score": pixelation_score
        }
    except ValueError as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")