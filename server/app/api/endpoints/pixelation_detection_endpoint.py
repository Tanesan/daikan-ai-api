import logging
from fastapi import FastAPI, HTTPException, UploadFile, Depends, File, Form, APIRouter
from app.services.pixelation_detection_service import PixelationDetectionService, PixelationDetectionResponse, get_pixelation_detection_service
from app.services.s3_service import S3Service
from app.api.deps import get_s3_service
from io import BytesIO

from app.utils.image_utils import uploadedfile_to_cv2_image
import cv2
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/detect_pixelation/", response_model=PixelationDetectionResponse, summary="Analyze image pixelation", description="Upload an image and check if it's pixelated based on a given threshold.")
async def analyze_pixelation(
    image: UploadFile = File(None, description="The image file to analyze"),
    image_url: str = Form(None, description="S3 URL to image file"),
    project_id: str = Form(..., description="Project ID for the image"),
    threshold: float = Form(0.05, description="Pixelation threshold (0-1). Lower values produce better results, but may also return false positives."),
    tolerance: float = Form(0.3, description="Tolerance for pixelation detection (0-1)"),
    s3_service: S3Service = Depends(get_s3_service),
    pixelation_detection_service: PixelationDetectionService = Depends(get_pixelation_detection_service)
):
    """
    Analyze if an uploaded image is pixelated based on given thresholds.

    Parameters:
    - **image**: The image file to be analyzed. Must be PNG, JPG, JPEG, GIF, BMP, or WebP.
    - **threshold**: Pixelation threshold. Scores above this value are considered pixelated. Default is 0.05.
    - **tolerance**: Tolerance for pixelation detection. Default is 0.3.

    Returns:
    A JSON object containing:
    - **is_pixelated** (bool): Indicates if the image is considered pixelated.
    - **pixelation_score** (float): The pixelation ratio of the image.

    Raises:
    - **HTTPException 400**: If the file type is invalid or there's an error processing the image.
    Analyze if an uploaded image is pixelated based on given thresholds.

    Parameters:
    - **image**: The image file to be analyzed. Must be PNG, JPG, JPEG, GIF, BMP, or WebP.
    - **threshold**: Pixelation threshold. Scores above this value are considered pixelated. Default is 0.05.
    - **tolerance**: Tolerance for pixelation detection. Default is 0.3.

    Returns:
    A JSON object containing:
    - **is_pixelated** (bool): Indicates if the image is considered pixelated.
    - **pixelation_score** (float): The pixelation ratio of the image.

    Raises:
    - **HTTPException 400**: If the file type is invalid or there's an error processing the image.
    """
    try:
        # Validate inputs
        if image is None and image_url is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'image' or 'image_url' must be provided.",
            )
        if image is not None and image_url is not None:
            raise HTTPException(
                status_code=400,
                detail="Provide only one of 'image' or 'image_url', not both.",
            )

        # Load image from file or S3 URL
        if image is not None:
            image_file = BytesIO(await image.read())
            logger.info(f"Processing uploaded image for project {project_id}")
        else:
            logger.info(f"Retrieving image from S3 URL for project {project_id}")
            # Get image content from S3 URL
            image_content = s3_service.get_image_from_url(image_url)
            image_file = BytesIO(image_content)

        cv2_image = uploadedfile_to_cv2_image(image_file)

        # Analyze the image for pixelation
        is_pixelated, pixelation_score = pixelation_detection_service.is_pixelated(cv2_image, threshold=threshold, tolerance=tolerance)

        return PixelationDetectionResponse(is_pixelated=is_pixelated, pixelation_score=pixelation_score)
    
    except ValueError as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        return PixelationDetectionResponse(is_pixelated=is_pixelated, pixelation_score=pixelation_score)
    
    except ValueError as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
