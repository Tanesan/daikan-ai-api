from io import BytesIO
import logging
import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import (
    get_manifest_service,
    get_s3_service,
    get_cropping_service,
)
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from app.services.cropping_service import CroppingService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/crop_image/",
    response_model=ImageProcessingResponse,
    summary="Crop an image",
    description="Upload an image or provide an S3 URL to crop it using specified corner points.",
)
async def crop_image(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="S3 URL to image file"),
    top_left_x: float = Form(..., description="Top-left x-coordinate"),
    top_left_y: float = Form(..., description="Top-left y-coordinate"),
    top_right_x: float = Form(..., description="Top-right x-coordinate"),
    top_right_y: float = Form(..., description="Top-right y-coordinate"),
    bottom_right_x: float = Form(..., description="Bottom-right x-coordinate"),
    bottom_right_y: float = Form(..., description="Bottom-right y-coordinate"),
    bottom_left_x: float = Form(..., description="Bottom-left x-coordinate"),
    bottom_left_y: float = Form(..., description="Bottom-left y-coordinate"),
    project_id: str = Form(..., description="Project ID for the image"),
    cropping_service: CroppingService = Depends(get_cropping_service),
    s3_service: S3Service = Depends(get_s3_service),
    manifest_service: ManifestService = Depends(get_manifest_service),
):
    """
    Crop an image using specified corner points and save the processed image to S3.
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
            cv2_image = uploadedfile_to_cv2_image(image_file)
        else:
            logger.info(f"Retrieving image from S3 URL for project {project_id}")
            # Get image content from S3 URL
            image_content = s3_service.get_image_from_url(image_url)
            image_file = BytesIO(image_content)
            cv2_image = uploadedfile_to_cv2_image(image_file)

        # Define corner points
        top_left = (top_left_x, top_left_y)
        top_right = (top_right_x, top_right_y)
        bottom_right = (bottom_right_x, bottom_right_y)
        bottom_left = (bottom_left_x, bottom_left_y)

        # Log the coordinates
        logger.info(f"Top-left coordinates: {top_left}")
        logger.info(f"Top-right coordinates: {top_right}")
        logger.info(f"Bottom-right coordinates: {bottom_right}")
        logger.info(f"Bottom-left coordinates: {bottom_left}")

        # Perform cropping
        cropped_image_cv2 = cropping_service.process(
            cv2_image, top_left, top_right, bottom_right, bottom_left
        )

        # Convert cropped image to byte array
        cropped_image_io = cv2_image_to_byte_array(cropped_image_cv2)
        cropped_image_value = cropped_image_io.getvalue()

        # Encode image to base64
        cropped_base64 = base64.b64encode(cropped_image_value).decode("utf-8")

        # Upload cropped image to S3
        s3_cropped_img_url = s3_service.upload_image(
            cropped_image_io, project_id, "cropped.png"
        )
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "cropped.png")

        return {
            "message": "Image cropped and uploaded successfully",
            "url": s3_cropped_img_url,
            "presigned_url": pre_signed_url,
            "image_base64": cropped_base64,
        }
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal server error.")
