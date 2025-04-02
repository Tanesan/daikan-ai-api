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
    get_scale_service,
)
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from app.services.scale_service import ScaleService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/scale_image/",
    response_model=ImageProcessingResponse,
    summary="Scale an image",
    description="Upload an image or provide an S3 URL to scale it using specified scale factors.",
)
async def scale_image(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="S3 URL to image file"),
    scale_factor_x: float = Form(..., description="Scale factor along x-axis"),
    scale_factor_y: float = Form(..., description="Scale factor along y-axis"),
    project_id: str = Form(..., description="Project ID for the image"),
    scale_service: ScaleService = Depends(get_scale_service),
    s3_service: S3Service = Depends(get_s3_service),
    manifest_service: ManifestService = Depends(get_manifest_service),
):
    """
    Scale an image using specified scale factors and save the processed image to S3.
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
        if scale_factor_x <= 0 or scale_factor_y <= 0:
            raise HTTPException(
                status_code=400,
                detail="Scale factors must be positive numbers.",
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

        # Perform scaling
        scaled_image_cv2 = scale_service.process(
            cv2_image, scale_factor_x, scale_factor_y
        )

        # Convert scaled image to byte array
        scaled_image_io = cv2_image_to_byte_array(scaled_image_cv2)
        scaled_image_value = scaled_image_io.getvalue()

        # Encode image to base64
        scaled_base64 = base64.b64encode(scaled_image_value).decode("utf-8")

        # Upload scaled image to S3
        s3_scaled_img_url = s3_service.upload_image(
            scaled_image_io, project_id, "scaled_nonsr.png"
        )
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "scaled_nonsr.png")

        return {
            "message": "Image scaled and uploaded successfully",
            "url": s3_scaled_img_url,
            "presigned_url": pre_signed_url,
            "image_base64": scaled_base64,
        }
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error scaling image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal server error.")
