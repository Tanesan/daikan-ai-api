# image_edge_enhancement.py

from io import BytesIO
import logging
import cv2
from fastapi import HTTPException, UploadFile, Depends, File, Form, APIRouter
import numpy as np
from app.services.image_edge_enhancement_service import ImageEdgeEnhancementService, get_image_edge_enhancement_service
from app.services.s3_service import S3Service
from app.api.deps import get_s3_service
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
import base64
from enum import Enum


logger = logging.getLogger(__name__)
router = APIRouter()

class EnhancementMethod(str, Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    UNSHARP_MASK = "unsharp_mask"
    HIGH_PASS_FILTER = "high_pass_filter"
    SHARPEN = "sharpen"
    LAPLACIAN_FILTER = "laplacian_filter"
    EDGE_ENHANCE = "edge_enhance"
    SOBEL_FILTER = "sobel_filter"

@router.post("/enhance_edges/", response_model=ImageProcessingResponse, summary="Enhance edges in an image", description="Upload an image and enhance its edges using the specified algorithm.")
async def enhance_edges(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="S3 URL to image file"),
    project_id: str = Form(..., description="Project ID for the image"),
    kernel_size: int = Form(5, description="Kernel size for edge enhancement algorithms. Default value: 5."),
    alpha: float = Form(1.5, description="Alpha parameter for unsharp mask algorithm. Default value: 1.5."),
    reps: int = Form(1, description="Number of times to run the algorithm"),
    method: EnhancementMethod = Form(EnhancementMethod.SHARPEN, description="Edge enhancement method. Options: gaussian_blur, unsharp_mask, high_pass_filter, sharpen, laplacian_filter, edge_enhance, sobel_filter. Default: edge_enhance."),
    image_edge_enhancement_service: ImageEdgeEnhancementService = Depends(get_image_edge_enhancement_service),
    s3_service: S3Service = Depends(get_s3_service),
):
    """
    Enhance edges in an uploaded image using the specified algorithm.

    Parameters:
    - **image**: The image file to be processed. Must be PNG, JPG, JPEG, GIF, BMP, or WebP.
    - **kernel_size**: Kernel size for edge enhancement algorithms. Default is 5.
    - **alpha**: Alpha parameter for unsharp mask algorithm. Default is 1.5.
    - **method**: Edge enhancement method. Options:
        - GUSSIAN_BLUR
        - UNSHARP_MASK
        - HIGH_PASS_FILTER
        - SHARPEN
        - LAPLACIAN_FILTER
        - EDGE_ENHANCE
        - SOBEL_FILTER
      Default is EDGE_ENHANCE.

    Returns:
    The enhanced image in PNG format as a binary response.

    Raises:
    - **HTTPException 400**: If the file type is invalid or there's an error processing the image.
    - **HTTPException 500**: If an unexpected error occurs during processing.
    """
    if reps < 1:
        raise HTTPException(status_code=400, detail="The parameter reps cannot be negative.")

    try:
        # Check if the image is uploaded or a URL is provided
        if image is not None:
            image_file = BytesIO(await image.read())
            logger.info(f"Processing uploaded image for project {project_id}")
        elif image_url is not None:
            logger.info(f"Retrieving image from S3 URL for project {project_id}")
            image_content = s3_service.get_image_from_url(image_url)
            image_file = BytesIO(image_content)
        else:
            raise HTTPException(status_code=400, detail="No image or image_url provided.")
        contents = image_file.read()
        cv2_image = uploadedfile_to_cv2_image(image_file)

        for i in range(0, reps):
            cv2_image = image_edge_enhancement_service.process(cv2_image, kernel_size, alpha, method.value)

        scaled_image = cv2_image_to_byte_array(cv2_image)
        scaled_image_value = scaled_image.getvalue()
        scaled_base64 = base64.b64encode(scaled_image_value).decode('utf-8')

        # Save the image to S3
        s3_mask_img_url = s3_service.upload_image(scaled_image, project_id, "enhance_edges.png")
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "enhance_edges.png")


        return {
            "image_base64": scaled_base64,
            "message": "Image processed successfully",
            "url": s3_mask_img_url,
            "presigned_url": pre_signed_url,
        }
    except ValueError as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image processing")