from io import BytesIO
import logging
import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import get_manifest_service, get_mask_service, get_s3_service, get_background_removal_service
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from app.services.background_removal_service import BackgroundRemovalService
from app.services.image_edge_enhancement_service import ImageEdgeEnhancementService, get_image_edge_enhancement_service
from app.services.mask_service import MaskService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/mask_image/", response_model=ImageProcessingResponse, summary="Mask an image", description="Upload an image or provide an S3 URL to mask the image using the specified algorithm.")
async def mask_image(
    image: UploadFile = File(None, description="Image file to process (optional)"),
    image_url: str = Form(None, description="URL of the image in S3 (optional)"),
    project_id: str = Form(..., description="Project ID for the image"),
    inverted: bool = Form(False, description="Invert the mask"),
    mask_service: MaskService = Depends(get_mask_service),
    image_edge_enhancement_service: ImageEdgeEnhancementService = Depends(get_image_edge_enhancement_service),
    s3_service: S3Service = Depends(get_s3_service),
    manifest_service: ManifestService = Depends(get_manifest_service)):
    """
    Mask an uploaded image or an image from S3 using a specified algorithm and save the processed image to S3.
    """
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
        
        # Convert the image to cv2 format
        cv2_image = uploadedfile_to_cv2_image(image_file)
        # The 'inverted' flag is used to correct the mask orientation.
        # In OpenCV, masks are typically represented with a white foreground on a black background.
        # However, we handle the mask with a black foreground on a white background.
        # The 'inverted' flag is negated here to adjust for this difference.
        cv2_mask_image = mask_service.process(cv2_image, inverted=not inverted)

        enhanced_image_cv2 = image_edge_enhancement_service.process(cv2_mask_image, 5, 1.5, "edge_enhance")
        enhanced_image = cv2_image_to_byte_array(enhanced_image_cv2)
        mask_image_value = enhanced_image.getvalue()
        mask_base64 = base64.b64encode(mask_image_value).decode('utf-8')
        
        s3_mask_img_url = s3_service.upload_image(enhanced_image, project_id, "mask.png")
        manifest_service.add_image_to_manifest(s3_mask_img_url, project_id, "mask")
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "mask.png")

        return {
            "message": "Image uploaded successfully",
            "url": s3_mask_img_url,
            "presigned_url": pre_signed_url,
            "image_base64": mask_base64
        }
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))