import base64
from io import BytesIO
import logging
from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import get_manifest_service, get_s3_service, get_smoothing_image_service
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from app.services.smoothing_image_service import SmoothingImageService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/smooth_mask/", response_model=ImageProcessingResponse, summary="Smooth a mask", description="Upload a mask and smooth it using the specified algorithm.")
async def smooth_mask(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="S3 URL to image file"),
    project_id: str = Form(..., description="Project ID for the image"),
    contour: int = Form(5, description="Contour value for the smoothing algorithm. Default value: 5. A number bigger than 0 for making the contour thicker and a number smaller than 0 for making the contour thinner."),
    smooth_mask_service: SmoothingImageService = Depends(get_smoothing_image_service),
    s3_service: S3Service = Depends(get_s3_service),
    manifest_service: ManifestService = Depends(get_manifest_service)):
    """
    Smooth a mask using a specified algorithm and save the processed image to S3.
    """
    try:
        if contour == 0:
            raise HTTPException(status_code=400, detail="Contour value must be different from 0")
        algo = SmoothingImageService.Algorithm.ERODE if contour >= 1 else SmoothingImageService.Algorithm.DILATE
        contour = abs(contour)

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
        cv2_smoothed_image = smooth_mask_service.process(cv2_image, algo, contour)
        smoothed_image = cv2_image_to_byte_array(cv2_smoothed_image)
        smooth_mask_image_value = smoothed_image.getvalue()
        smooth_mask_base64 = base64.b64encode(smooth_mask_image_value).decode('utf-8')
        
        s3_smoothed_img_url = s3_service.upload_image(smoothed_image, project_id, "smoothed.png")
        manifest_service.add_image_to_manifest(s3_smoothed_img_url, project_id, "smoothed")
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "smoothed.png")

        return {"message": "Image uploaded successfully",
                "url": s3_smoothed_img_url,
                "presigned_url": pre_signed_url,
                "image_base64": smooth_mask_base64}
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))