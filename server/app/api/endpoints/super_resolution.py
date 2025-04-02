import base64
from io import BytesIO
import logging
import os
from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import get_manifest_service, get_s3_service, get_super_resolution_service, get_super_resolution_service_mt
from app.api.deps import get_manifest_service, get_s3_service, get_super_resolution_service, get_super_resolution_service_mt
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.services.super_resolution_service import SuperResolutionService
from app.services.super_resolution_service import SuperResolutionService
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from .utils import super_resolution_estimate_execution_time



logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/super_resolution_estimation/")
async def super_resolution_estimation(width: int, height: int):
    # get the number of CPUs available
    num_cpu = os.cpu_count()
    estimated_execution_time = super_resolution_estimate_execution_time(num_cpu, width, height)
    return {"estimated_execution_time": estimated_execution_time}

@router.post("/super_resolution/", response_model=ImageProcessingResponse)
async def super_resolution(
    image: UploadFile = File(None, description="Image file to process"),                       
    image_url: str = Form(None, description="S3 URL to image file"),
    project_id: str = Form(..., description="Project ID for the image"),
    scale_factor: int = 2,
    parallel: bool = False,
    super_resolution_service: SuperResolutionService = Depends(get_super_resolution_service),
    super_resolution_service_mt: SuperResolutionService = Depends(get_super_resolution_service_mt),
   s3_service: S3Service = Depends(get_s3_service),
   manifest_service: ManifestService = Depends(get_manifest_service)):
    try:
        if scale_factor < 2:
            raise HTTPException(status_code=400, detail="Scale factor must be greater than or equal to 2")
        elif scale_factor > 8:
            raise HTTPException(status_code=400, detail="Scale factor must be less than or equal to 8")
        
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
        if parallel:
            logger.info("Running parallel super_resolution")
            cv2_scaled_image = super_resolution_service_mt.super_resolution_image(cv2_image, scale_factor)
        else:
            logger.info("Running sequential super_resolution")
            cv2_scaled_image = super_resolution_service.super_resolution_image(cv2_image, scale_factor)
        scaled_image = cv2_image_to_byte_array(cv2_scaled_image)
        scaled_image_value = scaled_image.getvalue()
        scaled_base64 = base64.b64encode(scaled_image_value).decode('utf-8')
        s3_scaled_img_url = s3_service.upload_image(scaled_image, project_id, "scaled.png")
        manifest_service.add_image_to_manifest(s3_scaled_img_url, project_id, "scaled")
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "scaled.png")

        logger.info("Finished super_resolution successfully, about to return a response")
        return {"message": "Image uploaded successfully",
                "url": s3_scaled_img_url,
                "presigned_url": pre_signed_url,
                "image_base64": scaled_base64}
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        logger.exception(e)

        raise HTTPException(status_code=500, detail=str(e))