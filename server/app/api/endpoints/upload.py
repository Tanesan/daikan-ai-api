from io import BytesIO
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.models.upload_models import UploadResponse
from app.services.s3_service import S3Service
from app.services.background_removal_service import BackgroundRemovalService
from app.services.super_resolution_service import SuperResolutionService
from app.services.mask_service import MaskService
from app.services.manifest_service import ManifestService
from app.utils.image_utils import uploadedfile_to_cv2_image, cv2_image_to_byte_array
from app.api.deps import get_s3_service, get_background_removal_service, get_super_resolution_service, get_mask_service, get_manifest_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload/", response_model=UploadResponse)
async def upload_image_to_s3(image: UploadFile, name: str,
                             s3_service: S3Service = Depends(get_s3_service),
                             background_removal_service: BackgroundRemovalService = Depends(get_background_removal_service),
                             super_resolution_service: SuperResolutionService = Depends(get_super_resolution_service),
                             mask_service: MaskService = Depends(get_mask_service),
                             manifest_service: ManifestService = Depends(get_manifest_service)):
    try:
        image_file = image_file = BytesIO(await image.read())
        logger.info(f"Uploading image {name}")
        s3_img_url = s3_service.upload_image(image.file, name, "original")
        manifest_service.add_image_to_manifest(s3_img_url, name, "original")

        logger.info(f"Processing image {name}")
        cv2_image = uploadedfile_to_cv2_image(image_file)
        cv2_scaled_image = super_resolution_service.super_resolution_image(cv2_image)
        scaled_image = cv2_image_to_byte_array(cv2_scaled_image)
        s3_scaled_img_url = s3_service.upload_image(scaled_image, name, "scaled")

        logger.info(f"Segmenting image {name}")
        cv2_segmented_image = background_removal_service.process(cv2_scaled_image, BackgroundRemovalService.Algorithm.AUTO_1)
        segmented_image = cv2_image_to_byte_array(cv2_segmented_image)
        s3_segmented_img_url = s3_service.upload_image(segmented_image, name, "segmented")

        logger.info(f"Masking image {name}")
        cv2_mask_image = mask_service.process(cv2_segmented_image, inverted=True)
        mask_image = cv2_image_to_byte_array(cv2_mask_image)
        s3_mask_img_url = s3_service.upload_image(mask_image, name, "mask")

        return {"message": "Image uploaded successfully",
                "url": s3_img_url,
                "scaled_img_url": s3_scaled_img_url,
                "segmented_url": s3_segmented_img_url,
                "mask_url": s3_mask_img_url}
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        logger.exception(e)
        
        raise HTTPException(status_code=500, detail=str(e))

