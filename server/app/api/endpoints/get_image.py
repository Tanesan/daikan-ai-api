import logging
from fastapi import APIRouter, HTTPException, File, Depends

from app.services import s3_service
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import get_s3_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/get_image/", response_model=ImageProcessingResponse)
def get_image(project_id: str, image_name: str,
              s3_service: s3_service.S3Service = Depends(get_s3_service),):
    """
    Get an image from S3 and return it as a byte array.
    
    project_id: The ID of the project to which the image belongs.
    image_name: The name of the image. It must include the extension. Otherwise, an exception will be raised.
    """
    try:
        # Get the image from S3
        s3_img_url = s3_service.get_s3_image_presigned_url(project_id, image_name)
        return {"message": "Image retrieved successfully",
                "url": s3_img_url}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"Error retrieving image: {e}")
        logger.exception(e)

        raise HTTPException(status_code=500, detail=str(e))