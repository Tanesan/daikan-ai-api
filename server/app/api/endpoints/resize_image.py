import base64
from fastapi import File, UploadFile, HTTPException, APIRouter, Depends, Form
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import logging
from app.services.resize_image_service import resize_image
from app.api.deps import get_s3_service
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/resize_image/", response_model=dict, summary="Resize an image", description="Upload an image and resize it to ensure the larger dimension is exactly 1000 pixels using the LANCZOS resizing algorithm.")
async def resize_image_endpoint(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="S3 URL to image file"),
    project_id: str = Form(..., description="Project ID for the image"),
    s3_service: S3Service = Depends(get_s3_service)
):
    """
    Receives an image file, resizes it so that the larger dimension is exactly 1000 pixels,
    and returns the resized image in Base64 format.
    
    Args:
        image (UploadFile): The image file to process.
    Returns:
        dict: A dictionary containing the Base64 encoded resized image.
    
    Raises:
        HTTPException: If there is an error processing the image.
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

        # cv2_image = uploadedfile_to_cv2_image(image_file)

        image = Image.open(image_file)
        
        # Resize the image regardless of its dimensions
        resized_image = resize_image(image, 1000)
        
        # Convert the resized image to Base64
        buffer = BytesIO()
        resized_image_format = resized_image.format if resized_image.format else 'PNG'
        resized_image.save(buffer, format=resized_image_format)
        resized_image_value = buffer.getvalue()
        buffer.seek(0)

        # Upload cropped image to S3
        s3_resize_img_url = s3_service.upload_image(
            buffer, project_id, "resized.png"
        )
        pre_signed_url = s3_service.get_s3_image_presigned_url(project_id, "resized.png")

        
        image_base64 = base64.b64encode(resized_image_value).decode('utf-8')
        return {
            "message": "Image resized and uploaded successfully",
            "url": s3_resize_img_url,
            "presigned_url": pre_signed_url,
            "image_base64": image_base64,
        }
    
    except UnidentifiedImageError as e:
        logger.error(f"Failed to identify the image format: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid image.")
    except Exception as e:
        logger.error(f"Unexpected error resizing image: {e}")
        logger.error(f"Unexpected error resizing image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image processing.")

        raise HTTPException(status_code=500, detail="An unexpected error occurred during image processing.")
