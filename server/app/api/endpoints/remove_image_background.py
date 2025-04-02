from io import BytesIO
import json
import logging
import base64
from fastapi import APIRouter, Form, HTTPException, UploadFile, Depends, File
from app.models.single_step_image_processing_model import ImageProcessingResponse, BackgroundRemovalRequest
from app.api.deps import get_manifest_service, get_s3_service, get_background_removal_service
from app.services.manifest_service import ManifestService
from app.services.s3_service import S3Service
from app.utils.image_utils import cv2_image_to_byte_array, uploadedfile_to_cv2_image
from app.services.background_removal_service import BackgroundRemovalService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/remove_background/", 
    response_model=ImageProcessingResponse, 
    summary="Remove background from an image", 
    description="""
    This endpoint processes the image to remove its background using the specified algorithm and parameters.
    
    
    Request Body Example for AUTO_1:
    json
    {
        "project_id": "project_12345",
        "algorithm": 1,
        "clusters": 4
    }
    

    Request Body Example for HINT_1:
    json
    {
        "project_id": "project_12345",
        "points": [[10, 20], [30, 40]],
        "tolerance": 0.5,
        "algorithm": 2
    }
    
    Request Body Example for AUTO_BY_COLOR_AT_POS:
    json
    {
        "project_id": "project_12345",
        "points": [[10, 20], [30, 40]],
        "tolerance": 0.5,
        "algorithm": 3
    }
    
    Request Body Example for AUTO_2:
    json
    {
        "project_id": "project_12345",
        "algorithm": 4
    }

    
    

    Request Body Example for HINT_1:
    json
    {
        "project_id": "project_12345",
        "points": [[10, 20], [30, 40]],
        "tolerance": 0.5,
        "algorithm": 2
    }
    
    Request Body Example for AUTO_BY_COLOR_AT_POS:
    json
    {
        "project_id": "project_12345",
        "points": [[10, 20], [30, 40]],
        "tolerance": 0.5,
        "algorithm": 3
    }
    
    Request Body Example for AUTO_2:
    json
    {
        "project_id": "project_12345",
        "algorithm": 4
    }

    
    Parameters:
    - image: The image file to process.
    - request_body: A JSON string containing the following parameters:
        - project_id: Identifier for the project.
        - points: Points for defining the region of interest in the image (optional for `HINT_1` algorithm).
        - tolerance: Tolerance level for the background removal algorithm (optional for `HINT_1` algorithm). A higher tolerance might remove more of the background.
                     When using  the tolerance must be a value between 0 and 1.
        - algorithm: Algorithm type for background removal. Valid values are:
            - 1: AUTO_1 - Automatic background removal using method 1.
            - 2: HINT_1 - Background removal using provided hint points and tolerance.
            - 3: AUTO_BY_COLOR_AT_POS - Automatic background removal using a list of points to sample the target colors.
            - 4: AUTO_2 - Automatic background removal using method 2, especialized for images with a gradient background.
        - clusters: number of clusters used when algorithm is AUTO_1. Default value is: 3
    
    Returns:
        A JSON response containing the message, URL of the processed image, a presigned URL for accessing the image, and a Base64 image.
    
    Raises:
        HTTPException: If an error occurs during processing, an HTTP 500 error is returned.
    """

)
async def remove_background(
    image: UploadFile = File(None, description="Image file to process (optional)"),
    image_url: str = Form(None, description="URL of the image in S3 (optional)"),
    request_body: str = Form(..., description="JSON string containing the request body"),
    background_removal_service: BackgroundRemovalService = Depends(get_background_removal_service),
    s3_service: S3Service = Depends(get_s3_service),
    manifest_service: ManifestService = Depends(get_manifest_service)
):
    try:
        # Parse the JSON string into a Pydantic model
        body = BackgroundRemovalRequest(**json.loads(request_body))
        
        # Convert algorithm to the appropriate enum type
        algorithm_enum = BackgroundRemovalService.Algorithm(body.algorithm)
        
        # Check if image or image_url is provided
        if image is not None:
            # Read the uploaded image file into a BytesIO object
            image_file = BytesIO(await image.read())
            logger.info(f"Processing uploaded image for project {body.project_id}")
        
        elif image_url is not None:
            # Retrieve the image from S3 using the URL
            logger.info(f"Retrieving image from S3 URL for project {body.project_id}")
            image_content = s3_service.get_image_from_url(image_url)
            image_file = BytesIO(image_content)
        else:
            raise HTTPException(status_code=400, detail="No image or image_url provided.")
        
        # Convert the image file to a cv2 image
        cv2_image = uploadedfile_to_cv2_image(image_file)
        
        # Process the image using the background removal service
        cv2_nobg_image = background_removal_service.process(
            cv2_image,
            algorithm_enum,
            points=body.points,
            tolerance=body.tolerance,
            clusters=body.clusters,
        )
        
        # Convert the processed image back to bytes
        nobg_image = cv2_image_to_byte_array(cv2_nobg_image)
        nobg_image_value = nobg_image.getvalue()
        nobg_base64 = base64.b64encode(nobg_image_value).decode('utf-8')
        
        # Upload the processed image to S3 and update the manifest
        s3_nobg_img_url = s3_service.upload_image(nobg_image, body.project_id, "segmented.png")
        manifest_service.add_image_to_manifest(s3_nobg_img_url, body.project_id, "segmented")
        
        # Get a presigned URL for the uploaded image
        pre_signed_url = s3_service.get_s3_image_presigned_url(body.project_id, "segmented.png")

        return {
            "message": "Image uploaded successfully",
            "url": s3_nobg_img_url,
            "presigned_url": pre_signed_url,
            "image_base64": nobg_base64
        }
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))