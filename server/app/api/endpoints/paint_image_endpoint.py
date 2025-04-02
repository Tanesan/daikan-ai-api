import base64
from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional
import logging
import cv2
import numpy as np
from io import BytesIO
import json

# Assuming these dependencies are defined elsewhere in your project
from app.api.deps import get_s3_service, get_paint_service
from app.models.paint_image_models import Layer
from app.services.s3_service import S3Service
from app.services.paint_service import PaintingService
from app.utils.image_utils import cv2_image_to_byte_array

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/paint_image/",
    summary="Process image with modifications",
    description="""Processes the image with the given modifications and uploads the result to S3.
        Examples of input for modifications:

    Example 1: 
    Painting a square on the top left corner
    {
        "modifications":[
        {
            "x":0,
            "y":0,
            "isEraser":false, 
            "color":"#FFFFFF",
            "shape":"square",
          "size":10
        }
        ]
    }

    Example 2:
    Painting and deleting a square on the top-left corner to verify no modification on the original image
    {
        "modifications":[
        {
            "x":0,
            "y":0,
            "isEraser":false, 
            "color":"#FFFFFF",
            "shape":"square",
          "size":10
        },
        {
            "x":0,
            "y":0,
            "isEraser":true, 
            "color":"",
            "shape":"square",
          "size":10
        }
        ]
    }
    
    """,
)
async def process_image(
    image: UploadFile = File(None, description="Image file to process"),
    image_url: str = Form(None, description="URL to image file"),
    modifications: str = Form(..., description="JSON string of modifications"),
    project_id: str = Form(..., description="Project ID for the image"),
    s3_service: S3Service = Depends(get_s3_service),
    painting_service: PaintingService = Depends(get_paint_service),
):
    """
    Process the image with the given modifications and upload the result to S3.
    """
    try:
        # Validate inputs
        if (image is None and not image_url) or (image is not None and image_url):
            raise HTTPException(
                status_code=400,
                detail="Provide either 'image' or 'image_url', but not both.",
            )

        # Load image
        if image is not None:
            image_bytes = await image.read()
            image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            # Load image from URL
            image_content = s3_service.get_image_from_url(image_url)
            image_cv2 = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
        if image_cv2 is None:
            raise HTTPException(status_code=400, detail="Invalid image file or URL")

        # Parse modifications
        try:
            modifications_data = json.loads(modifications)
            layer = Layer(**modifications_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in 'modifications'")
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Process modifications using the service
        image_processed = painting_service.process(image_cv2, layer.modifications)

        # Encode image
        _, encoded_image = cv2.imencode('.png', image_processed)

        # Convert to BytesIO
        image_io = BytesIO(encoded_image.tobytes())

        # Upload image to S3
        s3_image_url = s3_service.upload_image(image_io, project_id, 'painted.png')

        # Get presigned URL
        presigned_image_url = s3_service.get_s3_image_presigned_url(project_id, 'painted.png')

        # Get the bas64 version of the image
        painted_image_io = cv2_image_to_byte_array(image_processed)
        painted_image_value = painted_image_io.getvalue()
        painted_base64 = base64.b64encode(painted_image_value).decode("utf-8")

        return {
            "message": "Image processed and uploaded successfully",
            "url": s3_image_url,
            "presigned_url": presigned_image_url,
            "image_base64": painted_base64
        }
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal server error.")
