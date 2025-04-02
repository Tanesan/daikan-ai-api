from fastapi import APIRouter, HTTPException, UploadFile, Depends, File, Form
from app.models.single_step_image_processing_model import ImageProcessingResponse
from app.api.deps import get_upload_original_service
from app.services.upload_original_service import UploadOriginalService
#from app.utils.image_utils import uploadedfile_to_cv2_image

router = APIRouter()

@router.post(
    "/upload_original/",
    response_model=dict,
    summary="Upload an original image to S3",
    description="Upload an original image file, create a folder in S3 with the project ID, and store the image as 'original' in that folder.",
)
async def upload_original(
    original: UploadFile = File(..., description="Original image file to upload"),
    project_id: str = Form(..., description="Project ID for the original"),
    upload_service: UploadOriginalService = Depends(get_upload_original_service),
):
    try:
        # Read the uploaded file
        file_content = await original.read()
        
        # Process and upload the original
        upload_result = upload_service.process(file_content, project_id, original.filename)


        return {
            "message": "Original uploaded successfully",
            "url": upload_result["s3_url"],
            "presigned_url": upload_result["presigned_url"],
            "image_base64": upload_result["image_base64"],
            "folder_name": upload_result["folder_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))