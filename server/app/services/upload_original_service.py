import logging
import os
import base64
#from PIL import Image
#import numpy as np
from io import BytesIO
from app.services.s3_service import S3Service
from datetime import datetime

logger = logging.getLogger(__name__)


def get_timestamp_with_milliseconds():
    # Get the current date and time with microseconds
    now = datetime.now()
    # Format it with milliseconds by slicing the microsecond part to 3 digits
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp_str


class UploadOriginalService:
    def __init__(self, s3_service: S3Service):
        self.s3_service = s3_service

    def process(self, file_content: bytes, project_id: str, filename: str) -> dict:
        img_byte_arr = BytesIO()
        try:
            # Extract file extension and create folder name
            file_extension = os.path.splitext(filename)[1]
            
            # Create the new filename
            new_filename = f"original{file_extension}"

            # Create BytesIO object from file_content
            img_byte_arr = BytesIO(file_content)

            timestamp = get_timestamp_with_milliseconds()
            new_project_id = f"{project_id}-{timestamp}"

            # Upload original to S3
            s3_url = self.s3_service.upload_image(img_byte_arr, new_project_id, new_filename)

            # Generate base64 encoding of the original
            base64_original = base64.b64encode(file_content).decode('utf-8')

            # Get pre-signed URL
            presigned_url = self.s3_service.get_s3_image_presigned_url(new_project_id, new_filename)

            logger.info(f"Original uploaded successfully to S3: {s3_url}")
            return {
                "s3_url": s3_url,
                "presigned_url": presigned_url,
                "image_base64": base64_original,
                "folder_name": new_project_id
            }
        except Exception as e:
            logger.error(f"Error uploading original to S3: {e}")
            raise