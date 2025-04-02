from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    url: str
    scaled_img_url: str
    segmented_url: str
    mask_url: str