from pydantic import BaseModel
from typing import List, Tuple, Optional
from app.services.background_removal_service import BackgroundRemovalService

class ImageProcessingResponse(BaseModel):
    message: str
    url: str
    presigned_url: str
    image_base64: str

class SimpleBase64Response(BaseModel):
    message: str
    image_base64: str

class BackgroundRemovalRequest(BaseModel):
    project_id: str
    points: Optional[List[Tuple[int, int]]] = None
    tolerance: Optional[float] = None
    algorithm: int = BackgroundRemovalService.Algorithm.AUTO_1
    clusters: int = 3
    color: Optional[List[int]] = None