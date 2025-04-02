from app.services.s3_service import S3Service
from app.services.background_removal_service import BackgroundRemovalService
from app.services.cropping_service import CroppingService
from app.services.super_resolution_service import SuperResolutionService
from app.services.mask_service import MaskService
from app.services.manifest_service import ManifestService
from app.services.background_removal_service import BackgroundRemovalService
from app.services.scale_service import ScaleService
from app.services.smoothing_image_service import SmoothingImageService
from app.services.super_resolution_service_mt import SuperResolutionService as SuperResolutionServiceMT
from app.services.upload_original_service import UploadOriginalService
from app.services.paint_service import PaintingService

def get_s3_service():
    return S3Service()

def get_background_removal_service():
    return BackgroundRemovalService()

def get_super_resolution_service():
    return SuperResolutionService()

def get_super_resolution_service_mt():
    return SuperResolutionServiceMT()

def get_mask_service():
    return MaskService()

def get_manifest_service():
    return ManifestService()

def get_smoothing_image_service():
    return SmoothingImageService()

def get_cropping_service():
    return CroppingService()

def get_scale_service():
    return ScaleService()

def get_upload_original_service():
    s3_service = get_s3_service()
    return UploadOriginalService(s3_service)

def get_paint_service():
    return PaintingService()