import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.models.led_output import ImageHeightAndWidth, ImageRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/grouping_request/", response_model=ImageHeightAndWidth)
async def grouping_request(request: ImageRequest):
    try:
        data = request.image_dictionary
        scale_ratio = request.scale_ratio
        combined_image = None

        for key, image in data.items():
            image_array = np.where(np.array(image) == 0, 1, 0)

            if combined_image is None:
                combined_image = image_array
            else:
                combined_image = np.bitwise_xor(combined_image, image_array)
        rows, cols = np.where(combined_image == 1)

        max_height = 0
        max_width = 0

        if rows.size > 0:
            max_height = rows.max() - rows.min() + 1
        if cols.size > 0:
            max_width = cols.max() - cols.min() + 1

        return ImageHeightAndWidth(height=int(max_height * scale_ratio), width=int(max_width * scale_ratio))
    except Exception as e:
        logger.error(f"Error Grouping Request: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))