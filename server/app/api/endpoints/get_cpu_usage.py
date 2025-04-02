import logging

import psutil
from fastapi import APIRouter, HTTPException
from app.models.led_output import CPUUsageResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/get_cpu_usage/", response_model=CPUUsageResponse)
async def get_cpu_usage():
    try:
        """
            Returns the CPU usage as a percentage.
        """
        usage = psutil.cpu_percent(interval=1)

        if usage <= 30:
            announcement = "空いている"
        elif 30 < usage <= 70:
            announcement = "普通"
        else:
            announcement = "混んでいる"

        return CPUUsageResponse(cpu_usage=usage, announcement=announcement)

    except Exception as e:
        logger.error(f"Error Calcurate MAPE: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
