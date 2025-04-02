import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Dict

from app.models.led_output import PerImageParameter, ImageData, WholeImageParameter, ImageRequest, LedParameter, \
    WholeImagePredictParameter
from app.services.predict_led import predict_led
from app.models.led_output import LuminousModel

logger = logging.getLogger(__name__)
router = APIRouter()


def process_number(num):
    if num < 0:
        num = -num
    return round(num)


@router.post("/predict_led/", response_model=Dict[int, LedParameter])
async def calc_para_and_led(request: WholeImagePredictParameter):
    try:
        datas = request.perimages
        result = []
        for _, data in datas.items():
            predicted_led = predict_led(data.area, data.peri, data.distance, data.skeleton_length,
                                        request.scale_ratio, data.intersection_count3, data.intersection_count4,
                                        data.intersection_count5,
                                        data.intersection_count6, data.endpoints_count,
                                        LuminousModel(data.luminous_model))
            if LuminousModel(data.luminous_model) == LuminousModel.HYOMEN:
                result.append(LedParameter(
                    **{"nomal": process_number(predicted_led), "packed": process_number(predicted_led * 1.56)}))
            else:
                result.append(LedParameter(
                    **{"nomal": process_number(predicted_led), "packed": process_number(predicted_led * 1.25)}))
        result_data = {i + 1: v for i, v in enumerate(result)}
        return result_data

    except Exception as e:
        logger.error(f"Error predict LED: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
