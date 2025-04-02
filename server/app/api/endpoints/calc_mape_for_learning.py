import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Dict

from app.services.predict_led import predict_led
from app.models.led_output import PerImageParameter, WholeImageParameter, ImageRequest, LedParameter, \
    WholeImageParameterWithLED, Announcement

from app.models.led_output import LuminousModel


logger = logging.getLogger(__name__)
router = APIRouter()

def process_number(num):
    if num < 0:
        num = -num
    if num == 0:
        num = 1
    return round(num)

def get_led_list(whole_image_param: WholeImageParameterWithLED):
    predicted_led = [whole_image_param.perimages[key].led for key in sorted(whole_image_param.perimages.keys())]
    return np.array(predicted_led)

@router.post("/calc_mape_for_learning/", response_model=Dict[int, Announcement])
async def calc_mape_for_learning(request: WholeImageParameterWithLED):
    try:
        datas = request.perimages
        predicted_led_list = []
        for _, data in datas.items():
            predicted_led = predict_led(data.area, data.peri, data.distance, data.skeleton_length,
                                        request.scale_ratio, data.intersection_count3, data.intersection_count4,
                                        data.intersection_count5,
                                        data.intersection_count6, data.endpoints_count,
                                        LuminousModel(data.luminous_model))
            predicted_led_list.append(process_number(predicted_led))
        actual_led = get_led_list(request)
        if np.any(actual_led == 0):
            raise ValueError("actual の値に 0 が含まれています。MAPE は 0 の実測値では計算できません。")

        # MAPEの計算
        mape = (np.abs(actual_led - predicted_led_list) / actual_led) * 100
        mape = mape.astype(int)

        result = {}
        result[0] = {
                    "announcement": ""
                }
        for index, value in enumerate(mape):
            if value > 30:
                result[index + 1] = {
                    "announcement": "乖離有り"
                }
            else:
                result[index + 1] = {
                    "announcement": "乖離無し"
                }

        return result

    except Exception as e:
        logger.error(f"Error Calcurate MAPE: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
