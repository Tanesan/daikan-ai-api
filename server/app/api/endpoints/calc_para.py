import base64
import logging

import cv2
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from app.services.calc_thin import calc_thin
from app.utils.image_utils import read_image
import boto3
from typing import Dict
from datetime import datetime
from io import BytesIO
from app.models.led_output import PerImageParameter, ImageData, WholeImageParameter, WholeImageParameterWithLED

logger = logging.getLogger(__name__)
router = APIRouter()
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

def download_image_from_s3(url):
    try:
        object_name = f"{url.split('/')[-2]}/{url.split('/')[-1]}"
        # S3から画像をダウンロードしてバイトデータとして保持
        image_buffer = BytesIO()
        s3.download_fileobj(os.getenv('S3_BUCKET'), object_name, image_buffer)
        image_buffer.seek(0)
        image_np = np.frombuffer(image_buffer.read(), np.uint8)
        return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

@router.post("/calc_para/", response_model=WholeImageParameter)
async def calc_para_and_led(data: ImageData):
    """
    step5のセグメンテーションとLEDの計算を行うエンドポイント

    data: 画像データと画像の高さ(mm)
    """
    try:
        if data.url:
            image = download_image_from_s3(data.url)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to download image from S3")
        else:
            image_bytes = base64.b64decode(data.base64_image)
            with open("tmp.png", 'bw') as f3:
                f3.write(image_bytes)
            image = cv2.imread("tmp.png")

        image, binary = read_image(image)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        return calc_thin(image, binary, data.whole_hight_mm, data.url)

    except Exception as e:
        logger.error(f"Error calc Para: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/calc_para_with_led/", response_model=WholeImageParameterWithLED)
async def calc_para_with_led(data: ImageData):
    """
    セグメンテーションとLEDの数を同時に計算するエンドポイント

    data: 画像データと画像の高さ(mm)
    """
    try:
        if data.url:
            image = download_image_from_s3(data.url)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to download image from S3")
        else:
            image_bytes = base64.b64decode(data.base64_image)
            with open("tmp.png", 'bw') as f3:
                f3.write(image_bytes)
            image = cv2.imread("tmp.png")

        image, binary = read_image(image)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        return calc_thin(image, binary, data.whole_hight_mm, data.url, predict_led=True)

    except Exception as e:
        logger.error(f"Error calc Para with LED: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
