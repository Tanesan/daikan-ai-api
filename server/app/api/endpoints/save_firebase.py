import base64
import logging
import os
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from google.cloud import firestore
import pandas as pd
from app.models.led_output import WholeImageParameterWithLED
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
router = APIRouter()

credentials_info = {
        "type": os.getenv("GOOGLE_TYPE"),
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),  # 改行を適切に扱う
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
        "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
    }

credentials = service_account.Credentials.from_service_account_info(credentials_info)
db = firestore.Client(credentials=credentials)


def get_next_document_index(collection_name: str) -> int:
    """
    指定コレクション内で 'document_xxx' 形式のIDのうち、
    最大の xxx を探して +1 を返す。
    該当IDが存在しなければ 0 を返す。
    """
    collection_ref = db.collection(collection_name)
    docs = collection_ref.stream()

    max_index = -1
    for doc in docs:
        doc_id = doc.id
        if doc_id.startswith("document_"):
            try:
                # 「document_」以降を整数として取得
                current_index = int(doc_id.split("_")[1])
                if current_index > max_index:
                    max_index = current_index
            except ValueError:
                # document_ の後が数値でなければ無視
                pass

    return max_index + 1


def save_to_firestore(df: pd.DataFrame, collection_name: str):
    """
    コレクション内の現在の最大連番を取得し、
    連番を振りながらデータを Firestore に保存する。
    """
    start_index = get_next_document_index(collection_name)
    for i, row in df.iterrows():
        doc_index = start_index + i
        doc_ref = db.collection(collection_name).document(f'document_{doc_index}')
        doc_ref.set(row.to_dict())


def extract_relevant_data(whole_image_param_with_led: WholeImageParameterWithLED) -> pd.DataFrame:
    data_list = []
    for key, per_image_param in whole_image_param_with_led.perimages.items():
        filtered_data = {
            "area": per_image_param.area,
            "peri": per_image_param.peri,
            "distance": per_image_param.distance,
            "skeleton_length": per_image_param.skeleton_length,
            "intersection_count3": per_image_param.intersection_count3,
            "intersection_count4": per_image_param.intersection_count4,
            "intersection_count5": per_image_param.intersection_count5,
            "intersection_count6": per_image_param.intersection_count6,
            "endpoints_count": per_image_param.endpoints_count,
            "luminous_model": per_image_param.luminous_model,
            "led": per_image_param.led
        }
        data_list.append(filtered_data)

    df = pd.DataFrame(data_list)
    return df


@router.post("/save_firebase/")
async def save_firebase(request: WholeImageParameterWithLED):
    try:
        df = extract_relevant_data(request)
        save_to_firestore(df, "whole_image_parameters_led")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in save_firebase: {e}")
        raise HTTPException(status_code=500, detail=str(e))