# from dotenv import load_dotenv
import uvicorn
# load_dotenv()

import os
from fastapi import FastAPI
from app.api.endpoints import upload, super_resolution, get_image, remove_image_background, mask_image, smooth_mask, calc_para, predict_led, grouping_request, pixelation_endpoint, image_size_endpoint, resize_image, image_edge_enhancement, pixelation_detection_endpoint, get_learning_data, save_firebase, calc_mape_for_learning, get_cpu_usage, crop_endpoint, scale_endpoint, upload_original_endpoint, paint_image_endpoint
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk



app = FastAPI(
    title="daikan backend api",
    version="1.0.0",
    root_path="/api",
    openapi_version="3.0.0",

)

# same origin
origins = [
    "http://18.178.164.62",
    "https://18.178.164.62",
    "http://localhost:3000",
    "http://13.231.249.136",
    "https://13.231.249.136"
]

# ENV_MODEの値を取得
env_mode = os.getenv("ENV_MODE")
environment = env_mode
# 環境に応じたDSNを設定
if env_mode in ["development", "test"]:
    dsn = "https://c00592a68f2c6fa5857dc9aab4bc5fc2@o4507439513534464.ingest.us.sentry.io/4507441449336832"
elif env_mode in ["staging", "production"]:
    dsn = "https://d89be6c98237d03f8a3d667586290e8d@o4507439513534464.ingest.us.sentry.io/4508022094364672"
else:
    dsn = None  # 環境モードが未定義の場合

# Sentry SDKの初期化
if dsn:
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
else:
    print("No valid ENV_MODE or DSN provided.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(super_resolution.router)
app.include_router(get_image.router)
app.include_router(remove_image_background.router)
app.include_router(mask_image.router)
app.include_router(smooth_mask.router)
app.include_router(calc_para.router)
app.include_router(predict_led.router)
app.include_router(grouping_request.router)
app.include_router(pixelation_endpoint.router)
app.include_router(image_size_endpoint.router)
app.include_router(pixelation_detection_endpoint.router)
app.include_router(resize_image.router)
app.include_router(image_edge_enhancement.router)
app.include_router(get_learning_data.router)
app.include_router(save_firebase.router)
app.include_router(calc_mape_for_learning.router)
app.include_router(get_cpu_usage.router)
app.include_router(crop_endpoint.router)
app.include_router(scale_endpoint.router)
app.include_router(upload_original_endpoint.router)
app.include_router(paint_image_endpoint.router)

# Local実験用
# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8010, reload=True)