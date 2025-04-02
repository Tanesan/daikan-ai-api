from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET: str
    ENV_MODE: str
    GOOGLE_TYPE: str
    GOOGLE_PROJECT_ID: str
    GOOGLE_PRIVATE_KEY_ID: str
    GOOGLE_PRIVATE_KEY: str
    GOOGLE_CLIENT_EMAIL: str
    GOOGLE_CLIENT_ID: str
    GOOGLE_AUTH_URI: str
    GOOGLE_TOKEN_URI: str
    GOOGLE_AUTH_PROVIDER_CERT_URL: str
    GOOGLE_CLIENT_CERT_URL: str

    class Config:
        env_file = ".env"

settings = Settings()