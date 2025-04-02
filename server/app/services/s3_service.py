import boto3
import re
from botocore.exceptions import ClientError
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

    #--------------------------------------------------------------------------
    # Helper functions
    #--------------------------------------------------------------------------
    def _get_image_url(self, folder_name, image_name):
        return f"https://{settings.S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{folder_name}/{image_name}"

    #--------------------------------------------------------------------------
    # Main functions
    #--------------------------------------------------------------------------
    def upload_image(self, image, project_id, new_image_name):
        """
        Upload an image to S3 and return the URL of the uploaded image.

        Parameters:
            image (file-like object): The image to upload.
            project_id (str): The ID of the project, used as the folder name.
            new_image_name (str): The name to save the image as in S3.

        Returns:
            str: The URL of the uploaded image.
        """
        folder_name = project_id
        # Create the folder in S3 and upload the image
        self.s3_client.put_object(Bucket=settings.S3_BUCKET, Key=(folder_name + "/"))
        self.s3_client.upload_fileobj(
            image,
            settings.S3_BUCKET,
            f"{folder_name}/{new_image_name}"
        )
        return self._get_image_url(folder_name, new_image_name)

    def get_s3_image_presigned_url(self, project_id, image_name):
        """
        Get a presigned URL for an image in S3.

        Parameters:
            project_id (str): The ID of the project to which the image belongs.
            image_name (str): The name of the image, including the extension.

        Returns:
            str: The presigned URL for the image.

        Raises:
            FileNotFoundError: If the image does not exist in S3.
        """
        folder_name = project_id
        try:
            return self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.S3_BUCKET,
                    'Key': f"{folder_name}/{image_name}"
                },
                ExpiresIn=3600
            )
        except self.s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Image {image_name} not found in project {project_id}")

    def get_image_from_url(self, url):
        """
        Retrieves the image content from the given S3 URL.

        Parameters:
            url (str): The S3 URL of the image.

        Returns:
            bytes: The content of the image as bytes.

        Raises:
            ValueError: If the URL is invalid.
            HTTPException: If the image cannot be retrieved from S3.
        """
        try:
            # Extract bucket name and object key from the S3 URL
            bucket_name, object_key = self._parse_s3_url(url)
            # Download the image content from S3
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            image_content = response['Body'].read()
            return image_content
        except ClientError as e:
            logger.error(f"Failed to download image from S3: {e}")
            raise
        except ValueError as ve:
            logger.error(f"Invalid S3 URL: {ve}")
            raise

    def _parse_s3_url(self, url):
        """
        Parses an S3 URL and returns the bucket name and object key.

        Parameters:
            url (str): The S3 URL.

        Returns:
            Tuple[str, str]: A tuple containing the bucket name and object key.

        Raises:
            ValueError: If the URL is not a valid S3 URL.
        """
        # Patterns to match S3 URL formats
        s3_patterns = [
            r's3://(?P<bucket>[^/]+)/(?P<key>.+)',
            r'https?://(?P<bucket>[^.]+)\.s3\.amazonaws\.com/(?P<key>.+)',
            r'https?://(?P<bucket>[^.]+)\.s3\.(?P<region>[^.]+)\.amazonaws\.com/(?P<key>.+)',
            r'https?://s3\.(?P<region>[^.]+)\.amazonaws\.com/(?P<bucket>[^/]+)/(?P<key>.+)',
        ]

        for pattern in s3_patterns:
            match = re.match(pattern, url)
            if match:
                bucket_name = match.group('bucket')
                object_key = match.group('key')
                return bucket_name, object_key

        raise ValueError(f"Invalid S3 URL: {url}")