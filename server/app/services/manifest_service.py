import csv
import io
import boto3
from app.core.config import settings

class ManifestService:
    IMAGE_TYPES = ["original", "scaled", "segmented", "mask", "smoothed"]
    HEADER = ["s3_url", "project_id", "image_type"]

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET
        self.manifest_file = "manifest.csv"

    def _get_manifest_file(self):
        """Retrieve the manifest file if it exists, otherwise return None."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.manifest_file)
            return io.StringIO(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            return None

    def _write_manifest_file(self, rows):
        """Write the given rows to the manifest file, including the header."""
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        # Always write the header
        writer.writerow(self.HEADER)
        writer.writerows(rows)
        self.s3_client.put_object(Bucket=self.bucket, Key=self.manifest_file, Body=csv_buffer.getvalue())

    def add_image_to_manifest(self, s3_url, project_id, image_type):
        """Add or update an image entry in the manifest."""
        if image_type not in self.IMAGE_TYPES:
            raise ValueError(f"Invalid image type. Image type must be one of {self.IMAGE_TYPES}")

        rows = []
        manifest_file = self._get_manifest_file()
        if manifest_file:
            reader = csv.reader(manifest_file)
            next(reader)  # Skip the header
            rows = [row for row in reader]

        # Check if the image type already exists for the project, update if it does
        updated = False
        for row in rows:
            if row[1] == project_id and row[2] == image_type:
                row[0] = s3_url
                updated = True
                break

        if not updated:
            # Add the new image to the manifest if not updated
            rows.append([s3_url, project_id, image_type])

        self._write_manifest_file(rows)