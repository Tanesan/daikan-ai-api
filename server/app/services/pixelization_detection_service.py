from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Tuple
import tempfile
import os

app = FastAPI(
    title="Image Pixelation Analyzer",
    description="API to analyze images for pixelation",
    version="1.0.0"
)

def is_pixelated(image_path: str, threshold: float = 0.1) -> Tuple[bool, float]:
    """
    Detect if an image is pixelated within a certain threshold.

    This function analyzes an image to determine if it appears pixelated by examining
    edge sharpness and color transitions. It returns both a boolean indicating whether
    the image is considered pixelated and a float score quantifying the degree of pixelation.

    Args:
        image_path (str): The file path to the image to be analyzed.
        threshold (float, optional): The pixelation threshold. 
            Scores below this value are considered non-pixelated. 
            Defaults to 0.1.

    Returns:
        Tuple[bool, float]: A tuple containing:
            - bool: True if the image is considered pixelated, False otherwise.
            - float: The pixelation score, where lower values indicate less pixelation.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
        cv2.error: If there's an error reading or processing the image.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found or unable to read: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the magnitude of gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize the magnitude
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Calculate the average magnitude
        avg_magnitude = np.mean(magnitude)

        # Calculate color transitions
        diff_horiz = np.abs(img[:, 1:] - img[:, :-1]).mean()
        diff_vert = np.abs(img[1:, :] - img[:-1, :]).mean()
        color_transition = (diff_horiz + diff_vert) / 2

        # Normalize color transition
        color_transition = color_transition / 255

        # Combine edge sharpness and color transition
        pixelation_score = (1 - avg_magnitude/255 + color_transition) / 2

        # Check if the score is below the threshold
        is_pixelated = pixelation_score > threshold

        return is_pixelated, pixelation_score

    except cv2.error as e:
        raise cv2.error(f"Error processing the image: {str(e)}")

@app.post("/pixelization_detection")
async def pixelization_detection(
    image: UploadFile = File(...),
    threshold: float = Query(0.1, description="Pixelation threshold (0-1)")
):
    """
    Analyze an uploaded image for pixelation.

    - **image**: The image file to be analyzed.
    - **threshold**: Pixelation threshold. Scores above this value are considered pixelated. Default is 0.1.

    Returns:
        JSON object containing:
        - **is_pixelated**: Boolean indicating if the image is considered pixelated.
        - **pixelation_score**: Float representing the degree of pixelation (0-1).
    """
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG, and GIF are allowed.")

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await image.read())
        temp_file_path = temp_file.name

    try:
        is_pixelated_result, score = is_pixelated(temp_file_path, threshold)
        return JSONResponse({
            "is_pixelated": bool(is_pixelated_result),  # Explicitly convert to bool
            "pixelation_score": float(score)  # Explicitly convert to float
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)