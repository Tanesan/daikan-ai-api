import math
import os
import cv2
import logging
from super_resolution_service import SuperResolutionService
from background_removal_service import BackgroundRemovalService
from mask_service import MaskService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatchService:
    def __init__(self):
        self.super_resolution_service = SuperResolutionService()
        self.background_removal_service = BackgroundRemovalService()
        self.mask_service = MaskService()

    def process_images(self, folder_path: str):
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_path = os.path.join(root, filename)
                    logging.info(f"Processing {img_path}")
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        logging.warning(f"Failed to read image {img_path}")
                        continue

                    outscale = 2

                    # Assuming img is a numpy array representing the image
                    if img.shape[0] < 200 or img.shape[1] < 200:
                        outscale = math.ceil(200 / min(img.shape[0], img.shape[1]))
                    
                    logging.info(f"Using outscale {outscale}")

                    try:
                        img = self.super_resolution_service.super_resolution_image(img, outscale)
                        sr_path = os.path.join(root, "sr.png")
                        cv2.imwrite(sr_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        logging.info(f"Saved super resolution image to {sr_path}")
                    except Exception as e:
                        logging.error(f"Super resolution processing failed for {img_path}: {e}")
                        continue

                    try:
                        img = self.background_removal_service.process(img)
                        segmented_path = os.path.join(root, "segmented.png")
                        cv2.imwrite(segmented_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        logging.info(f"Saved segmented image to {segmented_path}")
                    except Exception as e:
                        logging.error(f"Background removal failed for {img_path}: {e}")
                        continue

                    try:
                        mask = self.mask_service.process(img, inverted=True)
                        mask_path = os.path.join(root, "mask.png")
                        cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        logging.info(f"Saved mask to {mask_path}")
                    except Exception as e:
                        logging.error(f"Mask processing failed for {img_path}: {e}")
                        continue
                
def main():
    batch_service = BatchService()
    batch_service.process_images("C:/Users/ontologist3/daikan-private/image-segmentation/no_noise_pictures-20240508T111642Z-001")

if __name__ == "__main__":
    main()
