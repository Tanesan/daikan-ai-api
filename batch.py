import math
import os
import cv2
import logging
from server.app.services.super_resolution_service import SuperResolutionService
from server.app.services.background_removal_service import BackgroundRemovalService
from server.app.services.mask_service import MaskService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchService:
    def __init__(self):
        self.super_resolution_service = SuperResolutionService()
        self.background_removal_service = BackgroundRemovalService()
        self.mask_service = MaskService()

    def process_images(self, folder_path: str):
        for root, _, files in os.walk(folder_path):
            for filename in files:
                basename_no_ext, ext = os.path.splitext(filename)
                if (filename.endswith(".png") or filename.endswith(".jpg")) and basename_no_ext == "original":
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
                    
                    logger.info(f"Image shape: {img.shape}")
                    
                    logging.info(f"Using outscale {outscale}")

                    try:
                        img = self.super_resolution_service.super_resolution_image(img, outscale)
                        logger.info(f"Super resolution image with new size = {img.shape}")
                        sr_path = os.path.join(root, "sr.png")
                        cv2.imwrite(sr_path, img)
                        logging.info(f"Saved super resolution image to {sr_path}")
                    except Exception as e:
                        logging.error(f"Super resolution processing failed for {img_path}: {e}")
                        continue

                    try:
                        img = self.background_removal_service.process(img)
                        segmented_path = os.path.join(root, "segmented.png")
                        cv2.imwrite(segmented_path, img)
                        logging.info(f"Saved segmented image to {segmented_path}")
                    except Exception as e:
                        logging.error(f"Background removal failed for {img_path}: {e}")
                        continue

                    try:
                        mask = self.mask_service.process(img, inverted=True)
                        mask_path = os.path.join(root, "mask.png")
                        cv2.imwrite(mask_path, mask)
                        logging.info(f"Saved mask to {mask_path}")
                    except Exception as e:
                        logging.error(f"Mask processing failed for {img_path}: {e}")
                        continue
                
def main():
    batch_service = BatchService()
    batch_service.process_images("C:/Users/ontologist3/daikan-private/image-segmentation/remove_difficult_0607")
    
    #batch_service.process_images("C:/Users/ontologist3/daikan-private/image-segmentation/few_noises/few_noises")
    #batch_service.process_images("C:/Users/ontologist3/daikan-private/image-segmentation/no_noise_from_illastrator_0529-20240529T135034Z-001/no_noise_from_illastrator_0529")
    #batch_service.process_images("C:/Users/ontologist3/daikan-private/image-segmentation/no_noise_pictures-20240508T111642Z-001_2/no_noise_pictures")
if __name__ == "__main__":
    main()
