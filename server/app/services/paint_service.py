import cv2
import numpy as np
from typing import List
from app.models.paint_image_models import Modification

class PaintingService:
    def process(
        self, 
        image: np.ndarray, 
        modifications: List[Modification]
    ) -> np.ndarray:
        """
        Processes the image with the given modifications.

        Returns:
            Updated image.
        """
        # Create an overlay image
        overlay = np.zeros_like(image, dtype=np.uint8)

        # Keep track of painted areas
        painted_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for mod in modifications:
            x = mod.x
            y = mod.y
            size = mod.size
            shape = mod.shape
            isEraser = mod.isEraser

            if isEraser:
                # Erase from overlay
                # Create a mask for the area to erase
                erase_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                if shape == 'square':
                    top_left = (x - size // 2, y - size // 2)
                    bottom_right = (x + size // 2, y + size // 2)
                    cv2.rectangle(erase_mask, top_left, bottom_right, color=255, thickness=-1)
                else:
                    # Should not reach here because eraser only supports 'square'
                    pass
                # Erase from painted_mask and overlay where erase_mask is set
                painted_mask = cv2.bitwise_and(painted_mask, cv2.bitwise_not(erase_mask))
                overlay = cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(erase_mask))
            else:
                # Paint onto overlay
                hex_color = mod.color.lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

                # Create a mask for the area to paint
                paint_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

                if shape == 'square':
                    top_left = (x - size // 2, y - size // 2)
                    bottom_right = (x + size // 2, y + size // 2)
                    cv2.rectangle(overlay, top_left, bottom_right, color=bgr_color, thickness=-1)
                    cv2.rectangle(paint_mask, top_left, bottom_right, color=255, thickness=-1)
                elif shape == 'circle':
                    center = (x, y)
                    radius = size // 2
                    cv2.circle(overlay, center, radius, color=bgr_color, thickness=-1)
                    cv2.circle(paint_mask, center, radius, color=255, thickness=-1)

                # Update painted_mask
                painted_mask = cv2.bitwise_or(painted_mask, paint_mask)

        # Composite overlay onto the original image using painted_mask as the mask
        # Where painted_mask is 255, we take the overlay; otherwise, we keep the original image
        result = image.copy()
        overlayed_area = cv2.bitwise_and(overlay, overlay, mask=painted_mask)
        background_area = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(painted_mask))
        result = cv2.add(overlayed_area, background_area)

        return result