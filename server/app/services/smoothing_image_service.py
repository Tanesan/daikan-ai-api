from enum import Enum
import cv2
import numpy as np

class SmoothingImageService:
    
    class Algorithm(Enum):
        ERODE = 1
        DILATE = 2
        OPEN = 3
        CLOSE = 4
        

    def __init__(self):
        pass

    def process(self, image, algo: Algorithm, element_size_pixels: int=5):
        print("smooth_image_edges")
        return self.smooth_edges(image, algo, element_size_pixels)

    def smooth_edges(self, image, algo, element_size_pixels: int):
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size_pixels, element_size_pixels))
        if algo == self.Algorithm.ERODE:
            return cv2.erode(image, element)
        elif algo == self.Algorithm.DILATE:
            return cv2.dilate(image, element)
        elif algo == self.Algorithm.OPEN:
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        elif algo == self.Algorithm.CLOSE:
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
        return image.copy()
