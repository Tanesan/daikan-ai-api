import cv2
import numpy as np

class MaskService:
    
    def __init__(self):
        pass
    
    def process(self, img, inverted=False):
        """This function creates a mask from the image.
        The mask is a binary image where the pixels are 0 or 1.
        If the image has an alpha channel, where the alpha channel is transparent, the mask will be black.
        
        img: opencv image
        """
        return self.__create_mask(img, inverted)

    def __create_mask(self, img, inverted=False):
        """Create a black and white mask from the image.
        If the image has an alpha channel, the mask will be white where the alpha channel is transparent.
        """
        #check if the image is already a mask then invert it by making the white pixels black and vice versa
        if self.is_mask(img):
            return self.invert_black_and_white(img)
        
        if img.shape[2] == 4:  # Check if the image has an alpha channel
            # Use the alpha channel as the mask. Invert it so transparency becomes white.
            if inverted:
                mask = cv2.bitwise_not(img[:, :, 3])
            else:
                mask = img[:, :, 3]
        else:
            # If no alpha channel, create a white mask of the same size as the image.
            mask = np.ones((img.shape[0], img.shape[1]), dtype="uint8") * 0

        return mask


    def is_mask(self, image):
        """
        Determines if an image is grayscale (black, white, and shades of gray).
        For color images, each pixel must have equal R, G, and B values.

        Parameters:
        - image: The image array.

        Returns:
        - True if the image is grayscale, False otherwise.
        """
        # Check if the image is grayscale or color
        if len(image.shape) == 2:
            # Image is already grayscale
            return True
        elif len(image.shape) == 3:
            # Color image
            # Get the number of channels
            num_channels = image.shape[2]
            # Split the image into its color channels
            channels = cv2.split(image)
            if num_channels == 3:
                b = channels[0]
                g = channels[1]
                r = channels[2]
                # Check if all channels are equal
                if np.array_equal(b, g) and np.array_equal(b, r):
                    return True
                else:
                    return False
            else:
                # Unexpected number of channels (e.g., less than 3)
                return False
        else:
            # Unsupported image format
            return False


    def invert_black_and_white(self, image):
        """
        Inverts black and white pixels in a purely black and white image.
        Converts black pixels to white and white pixels to black.

        Parameters:
        - image: The image array (as loaded by cv2.imread).

        Returns:
        - The inverted image array.

        Raises:
        - ValueError: If the image is not purely black and white.
        """
        if not self.is_mask(image):
            raise ValueError("The image is not purely black and white.")

        # Invert the image by subtracting pixel values from 255
        inverted_image = 255 - image
        return inverted_image