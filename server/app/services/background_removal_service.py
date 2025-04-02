from enum import Enum
import cv2
import numpy as np
import logging
from .rmbg_gradient_bg_helper import process_image as remove_gradient_background
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class BackgroundRemovalService:
    class Algorithm(Enum):
        AUTO_1 = 1
        HINT_1 = 2
        AUTO_BY_COLOR_AT_POS = 3
        AUTO_2 = 4 # Remove gradient background
        HINT_3 = 5 # Keep foreground(selected points)

    def __init__(self):
        pass

    def process(self, image, algo: Algorithm = Algorithm.AUTO_1, points: list = None, tolerance: float = None, clusters: int = 3):
        """This function removes the background from the image based on the points provided.
        The points are used as seed points for flood fill algorithm.

        image: opencv image
        points: list of (x, y) points
        tolerance: the tolerance for the flood fill algorithm. The smaller the value, the more similar the pixels need to be to be filled.
        """
        if algo == self.Algorithm.AUTO_1:
            return self.remove_background(image, clusters)
        if algo == self.Algorithm.HINT_1:
            if points is None or tolerance is None:
                raise ValueError("Missing points or tolerance")
            return self.__remove_similar_pixels_transparent(image, points, int(tolerance * 255))
        if algo == self.Algorithm.AUTO_BY_COLOR_AT_POS:
            if points is None:
                raise ValueError("Missing points")
            return self.__remove_background_by_points(image, points, int(tolerance * 255))
        if algo == self.Algorithm.AUTO_2:
            return self.__remove_gradient_background(image)
        if algo == self.Algorithm.HINT_3:
            if points is None or tolerance is None:
                raise ValueError("Missing points or tolerance")
            return self.__keep_similar_pixels(image, points, int(tolerance * 255))
    
    def __remove_similar_pixels_transparent(self, image, points, tolerance):
        """
        Makes pixels transparent starting from given points, consuming connected pixels
        within a specified color tolerance.

        :param image: Input image in BGR or BGRA format.
        :param points: List of (x, y) seed points.
        :param tolerance: Color tolerance value between 0 and 255.
        :return: Image with updated alpha channel.
        """
        # Validate tolerance
        if not (0 <= tolerance <= 255):
            raise ValueError("Tolerance must be between 0 and 255.")

        # Ensure the image is in BGRA format
        if image.ndim == 2:
            # Grayscale image
            image_bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            # BGR image
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            # BGRA image
            image_bgra = image.copy()
        else:
            raise ValueError("Unsupported image format.")

        height, width = image_bgra.shape[:2]

        # Prepare the final mask
        final_mask = np.zeros((height, width), dtype=np.uint8)

        for point in points:
            x_seed, y_seed = point
            if x_seed < 0 or y_seed < 0 or x_seed >= width or y_seed >= height:
                continue

            # Prepare a mask for floodFill - it needs to be 2 pixels wider and taller
            mask = np.zeros((height + 2, width + 2), np.uint8)

            # Create a contiguous BGR image for floodFill
            image_bgr = np.ascontiguousarray(image_bgra[:, :, :3])

            # Define flood fill flags
            flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (0 << 8)

            # Prepare the loDiff and upDiff tuples
            lo_diff = (tolerance, tolerance, tolerance)
            up_diff = (tolerance, tolerance, tolerance)

            # Perform flood fill
            _, _, _, _ = cv2.floodFill(
                image_bgr, mask, (x_seed, y_seed), newVal=0,
                loDiff=lo_diff,
                upDiff=up_diff,
                flags=flags
            )

            # Exclude the 1-pixel border added to the mask
            mask = mask[1:-1, 1:-1]

            # Combine the mask with the final mask
            final_mask = cv2.bitwise_or(final_mask, mask)

        # Update the alpha channel: set to 0 where final_mask is 1
        image_bgra[:, :, 3] = np.where(final_mask == 1, 0, image_bgra[:, :, 3])

        return image_bgra

    def remove_background(self, image, clusters=3):
        # Check the number of channels in the input image
        if len(image.shape) == 2:
            # If the image is already grayscale, no need to convert
            gray = image
        elif len(image.shape) == 3:
            # If the image is color, convert it to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 4:
            # If the image has an alpha channel, separate it and convert the rest to grayscale
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha_channel = image[:, :, 3]
        else:
            raise ValueError("Invalid input image")

        # Apply Gaussian blur to reduce noise
        blurred = gray.copy()

        # Perform color quantization using k-means clustering
        pixels = blurred.reshape((-1, 1))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((blurred.shape))

        # Find the background cluster
        background_label = np.argmax(np.bincount(labels.flatten()))

        # Create a mask based on the background cluster
        mask = (labels != background_label).astype(np.uint8) * 255

        # Apply connected component analysis to remove small noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        sizes = stats[1:, -1]
        min_size = 100  # Adjust this value based on your image size and noise level
        noise_labels = [i for i in range(1, num_labels) if sizes[i - 1] < min_size]
        mask[np.isin(labels, noise_labels)] = 0

        # Create an alpha channel where the mask is filled (background becomes transparent)
        alpha_channel = np.where(mask == 0, 0, 255).astype(np.uint8)

        # Create a BGRA image if not already present
        if len(image.shape) == 2:
            image_bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif len(image.shape) == 3:
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif len(image.shape) == 4:
            image_bgra = image.copy()

        image_bgra[:, :, 3] = alpha_channel

        return image_bgra

    def __remove_background_by_points(self, image, points: list, tolerance: int = 0):
        """
        Removes the background of the image based on the colors at specified points.

        This method identifies pixels in the image that are similar to the colors at the given points
        and makes those pixels fully transparent by adjusting the alpha channel.

        Parameters:
        ----------
        image : numpy.ndarray
            The input image in which the background needs to be removed. 
            The image should be in BGR format (3 channels) or BGRA format (4 channels).
            If the image is in BGR format, it will be converted to BGRA format internally.

        points : list of tuples
            A list of (x, y) coordinate tuples representing the points from which to sample colors.
            These points should be within the image boundaries.
            Example: [(100, 100), (200, 200), (300, 300)]

        tolerance : int
            A value representing the allowed variation from each sampled color. 
            Pixels within this range will be considered similar to the sampled colors 
            and will be made transparent. The tolerance is applied to each channel 
            (B, G, R) individually.

        Returns:
        -------
        numpy.ndarray
            The processed image with the background removed based on the colors at the specified points and tolerance.
            The returned image is in BGRA format, where the alpha channel has been adjusted to 
            make similar pixels transparent.

        Method Workflow:
        ----------------
        1. Image Format Handling:
            - The method first checks if the input image has three channels (BGR) or four channels (BGRA).
            - If the image has three channels (BGR), it is converted to BGRA format to include an alpha channel.
            If the image already has an alpha channel (BGRA), it is used as is.

        2. Color Sampling and Mask Creation:
            - For each point in the input list:
                a. The color at that point is sampled from the image.
                b. Lower and upper bounds for each color channel (B, G, R) are calculated based on 
                the sampled color and the provided tolerance.
                c. A binary mask is created for pixels similar to this color.
            - All individual masks are combined into a single mask using bitwise OR operation.

        3. Alpha Channel Adjustment:
            - The method uses the combined mask to adjust the alpha channel of the image. 
            - Pixels that match any of the sampled colors (i.e., those marked as 255 in the combined mask) 
            have their alpha value set to 0, making them fully transparent.
            - Pixels that do not match retain their original alpha values.

        4. Return the Processed Image:
            - The method returns the modified image, now with certain pixels made transparent based on their similarity 
            to the colors at the specified points.

        Example Usage:
        --------------
        image = cv2.imread('input_image.png')  # Load an image from file
        bg_service = BackgroundRemovalService()
        points = [(100, 100), (200, 200), (300, 300)]
        result_image = bg_service._BackgroundRemovalService__remove_background_by_color(
            image, points=points, tolerance=30)
        cv2.imwrite('output_image.png', result_image)  # Save the result to a new file

        In this example, the method processes the `input_image.png` and makes all pixels that are close to the colors
        at the specified points transparent, within the specified tolerance of 30.
        """
        # Ensure the image is in the correct format (BGRA)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image_bgra = image.copy()
        else:
            raise ValueError("Unsupported image format: the image must have 3 or 4 channels.")

        # Validate the points
        if not points:
            raise ValueError("Points list cannot be empty")
        
        height, width = image_bgra.shape[:2]
        for point in points:
            x, y = point
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError(f"Point {point} is out of image boundaries")

        # Initialize the combined mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for point in points:
            x, y = point
            # Extract the color at the specified point
            color = image_bgra[y, x, :3]

            # Calculate the lower and upper bounds for the color with tolerance
            lower_bound = np.array([max(c - tolerance, 0) for c in color], dtype=np.uint8)
            upper_bound = np.array([min(c + tolerance, 255) for c in color], dtype=np.uint8)

            # Create a mask where the pixels within the range are set to 255 (white)
            mask = cv2.inRange(image_bgra[:, :, :3], lower_bound, upper_bound)

            # Combine the mask with the previous masks
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Set the pixels' alpha channel to 0 (transparent) where the combined mask is 255
        image_bgra[:, :, 3] = np.where(combined_mask == 255, 0, image_bgra[:, :, 3])

        return image_bgra
    
    def __remove_gradient_background(self, image):
        result, *_ = remove_gradient_background(image)
        return result

    def __keep_similar_pixels(self, image, points, tolerance):
        """
        Preserves pixels starting from given points, consuming pixels similar to the seed colors
        based on Euclidean distance in color space. Uses 8-connected neighbors.
        Fills only small holes within the foreground mask without altering element widths,
        ensuring that intended holes (e.g., in letters like 'O') remain unfilled.

        :param image: Input image in BGR or BGRA format.
        :param points: List of (x, y) seed points.
        :param tolerance: Color tolerance value (0 to 255).
        :return: Image with updated alpha channel, keeping only similar pixels up to the max level.
        """
        # Constants
        MAX_LEVEL = 1  # Adjust this constant to set the maximum level

        # Validate tolerance
        if not (0 <= tolerance <= 255):
            raise ValueError("Tolerance must be between 0 and 255.")

        # Calculate the Euclidean tolerance
        euclidean_tolerance = tolerance * (3 ** 0.5)

        # Ensure the image is in BGRA format
        if image.ndim == 2:
            # Grayscale image
            image_bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            # BGR image
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            # BGRA image
            image_bgra = image.copy()
        else:
            raise ValueError("Unsupported image format.")

        height, width = image_bgra.shape[:2]

        # Prepare a mask to keep track of processed pixels
        processed_mask = np.zeros((height, width), dtype=np.uint8)

        # Prepare the final mask
        final_mask = np.zeros((height, width), dtype=np.uint8)

        # Initialize a queue with seed points, their levels, and their reference colors
        from collections import deque
        queue = deque()
        for point in points:
            x_seed, y_seed = point
            if x_seed < 0 or y_seed < 0 or x_seed >= width or y_seed >= height:
                continue
            seed_color = image_bgra[y_seed, x_seed, :3]
            queue.append((x_seed, y_seed, 0, seed_color))  # (x, y, level, reference_color)

        # Convert image to numpy array if not already
        image_bgr = np.ascontiguousarray(image_bgra[:, :, :3])

        # Process pixels in the queue
        while queue:
            x, y, level, ref_color = queue.popleft()

            if processed_mask[y, x]:
                continue

            if level > MAX_LEVEL:
                continue

            # Mark current pixel as processed
            processed_mask[y, x] = 1

            # Set pixel in final_mask
            final_mask[y, x] = 1

            # Get neighbors (8-connected)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append((nx, ny))

            for nx, ny in neighbors:
                if processed_mask[ny, nx]:
                    continue

                neighbor_color = image_bgr[ny, nx]

                # Calculate Euclidean distance
                color_diff = np.linalg.norm(neighbor_color.astype(np.int16) - ref_color.astype(np.int16))

                if color_diff <= euclidean_tolerance:
                    queue.append((nx, ny, level, ref_color))  # Same level, same reference color
                else:
                    if level + 1 <= MAX_LEVEL:
                        # Start a new level with the neighbor's color as reference
                        new_ref_color = neighbor_color
                        queue.append((nx, ny, level + 1, new_ref_color))

        # Invert the final mask to get the holes (unprocessed regions)
        holes_mask = (final_mask == 0).astype(np.uint8)

        # Label connected components in the holes mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)

        # Prepare a mask to hold small holes to fill
        small_holes_mask = np.zeros_like(final_mask, dtype=np.uint8)

        # Calculate the area threshold based on the image size or other criteria
        # For example, set threshold to a percentage of the total image area
        image_area = height * width
        hole_size_threshold = image_area * 0.0001  # Adjust this percentage as needed

        for i in range(1, num_labels):  # Skip the background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= hole_size_threshold:
                # Fill the hole
                small_holes_mask[labels == i] = 1

        # Update the final mask by filling the small holes
        final_mask = final_mask | small_holes_mask

        # Update the alpha channel: set to 0 where final_mask is 0
        image_bgra[:, :, 3] = np.where(final_mask == 1, image_bgra[:, :, 3], 0)

        return image_bgra