import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Global variables for the worker processes
upsampler = None
final_outscale = None
model = None

def init_worker(model_name, model_path, netscale, final_outscale_value):
    global upsampler, final_outscale, model

    final_outscale = final_outscale_value  # Set the global final_outscale

    # Initialize the model based on model_name
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2)
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=16, upscale=4, act_type='prelu')
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=16, upscale=4, act_type='prelu')
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type='prelu')
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type='prelu')
    else:
        raise ValueError(f"Model {model_name} is not supported")

    # Initialize the upsampler once per process
    denoise_strength = 0.5
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path_list = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]
    else:
        model_path_list = model_path

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path_list,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        half=False,
        gpu_id=-1  # Use CPU
    )

def process_tile(args):
    global upsampler, final_outscale
    tile, coord = args
    x, y, x_start, y_start, x_end, y_end = coord
    try:
        # Handle alpha channel if present
        if tile.shape[2] == 4:
            img_rgb = tile[:, :, :3]
            img_alpha = tile[:, :, 3]

            output_img_rgb, _ = upsampler.enhance(
                img_rgb, outscale=final_outscale)
            output_img_rgb, _ = upsampler.enhance(
                img_rgb, outscale=final_outscale)
            # Resize alpha channel
            output_alpha = cv2.resize(
                img_alpha,
                (output_img_rgb.shape[1], output_img_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Combine RGB and alpha channels
            output_tile = cv2.cvtColor(
                output_img_rgb, cv2.COLOR_BGR2BGRA)
            output_tile = cv2.cvtColor(
                output_img_rgb, cv2.COLOR_BGR2BGRA)
            output_tile[:, :, 3] = output_alpha
        else:
            output_tile, _ = upsampler.enhance(
                tile, outscale=final_outscale)
            output_tile, _ = upsampler.enhance(
                tile, outscale=final_outscale)

        return (output_tile, coord)
    except RuntimeError as error:
        raise error
    except RuntimeError as error:
        raise error
    except Exception as error:
        raise error

class SuperResolutionService:
    DEFAULT_OUTSCALE = 4

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model_name = "RealESRGAN_x4plus"
        self.model, self.netscale, self.file_url = self.get_model(self.model_name)
        self.model_path = self.download_model(self.model_name, self.file_url)

    def download_model(self, model_name, file_url):
        # Determine model path and ensure the model is downloaded
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # Model path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'),
                    progress=True, file_name=None)
        return model_path

    def get_model(self, model_name):
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4)
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2)
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=16, upscale=4, act_type='prelu')
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=4, act_type='prelu')
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
        else:
            raise ValueError(f"Model {model_name} is not supported")
        return model, netscale, file_url

    def super_resolution_image(self, img, outscale=None):
        global final_outscale, model  # Ensure global variables are accessible
        final_outscale = outscale if outscale is not None else self.DEFAULT_OUTSCALE
        model = self.model  # Needed in the worker initializer
        model = self.model  # Needed in the worker initializer

        # Prepare parameters for tiling
        tile_size = 256  # Adjust as needed
        tile_pad = 10    # Overlap to avoid seams
        tile_size = 256  # Adjust as needed
        tile_pad = 10    # Overlap to avoid seams
        h, w = img.shape[:2]
        tiles = []
        coordinates = []

        # Split the image into overlapping tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                x_start = max(x - tile_pad, 0)
                y_start = max(y - tile_pad, 0)
                x_end = min(x + tile_size + tile_pad, w)
                y_end = min(y + tile_size + tile_pad, h)

                tile = img[y_start:y_end, x_start:x_end]
                tiles.append(tile)
                coordinates.append((x, y, x_start, y_start, x_end, y_end))

        # Process tiles in parallel using ProcessPoolExecutor
        max_workers = min(2, os.cpu_count())  # Adjust the number of workers as needed

        # Prepare arguments for the initializer
        # Prepare arguments for the initializer
        initargs = (self.model_name, self.model_path, self.netscale, final_outscale)

        with ThreadPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=initargs) as executor:
            results = list(executor.map(process_tile, zip(tiles, coordinates)))

        # Prepare an empty array for the output image
        scale = final_outscale / self.netscale
        output_h = int(h * final_outscale)
        output_w = int(w * final_outscale)
        if img.shape[2] == 4:
            output_img = np.zeros((output_h, output_w, 4), dtype=np.uint8)
        else:
            output_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        # Stitch the processed tiles back together
        for output_tile, coord in results:
            x, y, x_start, y_start, x_end, y_end = coord

            # Calculate scaling factors
            input_tile_h = y_end - y_start
            input_tile_w = x_end - x_start
            tile_h, tile_w = output_tile.shape[:2]
            scale_h = tile_h / input_tile_h
            scale_w = tile_w / input_tile_w

            # Calculate positions in the output image
            x_out = int(x * scale_w)
            y_out = int(y * scale_h)
            x_out_end = x_out + int(tile_size * scale_w)
            y_out_end = y_out + int(tile_size * scale_h)

            # Calculate positions in the output tile
            x_tile_start = int((x - x_start) * scale_w)
            y_tile_start = int((y - y_start) * scale_h)
            x_tile_end = x_tile_start + (x_out_end - x_out)
            y_tile_end = y_tile_start + (y_out_end - y_out)

            # Handle boundary conditions
            x_out_end = min(x_out_end, output_img.shape[1])
            y_out_end = min(y_out_end, output_img.shape[0])
            x_tile_end = min(x_tile_end, output_tile.shape[1])
            y_tile_end = min(y_tile_end, output_tile.shape[0])

            # Place the tile into the output image
            output_img[y_out:y_out_end, x_out:x_out_end] = output_tile[
                y_tile_start:y_tile_end, x_tile_start:x_tile_end]
            output_img[y_out:y_out_end, x_out:x_out_end] = output_tile[
                y_tile_start:y_tile_end, x_tile_start:x_tile_end]

        return output_img