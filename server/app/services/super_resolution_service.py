import cv2
import numpy as np
import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class SuperResolutionService:
    DEFAULT_OUTSCALE = 4
    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model_name = "RealESRGAN_x4plus"
        self.model, self.netscale, self.file_url = self.get_model(self.model_name)

    def get_model(self, model_name):
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
            
        return model, netscale, file_url

    def super_resolution_image(self, img, outscale=None):
        final_outscale = outscale if outscale is not None else self.DEFAULT_OUTSCALE
        # determine model paths
        model_path = os.path.join('weights', "RealESRGAN_x4plus" + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in self.file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        
        tile = 0
        tile_pad = 4
        pre_pad = 0

        # use dni to control the denoise strength
        denoise_strength = 0.5
        dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]
        
            # restorer
        upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=self.model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=False,
            gpu_id=0,)

        try:
            output, _ = upsampler.enhance(img, outscale=final_outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            raise error
        except Exception as error:
            print('Error', error)
            raise error

        return output
