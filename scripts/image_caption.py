import os
import traceback
import sys

import gradio as gr
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from modules import devices, paths, lowvram
from modules.processing import Processed
import modules.scripts as scripts
from modules.shared import opts, cmd_opts, state
import modules.shared as shared

import models.blip

class Script(scripts.Script):
    """
    Image captioning using BLIP.
    """
    _blip_model = None
    BLIP_MODEL_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'

    def title(self):
        return "Image Caption"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        num_beams = gr.Slider(label="Num Beams", minimum=1, maximum=20, value=3)
        min_length = gr.Slider(label="Min Length", minimum=1, maximum=100, value=24)
        max_length = gr.Slider(label="Max Length", minimum=1, maximum=100, value=48)
        image_eval_size = gr.Slider(label="Image Eval Dim", minimum=100, maximum=512, value=480)

        return [image_eval_size, num_beams, min_length, max_length]

    def run(self, p, image_eval_size, num_beams, min_length, max_length):

        if self._blip_model is None:
            blip_model = models.blip.blip_decoder(
                pretrained=self.BLIP_MODEL_URL,
                image_size=int(image_eval_size),
                vit='base',
                med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json")
            )
            blip_model.eval()
            if not shared.cmd_opts.no_half:
                blip_model = blip_model.half()

            self._blip_model = blip_model

        pil_image = p.init_images[0]
        answer = ""

        try:
            self._blip_model = self._blip_model.to(shared.device)

            answer = self.generate_caption(pil_image, image_eval_size, num_beams, min_length, max_length)

            if not shared.opts.interrogate_keep_models_in_memory:
                self._blip_model = self._blip_model.to(devices.cpu)

            devices.torch_gc()

        except Exception:
            print(f"Error visual QA", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            answer += "ERROR:\n" + str(traceback.format_exc())

        return Processed(p, images_list=[], info=f"Caption -- {answer}")

    def generate_caption(self, pil_image, image_eval_size, num_beams, min_length, max_length):
        dtype = next(self._blip_model.parameters()).dtype
        image_eval_size = int(image_eval_size)
        num_beams = int(num_beams)
        min_length = int(min_length)
        max_length = int(max_length)

        gpu_image = transforms.Compose([
            transforms.Resize(
                (image_eval_size, image_eval_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(dtype).to(shared.device)

        with torch.no_grad():
            caption = self._blip_model.generate(
                gpu_image,
                # TODO(hayk): Add params for beam search vs nucleus?
                sample=False,
                num_beams=num_beams,
                min_length=min_length,
                max_length=max_length,
            )

        print(caption)

        return caption[0]
