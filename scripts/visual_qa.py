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

import models.blip_vqa
import models.blip

class Script(scripts.Script):
    """
    Visual QA feature using BLIP.
    """
    _blip_vqa_model = None
    BLIP_VQA_MODEL_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'

    def title(self):
        return "Visual QA"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        questions = gr.Textbox(label="Question", placeholder="How many people in the image?", lines=4)

        num_beams = gr.Slider(label="Num Beams", minimum=1, maximum=20, value=3)
        min_length = gr.Slider(label="Min Answer Length", minimum=1, maximum=100, value=1)
        max_length = gr.Slider(label="Max Answer Length", minimum=1, maximum=100, value=10)
        image_eval_size = gr.Slider(label="Image Eval Dim", minimum=100, maximum=512, value=480)

        return [questions, image_eval_size, num_beams, min_length, max_length]

    def run(self, p, questions, image_eval_size, num_beams, min_length, max_length):

        if self._blip_vqa_model is None:
            blip_vqa_model = models.blip_vqa.blip_vqa(
                pretrained=self.BLIP_VQA_MODEL_URL,
                image_size=int(image_eval_size),
                vit="base",
                med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json")
            )
            blip_vqa_model.eval()
            if not shared.cmd_opts.no_half:
                blip_vqa_model = blip_vqa_model.half()

            self._blip_vqa_model = blip_vqa_model

        pil_image = p.init_images[0]
        output_str = ""

        try:
            self._blip_vqa_model = self._blip_vqa_model.to(shared.device)

            for question in questions.split("\n"):
                # HACK(hayk): Parse out [0-20] as a prefix for min num words
                q_min_length = min_length
                for i in range(21):
                    prefix = f"[{i}]"
                    if question.startswith(prefix):
                        q_min_length = int(prefix[1:-1])
                        question = question[len(prefix):]
                        print(f"FOUND {prefix=} {q_min_length=} {question=}")
                        break

                answer = self.generate_answer(pil_image, question, image_eval_size, num_beams, q_min_length, max_length)

                output_str += question
                if q_min_length != min_length:
                    output_str += f" ({q_min_length} min words)"
                output_str += f" -- {answer}\n"

            if not shared.opts.interrogate_keep_models_in_memory:
                self._blip_vqa_model = self._blip_vqa_model.to(devices.cpu)

            devices.torch_gc()

        except Exception:
            print(f"Error visual QA", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            output_str += "ERROR:\n" + str(traceback.format_exc())

        return Processed(p, images_list=[], info=output_str)

    def generate_answer(self, pil_image, question, image_eval_size, num_beams, min_length, max_length):
        vqa = self._blip_vqa_model
        dtype = next(self._blip_vqa_model.parameters()).dtype
        image_eval_size = int(image_eval_size)
        num_beams = int(num_beams)
        min_length = int(min_length)
        max_length = int(max_length)

        transform_vq = transforms.Compose([
            transforms.Resize(
                (image_eval_size,image_eval_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform_vq(pil_image).unsqueeze(0).type(dtype).to(shared.device)

        with torch.no_grad():
            image_embeds = vqa.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            question = vqa.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                    return_tensors="pt").to(image.device)
            question.input_ids[:,0] = vqa.tokenizer.enc_token_id

            question_output = vqa.text_encoder(question.input_ids,
                                                attention_mask = question.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True)


            question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
            question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
            model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}

            bos_ids = torch.full((image.size(0),1),fill_value=vqa.tokenizer.bos_token_id,device=image.device)

            outputs = vqa.text_decoder.generate(input_ids=bos_ids,
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    num_beams=num_beams,
                                                    eos_token_id=vqa.tokenizer.sep_token_id,
                                                    pad_token_id=vqa.tokenizer.pad_token_id,
                                                    **model_kwargs)

            answers = []
            for output in outputs:
                answer = vqa.tokenizer.decode(output, skip_special_tokens=True)
                answers.append(answer)

        print(answers)

        return answers[0]
