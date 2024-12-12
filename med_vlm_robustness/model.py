import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import transformers.utils
from tqdm import tqdm
import torch

from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pytorch_lightning as pl
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.conversation import conv_templates, SeparatorStyle


class LLaVA_Med(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        disable_torch_init()
        self.model_path = cfg.model_path
        self.model_type = cfg.model_type
        self.is_medical = cfg.get("is_medical", True)
        if self.model_type == "full_finetuning" or self.model_type == "pretrained":
            self.model_base = cfg.get("model_base", None)
        else:
            if self.is_medical:
                self.model_base = cfg.get("model_base", os.getenv("LLAVA_MED_MODEL_PATH"))
            else:
                self.model_base = cfg.get("model_base", os.getenv("LLAVA_MODEL_PATH"))
        self.model_name = get_model_name_from_path(self.model_path)
        self.max_new_token = cfg.max_new_tokens
        print(f"Model base: {self.model_base}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self.model_name,
            is_medical=self.is_medical
        )
        self.test_results = []
        self.output_file = cfg.output_file
        if self.model_type == "prompt":
            self.prompt_embed = torch.load(f"{cfg.model_path}/adapter_model.bin")


    def test_step(self, batch, batch_idx):
        # Get the question and image pairs
        # Note: They load the questions and answers from a file (is it relevant here)
        images = batch["image"]
        questions = batch["question"]

        # TODO: Should we remove normalization in preprocessing here as well??
        #  self.image_processor.do_normalize = False
        # generate the question form with image token
        images = self.image_processor.preprocess(images=images, return_tensors="pt")["pixel_values"]
        images = images.type(torch.float16)
        # TODO: change this to batch inference
        qs = questions[0]
        # This is already done in dataset creation
        #qs = DEFAULT_IMAGE_TOKEN + "\n" + questions  # -> image and below the text (question)

        if type(self.model) == LlavaMistralForCausalLM:
            conv_mode = "mistral_instruct"
        else:
            conv_mode = "llava_v0"

        # get the conversation description(?) from the conversation templates and append it to the qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (  # IMAGE_TOKEN_INDEX = -200 (defined in the repo)
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        image_sizes = [x.size for x in images]

        # use functions from converstation class defined in converstation.py
        # set up the stopping string by using variables of the class
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # add stopping strings to stop model when it tries to generate aditional conversations after giving the answer
        # keywords = [stop_str, "###", "\n"]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        if self.model_type == "prompt":
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria],
                    max_new_tokens=self.max_new_token,
                    prompt_embeddings=self.prompt_embed["prompt_embeddings"]
                    # kwargs=generate_kwargs
                )
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria],
                    max_new_tokens=self.max_new_token,
                    # kwargs=generate_kwargs
                )

        # input_token_len = input_ids.shape[1]
        # outputs = self.tokenizer.batch_decode(
        #     output_ids[:, input_token_len:], skip_special_tokens=True
        # )[0]

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        print(outputs)

        # if outputs.endswith(stop_str):
        #     outputs = outputs[: -len(stop_str)]
        # outputs = outputs.strip()

        self.test_results.append({
            "qid": batch["qid"][0].item(),
            "question": batch["question"][0],
            "gt": batch["gt"][0],
            "pred": outputs,
            "answer_type": batch["answer_type"][0],
            "img_name": batch["img_name"][0],
        })

    def on_test_end(self):
        if not Path(self.output_file).parent.is_dir():
            os.makedirs(Path(self.output_file).parent)
        with open(self.output_file, 'w') as json_file:
            json.dump(self.test_results, json_file, indent=2)


