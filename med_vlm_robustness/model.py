import json
from pathlib import Path
from typing import TYPE_CHECKING

import transformers.utils
from tqdm import tqdm
import torch

from llava.model.builder import load_pretrained_model
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
    def __init__(self):
        super().__init__()

        # TODO: update with actual llava weights, not only delta
        disable_torch_init()
        self.model_path = "/nvme/VLMRobustness/Experiments/Slake/LoRA/llava-slake_train_iid_modality_X-Ray-finetune_lora"
        self.model_base = "liuhaotian/llava-v1.5-7b"
        self.model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self.model_name,
        )
        self.test_results = []


    def test_step(self, batch, batch_idx):
        # Get the question and image pairs
        # Note: They load the questions and answers from a file (is it relevant here)
        images = batch["image"]
        questions = batch["question"]

        # generate the question form with image token
        images = self.image_processor.preprocess(images=images, return_tensors="pt")["pixel_values"]
        images = images.type(torch.float16)
        # TODO: change this to batch inference
        questions = questions[0]
        qs = DEFAULT_IMAGE_TOKEN + "\n" + questions  # -> image and below the text (question)

        # set up the conversation mode depending on the model
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        # TODO: replace conv mode by this part
        # # if a new conv.mode is defined in addition to the predefined options follow this
        # if args.conv_mode is not None and conv_mode != args.conv_mode:
        #     print(
        #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
        #             conv_mode, args.conv_mode, args.conv_mode
        #         )
        #     )
        # else:
        #     args.conv_mode = conv_mode

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

        # use functions from converstation class defined in converstation.py
        # set up the stopping string by using variables of the class
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        # TODO: change it to this later
        # with torch.inference_mode():
        #     output_ids = self.model.generate(
        #         input_ids,
        #         images=images,
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         max_new_tokens=args.max_new_tokens,
        #         use_cache=True,
        #         stopping_criteria=[stopping_criteria],
        #     )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(qs)
        # print(outputs)
        self.test_results.append({
            "qid": batch["qid"][0].item(),
            "question": batch["question"][0],
            "gt": batch["gt"][0],
            "pred": outputs,
            "answer_type": batch["answer_type"][0],
            "img_name": batch["img_name"][0],
        })


    def on_test_end(self):
        json_path = "/nvme/VLMRobustness/test_results.json"
        with open(json_path, 'w') as json_file:
            json.dump(self.test_results, json_file, indent=2)


