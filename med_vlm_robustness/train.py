from llava.train.train import *

from med_vlm_robustness.datamodule import get_json_filename


#from llava.train.train import ModelArguments, DataArguments, TrainingArguments


def main():
    data_root_dir = "/nvme/VLMRobustness"
    dataset = "Slake"
    split_file = "slake_train_all"
    output_dir = f"{data_root_dir}/{dataset}/Experiments/LoRA"

    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments(output_dir=output_dir)

    # add lora
    training_args.lora_enable = True
    training_args.lora_r = 128
    training_args.lora_alpha = 256
    training_args.mm_projector_lr = 2e-5

    # TODO: what about deepspeed?

    model_args.model_name_or_path = "liuhaotian/llava-v1.5-7b"
    model_args.version = "v1"

    # get dataset json
    data_args.data_path = get_json_filename(split_file)
    data_args.image_folder = f"{data_root_dir}/{dataset}"

    model_args.vision_tower = "openai/clip-vit-large-patch14-336"
    model_args.mm_projector_type = "mlp2x_gelu"
    # Added for trying to get it running
    model_args.mm_vision_select_layer = -2
    model_args.mm_use_im_start_end = False
    model_args.mm_use_im_patch_token = False

    training_args.image_aspect_ratio = "pad"
    training_args.group_by_modality_length = True
    training_args.bf16 = True
    training_args.output_dir = output_dir
    training_args.num_train_epochs = 1
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 4
    training_args.gradient_accumulation_steps = 1
    training_args.evaluation_strategy = "no"
    training_args.save_strategy = "steps"
    training_args.save_steps = 50000
    training_args.save_total_limit = 1
    # TODO was 2e-5 before
    training_args.learning_rate = 2e-4
    training_args.weight_decay = 0.
    training_args.warmup_ratio = 0.03
    training_args.lr_scheduler_type = "cosine"
    training_args.logging_steps = 1
    # TODO was False before
    training_args.tf32 = True
    training_args.model_max_length = 2048
    # TODO was False before
    training_args.gradient_checkpointing = True
    training_args.dataloader_num_workers = 4
    training_args.lazy_preprocess = True

    train(data_args=data_args, model_args=model_args, training_args=training_args)

if __name__=="__main__":
    main()