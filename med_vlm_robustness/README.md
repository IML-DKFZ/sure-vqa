# SURE-VQA: SYSTEMATIC UNDERSTANDING OF ROBUSTNESS EVALUATION IN MEDICAL VQA TASKS

> Vision-Language Models (VLMs) have great potential in medical tasks, like Visual Question Answering (VQA), where they could act as interactive assistants for both patients and clinicians. Yet their robustness to distribution shifts on unseen data remains a critical concern for safe deployment. Evaluating such robustness requires a controlled experimental setup that allows for systematic insights into the model’s behavior. However, we demonstrate that current setups fail to offer sufficiently thorough evaluations, limiting their ability to accurately assess model robustness. To address this gap, our work introduces a novel framework, called SURE-VQA, centered around three key requirements to overcome the current pitfalls and systematically analyze the robustness of VLMs: 1) Since robustness on synthetic shifts does not necessarily translate to real-world shifts, robustness should be measured on real-world shifts that are inherent to the VQA data; 2) Traditional token-matching metrics often fail to capture underlying semantics, necessitating the use of large language models (LLMs) for more accurate semantic evaluation; 3) Model performance often lacks interpretability due to missing sanity baselines, thus meaningful baselines should be reported that allow assessing the multimodal impact on the VLM. To demonstrate the relevance of this framework, we conduct a study on the robustness of various Parameter-Efficient Fine-Tuning (PEFT) methods across three medical datasets with fourdifferent types of distribution shifts. Our study reveals several important findings:
>1) Sanity baselines that do not utilize image data can perform surprisingly well
>2) We confirm LoRA as the best-performing PEFT method
>3) No PEFT method consistently outperforms others in terms of robustness to shifts.

<div align="center">
  <img width="70%" src="figures/Pitfalls_Requirements.png">
</div>

Pitfalls and Requirements for Systematic Evaluating the Robustness of VLMs in
VQA Tasks. We aim to overcome pitfalls (P1-P3) in the current evaluation of VLM robustness by
satisfying the three requirements (R1-R3): We define a diverse set of realistic shifts (R1). We use
appropriate metrics for evaluation by using an LLM as evaluator of the VLM output (R2). Finally,
we compare the results of the VLM with relevant sanity baselines to see the performance gains over
such baselines like e.g. considering the text of the question only (R3).

## Table of Contents

- [Setup](#setup)
- [Finetuning](#finetune)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

<a name="setup"></a>
## Setup
The code is tested with python version 3.10.
1) Clone this repository
2) Install the requirements 
```
pip install -r requirements.txt
```
3) Clone the LlaVA-Med v1.5 repository [here](https://github.com/microsoft/LLaVA-Med)
4) Download LlaVA-Med v1.5 weights [here](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)
5) Add the relevant paths to your `.env` file. An example of this file is provided under `med_vlm_robustness/example.env`. You need to mainly add the paths of your; 
    - Llava-Med v1.5 weights
    - Experiment root directory
    - Dataset root directory

<a name="finetune"></a>
## Fine-tuning
To run fine-tuning you need to execute the `train.py` python file in the repository. The configurations for fine-tuning can be found in the file `config/train/training_defaults.yaml`. This `.yaml` file already contains example configurations for you, however, you can change these settings easily for your own use case.

To run fine-tuning without using any images but only the question-answer pairs set the parameter `no_image` to `True`. Note that when you set this parameter to `True`, your fine-tuned model will have `no-image` tag in its name and if you want to run inference and evaluation on this model, you need to set `train_no_image` parameter in inference and evaluation config files to also `True`.

After fine-tuning, the results are stored in a folder named using the hyperparameters you specified. This folder name is your fine-tuned model name and it will have the following structure:

```
llava-SLAKE_train_iid_content_type_Size-finetune_lora_rank128_lr3e-5_seed123
├── init_weight
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── non_lora_trainables.bin
    ├── README.md
├── adapter_config.json
├── adapter_model.bin
├── config.json
├── non_lora_trainables.bin
├── README.md
├── trainer_state.json
```

<a name="inference"></a>
## Inference
To run inference you need to execute the `inference.py` python file in the repository. The configurations for inference can be found in the file `config/inference/inference_defaults.yaml`. This `.yaml` file is used for running inference after you fine-tune a model. If you want to run the inference on the pretrained (no fine-tune) model you ca nuse `config/inference/inference_pretrained_defaults.yaml` file. Both files already contain example configurations for you as in the fine-tuning case. Note that you can change these settings easily for your own use case. 

To run the inference for `no-image` baseline where the images are not utilized during inference and the model only uses question and answer pairs, set the parameter `no-image` to `True`. If you are running this baseline on a model which is also finetuned without images set the `train_no_image` parameter to also `True`. 

To run the inference with corrupted images where the dataloader corrupts the images during inference with the given probability and strength level, set the parameter `corruption` to `True` and specify the strength and probability of each corruption in the related parameters as follows;
```
corruption_probabilities: {
    'blur': 0.5,
    'brightness': 0.5,
    'noise': 0.5,
}
corruption_strength: {
    'blur':'low', 
    'brightness': 'low',
    'noise': 'low',
}
```
After inference, the results are stored in a subfolder called `eval` under your fine-tuned model folder, which has the following structure:

```
eval
├── <type_of_inference_you_run>
    ├── test_results.json
```

<a name="evaluation"></a>
## Evaluation
After evaluation, the results are stored in the eval subfolder, which will have the following updated structure:
```
eval
├── <type_of_inference_you_run>
    ├── closed_ended_metrics.json
    ├── mistral_metrics_closed.json
    ├── mistral_metrics.json
    ├── open_ended_metrics.json
    ├── test_results.json
```

<a name="acknowledgement"></a>
## Acknowledgement
<!-- To enable the gpt4 evaluation functionality in this repo you need to have an API key from Openai.

To get an API key please go to the following webpage: https://platform.openai.com/docs/overview

After getting a key, you need to add this key to the .env file in your repository. The file, *template.env*, is created for you as a guide which shows how your .env file should look like. You can copy the context of this file to your .env file or basically change the file name to .env by removing the template part. Note that the repository will not run properly if you don't set all the parameters spesified in this file.

You can add your API key to the following variable in the .env file:

```shell
export OPEN_AI_API_KEY= your_api_key
```

Once you have set your key you need to run the following before starting the program.

```shell
source .env 
``` -->

## Setting up OVQA Dataset
- change the train, val and test set names as test.json train.json validate.json
