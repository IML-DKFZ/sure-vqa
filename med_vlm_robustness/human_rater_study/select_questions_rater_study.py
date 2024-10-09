import json
from pathlib import Path
import random

import pandas as pd

from med_vlm_robustness.utils import set_seed, get_config


def select_questions(cfg):
    set_seed(cfg.seed)
    dataset = cfg.dataset
    answer_type = cfg.answer_type
    train_set = cfg.train_set
    val_set = cfg.val_set
    num_questions = cfg.num_questions
    balanced = getattr(cfg, "balanced", False)

    scores_path = Path(
        f"{cfg.experiment_root_dir}/{dataset}/{cfg.finetune_method}/llava-{dataset}_train_{train_set}-finetune_{cfg.finetune_method}_{cfg.hyperparam_str}/eval/{dataset}_val_{val_set}/mistral_metrics.json")
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    scores_df = pd.DataFrame(scores[1:]).set_index("qid")
    scores_df_non_equal = scores_df.loc[scores_df["gt"] != scores_df["pred"]]
    scores_df_non_equal['mistral_score'] = pd.to_numeric(scores_df_non_equal['mistral_score'], errors='coerce')
    scores_df_non_equal = scores_df_non_equal.dropna(subset=['mistral_score'])
    num_ratings = []
    for score in range(1, 6):
        num_ratings.append(len(scores_df_non_equal.loc[scores_df_non_equal["mistral_score"] == score]))
    print(num_ratings)
    json_file_path = cfg.dataset_dir / "validate.json"
    output_file_path = cfg.output_dir / f"rater_study_{answer_type.lower()}.json"

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    data = [element for element in data if element.get("answer_type") == answer_type]
    data = [element for element in data if element.get("qid") in list(scores_df_non_equal.index)]
    selected_questions_id = []
    if balanced:
        for score in range(1, 6):
            num_samples = int(num_questions / 5) if num_ratings[score-1] > num_questions / 5 else num_ratings[score-1]
            random_qids = scores_df_non_equal.loc[(scores_df_non_equal["mistral_score"] == score)].sample(n=num_samples).index.values
            selected_questions_id.extend(random_qids)
    else:
        selected_questions = scores_df_non_equal.sample(n=num_questions)
        selected_questions_id = selected_questions.index.values
        num_ratings = []
        for score in range(1, 6):
            num_ratings.append(len(selected_questions.loc[selected_questions["mistral_score"] == score]))
        print(num_ratings)
    selected_questions = [element for element in data if element.get("qid") in selected_questions_id]
    random.shuffle(selected_questions)
    print(f"Number of selected questions: {len(selected_questions)}")
    with open(output_file_path, 'w') as f:
        json.dump(selected_questions, f, indent=4)


def main(cfg):
    cfg.data_root_dir = Path(cfg.data_root_dir)
    cfg.dataset_dir = cfg.data_root_dir / cfg.dataset
    if cfg.output_dir is None:
        cfg.output_dir = cfg.dataset_dir
    else:
        cfg.output_dir = Path(cfg.output_dir)
    select_questions(cfg)


if __name__=="__main__":
    config = get_config()
    main(cfg=config)