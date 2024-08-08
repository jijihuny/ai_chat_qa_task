from transformers.hf_argparser import HfArgumentParser
from tqdm import tqdm
from typing import Self, Dict
from omegaconf import OmegaConf
import pandas as pd
from base import Base
from arguments import Arguments, Config


class Evaluator(Base):
    def __init__(self: Self, args: Config):
        super().__init__(args=args)
        self.prepare_model()
        self.prepare_pipeline()
        self.prepare_data()
        self.prepare_metric()

    def __call__(self: Self):
        eval_sample = self.dataset["test"]
        predictions = []

        for example in tqdm(eval_sample["text"]):
            generated = self.generator(example, **self.args.generation)[0][
                "generated_text"
            ]
            predictions += [generated]

        if self.args.metric.only_inference:
            return list(zip(eval_sample['id'], predictions))
        else:
            references = eval_sample["answer"]

            self.results = self.metric.compute(
                predictions=predictions, references=references
            )

            return self.results, list(zip(eval_sample['id'], predictions, references))

import yaml
import os
from os.path import join
from pathlib import Path

def main():
    parser = HfArgumentParser(dataclass_types=[Arguments, Config])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    
    cwd = os.getcwd()
    base = Path(cwd)
    config_path = base / "config.yaml"
    config_yaml = None
    with config_path.open('r') as input:
        config_yaml = yaml.load(input)

    config: Config = parser.parse_dict(config_yaml)[1]
    evaluator = Evaluator(config)
    results, frame = evaluator()

    output_path = base / "eval" / str(args.name)
    output_path.mkdir(exist_ok=True)
    with (output_path / "result.yaml").open('w') as output:
        yaml.dump(results, output)
    with (output_path / "config_yaml").open('w') as output:
        yaml.dump(config_yaml, output)

    if config.metric.only_inference:
        columns=["id", "answer"]
    else:
        columns=["id", "pred", "label"]
    
    df = pd.DataFrame(
        frame,
        columns=columns
    )
    df.to_csv((output_path / "result.csv"))


if __name__ == "__main__":
    main()
