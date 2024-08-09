from transformers.hf_argparser import HfArgumentParser
from datasets import Dataset
from tqdm import tqdm
from typing import Self, Dict, Iterable
import pandas as pd
from base import Base
from arguments import Arguments, Config, GenerationConfig


class Evaluator(Base):
    def __init__(self: Self, args: Config):
        super().__init__(args=args)
        self.prepare_model()
        self.prepare_pipeline()
        self.prepare_data()
        self.prepare_metric()

    def __call__(self: Self):
        formatter = self._chat_prompt_format_func()
        eval_sample = self.dataset["test"].map(lambda x: {"text": formatter(x)})
        predictions = []

        if isinstance(self.args.generation, GenerationConfig):
            self.args.generation = self.args.generation.__dict__

        def batch(iterable: Dataset, size: int = 4):
            total_batches = (len(iterable) + size - 1) // size
            return tqdm(iterable.iter(size), total=total_batches, desc="evaluation..")

        for examples in batch(eval_sample, 4):
            generated = self.generator(examples["text"], **self.args.generation)
            predictions += [gen[0]["generated_text"] for gen in generated]

        if self.args.metric.only_inference:
            return {}, list(zip(eval_sample["id"], predictions))
        else:
            references = eval_sample["answer"]

            self.results = self.metric.compute(
                predictions=predictions, references=references
            )

            return self.results, list(zip(eval_sample["id"], predictions, references))


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
    with config_path.open("r") as input:
        config_yaml = yaml.load(input, Loader=yaml.FullLoader)

    config: Config = parser.parse_dict(config_yaml)[1]
    evaluator = Evaluator(config)

    results, frame = evaluator()

    output_path = base / "eval" / str(args.name)
    output_path.mkdir(exist_ok=True, parents=True)
    if config.metric.only_inference != True:
        with (output_path / "result.yaml").open("w") as output:
            yaml.dump(results, output)
    with (output_path / "config_yaml").open("w") as output:
        yaml.dump(config_yaml, output)

    if config.metric.only_inference:
        columns = ["id", "answer"]
    else:
        columns = ["id", "pred", "label"]

    df = pd.DataFrame(frame, columns=columns)
    df.to_csv((output_path / "result.csv"), index=False)


if __name__ == "__main__":
    main()
