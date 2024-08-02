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

        references = eval_sample["answer"]

        self.results = self.metric.compute(
            predictions=predictions, references=references
        )

        return self.results, {"predictions": predictions, "references": references}


import os
from os.path import join


def main():
    parser = HfArgumentParser(dataclass_types=[Arguments, Config])
    cwd = os.getcwd()
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    path = join(cwd, args.config, "config.yaml")
    pargs: Config = parser.parse_yaml_file(path)[1]
    evaluator = Evaluator(pargs)
    results, frame = evaluator()

    import yaml

    with open(join(cwd, args.config, "results.yaml"), "w") as output:
        yaml.dump(results, output)

    df = pd.DataFrame(
        list(zip(frame["predictions"], frame["references"])),
        columns=["predictions", "references"],
    )
    df.to_csv(join(cwd, args.config, "result.csv"))


if __name__ == "__main__":
    main()
