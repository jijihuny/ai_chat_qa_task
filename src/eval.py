from transformers.hf_argparser import HfArgumentParser
from datasets import Dataset
from tqdm import tqdm
from typing import Self, Dict
import pandas as pd
from functools import partial
from base import Base
from arguments import Arguments, Config, GenerationConfig
from beam_generation import get_beam_search_sequences


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

        def batch(iterable: Dataset, size: int = 4):
            total_batches = (len(iterable) + size - 1) // size
            return tqdm(iterable.iter(size), total=total_batches, desc="evaluation..")

        generation_kwargs = None
        if isinstance(self.args.generation, GenerationConfig):
            generation_kwargs = self.args.generation.__dict__
        else:
            generation_kwargs = self.args.generation

        beam_search = None
        if (
            self.args.generation.num_beams
            and isinstance(self.args.generation.num_return_sequences, int)
            and self.args.generation.num_return_sequences > 1
        ):
            beam_search = True
            get_sequences = partial(
                get_beam_search_sequences, model=self.model, tokenizer=self.tokenizer
            )

            for examples in batch(
                eval_sample, self.args.train.args.per_device_eval_batch_size
            ):
                generated = get_sequences(inputs=examples["text"], **generation_kwargs)
                predictions += generated

        else:
            for examples in batch(
                eval_sample, self.args.train.args.per_device_eval_batch_size
            ):
                generated = self.generator(examples["text"], **generation_kwargs)
                predictions += [gen[0]["generated_text"] for gen in generated]

        if self.args.metric.only_inference:
            return {}, list(zip(eval_sample["id"], predictions))
        else:
            references = eval_sample["answer"]
            if beam_search:
                return {}, list(zip(eval_sample["id"], references, predictions))

            self.results = self.metric.compute(
                predictions=predictions, references=references
            )
            return self.results, list(zip(eval_sample["id"], references, predictions))


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
    if config.metric.only_inference != True and results:
        with (output_path / "result.yaml").open("w") as output:
            yaml.dump(results, output)
    with (output_path / "config.yaml").open("w") as output:
        yaml.dump(config_yaml, output, allow_unicode=True)

    if (
        config.generation.num_beams
        and isinstance(config.generation.num_return_sequences, int)
        and config.generation.num_return_sequences > 1
    ):
        with (output_path / "candidates.yaml").open("w") as output:
            obj = [
                {
                    "id": id,
                    "candidates": {
                        "generated_texts": candidates["generated_texts"],
                        "scores": candidates["scores"].tolist(),
                    },
                }
                for id, candidates in frame
            ]
            yaml.dump(obj, output, allow_unicode=True)
            return
    elif config.metric.only_inference:
        columns = ["id", "answer"]
    else:
        columns = ["id", "pred", "label"]

    df = pd.DataFrame(frame, columns=columns)
    df.to_csv((output_path / "result.csv"), index=False)


if __name__ == "__main__":
    main()
