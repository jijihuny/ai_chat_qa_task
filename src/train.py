from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
)
from transformers.hf_argparser import HfArgumentParser
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Self, Tuple
from base import Base
from arguments import Arguments, Config


class Trainer(Base):
    def __init__(self: Self, args: Config):
        super().__init__(args=args)

        self.prepare_model()
        self.prepare_data()
        self.prepare_metric()
        self._prepare_data_collator()
        self.prepare_trainer()

    def prepare_model(self: Self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model.path,
            torch_dtype=self.args.model.torch_dtype,
            device_map=self.args.model.device_map,
            attn_implementation=self.args.model.attn_implementation,
            quantization_config=self.args.train.quantization,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model.path, device_map=self.args.model.device_map
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.args.train.lora)

    def _prepare_data_collator(self: Self) -> DataCollatorForCompletionOnlyLM:
        self.data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            response_template=self.args.train.response_template,
        )
        return self.data_collator

    def prepare_trainer(self: Self):
        def formatting_func(dataset) -> list[str]:
            return dataset["text"]

        def compute_metrics(eval_pred: EvalPrediction):
            predictions = eval_pred.predictions
            label_ids = eval_pred.label_ids

            print(predictions, label_ids)

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args.train.args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            formatting_func=formatting_func,
            peft_config=self.args.train.lora,
            compute_metrics=compute_metrics,
        )

    def __call__(self: Self):
        try:
            self.trainer.train(resume_from_checkpoint=True)
        except:
            self.trainer.train()


import os
from os.path import join


def main():
    parser = HfArgumentParser([Arguments, Config])
    cwd = os.getcwd()
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    path = join(cwd, args.config, "config.yaml")
    pargs: Config = parser.parse_yaml_file(path)[1]

    trainer = Trainer(pargs)
    trainer()


if __name__ == "__main__":
    main()
