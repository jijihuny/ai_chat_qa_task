from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    DataCollatorForLanguageModeling,
)
from transformers.hf_argparser import HfArgumentParser
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DataCollatorForCompletionOnlyLM
from torch import Tensor, LongTensor
from torch.nn.functional import cross_entropy
import numpy as np
from typing import Self, Tuple
from base import Base
from arguments import Arguments, Config
from scheduler import CosineScheduleTrainer


class Trainer(Base):
    def __init__(self: Self, args: Config):
        super().__init__(args=args)

        self.prepare_model()
        self.prepare_data()
        self.prepare_metric()
        self._prepare_data_collator()
        self.prepare_trainer()

    def prepare_model(self: Self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.args.model.path,
            torch_dtype=self.args.model.torch_dtype,
            device_map=self.args.model.device_map,
            attn_implementation=self.args.model.attn_implementation,
            revision=self.args.model.revision,
            quantization_config=self.args.train.quantization,
        )
        self.model: PreTrainedModel = prepare_model_for_kbit_training(self.model)
        if self.args.train.peft_model_path:
            self.model: PreTrainedModel = PeftModel.from_pretrained(
                self.model,
                self.args.train.peft_model_path,
                is_trainable=True,
                revision=self.args.model.peft_revision,
            )
        else:
            self.model: PreTrainedModel = get_peft_model(
                self.model, self.args.train.lora
            )

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.args.model.path, device_map=self.args.model.device_map
        )

        return self.model, self.tokenizer

    def _prepare_data_collator(
        self: Self,
    ) -> DataCollatorForCompletionOnlyLM | DataCollatorForLanguageModeling:
        if self.args.train.use_completion_only_data_collator:
            self.data_collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template=self.args.train.response_template,
            )
        else:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        return self.data_collator

    def prepare_trainer(self: Self):
        formatter = self._chat_prompt_format_func()

        def formatting_func(examples) -> list[str]:
            return [
                formatter({"context": context, "question": question, "answer": answer})
                for context, question, answer in zip(
                    examples["context"], examples["question"], examples["answer"]
                )
            ]

        # ANSWER_REGEXP = "\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n\n(.+)\<\|eot_id\|\>"
        if self.metric:

            def compute_metrics(eval_pred: EvalPrediction):
                predictions, label_ids = eval_pred
                N, length = label_ids.shape
                # TODO memory issue, generation
                total_loss = cross_entropy(
                    Tensor(predictions).view(N, -1, length), LongTensor(label_ids)
                )
                if isinstance(predictions, np.ndarray):
                    predictions = [predictions[predictions >= 0].astype(np.int32)]
                elif isinstance(predictions, tuple):
                    predictions = [
                        pred[pred >= 0].astype(np.int32) for pred in predictions
                    ]

                if isinstance(label_ids, np.ndarray):
                    label_ids = [label_ids[label_ids >= 0].astype(np.int32)]
                elif isinstance(label_ids, tuple):
                    label_ids = [
                        label[label >= 0].astype(np.int32) for label in label_ids
                    ]

                predictions = self.tokenizer.batch_decode(predictions)
                references = self.tokenizer.batch_decode(label_ids)

                results = self.metric.compute(
                    predictions=predictions, references=references
                )

                return {"total_loss": total_loss, **results}

            def preprocess_logits_for_metrics(
                logits: Tensor, label_ids: Tensor
            ) -> Tensor:
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1)

        self.init_seed()

        self.trainer = CosineScheduleTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args.train.args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            formatting_func=formatting_func,
            peft_config=self.args.train.lora,
            # compute_metrics=compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def __call__(self: Self):
        try:
            self.trainer.train(resume_from_checkpoint=True)
        except:
            self.trainer.train()


import yaml
import os
from pathlib import Path


def main():
    parser = HfArgumentParser(dataclass_types=[Arguments, Config])
    args: Arguments = parser.parse_args_into_dataclasses()[0]

    cwd = os.getcwd()
    base = Path(cwd)
    config_path = base / args.config
    config_yaml = None
    with config_path.open("r") as input:
        config_yaml = yaml.load(input, Loader=yaml.FullLoader)

    config: Config = parser.parse_dict(config_yaml)[1]
    output_path = base / "train" / str(args.name)
    output_path.mkdir(exist_ok=True, parents=True)
    with (output_path / "config_yaml").open("w") as output:
        yaml.dump(config_yaml, output)

    config.train.args.output_dir = output_path
    trainer = Trainer(config)
    trainer()


if __name__ == "__main__":
    main()
