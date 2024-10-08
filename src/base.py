from transformers import (
    set_seed,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)
from datasets import load_dataset, Dataset
from evaluate import load, Metric
from typing import Self, Tuple, Callable, TypedDict
from arguments import Config


class Example(TypedDict):
    context: str
    question: str
    answer: str


class Base:
    r"""
    base class for inference and training
    """

    def __init__(self: Self, args: Config):
        self.args = args

    def init_seed(self: Self):
        if self.args.seed is not None:
            self.seed = self.args.seed
        else:
            self.seed = 0
        set_seed(self.seed)

    def prepare_model(self: Self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.args.model.path,
            torch_dtype=self.args.model.torch_dtype,
            device_map=self.args.model.device_map,
            attn_implementation=self.args.model.attn_implementation,
            revision=self.args.model.revision,
            adapter_kwargs={"revision": self.args.model.peft_revision},
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.args.model.path, device_map=self.args.model.device_map
        )

        return self.model, self.tokenizer

    def prepare_pipeline(self: Self) -> TextGenerationPipeline:
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            self.prepare_model()
        self.generator: TextGenerationPipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.generator

    def prepare_data(self: Self) -> Dataset:
        if self.args.dataset.data_files != None:
            train = str(self.args.dataset.data_files.get("train"))
            test = str(self.args.dataset.data_files.get("test"))
            assert (train.endswith(".csv") and test.endswith(".csv")) or (
                not train.endswith(".csv") and not test.endswith(".csv")
            ), ValueError(f"see {train, test} format")

            self.dataset = load_dataset(
                "csv", data_files=self.args.dataset.data_files, num_proc=1
            )
        elif self.args.dataset.path.endswith(".csv"):
            self.dataset = load_dataset(
                "csv", data_files={self.args.dataset.name: self.args.dataset.path}
            )
        else:
            self.dataset = load_dataset(self.args.dataset.path, self.args.dataset.name)

        if self.args.dataset.test_size:
            self.dataset = self.dataset["train"].train_test_split(
                test_size=self.args.dataset.test_size,
                shuffle=self.args.dataset.shuffle,
                seed=self.args.seed,
            )
        elif self.args.dataset.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.args.seed)

        return self.dataset

    def prepare_metric(self: Self) -> Metric:
        if self.args.metric.path:
            self.metric = load(path=self.args.metric.path)
        else:
            self.metric = None
        return self.metric

    def _chat_prompt_format_func(self: Self) -> Callable[[Example], str]:
        if not hasattr(self, "user_message_template"):
            self.user_message_template = """Context: {context}\nQuestion: {question}"""

        def formatter(example: Example) -> str:
            conversation = [
                {"role": "system", "content": self.args.model.system_prompt},
                {
                    "role": "user",
                    "content": self.user_message_template.format(
                        context=example["context"], question=example["question"]
                    ),
                },
            ]
            if self.args.dataset.include_answer:
                conversation += [{"role": "assistant", "content": example["answer"]}]
            return self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=(
                    False if self.args.dataset.include_answer else True
                ),
            )

        return formatter
