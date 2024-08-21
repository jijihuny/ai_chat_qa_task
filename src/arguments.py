from curses import noecho
from transformers.hf_argparser import HfArg, HfArgumentParser
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Arguments:
    name: str | None = HfArg(default="mymodel", aliases=["-n"])
    config: str = HfArg(default="config.yaml", aliases=["-c"])


@dataclass
class ModelConfig:
    task: str = HfArg(default="text-generation")
    system_prompt: str = HfArg(default="너는 유능한 챗봇이야")
    path: str = HfArg(default="meta-llama/Meta-Llama-3-8B")
    torch_dtype: str = HfArg(default="auto")
    device_map: str = HfArg(default="auto")
    attn_implementation: str | None = HfArg(default=None)
    revision: str | None = HfArg(default=None)
    peft_revision: str | None = HfArg(default=None)


class DataFiles(TypedDict):
    train: str
    test: str


@dataclass
class DatasetConfig:
    path: str | None = HfArg(default=None)
    name: str | None = HfArg(default=None)
    data_files: DataFiles | None = HfArg(default=None)
    shuffle: bool = HfArg(default=True)
    test_size: float | None = HfArg(default=0.9)
    include_answer: bool | None = HfArg(default=False)


@dataclass
class MetricConfig:
    path: str | None = HfArg(default=None)
    per_device_inference_batch_size: int = HfArg(default=2)
    only_inference: bool = HfArg(default=False)

@dataclass
class GenerationConfig:
    return_full_text: bool = HfArg(default=False)

    # parameters that define the output variables of generate
    num_return_sequences: int | None = HfArg(default=None)
    output_scores: bool | None = HfArg(default=False)
    output_logits: bool | None = HfArg(default=None)
    return_dict_in_generate: bool | None = HfArg(default=False)

    renormalize_logits: bool | None = HfArg(default=None)

    max_new_tokens: int | None = HfArg(default=None)

    do_sample: bool = HfArg(default=False, help="sampling 여부")
    top_k: int | None = HfArg(default=1, help="상위 K", metadata={"type": int})
    top_p: float | None = HfArg(
        default=0.95,
        help="smallest subset of vocabrary such that sum of total probilities >= p",
        metadata={"type": float},
    )
    temperature: float | None = HfArg(default=1.0, metadata={"type": float})

    penalty_alpha: float | None = HfArg(
        default=None, help="Degeneration penalty. 0.6", metadata={"type": float}
    )

    repetition_penalty: float | None = HfArg(
        default=None, help="repetition penalty. 1.2", metadata={"type": float}
    )

    dola_layers: str | None = HfArg(default=None, help="DoLa")

    num_beams: int | None = HfArg(default=None)
    num_beam_groups: int | None = HfArg(default=None)
    diversity_penalty: float | None = HfArg(default=None)
    length_penalty: float | None = HfArg(default=None)


@dataclass
class TrainConfig:
    peft_model_path: str | None = HfArg(default=None)
    instruction_template: str | None = HfArg(default=None)
    response_template: str | None = HfArg(default=None)
    use_completion_only_data_collator: bool = HfArg(default=True)
    quantization: BitsAndBytesConfig = HfArg(
        default_factory=lambda: BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    )
    lora: LoraConfig = HfArg(default_factory=lambda: LoraConfig())
    args: SFTConfig = HfArg(default_factory=lambda: SFTConfig(output_dir="output"))

    def __post_init__(self):
        self.quantization = (
            BitsAndBytesConfig(**self.quantization)
            if isinstance(self.quantization, dict)
            else self.quantization
        )
        self.lora = (
            LoraConfig(**self.lora) if isinstance(self.lora, dict) else self.lora
        )
        self.args = SFTConfig(**self.args) if isinstance(self.args, dict) else self.args


@dataclass
class EnsembleConfig:
    models: list[str] = HfArg(default_factory=lambda: [])
    weighted_voting: bool | None = HfArg(default=True)


@dataclass
class Config:
    """
    TODO
    """

    model: ModelConfig = HfArg(default_factory=ModelConfig)
    dataset: DatasetConfig = HfArg(default_factory=DatasetConfig)
    metric: MetricConfig = HfArg(default_factory=MetricConfig)
    generation: GenerationConfig = HfArg(default_factory=GenerationConfig)
    train: TrainConfig = HfArg(default_factory=TrainConfig)
    ensemble: EnsembleConfig = HfArg(default_factory=EnsembleConfig)
    seed: int = HfArg(default=42)

    def __post_init__(self):
        self.model = (
            ModelConfig(**self.model) if isinstance(self.model, dict) else self.model
        )
        self.dataset = (
            DatasetConfig(**self.dataset)
            if isinstance(self.dataset, dict)
            else self.dataset
        )
        self.metric = (
            MetricConfig(**self.metric)
            if isinstance(self.metric, dict)
            else self.metric
        )
        self.generation = (
            GenerationConfig(**self.generation)
            if isinstance(self.generation, dict)
            else self.generation
        )
        self.train = (
            TrainConfig(**self.train) if isinstance(self.train, dict) else self.train
        )
        self.ensemble = (
            EnsembleConfig(**self.ensemble)
            if isinstance(self.ensemble, dict)
            else self.ensemble
        )
