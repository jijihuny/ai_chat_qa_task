from transformers.hf_argparser import HfArg, HfArgumentParser
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from dataclasses import dataclass


@dataclass
class Arguments:
    config: str | None = HfArg(default="experiment")


@dataclass
class ModelConfig:
    _argument_group_name: str = "model"
    task: str = HfArg(default="text-generation")
    system_prompt: str = HfArg(default="너는 유능한 챗봇이야")
    path: str = HfArg(default="meta-llama/Meta-Llama-3-8B")
    torch_dtype: str = HfArg(default="auto")
    device_map: str = HfArg(default="auto")
    attn_implementation: str | None = HfArg(default=None)


@dataclass
class DatasetConfig:
    path: str | None = HfArg(default=None)
    name: str | None = HfArg(default=None)
    shuffle: bool = HfArg(default=True)
    test_size: float | None = HfArg(default=0.9)


@dataclass
class MetricConfig:
    path: str | None = HfArg(default=None)


@dataclass
class GenerationConfig:
    return_full_text: bool = HfArg(default=False)
    max_new_token: int | None = HfArg(default=None)

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


@dataclass
class TrainConfig:
    instruction_template: str | None = HfArg(default=None)
    response_template: str | None = HfArg(default=None)
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
class Config:
    """
    TODO
    """

    model: ModelConfig = HfArg(default_factory=ModelConfig)
    dataset: DatasetConfig = HfArg(default_factory=DatasetConfig)
    metric: MetricConfig = HfArg(default_factory=MetricConfig)
    generation: GenerationConfig = HfArg(default_factory=GenerationConfig)
    train: TrainConfig = HfArg(default_factory=TrainConfig)
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
