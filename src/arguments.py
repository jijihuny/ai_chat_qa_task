from transformers.hf_argparser import (
    HfArg
)
from dataclasses import dataclass, MISSING
from omegaconf import OmegaConf
@dataclass
class Arguments:
    config: str | None = HfArg(default='experiment')

@dataclass
class ModelPreset:
    task: str = HfArg(default="text-generation")
    system_prompt: str = HfArg(default='너는 유능한 챗봇이야')
    path: str = HfArg(default="meta-llama/Meta-Llama-3-8B")
    torch_dtype: str = HfArg(default='auto')
    device_map: str = HfArg(default='auto')
    attn_implementation: str | None = HfArg(default=None)


@dataclass
class DatasetPreset:
    path: str | None = HfArg(default=None)
    name: str | None = HfArg(default=None)
    shuffle: bool = HfArg(default=True)
    test_size: float = HfArg(default=0.9)

@dataclass
class MetricPreset:
    path: str | None = HfArg(default=None)

@dataclass
class GenerationPreset:
    return_full_text: bool = HfArg(default=False)
    max_new_token: int | None = HfArg(default=None)

    do_sample: bool = HfArg(default=False, help='sampling 여부')
    top_k: int = HfArg(default=1, help='상위 K', metadata={'type': int})
    top_p: float = HfArg(default=0.95, help='smallest subset of vocabrary such that sum of total probilities >= p', metadata={'type': float})
    temperature: float | None = HfArg(default=1.0, metadata={'type': float})
    
    penalty_alpha: float | None = HfArg(default=None, help='Degeneration penalty. 0.6', metadata={'type': float})
    
    repetition_penalty: float | None = HfArg(default=None, help='repetition penalty. 1.2', metadata={'type': float})
    
    dola_layers: str | None = HfArg(default=None, help='DoLa')
    

@dataclass
class Preset:
    """
    TODO
    """
    model: ModelPreset = HfArg(default_factory=lambda : ModelPreset())
    dataset: DatasetPreset = HfArg(default_factory=lambda : DatasetPreset())
    metric: MetricPreset = HfArg(default_factory=lambda : MetricPreset())
    generation: GenerationPreset = HfArg(default_factory=lambda : GenerationPreset())

    seed: int = HfArg(default=42)