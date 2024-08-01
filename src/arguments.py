from transformers.hf_argparser import (
    HfArg
)
from dataclasses import dataclass, MISSING
from typing import (
    Callable
)


@dataclass
class Arguments:
    """
    TODO
    """
    task: str = HfArg(default="text-generation")
    model: str = HfArg(default="meta-llama/Meta-Llama-3-8B")
    torch_dtype: str = HfArg(default='auto')
    device_map: str = HfArg(default='auto')

    dataset: str | None = HfArg(default=None)
    dataset_name: str | None = HfArg(default=None)
    shuffle: bool = HfArg(default=True)
    test_size: float = HfArg(default=0.9)
    seed: int = HfArg(default=42)

    metric: str = HfArg(default=None)

    max_new_token: int = HfArg(default=None)

    do_sample: bool = HfArg(default=False, help='sampling 여부')
    top_k: int = HfArg(default=1, help='상위 K', metadata={'type': int})
    top_p: float = HfArg(default=0.95, help='smallest subset of vocabrary such that sum of total probilities >= p', metadata={'type': float})
    temperature: float = HfArg(default=1.0, metadata={'type': float})

    penalty_alpha: float = HfArg(default=None, help='Degeneration penalty. 0.6', metadata={'type': float})
    repetition_penalty: float = HfArg(default=None, help='repetition penalty. 1.2', metadata={'type': float})

    dora_layers: str = HfArg(default=None, help='DoLa')

    system_prompt: str = HfArg(default='너는 유능한 챗봇이야')