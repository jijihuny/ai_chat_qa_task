from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)

from transformers.hf_argparser import (
    HfArgumentParser,
    HfArg
)

from datasets import (
    load_dataset,
    Dataset
)

from evaluate import (
    load,
    Metric
)

from tqdm import tqdm

from typing import (
    Tuple,
    Union,
    Optional,
    List,
    Dict,
    Callable
)

from dataclasses import dataclass

@dataclass
class ModelArgs:
    base_model_id: str = HfArg(
        default='meta-llama/Meta-Llama-3-8B',
        aliases=['--model', '-m'],
        help='모델 이름 설정'
    )

def prepare_model(base_model_id: str = 'meta-llama/Meta-Llama-3-8B')-> Tuple[
    PreTrainedModel,
    PreTrainedTokenizer
]:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        device_map='auto'
    )

    return model, tokenizer

@dataclass
class GenerationStrategiesArgs:
    num_beams: int = HfArg(default=1, aliases=['--num_beams', '-b'], help='beam search')

    do_sample: bool = HfArg(default=False, aliases=['--do_sample', '-s'], help='stochastic')
    
    # https://arxiv.org/pdf/2202.06417
    penalty_alpha: float = HfArg(default=0.0, aliases=['--penalty_alpha', '-p'], help='contrastive penalty')

    top_k: int = HfArg(default=1, aliases=['--top_k', '-k'], help='selection')

    dora_layers: List[int] | str | None = HfArg(default=None, aliases=['--dora_layers'], help='dora layers. high, low, list[int]')

def prepare_pipeline(
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizer],
        generation_config: Optional[GenerationConfig] = None
)-> TextGenerationPipeline:
    return pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer
        (**generation_config if generation_config is not None else None)
    )

@dataclass
class DatasetArgs:
    path: str = HfArg(default='dataset', aliases=['--dataset', '-d'], help='dataset path')
    train_size: float | None = HfArg(default=None, aliases=['--train_size'], help='split')
    shuffle: bool | None = HfArg(default=None, aliases=['--shuffle'], help='shuffle dataset')
    seed: int | None = HfArg(default=None, aliases=['--seed', '-s'], help='dataset seed')

def prepare_data(
        path: str, 
        name: str | None = None,
        train_size: float | None = None,
        shuffle: bool | None = None,
        seed: int | None = None
    )-> Dataset:
    dataset = load_dataset(
        path,
        name
    )
    
    if train_size is not None:
        dataset = dataset.train_test_split(
            train_size=train_size,
            shuffle=shuffle,
            seed=seed
        )

    return dataset

@dataclass
class MetricArgs:
    metric: str = HfArg(default='metric', aliases=['--metric', '-m'], help='metric')

def prepare_metric(path: str)-> Metric:
    return load(path=path)

@dataclass
class InferenceArgs:
    system_prompt: str = HfArg(default='너는 친절한 챗봇이야', aliases=['--system_prompt'], help='system prompt')

def get_chat_prompt_func(tokenizer: PreTrainedTokenizer, prompt: str)-> Callable[[Dict[str, str]], str]:
    user_message_template = """Context: {context}\nQuestion: {question}"""
    def format(example: Dict[str, str]):
        conversation = [
            { 'role': 'system', 'content': prompt },
            { 'role': 'user', 'content': user_message_template.format(context=example['context'], question=example['question']) }
        ]
        return tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=True)

    return format

def main():
    parser = HfArgumentParser(dataclass_types=[ModelArgs, GenerationStrategiesArgs, DatasetArgs, MetricArgs, InferenceArgs])
    (
        model_args, 
        generation_args, 
        dataset_args, 
        metric_args,
        infer_args
    ) = parser.parse_args_into_dataclasses()

    model, tokenizer = prepare_model(**model_args)
    generator = prepare_pipeline(model, tokenizer, **generation_args)

    data = prepare_data(**dataset_args)
    metric = prepare_metric(path=metric_args.metric)

    formatter = get_chat_prompt_func(tokenizer=tokenizer, prompt=infer_args.system_prompt)
    data = data.map(lambda x : { 'text': formatter(x) }, num_proc=4)
    predictions = []

    for i, example in tqdm(data['test']):
        generated = generator(example['text'], return_full_text=False)[0]['generated_text']
        predictions += [ generated ]

    results = metric.compute(
        predictions=predictions,
        references=data['test']['answer']
    )

    print(results)

if __name__ == '__main__':
    main()