from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)

from transformers.hf_argparser import (
    HfArgumentParser
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
    List,
    Dict,
    Callable
)

from arguments import Arguments

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


def prepare_pipeline(
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizer]
)-> TextGenerationPipeline:
    return pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer
    )


def prepare_data(
        path: str, 
        name: str | None = None,
        test_size: float | None = None,
        shuffle: bool | None = None,
        seed: int | None = None
    )-> Dataset:
    dataset = load_dataset(
        path,
        name
    )
    
    if test_size is not None:
        dataset = dataset.train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )

    return dataset


def prepare_metric(path: str)-> Metric:
    return load(path=path)


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
    parser = HfArgumentParser(dataclass_types=[Arguments])
    args: Arguments = parser.parse_yaml_file('config.yaml')

    model, tokenizer = prepare_model(args.model)
    generator = prepare_pipeline(model, tokenizer, )
    data = prepare_data(
        path=args.dataset,
        name=args.dataset_name,
        train_size=args.test_size,
        shuffle=args.shuffle,
        seed=args.seed
    )
    metric = prepare_metric(path=args.metric)

    formatter = get_chat_prompt_func(tokenizer=tokenizer, prompt=args.system_prompt)
    data = data.map(lambda x : { 'text': formatter(x) }, num_proc=4)
    predictions = []

    for i, example in tqdm(data['test']):
        generated = generator(
            example['text'], 
            return_full_text=False,
            generate_kwargs=args
        )[0]['generated_text']
        predictions += [ generated ]

    results = metric.compute(
        predictions=predictions,
        references=data['test']['answer']
    )

    import time
    import json
    with open(f'result_{args.model}_{time.time()}.json', 'w') as output:
        json.dump(results, output, indent=4)

if __name__ == '__main__':
    main()