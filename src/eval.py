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

def prepare_model(args: Arguments)-> Tuple[
    PreTrainedModel,
    PreTrainedTokenizer
]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        device_map=args.device_map
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
        args: Arguments
    )-> Dataset:
    dataset = load_dataset(
        args.dataset,
        args.dataset_name
    )
    
    if args.test_size is not None:
        dataset = dataset.train_test_split(
            test_size=args.test_size,
            shuffle=args.shuffle,
            seed=args.seed
        )

    return dataset


def prepare_metric(args: Arguments)-> Metric:
    return load(path=args.metric)


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
    args: Arguments = parser.parse_yaml_file('config.yaml')[0]

    model, tokenizer = prepare_model(args)
    args.model = model
    generator = prepare_pipeline(args)
    data = prepare_data(args)
    metric = prepare_metric(args)

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