from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed
)
from transformers.hf_argparser import HfArgumentParser
from datasets import load_dataset, Dataset
from evaluate import load, Metric
from tqdm import tqdm
from typing import (
    Tuple,
    Self,
    Callable,
    Dict
)
from arguments import Arguments, Preset

class Evaluator:
    def __init__(self: Self, args: Preset):
        if args.seed is not None:
            self.seed = args.seed
        else:
            self.seed = 0
        set_seed(self.seed)
        self.args = args

        self.prepare_model()
        self.prepare_pipeline()
        self.prepare_data()
        self.prepare_metric()


    def prepare_model(self: Self)-> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=self.args.torch_dtype,
            device_map=self.args.device_map
        )
        self.tokenizer: PreTrainedModel = AutoTokenizer.from_pretrained(
            self.args.model,
            device_map=self.args.device_map
        )

        return self.model, self.tokenizer

    def prepare_pipeline(
            self: Self
    )-> TextGenerationPipeline:
        if not(hasattr(self, 'model')) or not(hasattr(self, 'tokenizer')):
            self.prepare_model()
        self.generator: TextGenerationPipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return self.generator

    def prepare_data(self: Self)-> Dataset:
        self.dataset = load_dataset(
            self.args.dataset,
            self.args.dataset_name
        )
        
        if self.args.test_size is not None:
            self.dataset = self.dataset['train'].train_test_split(
                test_size=self.args.test_size,
                shuffle=self.args.shuffle,
                seed=self.args.seed
            )

        return self.dataset


    def prepare_metric(self: Self)-> Metric:
        self.metric = load(path=self.args.metric)
        return self.metric


    def __chat_prompt_format(self: Self, example: Dict[str, str])-> str:
        if not(hasattr(self, 'user_message_template')):
            self.user_message_template = """Context: {context}\nQuestion: {question}"""

        conversation = [
            { 'role': 'system', 'content': self.args.system_prompt },
            { 'role': 'user', 'content': self.user_message_template.format(context=example['context'], question=example['question']) }
        ]
        return self.tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=True)

    def __call__(self: Self):
        eval_sample = self.dataset['test'].map(lambda example: { "text": self.__chat_prompt_format(example) }, num_proc=4)
        predictions = []

        for example in tqdm(eval_sample['text']):
            generated = self.generator(
                example,
                return_full_text=False,
                do_sample=False
            )[:]['generated_text']
            predictions += generated

        self.results = self.metric.compute(
            predictions=predictions,
            references=eval_sample['answer']
        )

        return self.results
    

import os
from os.path import join

def main():
    parser = HfArgumentParser(dataclass_types=[Arguments, Preset])
    cwd = os.getcwd()
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    path = join(cwd, args.config, 'config.yaml')
    pargs: Preset = parser.parse_yaml_file(path)[1]

    evaluator = Evaluator(pargs)
    results = evaluator()

    import yaml
    with open(join(cwd, args.config, 'results.yaml'), 'w') as output:
        yaml.dump(results, output)

if __name__ == '__main__':
    main()