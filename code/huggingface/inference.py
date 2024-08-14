import os
import json
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    TaskType,
    get_peft_model,
    PeftModelForCausalLM,
)

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator

TEST_fOLDER = '/home/jovyan/work/prj_data/open/test.csv'
MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
CHECK_POINT = "llama3/checkpoint-16858/"
OUTPUT = "test"

csv = pd.read_csv(TEST_fOLDER)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = PeftModelForCausalLM.from_pretrained(base_model, CHECK_POINT)

accelerater = Accelerator()
model, tokenizer = accelerater.prepare(model, tokenizer)

model.merge_and_unload()
pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, device=0)
print(pipe.device)

def get_prompt(data):
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

너는 주어진 Context에서 Question에 대한 Answer를 찾는 챗봇이야. Answer를 간결하게 문장이 아니라 정확한 짧은 표현으로 말해줘.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {data['context']}
Question: {data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

submission_dict = {}

for i, data in tqdm(csv.iterrows()):
    prompt = get_prompt(data)
    generated = pipe(prompt, num_return_sequences=1, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    generated = generated[len(prompt):]
    submission_dict[data['id']] = generated
    print(f"ID: {data['id']} Question: {data['question']} Generated answer: {generated}")
    
df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
df.to_csv(f'{OUTPUT}.csv', index=False)