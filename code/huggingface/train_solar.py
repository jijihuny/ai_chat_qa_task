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
    DataCollatorForLanguageModeling, 
    DataCollatorWithPadding,
    PreTrainedTokenizerFast
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    TaskType,
    get_peft_model
)

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator

# 모델 정의

repo = 'LDCC/LDCC-SOLAR-10.7B'

# accelerater = Accelerator()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=quantization_config,
        device_map={"":0},
        torch_dtype="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=['up_proj', 
                    'down_proj', 
                    'gate_proj', 
                    'k_proj', 
                    'q_proj', 
                    'v_proj', 
                    'o_proj'],
    task_type=TaskType.CAUSAL_LM
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = PreTrainedTokenizerFast.from_pretrained(repo, eos_token='</s>')
tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False 

# 데이터셋 정의

train_dataset = load_dataset("csv", data_files="/home/jovyan/work/prj_data/open/train.csv")
max_length = 512

formatted_data = []
for data in tqdm(train_dataset['train']):
    c = data['context']
    q = data['question']
    a = data['answer']
    input_text = f"{tokenizer.bos_token}Context: {c}\nQuestion: {q}{tokenizer.eos_token}{a}"
    encoded_text = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    formatted_data.append(encoded_text)

formatted_data = torch.cat(formatted_data, dim=0)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess_function(examples):
    questions = examples['question']
    contexts = examples['context']
    answers = examples['answer']
    inputs = tokenizer(questions, contexts, truncation=True, padding='max_length', max_length=512)
    start_positions = []
    end_positions = []
    
    for i in range(len(answers)):
        answer = answers[i]
        context = contexts[i]
        start_idx = context.find(answer)
        end_idx = start_idx + len(answer)
        
        start_positions.append(start_idx)
        end_positions.append(end_idx)
    
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

# 데이터셋 전처리
# tokenized_datasets = train_dataset.map(preprocess_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer)

# Train

import wandb
wandb.login()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 훈련 설정
training_args = TrainingArguments(
    output_dir='solar',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    save_steps=0.2,
    weight_decay=0.01,
    optim="paged_adamw_8bit"
)

# Trainer 생성
trainer = Trainer(
    model=model,
    train_dataset=formatted_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 모델 훈련
model.config.use_cache = False 
trainer.train()