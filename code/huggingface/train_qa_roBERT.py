import os
import re
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from tokenizers.processors import TemplateProcessing
from trl import SFTConfig, SFTTrainer
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model,
    TaskType,
)
from datasets import load_dataset
from accelerate import Accelerator

repo = 'uomnf97/klue-roberta-finetuned-korquad-v2' #"CurtisJeon/klue-roberta-large-korquad_v1_qa"
tokenizer = AutoTokenizer.from_pretrained(repo)
dataset = load_dataset("csv", data_files="/home/jovyan/work/prj_data/open/train.csv")
max_length = 512
stride = 160

def preprocess_function(examples):
    questions, contexts, answers = examples["question"], examples["context"], examples["answer"]
    def preprocess_text(text):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    questions = list(map(preprocess_text, questions))
    contexts = list(map(preprocess_text, contexts))
    answers = list(map(preprocess_text, answers))
    
    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # offset_mapping: [(token1 start, token1 end), (token2 ~, ), ...]
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        # sequence_ids: (token=None, question=0, context=1)
        sequence_ids = inputs.sequence_ids(i)

        # 컨텍스트의 시작 및 마지막을 찾는다.
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        
        start_char = contexts[sample_idx].find(answer, offset[context_start][0], offset[context_end][1])
        end_char = start_char + len(answer)

        # 만일 정답이 컨텍스트에 완전히 포함되지 않는다면, 레이블은 (0, 0)임
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 그렇지 않으면 정답의 시작 및 마지막 인덱스
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
    
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 데이터 프레임을 전처리합니다
# preprocess_function(dataset["train"][15281])
train_dataset = dataset["train"].map(
    preprocess_function,
    batched = True,
    remove_columns=dataset["train"].column_names,
)

# roBERTa에서는 삭제, BERT에서는 중요한 역할
train_dataset = train_dataset.remove_columns("token_type_ids")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForQuestionAnswering.from_pretrained(
        repo,
        device_map="cuda:0",
        torch_dtype=torch.float32,
        # quantization_config=quantization_config,
)
print("Model Set Completed.")
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "key", "value"],
    task_type="QUESTION_ANSWERING"
)

# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

accelerater = Accelerator()
model, tokenizer = accelerater.prepare(model, tokenizer)

import wandb
wandb.login()

torch.cuda.empty_cache()
training_args = TrainingArguments(
    output_dir="roBERTa_v2_answer",
    evaluation_strategy="no",
    num_train_epochs=4,
    save_steps=0.1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()