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

resume = False
repo = 'charlieCs/Open-Solar-ko-10B-dacon-qa'

data_path = "/home/jovyan/work/prj_data/open/train.csv"
# max_length = 1280 # max_token

#################
# 데이터셋 정의 #
#################

tokenizer = AutoTokenizer.from_pretrained(repo)
dataset = load_dataset("csv", data_files=data_path)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A <s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
        ("</s>", tokenizer.convert_tokens_to_ids("</s>"))
    ],
)

def preprocess_function(examples):
    question, context, answer = examples["question"], examples["context"], examples["answer"]
    def preprocess_text(text):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    question = preprocess_text(question)
    context = preprocess_text(context)
    answer = preprocess_text(answer)
    
    inputs = tokenizer(
        question,
        context,
        return_offsets_mapping=True,
        truncation=False,
        # truncation=True
        # max_length=max_length, 
        # padding="max_length",
    )

    start_char = context.find(answer)
    end_char = start_char + len(answer)

    # offset_mapping: [(token1 start, token1 end), (token2 ~, ), ...]
    offset= inputs.pop("offset_mapping")
    
    # sequence_ids: (token=None, question=0, context=1)
    sequence_ids = inputs.sequence_ids(0)

    # 컨텍스트의 시작 및 마지막을 찾는다.
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # 만일 정답이 컨텍스트에 완전히 포함되지 않는다면, 레이블은 (0, 0)임
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_position = 0
        end_position = 0
    else:
        # 그렇지 않으면 정답의 시작 및 마지막 인덱스
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_position = idx - 1

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_position = idx + 1

    inputs["start_positions"] = start_position
    inputs["end_positions"] = end_position
    return inputs

# 데이터 프레임을 전처리합니다
train_dataset = dataset["train"].map(preprocess_function)
train_dataset = train_dataset.remove_columns(['id', 'context', 'question', 'answer'])

#############
# 모델 정의 #
#############

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
print("start")
model = AutoModelForQuestionAnswering.from_pretrained(
        repo,
        quantization_config=quantization_config,
        device_map={"":0},
        torch_dtype="auto",
)
print("end")

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
    task_type="QUESTION_ANSWERING"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

accelerater = Accelerator()
model, tokenizer = accelerater.prepare(model, tokenizer)

########
# 학습 #
########

import wandb
wandb.login()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

training_args = TrainingArguments(
    output_dir="test",
    num_train_epochs=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=0.1,
    dataloader_num_workers=4 
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

if resume:
    trainer.train(resume_from_checkpoint=repo)
else:
    trainer.train()