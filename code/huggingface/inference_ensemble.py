import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    get_peft_model,
    PeftModelForCausalLM,
)

from accelerate import Accelerator

TEST_fOLDER = '/home/jovyan/work/prj_data/open/test.csv'
MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
CHECK_POINT = "/home/jovyan/work/ai_chat_qa_task/code/huggingface/llama3/checkpoint-16858"
OUTPUT = "ensemble"

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

def get_prompt(data, guide):
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

너는 주어진 Context에서 Question에 대한 Answer를 찾는 챗봇이야. 힌트는 '{guide}'야. 이 부분들을 참고해서 Answer를 Context에 있는 표현 그대로 답변해줘.

예를 들어 다음과 같이 답변하면 돼. 
질문1: 어떤 분야의 기술 개발을 돕기 위해 4군데의 과기원이 모였어
답변1: 첨단소재, 공정장비, 바이오·헬스케어, 정보통신기술(ICT)·소프트웨어(SW), 기계항공, 제조 자동화·지능화 등

질문2: 해양조사로 도출된 내용이 활용되는 곳은 어디야
답변2: 조사를 통해 나온 정보는 바다 보전, 이용, 개발뿐 아니라 선박 교통안전, 해양관할권 확보 등에 광범위하게 활용

질문3: 참존에 따르면 톤업핏 마스크의 모델로 뽑힌 연예인은 누구야
답변3: 비

질문4: 종신보험 리모델링을 권장하는 보험사들은 왜 그런 거야
답변4: 판매수수료 증대
<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {data['context']}
Question: {data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

submission_dict = {}
guide_path = [
    "/home/jovyan/work/ai_chat_qa_task/code/submit/test_91.csv",
    "/home/jovyan/work/ai_chat_qa_task/code/submit/beam_089.csv",
    "/home/jovyan/work/ai_chat_qa_task/code/submit/best_0906.csv"
]
guide_table = []
for i in range(len(guide_path)):
    guide_table.append(pd.read_csv(guide_path[i]))

for i, data in tqdm(csv.iterrows()):
    refer = []
    for j in range(len(guide_table)):
        table = guide_table[j]
        guide = table[table['id'] == data['id']]['answer'].values[0]
        refer.append(guide)
    # print(refer)
    prompt = get_prompt(data, refer)
    generated = pipe(prompt, num_return_sequences=1, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    generated = generated[len(prompt):]
    submission_dict[data['id']] = generated
    print(f"ID: {data['id']} Question: {data['question']} Generated answer: {generated}")
    
df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
df.to_csv(f'{OUTPUT}.csv', index=False)