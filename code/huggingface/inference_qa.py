import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

CHECK_POINT = "/home/jovyan/work/ai_chat_qa_task/code/huggingface/roBERTa_v2/checkpoint-23972"
TEST_fOLDER = '/home/jovyan/work/prj_data/open/test.csv'
OUTPUT = "roBERTa_v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
csv = pd.read_csv(TEST_fOLDER)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 모델 및 토크나이저 로드
config = PeftConfig.from_pretrained(CHECK_POINT)
model = AutoModelForQuestionAnswering.from_pretrained(
    config.base_model_name_or_path,
    # quantization_config=quantization_config,
    device_map="cuda:0",
    torch_dtype=torch.float16
)
# model = PeftModel.from_pretrained(model, CHECK_POINT)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

TEST_fOLDER = '/home/jovyan/work/prj_data/open/test.csv'
csv = pd.read_csv(TEST_fOLDER)
idx = 5

def get_prediction(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        return_tensors="pt",
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    inputs.pop("token_type_ids")
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    
    with torch.no_grad():
        outputs = model(**inputs)

    # 모든 청크에 대한 start/end 로짓 가져오기
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # 가장 높은 점수의 답변 찾기
    max_answer_score = -float('inf')
    best_answer = ""

    for i in range(len(start_logits)):
        start_indexes = torch.argsort(start_logits[i], descending=True)[:20]
        end_indexes = torch.argsort(end_logits[i], descending=True)[:20]
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 답변의 길이를 50 토큰으로 제한
                if end_index < start_index or end_index - start_index + 1 > 50 or end_index - start_index <= 1:
                    continue
                # 답변이 CLS일때는 제외
                if start_index==0 or end_index==0:
                    continue
                
                answer_score = start_logits[i][start_index] + end_logits[i][end_index]
                if answer_score > max_answer_score:
                    max_answer_score = answer_score
                    best_answer = tokenizer.decode(inputs["input_ids"][i][start_index:end_index+1])
    return best_answer
    
submission_dict = {}
for _, row in tqdm(csv.iterrows(), total=len(csv)):
    answer = get_prediction(row['question'], row['context'])
    
    def clean_prediction(text):
        special_tokens = list(tokenizer.special_tokens_map.values())
        pattern = '|'.join(map(re.escape, special_tokens))
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    answer = clean_prediction(answer)
    submission_dict[row['id']] = answer
    print(f"ID: {row['id']} Question: {row['question']} Generated answer: {answer}")
df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
df.to_csv(f'{OUTPUT}.csv', index=False)