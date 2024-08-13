from transformers.hf_argparser import HfArgumentParser, HfArg
from typing import Self
from pathlib import Path
from dataclasses import dataclass
import yaml
import re
import string
from collections import Counter
from arguments import Config

def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    def remove_punc(text):
        '''구두점 제거'''
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        '''소문자 전환'''
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # 문자 단위로 f1-score를 계산 합니다.
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

@dataclass
class TextsAndScores:
    generated_texts: list[str]
    scores: list[float]

    def __post_init__(self: Self):
        assert len(self.generated_texts) == len(self.scores)
        assert abs(sum(self.scores) - 1.0) < 1e-5 # softmax
count = 0

class Candidates:
    def __init__(self: Self, candidates: dict | TextsAndScores | None, weight: float | None):
        self.__table__ = {key: None for key in string.punctuation}
        self.__candidates__ = {}

        if candidates:
            self.add_candidates(candidates=candidates, weight=weight)
    
    def __getitem__(self: Self, key: str):
        if not (key in self):
            return None
        return self.__candidates__[key]
        
    def __contains__(self: Self, key: str):
        return key.translate(self.__table__) in self.__candidates__.keys()

    def add_candidates(self: Self, candidates: dict | TextsAndScores, weight: float | None):
        if not isinstance(candidates, TextsAndScores):
            candidates = TextsAndScores(**candidates)
        if not isinstance(weight, float):
            weight = 1.0
        
        for text, score in zip(candidates.generated_texts, candidates.scores):
            text = text.translate(self.__table__)
            if text in self.__candidates__.keys():
                self.__candidates__[text] += score * float(weight)
            else:
                self.__candidates__[text] = score * float(weight)

    def get_best_candidate(self: Self)->str:
        return max(self.__candidates__, key=self.__candidates__.get)

    def get_best_candidate_using_similarity(self: Self)->str:
        table = {}
        # O(n^2)
        global count
        for current_word, current_score in self.__candidates__.items():
            score = float(current_score)
            for target_word, target_score in self.__candidates__.items():
                count += 1
                distance = f1_score(current_word, target_word)
                score += float(distance * target_score)

            table[current_word] = score
        return max(table, key=table.get)

class Ensembler:
    def __init__(self: Self, config: Config, base: str | Path):
        self.config = config.ensemble
        self.base = base
        self.models = []
        self.candidates: dict[str, Candidates] = {}

        for model in self.config.models:
            self.models += [[self.base / "eval" / model['name'], model.get('weight')]]

        for model in self.models:
            self.__load_yaml(model)

    def __load_yaml(self: Self, model: list[tuple[str | Path, float]]):
        if not hasattr(self, 'candidates'):
            self.candidates = {}
        path, weight = model
        with (path / "candidates.yaml").open('r') as input:
            table = yaml.load(input, yaml.FullLoader)
            for row in table:
                if not row['id'] in self.candidates.keys():
                    self.candidates[row['id']] = Candidates(row['candidates'], weight=weight)
                else:
                    self.candidates[row['id']].add_candidates(row['candidates'], weight=weight)

    def get_best_candidates(self: Self):
        self.results = {}
        for id, candidates in self.candidates.items():
            self.results[id] = candidates.get_best_candidate_using_similarity()
        return self.results
    
    def __call__(self: Self):
        return self.get_best_candidates()

import os   
import pandas as pd

def main():
    base_path = os.getcwd()
    base = Path(base_path)

    parser = HfArgumentParser(Config)
    config = parser.parse_yaml_file(str(base / "config.yaml"))[0]

    ensembler = Ensembler(config, base)

    results = ensembler()
    print(count)
    df = pd.DataFrame(results.items(), columns=['id', 'answer'])
    name = "-".join(["ensemble", *(model['name'] for model in config.ensemble.models)]) + ".csv"
    df.to_csv(name, index=False)

if __name__=='__main__':
    main()