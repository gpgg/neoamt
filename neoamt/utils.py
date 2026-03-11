import re
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_solution(solution_str: str):
    answer_pattern = r'<translation>(.*?)</translation>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()