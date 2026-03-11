import os
import re
import json
import argparse
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM

from neoamt.utils import load_jsonl
from neoamt.utils import extract_solution
load_dotenv()  # take environment variables

client = OpenAI()


def extract_evaluation(response: str):
    pattern = r"<evaluation>(.*?)</evaluation>"
    # file all matches
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # return the last match
    else:
        return None


PROMPT_TEMP_POINT_WISE = """You are an expert in evaluating the quality of translations.
You will be given a source sentence, a reference translation, and a candidate translation.
The source sentence contains a neologism (a newly coined word or expression).
Your task is to determine how well the candidate translation captures the meaning of the source sentence, especially focusing on the neologism.
Please consider the following criteria when conducting your evaluation:
1. Neologism Quality (score: 0-50).
2. Overall Translation Quality (score: 0-50).

After evaluating the candidate translation based on the above criteria, please provide your assessment in the following format:
<evaluation> score </evaluation>. 
The final 'score' is a numerical value between 0 and 100. A higher score indicates a better translation.

Here is the information you will need for your evaluation:
Source Sentence: {source_sentence}
Neologism and Its Meaning: {neologism} ({neologism_meaning})
Reference Translation: {reference_translation}
Candidate Translation: {candidate_translation}"""

PROMPT_TEMP_LIST_WISE = """You are an expert in evaluating the quality of translations.
You will be given a source sentence, a reference translation, and three candidate translations (A, B and C).
The source sentence contains a neologism (a newly coined word or expression).
Your task is to determine which candidate translation captures the meaning of the source sentence, especially focusing on the neologism, most accurately and naturally.

Please consider the following criteria when conducting your evaluation:
1. Neologism Quality (score: 0-50).
2. Overall Translation Quality (score: 0-50).

After evaluating the candidates based on the above criteria, please provide your assessment in the following format:
<evaludation> A > B > C </evaludation> if you think A is the best, followed by B and then C.
<evaludation> B > A > C </evaludation> if you think B is the best, followed by A and then C.
<evaludation> C > A > B </evaludation> if you think C is the best, followed by A and then B.
Here is the information you will need for your evaluation:
Source Sentence: {source_sentence}
Neologism and Its Meaning: {neologism} ({neologism_meaning})
Reference Translation: {reference_translation}
Candidate A: {candidate_a}
Candidate B: {candidate_b}
Candidate C: {candidate_c}
"""


def transformers_gen(
    model,
    tokenizer,
    prompt,
    # enable_thinking=False,
    max_new_tokens=4096,
    temperature=0.2,
    top_p=0.95,
):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


def gen(
    prompt, 
    model_path=None, 
    tokenizer=None, 
    model=None,
    max_new_tokens=4096,
    temperature=0.2,
    top_p=0.95,
):
    print("Generating with model:", model_path)
    response = None
    if model_path.startswith("/lustre/miao/models"):
        response = transformers_gen(
            model, 
            tokenizer, 
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    elif model_path.startswith("gpt-5-"):
        response = client.responses.create(
            model=model_path,
            input=prompt,
        )
        response = response.output_text
    return response


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", default="eval/model_gen/TowerInstruct-7B-v0.2_gen_2025-10-09-04-15.raw_scores.json", type=str)
    parser.add_argument("--raw_test_set_path", default="data/retrieval_data/wikidict/output/enwikidict_20250823/neo/trans_pair_w_alignment_no_empty_line.jsonl", type=str)
    
    parser.add_argument("--test_file_path", default="data/wiktionary/think_search_translate/neologism2plain/test.jsonl", type=str)
    parser.add_argument("--test_output_file", default="output/test-20250925_1031-neologism2plain-qwen3-8b-rqe-search-tools-enabled-True-sync_with_tools-think_search_translate-xcomet_xl-enable_thinking-True-0-1.0--1-20250927_0211/0.jsonl", type=str)
    
    parser.add_argument("--eval_format", default="amt", type=str, choices=["amt", "other"])
    parser.add_argument("--method", default="point-wise", type=str, choices=["point-wise", "list-wise", "gemba"])
    parser.add_argument("--model_path", default="/lustre/miao/models/openai/gpt-oss-120b", type=str)
    parser.add_argument("--langs", type=str, default="zh-en,ru-en,fr-en,uk-en,ja-en,de-en,cs-en,pl-en")
    
    args = parser.parse_args()
    
    lang_pairs = args.langs.split(",")
    model_name = args.model_path.split("/")[-1]
    
    if args.model_path.startswith("/lustre/miao/models"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype="auto",
        )
    else:
        tokenizer = None
        model = None
        
    # assemble data from test indices.
    max_tries = 3
    
    if args.method == "point-wise":
        if args.eval_format == "amt":
            test_data = load_jsonl(args.test_file_path)
            output_data = load_jsonl(args.test_output_file)
            
            filtered_output_data = [item for item in output_data if item["lp"] in lang_pairs]
            
            assert len(test_data) == len(filtered_output_data)
            
            data = []
            for i, item in enumerate(filtered_output_data):
                input = item["input"]
                source_text = input.split(" text:")[1].split("\nassistant\n")[0].strip()
                output = item["output"]
                candidate = extract_solution(output).strip()
                ref = item["gts"]
                
                extra_info = test_data[i]["extra_info"]
                gloss_meaning = str(extra_info["glosses"])
                neologism = extra_info["src_neologisms"][0]
                
                data.append({
                    "source_text": source_text,
                    "reference_translation": ref,
                    "candidate_translation": candidate,
                    "gloss_meaning": gloss_meaning,
                    "neologism": neologism,
                    "extra_info": extra_info,
                })
                print(f"Data {i}: {data[-1]}")

        elif args.eval_format == "other":
            data = load_jsonl(args.input_path)
            raw_test_set = load_jsonl(args.raw_test_set_path)
        
        for i, item in enumerate(tqdm(data)):
            if args.eval_format == "amt":
                gloss_meaning = item["gloss_meaning"]
                source_text = item["source_text"]
                reference_translation = item["reference_translation"]
                candidate_translation = item["candidate_translation"]
                neologism = item["neologism"]
            elif args.eval_format == "other":
                gloss_meaning = str(raw_test_set[i]["glosses"])
                source_text = item["source_text"]
                reference_translation = item["target_text"]
                candidate_translation = item["output"]
                neologism = item["src_neologisms"][0]
                
            prompt = PROMPT_TEMP_POINT_WISE.format(
                source_sentence=source_text,
                neologism=neologism,
                neologism_meaning=gloss_meaning,
                reference_translation=reference_translation,
                candidate_translation=candidate_translation,
            )
            
            print("Prompt:", prompt)
            for _ in range(max_tries):
                response = gen(
                    prompt=prompt,
                    model_path=args.model_path,
                    tokenizer=tokenizer,
                    model=model,
                )
                print("LLM-as-judge response:", response)
                
                score_str = extract_evaluation(response)
                try:
                    score = float(score_str)
                except Exception as e:
                    print(f"Error parsing score: {score_str}, error: {e}")
                    score = None
                if score is not None:
                    break
            item[f"llm_judge_{args.method}_{model_name}_prompt"] = prompt
            item[f"llm_judge_{args.method}_{model_name}_response"] = response
            item[f"llm_judge_{args.method}_{model_name}"] = score
        
        if args.eval_format == "amt":
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d-%H-%M")
            output_path = args.test_output_file.replace(".jsonl", f"_llm_judge_{args.method}_{model_name}_{str(now)}.jsonl")
        elif args.eval_format == "other":
            output_path = args.input_path.replace(".json", f"_llm_judge_{args.method}_{model_name}.jsonl")
        
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()


